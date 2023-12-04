""" Utilities for running training and evaluation loops """
import torch
from tqdm import tqdm
import numpy as np
from terminator.utils.common import seq_to_ints

# pylint: disable=no-member


def _to_dev(data_dict, dev):
    """ Push all tensor objects in the dictionary to the given device.

    Args
    ----
    data_dict : dict
        Dictionary of input features to TERMinator
    dev : str
        Device to load tensors onto
    """
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(dev)
        if key == 'gvp_data':
            data_dict['gvp_data'] = [data.to(dev) for data in data_dict['gvp_data']]


def _ld_item_values(ld):
    """ Convert all 0-dim tensors in a loss dictionary into python native types.

    Args
    ----
    ld : dict
        Dictionary with keys :code:`loss_fn_name` and values of dictionaries with
        - :code:`loss` corresponding to the loss value
        - :code:`count` corresponding to the normalizing factor
        - :code:`scaling_factor` corresponding to the scaling coefficient in the loss function

    Returns
    -------
    ld_copy
        Updated loss dictionary with all 0-dim tensors converted to python native types.
    """
    ld_copy = ld.copy()
    for sub_ld in ld_copy.values():
        for key, val in sub_ld.items():
            if isinstance(val, torch.Tensor):
                if len(val.shape) == 0:
                    sub_ld[key] = val.item()
                else:
                    raise RuntimeError("loss dictionary contains non-0-dim tensor value")
    return ld_copy


def _compute_loss(loss_dict):
    """ Compute the total loss given a loss dictionary

    Args
    ----
    loss_dict : dict
        Dictionary with keys :code:`loss_fn_name` and values of dictionaries with
        - :code:`loss` corresponding to the loss value
        - :code:`count` corresponding to the normalizing factor
        - :code:`scaling_factor` corresponding to the scaling coefficient in the loss function

    Returns
    -------
    loss : torch.Tensor of size 1
        Computed loss
    """
    loss = 0
    for subloss_dict in loss_dict.values():
        loss += subloss_dict["loss"] * subloss_dict["scaling_factor"]
    return loss


def _sum_loss_dicts(total_ld, batch_ld):
    """ Add all values in :code:`batch_ld` into the corresponding values in :code:`total_ld`

    Args
    ----
    total_ld, batch_ld : dict
        Dictionary with keys :code:`loss_fn_name` and values of dictionaries with
        - :code:`loss` corresponding to the loss value
        - :code:`count` corresponding to the normalizing factor
        - :code:`scaling_factor` corresponding to the scaling coefficient in the loss function

    Returns
    -------
    combined_ld : dict
        Combined loss dictionary with the same structure as the input dictionaries
    """
    def _weighted_loss_sum(ld1, ld2):
        """ Compute the weighted loss between two loss dictionaries """
        c1, c2 = ld1["count"], ld2["count"]
        return (ld1["loss"] * c1 + ld2["loss"] * c2)/(c1 + c2)
    # combine the two loss dictionaries
    combined_ld = total_ld.copy()
    for loss_fn_name, batch_subld in batch_ld.items():
        if loss_fn_name not in combined_ld.keys():
            combined_ld[loss_fn_name] = batch_subld
        else:
            combined_subld = combined_ld[loss_fn_name]
            assert combined_subld["scaling_factor"] == batch_subld["scaling_factor"]
            combined_subld["loss"] = _weighted_loss_sum(combined_subld, batch_subld)
            combined_subld["count"] += batch_subld["count"]
    return combined_ld

def _log_rank_0(message):
    """ Logs input message if local process rank is 0 or ddp not enabled

    Args
    ----
    message : str
        String containing message to log

    Returns
    -------
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message)
    else:
        print(message)

def run_epoch(model, dataloader, loss_fn, optimizer=None, scheduler=None, grad=False, test=False, dev="cuda:0", isDataParallel=False, finetune=False, wds_dataloader=False):
    """ Run :code:`model` on one epoch of :code:`dataloader`

    Args
    ----
    model : terminator.model.TERMinator.TERMinator
        An instance of TERMinator
    dataloader : torch.utils.data.DataLoader
        A torch DataLoader that wraps either terminator.data.data.TERMDataLoader or terminator.data.data.TERMLazyDataLoader
    loss_fn : function
        Loss function with signature :code:`loss_fn(etab, E_idx, data)` and returns :code`loss, batch_count`,
        where
        - :code:`etab, E_idx` is the outputted Potts Model
        - :code:`data` is the input data produced by :code:`dataloader`
        - :code:`hparams` is the model hyperparameters
        - :code:`loss` is the loss value
        - :code:`batch_count` is the averaging factor
    optimizer : torch optimizer or None
        An optimizer for :code:`model`. Used when :code:`grad=True, test=False`
    scheduler : torch scheduler or None
        The associted scheduler for the given optimizer
    grad : bool
        Whether or not to compute gradients. :code:`True` to train the model, :code:`False` to use model in evaluation mode.
    test : bool
        Whether or not to save the outputs of the model. Requires :code:`grad=False`.
    dev : str, default="cuda:0"
        What device to compute on

    Returns
    -------
    epoch_loss : float
        Loss on the run epoch
    running_loss_dict : dict
        Loss breakdown into component sublosses and scaling factors of epoch_loss
    dump : list of dicts, conditionally present
        Outputs of the model. Present when :code:`test=True`
    """
    # arg checking
    if test:
        assert not grad, "grad should not be on for test set"
    if grad:
        assert optimizer is not None, "require an optimizer if using grads"
    if scheduler is not None:
        assert optimizer is not None, "using a scheduler requires an optimizer"

    running_loss_dict = {}

    # set grads properly
    if grad:
        model.train()
        if finetune: # freeze all but the last output layer
            if isDataParallel: # TODO cleaner way to do this?
                for (name, module) in model.module.named_children():
                    if name == "top":
                        for (n, m) in module.named_children():
                            if n == "W_out":
                                print(f"Reinitializing the weights of layer {n}")
                                torch.nn.init.xavier_uniform_(m)
                                m.requires_grad = True
                            else:
                                m.requires_grad = False
                    else:
                        module.requires_grad = False
            else:
                for (name, module) in model.named_children():
                    if name == "top":
                        for (n, m) in module.named_children():
                            if n == "W_out":
                                m.requires_grad = True
                            else:
                                m.requires_grad = False
                    else:
                        module.requires_grad = False
        else:
            torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    # record inference outputs if necessary
    if test:
        dump = []
    if wds_dataloader:
        dataloader = dataloader.with_length(2*692) ## Approximate number of batches with max_seq_tokens = 6000
    progress = tqdm(total=len(dataloader))
    for data in dataloader:
        # Extract info from batch in readable format if using wds_dataloader
        if wds_dataloader:
            batch = data
            data = {
                    'X': batch[0],
                    'x_mask': batch[1],
                    'chain_idx': batch[2],
                    'seq_lens': batch[3],
                    'seqs': [u['seq'] for u in batch[4]],
                    'ids': [u['pdb'] for u in batch[5]],
                }
            # Convert seqs to ints and pad
            seq_list = []
            for i_data in range(len(data['seqs'])):
                int_seqs = torch.from_numpy(np.array(seq_to_ints(data['seqs'][i_data])))
                seq_list.append(torch.nn.functional.pad(int_seqs, (0, max(batch[3] - len(int_seqs)))))
            data['seqs'] = torch.stack(seq_list)

            
        # a small hack for DataParallel to know which device got which proteins
        data['scatter_idx'] = torch.arange(len(data['seq_lens']))
        _to_dev(data, dev)
        max_seq_len = max(data['seq_lens'].tolist())
        ids = data['ids']

        try:
            etab, E_idx, _ = model(data, max_seq_len)
            batch_loss_dict = loss_fn(etab, E_idx, data)
            loss = _compute_loss(batch_loss_dict)
        except Exception as e:
            print(ids)
            raise e

        if grad:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss_dict = _sum_loss_dicts(running_loss_dict,
                                            _ld_item_values(batch_loss_dict))

        if test:
            n_batch, l, n = etab.shape[:3]
            dump.append({
                'loss': loss,
                'out': etab.view(n_batch, l, n, 20, 20).cpu().numpy(),
                'idx': E_idx.cpu().numpy(),
                'ids': ids
            })

        # term_mask_eff = int((~data['src_key_mask']).sum().item() / data['src_key_mask'].numel() * 100)
        res_mask_eff = int(data['x_mask'].sum().item() / data['x_mask'].numel() * 100)

        # compactify what's printed to stdout
        loss_breakdown = {}
        for loss_fn_name, subloss_dict in running_loss_dict.items():
            loss_breakdown[loss_fn_name] = round(subloss_dict["loss"], 2)
        avg_loss = round(_compute_loss(running_loss_dict), 2)

        progress.update(1)
        progress.refresh()
        progress.set_description_str(f'avg loss {avg_loss} {loss_breakdown} | eff 0.{res_mask_eff}')
        # progress.set_description_str(f'avg loss {avg_loss} {loss_breakdown} | eff 0.{term_mask_eff}, 0.{res_mask_eff}')

    progress.close()
    epoch_loss = _compute_loss(running_loss_dict)

    if scheduler is not None:
        scheduler.step(epoch_loss)

    torch.set_grad_enabled(True)  # just to be safe
    if test:
        return epoch_loss, running_loss_dict, dump
    return epoch_loss, running_loss_dict, None
