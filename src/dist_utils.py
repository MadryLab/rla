import os
import torch.distributed as dist
import torch as ch

def make_training_dir(outdir, exp_name):
    path = os.path.join(outdir, exp_name)
    os.makedirs(path, exist_ok=True)
    existing_count = -1

    for f in os.listdir(path):
        if f.startswith('version_'):
            version_num = f.split('version_')[1]
            if version_num.isdigit() and existing_count < int(version_num):
                existing_count = int(version_num)
    version_num = existing_count + 1
    new_path = os.path.join(path, f"version_{version_num}")
    print("logging in ", new_path)
    os.makedirs(new_path)
    os.makedirs(os.path.join(new_path, 'checkpoints'))
    return new_path

def exec_wrapper(gpu, address, port, world_size, exec_fn, *args):
    # initialize distributed
    os.environ['MASTER_ADDR'] = address
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=gpu, world_size=world_size)
    ch.cuda.set_device(gpu)

    # run function
    exec_fn(gpu, *args)

    # teardown
    dist.destroy_process_group()

class DistributedManager():
    def __init__(self, world_size, address, port):
        self.distributed = world_size > 1
        self.world_size = world_size
        self.address = address
        self.port = port
    
    def launch_from_args(self, exec_fn, cargs):
        distributed = self.distributed
            
        if distributed:
            spawn_args = [self.address, self.port, self.world_size, exec_fn] + cargs
            ch.multiprocessing.spawn(exec_wrapper, args=spawn_args, 
                                     nprocs=self.world_size,
                                     join=True)
        else:
            exec_fn(0, *cargs)
            