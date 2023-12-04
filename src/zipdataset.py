import webdataset.filters as filters
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Set, Tuple

import random, re, tarfile

import braceexpand
import torch

from webdataset import filters, gopen
from webdataset.handlers import reraise_exception
import webdataset as wds
from webdataset.filters import reraise_exception
from webdataset.tariterators import base_plus_ext, tar_file_expander, url_opener, meta_prefix, meta_suffix, valid_sample
import zipfile
from zipfile import ZipFile
import numpy as np

trace = True

def group_by_keys(
    data: Iterable[Dict[str, Any]],
    keys: Callable[[str], Tuple[str, str]] = base_plus_ext,
    lcase: bool = True,
    suffixes: Optional[Set[str]] = None,
    handler: Callable[[Exception], bool] = reraise_exception,
) -> Iterator[Dict[str, Any]]:
    """Group tarfile contents by keys and yield samples.
    Args:
        data: iterator over tarfile contents
        keys: function that takes a file name and returns a key and a suffix.
        lcase: whether to lowercase the suffix.
        suffixes: list of suffixes to keep.
        handler: exception handler.
    Raises:
        ValueError: raised if there are duplicate file names in the tar file.
    Yields:
        iterator over samples.
    """
    current_sample = None
    count = 0
    for filesample in data:
        try:
            assert isinstance(filesample, dict)
            fname, value = filesample["fname"], filesample["data"]
            prefix, suffix = keys(fname)
            yield {
                "__key__": str(count),
                suffix: value
            }
            count += 1
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break
    if valid_sample(current_sample):
        yield current_sample


def custom_zip_file_iterator(
    fileobj: zipfile.ZipFile,
    stream_order,
    format_string,
    skip_meta: Optional[str] = r"__[^/]*__($|/)",
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.
    Args:
        fileobj: the tar file stream.
        skip_meta: regexp for keys that are skipped entirely. Defaults to r"__[^/]*__($|/)".
        handler: exception handler. Defaults to reraise_exception.
        select: predicate for selecting files. Defaults to None.
    Yields:
        a stream of samples.
    """
    for index in stream_order:
        fname = format_string.format(index,)
        try:
            if fname is None:
                continue
            if (
                "/" not in fname
                and fname.startswith(meta_prefix)
                and fname.endswith(meta_suffix)
            ):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, fname):
                continue
            if rename_files:
                fname = rename_files(fname)
            if select_files is not None and not select_files(fname):
                continue
            
            with fileobj.open(fname, mode='r') as sample_file:
                data = sample_file.read()
            result = dict(fname=fname, data=data)
            yield result
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    fileobj.close()


def custom_tar_file_expander(
    data: Iterable[Dict[str, Any]],
    stream_order,
    format_string,
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Expand tar files.
    Args:
        data: iterator over opened tar file streams.
        handler: exception handler.
        select_files: select files from tarfiles by name (permits skipping files).
    Yields:
        a stream of samples.
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in custom_zip_file_iterator(
                source["stream"],
                handler=handler,
                select_files=select_files,
                rename_files=rename_files,
                stream_order=stream_order,
                format_string=format_string,
            ):
                assert (
                    isinstance(sample, dict) and "data" in sample and "fname" in sample
                )
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


                
def zip_opener(
    data: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    **kw: Dict[str, Any],
):
    """Open URLs and yield a stream of url+stream pairs.
    Args:
        data: iterator over dict(url=...)
        handler: exception handler.
        kw: keyword arguments for gopen.gopen.
    Yields:
        a stream of url+stream pairs.
    """
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        try:
            stream = ZipFile(url)
            sample.update(stream=stream)
            yield sample
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break

def tarfile_samples(
    src: Iterable[Dict[str, Any]],
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
    stream_order_args=None,
    format_string=None,
) -> Iterable[Dict[str, Any]]:
    """Given a stream of tar files, yield samples.
    Args:
        src: stream of tar files
        handler: exception handler
        select_files: function that selects files to be included
    Returns:
        stream of samples
    """
    streams = zip_opener(src, handler=handler)
    
    if stream_order_args is not None:
        stream_order = get_stream_order(**stream_order_args)
        assert format_string is not None
        files = custom_tar_file_expander(
            streams, handler=handler, select_files=select_files, rename_files=rename_files,
            stream_order=stream_order, format_string=format_string
        )
    else:
        files = tar_file_expander(
            streams, handler=handler, select_files=select_files, rename_files=rename_files,
        )
    samples = group_by_keys(files, handler=handler)
    return samples


custom_tarfile_to_samples = filters.pipelinefilter(tarfile_samples)

def get_zipdataset(path, format_string, stream_order_args):
    cols = ['inp.pyd']
    wd_ds = wds.WebDataset(path).decode().to_tuple(*cols)
    wd_ds.pipeline[3] = custom_tarfile_to_samples(handler=reraise_exception, stream_order_args=stream_order_args, 
                                                  format_string=format_string)
    return wd_ds

def get_stream_order(cath_path, num_total_steps, batch_size, rng, dist_args=None):
    cath_info_dict = torch.load(cath_path)

    cath_dict_to_pdb = cath_info_dict['cath_dict_to_pdb']
    cath_dict_to_index = cath_info_dict['cath_dict_to_index']

    clusters = list(cath_dict_to_pdb.keys())
    lengths = np.array([len(cath_dict_to_pdb[c]['train']) for c in clusters])
    cluster_weights = lengths/sum(lengths)
    cluster_picks = rng.choice(np.arange(len(clusters)), size=num_total_steps, p=cluster_weights)

    order = []
    for c in cluster_picks:
        cluster = clusters[c]
        members = cath_dict_to_index[cluster]
        if dist_args is None:
            batch = rng.choice(members, replace=False, size=batch_size)
        else:
            batch = rng.choice(members, replace=False, size=batch_size*dist_args['world_size'])
            batch = batch[::dist_args['rank']]
        order.append(batch)
    order = np.concatenate(order)
    return order

def get_clip_webdataset(path, format_string, cath_path, num_total_steps, batch_size, rng, dist_args=None):
    stream_order_args = {
        'cath_path': cath_path, 'num_total_steps': num_total_steps, 'batch_size': batch_size, 'rng': rng, 
        'dist_args': dist_args
    }
    return get_zipdataset(path, format_string, stream_order_args)