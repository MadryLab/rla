import argparse
from types import SimpleNamespace

import yaml
from fastargs import get_current_config

def process_args_and_config():
    config = get_current_config()
    parser = argparse.ArgumentParser(description="Feature Datamodels")
    config.augment_argparse(parser)
    config.validate(mode='stderr')
    config.summary()

    config_args = config.get()

    args = convert_fastargs(config_args)
    args = SimpleNamespace(**args)
    
    return args

def convert_fastargs(fast_args):
    args_dict = {}
    fast_args_vars = vars(fast_args)

    for key, value in fast_args_vars.items():
        if isinstance(value, SimpleNamespace):
            args_dict.update(convert_fastargs(value))
        else:
            args_dict[key] = value

    return args_dict

def convert_arg_to_list(args, keys):
    for key in keys:
        if getattr(args, key) == "":
            setattr(args, key, [])
        else:
            val_list = [int(x) for x in getattr(args, key).split(",")]
            setattr(args, key, val_list)

def convert_emptystr_to_None(args, keys):
    for key in keys:
        if getattr(args, key) == "":
            setattr(args, key, None)

def read_yaml(yaml_dir):
    with open(yaml_dir, "r") as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_file
