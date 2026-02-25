import os
import ast
import sys
import shutil
import glob
import functools
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset.dataset import SimpleIterDataset
from src.utils.import_tools import import_module
from src.dataset.functions_graph import graph_batch_func

def set_gpus(args):
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]
        dev = torch.device(gpus[0])
        print("Using GPUs:", gpus)
    else:
        print("No GPUs flag provided - Setting GPUs to [0]")
        gpus = [0]
        dev = torch.device(gpus[0])
        raise Exception("Please provide GPU number")
    return gpus, dev



def get_gpu_dev(args):
    if args.gpus != "":
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = 0
        devices = 0
    return accelerator, devices
# TODO change this to use it from config file

def model_setup(args, data_config):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module
    """
    network_module = import_module(args.network_config, name="_network_module")
   
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]  # ?
        dev = torch.device(gpus[0])
        print("using GPUs:", gpus)
    else:
        gpus = None
        local_rank = 0
        dev = torch.device("cpu")
    model, model_info = network_module.get_model(
        data_config, args=args, dev=dev
    )
    return model.mod


def get_samples_steps_per_epoch(args):
    if args.samples_per_epoch is not None:
        if args.steps_per_epoch is None:
            args.steps_per_epoch = args.samples_per_epoch // args.batch_size
        else:
            raise RuntimeError(
                "Please use either `--steps-per-epoch` or `--samples-per-epoch`, but not both!"
            )
    if args.samples_per_epoch_val is not None:
        if args.steps_per_epoch_val is None:
            args.steps_per_epoch_val = args.samples_per_epoch_val // args.batch_size
        else:
            raise RuntimeError(
                "Please use either `--steps-per-epoch-val` or `--samples-per-epoch-val`, but not both!"
            )
    if args.steps_per_epoch_val is None and args.steps_per_epoch is not None:
        args.steps_per_epoch_val = round(
            args.steps_per_epoch * (1 - args.train_val_split) / args.train_val_split
        )
    if args.steps_per_epoch_val is not None and args.steps_per_epoch_val < 0:
        args.steps_per_epoch_val = None
    return args

def to_filelist(args, mode="train"):
    if mode == "train":
        flist = args.data_train
    elif mode == "val":
        flist = args.data_val
    else:
        raise NotImplementedError("Invalid mode %s" % mode)

    # keyword-based: 'a:/path/to/a b:/path/to/b'
    file_dict = {}
    for f in flist:
        if ":" in f:
            name, fp = f.split(":")
        else:
            name, fp = "_", f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    if args.local_rank is not None:
        if mode == "train":
            gpus_list, _ = set_gpus(args)
            local_world_size = len(gpus_list)  # int(os.environ['LOCAL_WORLD_SIZE'])
            new_file_dict = {}
            for name, files in file_dict.items():
                new_files = files[args.local_rank :: local_world_size]
                assert len(new_files) > 0
                np.random.shuffle(new_files)
                new_file_dict[name] = new_files
            file_dict = new_file_dict
            print(args.local_rank, len(file_dict["_"]))

    
    filelist = sum(file_dict.values(), [])
    assert len(filelist) == len(set(filelist))
    return file_dict, filelist


def train_load(args):
    """
    Loads the training data.
    :param args:
    :return: train_loader, val_loader, data_config, train_inputs
    """
    train_file_dict, train_files = to_filelist(args, "train")
    if args.data_val:
        val_file_dict, val_files = to_filelist(args, "val")
        train_range = val_range = (0, 1)
    else:
        val_file_dict, val_files = train_file_dict, train_files
        train_range = (0, args.train_val_split)
        val_range = (args.train_val_split, 1)
    

    
    train_data = SimpleIterDataset(
        train_file_dict,
        args.data_config,
        for_training=True,
        extra_selection=None,
        remake_weights=False,
        load_range_and_fraction=(train_range, args.data_fraction),
        file_fraction=args.file_fraction,
        fetch_by_files=args.fetch_by_files,
        fetch_step=args.fetch_step,
        infinity_mode=args.steps_per_epoch is not None,
        name="train" + ("" if args.local_rank is None else "_rank%d" % args.local_rank),
        args_parse=args
    )
    val_data = SimpleIterDataset(
        val_file_dict,
        args.data_config,
        for_training=True,
        extra_selection=None,
        load_range_and_fraction=(val_range, args.data_fraction),
        file_fraction=args.file_fraction,
        fetch_by_files=args.fetch_by_files,
        fetch_step=args.fetch_step,
        infinity_mode=args.steps_per_epoch_val is not None,
        name="val" + ("" if args.local_rank is None else "_rank%d" % args.local_rank),
        args_parse=args
    )

    collator_func = graph_batch_func
    # train_data_arg = train_data
    # val_data_arg = val_data
    # if args.train_cap == 1:
    #    train_data_arg = [next(iter(train_data_arg))]
    # if args.val_cap == 1:
    #    val_data_arg = [next(iter(val_data_arg))]
    prefetch_factor = None
    if args.num_workers > 0:
        prefetch_factor = args.prefetch_factor
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=min(args.num_workers, int(len(train_files) * args.file_fraction)),
        collate_fn=collator_func,
        persistent_workers=False,
        prefetch_factor=prefetch_factor
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        collate_fn=collator_func,
        num_workers=min(args.num_workers, int(len(val_files) * args.file_fraction)),
        persistent_workers=args.num_workers > 0
        and args.steps_per_epoch_val is not None,
        prefetch_factor=prefetch_factor
    )

    data_config = 0 #train_data.config
    train_input_names = 0 #train_data.config.input_names
    train_label_names = 0  # train_data.config.label_names

    return train_loader, val_loader, data_config, train_input_names


def test_load(args):
    """
    Loads the test data.
    :param args:
    :return: test_loaders, data_config
    """
    # keyword-based --data-test: 'a:/path/to/a b:/path/to/b'
    # split --data-test: 'a%10:/path/to/a/*'
    file_dict = {}
    split_dict = {}
    for f in args.data_test:
        if ":" in f:
            name, fp = f.split(":")
            if "%" in name:
                name, split = name.split("%")
                split_dict[name] = int(split)
        else:
            name, fp = "", f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    # apply splitting
    for name, split in split_dict.items():
        files = file_dict.pop(name)
        for i in range((len(files) + split - 1) // split):
            file_dict[f"{name}_{i}"] = files[i * split : (i + 1) * split]

    def get_test_loader(name):
        filelist = file_dict[name]
        num_workers = min(args.num_workers, len(filelist))
        test_data = SimpleIterDataset(
            {name: filelist},
            args.data_config,
            for_training=False,
            extra_selection=None,
            load_range_and_fraction=((0, 1), args.data_fraction),
            fetch_by_files=True,
            fetch_step=1,
            name="test_" + name,
            args_parse=args
        )
        test_loader = DataLoader(
            test_data,
            num_workers=num_workers,
            batch_size=args.batch_size,
            drop_last=False,
            pin_memory=True,
            collate_fn=graph_batch_func,
        )
        return test_loader

    test_loaders = {
        name: functools.partial(get_test_loader, name) for name in file_dict
    }
    #data_config = SimpleIterDataset({}, args.data_config, for_training=False).config
    data_config = 0
    return test_loaders, data_config


def count_parameters(model):
    return sum(p.numel() for p in model.mod.parameters() if p.requires_grad)


