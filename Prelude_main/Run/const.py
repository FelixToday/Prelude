# -*- coding: utf-8 -*-
"""Global constants for dataset specification and data paths.

The repository works with a set of pre-processed datasets (see
``data_process/``). Each dataset is identified by a short name that
maps to the number of tabs, the maximum load time, and the number of
classes used by the WF attack.
"""
import os
import socket

dataset_lib = {
    "CW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'CW', 'num_classes': 95},
    "trafficsilver_bwr_CW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'CW_BWR', 'num_classes': 95},
    "trafficsilver_rb_CW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'CW_RB', 'num_classes': 95},
    "trafficsilver_bd_CW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'CW_BD', 'num_classes': 95},
    "wtfpad_CW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'CW_Pad', 'num_classes': 95},
    "front_CW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'CW_Front', 'num_classes': 95},
    "regulator_CW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'CW_Regula', 'num_classes': 95},
    "tamaraw_CW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'CW_Tamaraw', 'num_classes': 95},

    "OW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'OW', 'num_classes': 96},
    "trafficsilver_bwr_OW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'OW_BWR', 'num_classes': 96},
    "trafficsilver_rb_OW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'OW_RB', 'num_classes': 96},
    "trafficsilver_bd_OW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'OW_BD', 'num_classes': 96},
    "wtfpad_OW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'OW_Pad', 'num_classes': 96},
    "front_OW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'OW_Front', 'num_classes': 96},
    "regulator_OW": {'num_tabs': 1, 'maximum_load_time': 80, 'name': 'OW_Regula', 'num_classes': 96},
}

# Pre-configured absolute dataset paths for specific servers.
# Users can register their own hostname -> dataset path here; otherwise
# ``get_filebase_dir`` falls back to the repository-local ``npz_dataset``.
filebase_dir_dict = {}

machine_name_dict = {}


def get_filebase_dir():
    """Return the directory that contains the ``{dataset}/train|valid|test.npz`` folders.

    Registered servers use their pre-configured path; for any other machine
    we fall back to the repository-local ``npz_dataset/`` directory.
    """
    hostname = socket.gethostname()
    if hostname in filebase_dir_dict:
        return filebase_dir_dict[hostname]
    local_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "npz_dataset"))
    print(f"[const] Host '{hostname}' is not registered, "
          f"using the repository-local dataset path: {local_dir}")
    return local_dir


def get_machine_name():
    hostname = socket.gethostname()
    return machine_name_dict.get(hostname, hostname)