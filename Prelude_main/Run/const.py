# -*- coding: utf-8 -*-

# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2025/12/1 下午4:50
dataset_lib = {
    "CW":{'num_tabs':1,'maximum_load_time': 80,'name':'CW', 'num_classes': 95},
    "trafficsilver_bwr_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'CW_BWR'},
    "trafficsilver_rb_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'CW_RB', 'num_classes': 95},
    "trafficsilver_bd_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'CW_BD'},
    "wtfpad_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'CW_Pad', 'num_classes': 95},
    "front_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'CW_Front', 'num_classes': 95},
    "regulator_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'CW_Regula', 'num_classes': 95},
    "tamaraw_CW":{'num_tabs':1,'maximum_load_time': 80,'name':'CW_Tamaraw'},

    "OW":{'num_tabs':1,'maximum_load_time': 80,'name':'OW', 'num_classes': 96},
    "trafficsilver_bwr_OW":{'num_tabs':1,'maximum_load_time': 80,'name':'OW_BWR'},
    "trafficsilver_rb_OW":{'num_tabs':1,'maximum_load_time': 80,'name':'OW_RB'},
    "trafficsilver_bd_OW":{'num_tabs':1,'maximum_load_time': 80,'name':'OW_BD'},
    "wtfpad_OW":{'num_tabs':1,'maximum_load_time': 80,'name':'OW_Pad'},
    "front_OW":{'num_tabs':1,'maximum_load_time': 80,'name':'OW_Front'},
    "regulator_OW":{'num_tabs':1,'maximum_load_time': 80,'name':'OW_Regula'},


    "Closed_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'CW_2tab'},
    "Closed_3tab":{'num_tabs':3,'maximum_load_time': 120,'name':'CW_3tab'},
    "Closed_4tab":{'num_tabs':4,'maximum_load_time': 120,'name':'CW_4tab'},
    "Closed_5tab":{'num_tabs':5,'maximum_load_time': 120,'name':'CW_5tab'},
    "wtfpad_Closed_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'CW_Pad'},
    "front_Closed_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'CW_Fro'},
    "regulator_Closed_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'CW_Reg'},
    "Open_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'OW_2tab'},
    "Open_3tab":{'num_tabs':3,'maximum_load_time': 120,'name':'OW_3tab'},
    "Open_4tab":{'num_tabs':4,'maximum_load_time': 120,'name':'OW_4tab'},
    "Open_5tab":{'num_tabs':5,'maximum_load_time': 120,'name':'OW_5tab'},
    "wtfpad_Open_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'OW_Pad'},
    "front_Open_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'OW_Fro'},
    "regulator_Open_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'OW_Reg'},
    # "":{'num_tabs':,'maximum_load_time': ,'name':''},
}

valid_note = {
    "ExploreModel":
        [
            'baseline_same',
            'baseline_optim',
            'baseline_same_ablation_no_no',
            'baseline_same_ablation_SWIFT_no',
            'baseline_same_ablation_no_LINA',
            'baseline_same_ablation_SWIFT_no',
        ],
    "other":
        [
            'baseline_same'
        ]
}

filebase_dir_dict = {
    "2e83997a5bca":"/data/lixianjun/dataset/wfa/npz_dataset",
    "2db02d80faff":"/home/lixianjun/work_dir/dataset/wfa/npz_dataset",
    "autodl-container-060c4ba85a-c8e4a13b":"/root/autodl-tmp/lixianjun/dataset/wfa/npz_dataset",
    "autodl-container-15ef4cab54-a53eb79e":"/root/autodl-tmp/lixianjun/dataset/npz_dataset"
}
machine_name_dict = {
    "2e83997a5bca":"S92-2e83997a5bca",
    "2db02d80faff":"S94-2db02d80faff",
    "autodl-container-060c4ba85a-c8e4a13b":"AliMe",
    "autodl-container-15ef4cab54-a53eb79e":"AliHuo"
}


def get_filebase_dir():
    import socket
    return filebase_dir_dict[socket.gethostname()]
def get_machine_name():
    import socket
    return machine_name_dict[socket.gethostname()]