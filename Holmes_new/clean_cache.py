# -*- coding: utf-8 -*-
# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2026/1/12 上午9:42

import os
import argparse
from lxj_utils_sys import str_to_bool
parser = argparse.ArgumentParser(description="Clean Cache")
parser.add_argument('--dataset', default="all",help="清除缓存数据集名称")
parser.add_argument('--clean_aug_data', type=str_to_bool, default="False",help="是否清除生成的数据集")
parser.add_argument('--file_base_dir', default="auto",help="数据集存储路径")
from lxj_holmes_utils import adjust_args
from lxj_utils_sys import parse_args, print_config_info
args, args_help = parse_args(parser, is_print_help=False)
args = adjust_args(args)
print_config_info(args, args_help, sorted_keys=True)

database_dir = args['file_base_dir']
dataset_list = os.listdir(database_dir) if args['dataset'] == "all" else [args['dataset']]

for dataset in dataset_list:

    delete_files = [
        'taf_aug_train.npz',
        'taf_aug_valid.npz',
        'taf_test.npz',
    ]+[f'taf_test_p{int(load_ratio)}.npz' for load_ratio in range(10,100+1,10)]
    if args['clean_aug_data']:
        print("完全清除缓存（aug_train, aug_valid)")
        delete_files = ['aug_train.npz', 'aug_valid.npz'] + delete_files
    else:
        print("保留缓存（aug_train, aug_valid)")


    for file in delete_files:
        if os.path.exists(os.path.join(database_dir, dataset, file)):
            print(f"正在删除文件: -- {dataset} -- {file}")
            os.remove(os.path.join(database_dir, dataset, file))
    print("="*20 + f" {dataset}缓存已清除！" + "="*20)