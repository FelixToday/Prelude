# -*- coding: utf-8 -*-

# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2025/12/11 下午4:32


import warnings
import os
import torch
from lxj_utils_sys import BaseLogger, ModelCheckpoint, same_seed, BaseLogger_v2
from lxj_utils_sys import print_colored, print_config_info
from utils_dataset_metric import get_model_and_dataloader
from utils_porcess import *

warnings.filterwarnings("ignore")
fix_seed = 2025
same_seed(fix_seed)


def load_model(args, config, device, valid_X, valid_y, test_X, test_y, num_classes):
    """加载模型"""
    ckp_path = os.path.join("../../checkpoints", args['dataset'], config['model'], args['note']).rstrip('/')
    mode = 'max'

    # 修改点：移除 num_tabs 的判断逻辑，固定使用 f1
    metric_name = "f1"

    modelsaver = ModelCheckpoint(filename=os.path.join(ckp_path, f"model.pth"),
                                 mode=mode, metric_name=metric_name)

    # 获取模型结构
    model, _, _ = get_model_and_dataloader(valid_X, valid_y, test_X, test_y, num_classes, config, args)
    model = modelsaver.load(model, device)[0]
    return model


def main():
    parser = get_parser("test")
    args, args_help = parse_args(parser, is_print_help=True)
    config = load_config_file(args['config'])
    args, config = adjust_system_args(args, config)

    # 打印配置（args 中仍包含 num_tabs，但不再影响逻辑分支）
    print_config_info({**config, **args}, {**param_description, **args_help})

    # 设备设置
    device = torch.device(args['device'])

    # 路径设置
    ckp_path = os.path.join(args['checkpoint_path'], args['dataset'], config['model'], args['note']).rstrip('/')
    # test_path = os.path.join(str(ckp_path), "test", f"test_p{args['load_ratio']:.2f}")
    # print_colored(f"保存位置：{test_path}", 'red')
    #
    # # 日志初始化
    # logger = BaseLogger(json_save_path=os.path.join(test_path, "result.json"),
    #                     log_save_path=os.path.join(test_path, "log.txt"))
    test_path = os.path.join(ckp_path, "test")
    print_colored(f"保存位置：{test_path}", 'red')
    print_colored(f"加载比例：{args['load_ratio']} %", 'red')
    logger_v2 = BaseLogger_v2(json_path=os.path.join(test_path, "result.json"),
                              log_path=os.path.join(test_path, "log.txt"))
    if args['load_ratio'] != 10:
        logger_v2.load()

    # 数据加载
    # dataname = ['valid', 'test']
    # valid_X, valid_y, test_X, test_y = load_dataset_data(args, dataname)

    # 保持原有的测试集加载逻辑
    test_X, test_y = load_data(os.path.join(args['file_base_dir'], args['dataset'], f"test.npz"),
                               drop_extra_time=True, load_time=80)
    valid_X, valid_y = test_X, test_y

    # 类别数确定
    num_classes = dataset_lib[args['dataset']]['num_classes']

    # 模型和数据加载器
    model, val_loader, test_loader = get_model_and_dataloader(valid_X, valid_y,
                                                              test_X, test_y,
                                                              num_classes, config, args)

    # 加载模型
    mode = 'max'
    # 修改点：固定使用 f1
    metric_name = "f1"

    modelsaver = ModelCheckpoint(filename=os.path.join(ckp_path, f"model.pth"),
                                 mode=mode, metric_name=metric_name)
    model = modelsaver.load(model, device)[0]

    # 运行测试
    result, main_metric = evaluate_one_epoch(model, test_loader, device, config, args, num_classes)

    # 保存结果
    # logger.log('test.metrics', result)
    # logger.info("Test metrics: %s", result, is_logfile=True)
    logger_v2.record("load_ratio", args['load_ratio'])
    logger_v2.record("metrics", result, unpack_dict=True)
    logger_v2.print("Test metrics: %s", result, save_to_file=True)

    # PR AUC计算
    if args.get('is_pr_auc'):
        from lxj_utils_sys import compute_pr_result
        # 修改点：task_type 从 "multilabel" 改为 "multiclass" 或根据你的单标签业务调整
        pr_auc_result = compute_pr_result(dataloader=test_loader,
                                          model=model,
                                          task_type="multilabel",
                                          average="macro")  # 通常单标签推荐用 macro
        #logger.log('test.pr_auc', pr_auc_result, unzip_dict=True)
        logger_v2.record(f"pr_auc_{int(args['load_ratio'])}", pr_auc_result, unpack_dict=True)
    print("\n\n" + "=" * 20 + " end " + "=" * 20 + "\n")


if __name__ == "__main__":
    main()
    print_colored("test 全部运行结束", "green")