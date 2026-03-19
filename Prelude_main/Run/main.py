import time
import os
import torch
from tqdm import tqdm
from pytorch_metric_learning import miners, losses
import warnings

# 假设这些是从你的自定义库中导入的
from lxj_utils_sys import BaseLogger, ModelCheckpoint, same_seed, LearningRateScheduler, print_dict
from lxj_utils_sys import IncrementalMeanCalculator
from lxj_utils_sys import print_colored, print_config_info
from utils_dataset_metric import get_model_and_dataloader
from utils_porcess import *

warnings.filterwarnings("ignore")
fix_seed = 2025
same_seed(fix_seed)


def get_criterion(config, args, num_classes):
    """根据模型和任务类型获取损失函数"""
    # 保持 Metric Learning 的逻辑
    if config['model'] == "TF":
        return losses.TripletMarginLoss(margin=0.1), miners.TripletMarginMiner(margin=0.1, type_of_triplets="semihard")

    # 移除原有的 num_tabs > 1 的 MultiLabelSoftMarginLoss 判断
    # 统一走单标签分类逻辑（CrossEntropy 或 LabelSmoothing）
    if args.get('optim'):
        from timm.loss import LabelSmoothingCrossEntropy
        return LabelSmoothingCrossEntropy(smoothing=0.1), None
    else:
        return torch.nn.CrossEntropyLoss(), None


def train_one_epoch(model, train_loader, criterion, optimizer, device, config, args, epoch, scheduler=None, miner=None):
    """执行单个训练轮次"""
    model.train()
    sum_loss = 0
    sum_count = 0

    for index, cur_data in enumerate(tqdm(train_loader)):
        if args.get('optim') and scheduler:
            # 兼容 timm 风格的 scheduler step
            scheduler.step(epoch + index / len(train_loader))

        optimizer.zero_grad()

        # 数据准备
        cur_X, cur_y, idx = prepare_batch_data(cur_data, device)
        if not args.get('use_idx'):
            idx = None

        # 前向传播
        outs = model_forward(model, config, cur_X, idx)

        # 损失计算
        if config['model'] == "TF":
            hard_pairs = miner(outs, cur_y)
            loss = criterion(outs, cur_y, hard_pairs)
        elif config['model'] in ["CountMamba", "Prelude"] and config.get('early_stage'):
            # 特殊模型的 Masked Augmentation 逻辑
            N, L, D = outs.shape
            noise = torch.rand(N, L, device=outs.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :config['num_aug']]
            outs_masked = torch.gather(outs, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # 这里的 cur_y 扩充逻辑保持不变
            cur_y_expanded = cur_y.unsqueeze(1).repeat(1, config['num_aug'])
            loss = criterion(outs_masked.contiguous().view(-1, D), cur_y_expanded.contiguous().view(-1))
        else:
            loss = criterion(outs, cur_y)

        if torch.isnan(loss):
            print("Warning: loss is nan")

        # 反向传播
        loss.backward()
        optimizer.step()

        sum_loss += loss.item() * outs.shape[0]
        sum_count += outs.shape[0]

    return round(sum_loss / sum_count, 4)


def main():
    parser = get_parser("train")
    args, args_help = parse_args(parser, is_print_help=False)
    config = load_config_file(args['config'])
    args, config = adjust_system_args(args, config)

    # 打印配置信息，num_tabs 依然会在 config/args 中被打印
    print_config_info({**config, **args}, {**param_description, **args_help})

    device = torch.device(args['device'])
    ckp_path = os.path.join(args['checkpoint_path'], args['dataset'], config['model'], args['note']).rstrip('/')
    os.makedirs(ckp_path, exist_ok=True)

    logger = BaseLogger(json_save_path=os.path.join(ckp_path, "result.json"),
                        log_save_path=os.path.join(ckp_path, "log.txt"))

    dataname = ['train', args['valid_name']] if not args['test_flag'] else ['valid', 'test']
    train_X, train_y, valid_X, valid_y = load_dataset_data(args, dataname)

    num_classes = dataset_lib[args['dataset']]['num_classes']

    model, train_loader, val_loader = get_model_and_dataloader(train_X, train_y,
                                                               valid_X, valid_y,
                                                               num_classes, config, args)

    logger.log('config.config', config, True)
    logger.log('config.args', args, True)

    model.to(device)
    optimizer = eval(f"torch.optim.{config['optimizer']}")(model.parameters(), lr=float(config['learning_rate']))
    logger.log("config.model", str(model.__class__))

    criterion, miner = get_criterion(config, args, num_classes)

    scheduler = None
    if args.get('optim'):
        scheduler = LearningRateScheduler(optimizer, lr=float(config['learning_rate']),
                                          min_lr=args['min_lr'], warmup_epochs=args['warmup_epochs'],
                                          total_epochs=args['train_epochs'])

    # 修改点：固定使用 f1 作为主评价指标，不再判断 num_tabs
    mode = 'max'
    metric_name = "f1"
    modelsaver = ModelCheckpoint(filename=os.path.join(ckp_path, f"model.pth"),
                                 mode=mode, metric_name=metric_name,
                                 max_stagnation_epochs=args['stag_epochs'] if args.get('optim') else None)

    train_timmer = IncrementalMeanCalculator()
    valid_timmer = IncrementalMeanCalculator()
    metric_best_value = 0

    logger.info("\n\n" + "-" * 20 + " start " + "-" * 20 + "\n", is_logfile=True)

    for epoch in range(args['train_epochs']):
        print_colored(f"Epoch {epoch + 1}/{args['train_epochs']}", "blue")

        # 训练
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config, args, epoch, scheduler,
                                     miner)
        logger.log("train.loss", train_loss)
        train_timmer.add(time.time() - start_time)

        # 验证
        start_time = time.time()
        valid_result, main_metric = evaluate_one_epoch(model, val_loader, device, config, args, num_classes)

        logger.log("valid.result", valid_result, unzip_dict=True)
        logger.info(f"Epoch {epoch + 1}: train_loss = {train_loss}, {metric_name} = {main_metric}", is_logfile=True)

        # 保存模型
        should_stop = modelsaver.save(main_metric, model, epoch + 1, final=(epoch + 1) == args['train_epochs'])
        if main_metric > metric_best_value:
            metric_best_value = main_metric

        valid_timmer.add(time.time() - start_time)

        # 打印日志
        log_str = f"epoch {epoch + 1}: time.train = {train_timmer.get():.2f}s, time.valid = {valid_timmer.get():.2f}s, best_{metric_name} = {metric_best_value:.4f}"
        print(log_str)
        if should_stop:
            print_colored("训练早停：达到最大停滞epoch次数", "yellow")
            break
    logger.info("\n\n" + "=" * 20 + " end " + "=" * 20 + "\n", is_logfile=True)
if __name__ == "__main__":
    main()
    print_colored("train 全部运行结束", "green")