import os
import torch
import random
import argparse
import numpy as np
from pytorch_metric_learning import losses
from tqdm import tqdm
import torch.nn.functional as F
from lxj_holmes_utils import measurement

from Holmes_model import Holmes

from lxj_utils_sys import BaseLogger

torch.multiprocessing.set_sharing_strategy('file_system')
# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, default="CW", help="Dataset name")
parser.add_argument("--train_epochs", type=int, default=30, help="Train epochs")
parser.add_argument("--checkpoints", type=str, default="../checkpoints/", help="Location of model checkpoints")
parser.add_argument('--file_base_dir', default="auto", help="数据集存储路径")
parser.add_argument('--valid_name', default="aug_valid", help="验证集")
parser.add_argument('--note', default="test_exp", help="运行保存文件夹")

from lxj_holmes_utils import adjust_args
from lxj_utils_sys import parse_args, print_config_info
args, args_help = parse_args(parser, is_print_help=False)
args = adjust_args(args)
print_config_info(args, args_help, sorted_keys=True)

device = torch.device("cuda")

in_path = os.path.join(args['file_base_dir'], args['dataset'])
ckp_path = os.path.join(args['checkpoints'], args['dataset'], "Holmes", args['note'])
os.makedirs(ckp_path, exist_ok=True)

out_file = os.path.join(ckp_path, f"max_f1.pth")
logger = BaseLogger(json_save_path=os.path.join(ckp_path, "result.json"),
                        log_save_path=os.path.join(ckp_path, "log.txt"))

from lxj_holmes_utils import TrafficDataset

train_dataset = TrafficDataset(os.path.join(in_path, f"aug_train.npz"), drop_extra_time=True, load_time=80, load_ratio=100)
valid_dataset = TrafficDataset(os.path.join(in_path, f"{args['valid_name']}.npz"), drop_extra_time=True, load_time=80, load_ratio=100)
num_classes = len(np.unique(train_dataset.y))
print(f"num_classes: {num_classes}")
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True, drop_last=True, num_workers=2)
valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=50, shuffle=False, drop_last=False, num_workers=2)

model = Holmes(num_classes)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

criterion = losses.SupConLoss(temperature=0.1)


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)

    return pred_labels


def knn_monitor(net, device, memory_data_loader, test_data_loader, num_classes, k=200, t=0.1):
    net.eval()
    total_num = 0
    feature_bank, feature_labels = [], []
    y_pred = []
    y_true = []

    with torch.no_grad():
        # Generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().to(device)
        feature_labels = torch.cat(feature_labels, dim=0).t().contiguous().to(device)

        # Loop through test data to predict the label by weighted kNN search
        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, k, t)
            total_num += data.size(0)
            y_pred.append(pred_labels[:, 0].cpu().numpy())
            y_true.append(target.cpu().numpy())

    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()

    return y_true, y_pred



metric_best_value = 0
for epoch in range(args['train_epochs']):
    model.train()
    sum_loss = 0
    sum_count = 0

    for cur_data in tqdm(train_iter):
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
        optimizer.zero_grad()
        outs = model(cur_X)

        loss = criterion(outs, cur_y)

        loss.backward()
        optimizer.step()
        sum_loss += loss.data.cpu().numpy() * outs.shape[0]
        sum_count += outs.shape[0]

    train_loss = round(sum_loss / sum_count, 3)
    print(f"epoch {epoch}: train_loss = {train_loss}")

    valid_true, valid_pred = knn_monitor(model, device, train_iter, valid_iter, num_classes, 10)

    valid_result = measurement(valid_true, valid_pred)
    print(f"{epoch}: {valid_result}")
    logger.log("valid.result", valid_result, unzip_dict=True)
    if valid_result["F1-score"] > metric_best_value:
        metric_best_value = valid_result["F1-score"]
        torch.save(model.state_dict(), out_file)
