import math
import numpy as np
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tqdm
import torch


def gen_one_hot(arr, num_classes):
    binary = np.zeros((arr.shape[0], num_classes))
    for i in range(arr.shape[0]):
        binary[i, arr[i]] = 1

    return binary


def compute_metric(y_true_fine, y_pred_fine):
    y_true_fine = y_true_fine.reshape(-1, y_true_fine.shape[-1])
    y_pred_fine = y_pred_fine.reshape(-1, y_pred_fine.shape[-1])

    num_classes = np.max(y_true_fine) + 1
    y_true_fine = gen_one_hot(y_true_fine, num_classes)
    y_pred_fine = gen_one_hot(y_pred_fine, num_classes)

    result = measurement(y_true_fine, y_pred_fine, eval_metrics="Accuracy Precision Recall F1-score")
    return result


def process_addition_class(one_hot_tensor):
    num_ones = one_hot_tensor.sum(dim=2)

    only_one = (num_ones == 1).float()
    one_hot_tensor[..., -2] = only_one

    no_ones = (num_ones == 0).float()
    one_hot_tensor[..., -1] = no_ones
    one_hot_tensor[..., -2] += no_ones

    return one_hot_tensor


def process_one_hot(BAPM, num_classes, num_patches):
    part_length = BAPM.shape[-1] // num_patches
    one_hot = torch.zeros((BAPM.shape[0], num_patches, num_classes + 2), dtype=torch.float32)
    for i in range(num_patches):
        start_idx = i * part_length
        end_idx = start_idx + part_length

        current_part = BAPM[:, :, start_idx:end_idx]
        for cls in range(num_classes):
            cls_out = (current_part == cls).any(dim=1).float().any(dim=1).float()
            one_hot[:, i, cls] = cls_out

    one_hot = process_addition_class(one_hot)
    return one_hot


def process_BAPM(BAPM, num_classes):
    neg_ones_mask = (BAPM == -1)
    neg_ones_count = neg_ones_mask.sum(dim=1)
    BAPM[neg_ones_mask & (neg_ones_count.unsqueeze(1) == 1)] = num_classes
    indices_double_neg = neg_ones_mask & (neg_ones_count.unsqueeze(1) == 2)
    for b in range(BAPM.shape[0]):
        if indices_double_neg[b].any():
            BAPM[b, 0, indices_double_neg[b, 0]] = num_classes
            BAPM[b, 1, indices_double_neg[b, 1]] = num_classes + 1

    return BAPM


def measurement(y_true, y_pred, eval_metrics):
    eval_metrics = eval_metrics.split(" ")
    results = {}
    for eval_metric in eval_metrics:
        if eval_metric == "Accuracy":
            results[eval_metric] = round(accuracy_score(y_true, y_pred) * 100, 2)
        elif eval_metric == "Precision":
            results[eval_metric] = round(precision_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "Recall":
            results[eval_metric] = round(recall_score(y_true, y_pred, average="macro") * 100, 2)
        elif eval_metric == "F1-score":
            results[eval_metric] = round(f1_score(y_true, y_pred, average="macro") * 100, 2)
        else:
            raise ValueError(f"Metric {eval_metric} is not matched.")
    return results


def evaluate(model, loader, num_tabs, num_classes, fine_predict):
    if num_tabs > 1:
        y_pred_score = np.zeros((0, num_classes))
        y_true = np.zeros((0, num_classes))
        if fine_predict:
            y_pred_fine = []
            y_true_fine = []

        with torch.no_grad():
            model.eval()
            for index, cur_data in enumerate(tqdm.tqdm(loader)):
                cur_X, cur_y = cur_data[0][0].cuda(), cur_data[1].cuda()
                idx = cur_data[0][1].cuda()

                if fine_predict:
                    outs, outs_fine, _ = model(cur_X, idx)

                    BAPM = cur_data[0][2]
                    BAPM = process_BAPM(BAPM, num_classes - 1)
                    one_hot = process_one_hot(BAPM, num_classes - 1, model.num_patches)
                    one_hot = one_hot.cuda()

                    fine_pred = torch.argsort(outs_fine, dim=-1)[:, :, -2:]
                    fine_label = torch.tensor([torch.nonzero(sample).squeeze().tolist() for sample in one_hot.view(-1, one_hot.shape[-1])])
                    fine_label = fine_label.view(-1, model.num_patches, 2)

                    sorted_fine_pred = torch.gather(fine_pred, dim=-1, index=torch.argsort(fine_pred, dim=-1))
                    sorted_fine_label = torch.gather(fine_label, dim=-1, index=torch.argsort(fine_label, dim=-1))
                    y_pred_fine.append(sorted_fine_pred.cpu().numpy())
                    y_true_fine.append(sorted_fine_label.cpu().numpy())
                else:
                    outs = model(cur_X, idx)
                y_pred_score = np.append(y_pred_score, outs.cpu().numpy(), axis=0)
                y_true = np.append(y_true, cur_y.cpu().numpy(), axis=0)

            max_tab = 5
            tp = {}
            for tab in range(1, max_tab + 1):
                tp[tab] = 0

            for idx in range(y_pred_score.shape[0]):
                cur_pred = y_pred_score[idx]
                for tab in range(1, max_tab + 1):
                    target_webs = cur_pred.argsort()[-tab:]
                    for target_web in target_webs:
                        if y_true[idx, target_web] > 0:
                            tp[tab] += 1
            mapk = .0
            for tab in range(1, max_tab + 1):
                p_tab = tp[tab] / (y_true.shape[0] * tab)
                mapk += p_tab
                if tab == num_tabs:
                    result = {
                        f"p@{tab}": round(p_tab, 4) * 100,
                        f"ap@{tab}": round(mapk / tab, 4) * 100
                    }

        if fine_predict:
            y_pred_fine = np.concatenate(y_pred_fine, axis=0)
            y_true_fine = np.concatenate(y_true_fine, axis=0)

            metric_result = compute_metric(y_true_fine, y_pred_fine)
            for k in list(metric_result.keys()):
                metric_result["fine_" + k] = metric_result[k]
                del metric_result[k]
            result.update(metric_result)

        y_pred_coarse = y_pred_score.argsort()[:, -2:]
        y_true_coarse = [torch.nonzero(sample).squeeze().tolist() for sample in torch.tensor(y_true)]
        y_true_coarse = np.array(y_true_coarse)

        metric_coarse_result = compute_metric(y_true_coarse, y_pred_coarse)
        result.update(metric_coarse_result)
        return result
    else:
        with torch.no_grad():
            model.eval()
            valid_pred = []
            valid_true = []

            for index, cur_data in enumerate(tqdm.tqdm(loader)):
                cur_X, cur_y = cur_data[0][0].cuda(), cur_data[1].cuda()
                idx = cur_data[0][1].cuda()

                outs = model(cur_X, idx)

                outs = torch.argsort(outs, dim=1, descending=True)[:, 0]
                valid_pred.append(outs.cpu().numpy())
                valid_true.append(cur_y.cpu().numpy())

            valid_pred = np.concatenate(valid_pred)
            valid_true = np.concatenate(valid_true)

        valid_result = measurement(valid_true, valid_pred, "Accuracy Precision Recall F1-score")
        return valid_result


def get_layer_id_for_vit(name, num_layers):
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed') or name.startswith('PL_pos_embed'):
        return 0
    elif name.startswith('local_model'):
        return 0
    elif name.startswith('blocks') or name.startswith('PL_blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, prefix=""):
    param_groups = {}
    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = prefix + "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]

            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid = np.arange(grid_size, dtype=np.float32)
    # print(grid.shape)

    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
