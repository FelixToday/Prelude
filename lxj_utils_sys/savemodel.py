import json

import torch
import os
from .utils import print_colored

class ModelCheckpoint:
    def __init__(self, filename, mode='min', metric_name='metric', max_stagnation_epochs=None):
        """
        模型检查点初始化
        :param filename: 保存文件路径（包含文件夹）
        :param mode: 评价指标类型，'min'表示越小越好，'max'表示越大越好
        :param metric_name: 评价指标名称，用于记录文件
        :param max_stagnation_epochs: 最大停滞epoch次数，用于早停检查
        """
        self.filename = filename
        self.mode = mode
        self.metric_name = metric_name
        self.max_stagnation_epochs = max_stagnation_epochs
        self.best_metric = float('inf') if mode == 'min' else -float('inf')
        self.metric_file = os.path.splitext(self.filename)[0] + '_metric.json'
        self.stagnation_count = 0  # 当前停滞epoch次数

        self.output_dict = {
                'best_metric': self.best_metric,
                'save_epoch': 0,
                'metric_name': self.metric_name,
                'Mode': self.mode,
                'Complete': False
            }

        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def save(self, metric, model, epoch=None, final=False):
        """
        根据评价指标保存模型和metric记录，并检查停滞次数
        :param metric: 当前评价指标值
        :param model: 要保存的模型
        :param epoch: 当前轮次
        :return: 如果停滞次数达到最大值返回True，否则返回False
        """
        # 检查是否应该保存模型
        print(f"当前轮次: {epoch}")
        if (self.mode == 'min' and metric < self.best_metric) or \
                (self.mode == 'max' and metric > self.best_metric):
            self.best_metric = metric
            torch.save(model.state_dict(), self.filename)

            # 保存metric到json文件
            self.output_dict = {
                'best_metric': metric,
                'save_epoch': 0 if epoch is None else epoch,
                'metric_name': self.metric_name,
                'Mode': self.mode,
                'complete': final
            }
            self.write_to_json()

            print(f'模型已保存到: {self.filename}')
            print(f'指标记录已保存到: {self.metric_file}')
            print(f"保存了新模型（当前{self.metric_name}: {metric}）")
            # 重置停滞次数
            self.stagnation_count = 0
        else:
            print(f"未达到保存条件（当前{self.metric_name}: {metric}，最佳: {self.best_metric}）")
            self.output_dict['complete'] = final
            self.write_to_json()
            # 增加停滞次数
            self.stagnation_count += 1

        # 显示当前停滞次数
        print(f"停滞的epoch次数: {self.stagnation_count}/{self.max_stagnation_epochs if self.max_stagnation_epochs is not None else '无限制'}")


        # 检查是否达到最大停滞次数
        if self.max_stagnation_epochs is not None and self.stagnation_count == self.max_stagnation_epochs:
            print(f"达到最大停滞epoch次数: {self.max_stagnation_epochs}")
            self.output_dict['complete'] = True
            self.write_to_json()
            return True
        else:
            return False

    def load(self, model, device="cpu"):
        """
        从文件中加载模型
        :param model: 要加载状态的模型
        :param device: 设备，如'cpu'或'cuda'
        :return: 加载的模型, 最佳指标值, 保存时的轮次
        """
        if os.path.exists(self.filename):
            model.to(device)
            model_state_dict = torch.load(self.filename, map_location=device)
            model.load_state_dict(model_state_dict)

            with open(self.metric_file, 'r', encoding='utf-8') as f:
                output_dict = json.load(f)
                print_colored(f'模型已从: {self.filename} 加载，\n加载轮次 {output_dict["save_epoch"]}，{self.metric_name}: {output_dict["best_metric"]:.4f}', 'green')
                self.best_metric = output_dict["best_metric"]
                return model, output_dict["best_metric"], output_dict["save_epoch"]
        else:
            print(f'模型文件: {self.filename} 不存在')
            return model.to(device), None, None
    def write_to_json(self):
        with open(self.metric_file, 'w', encoding='utf-8') as f:
            json.dump(self.output_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    checkpoint = ModelCheckpoint(
        filename=r'./saved_models/best_model.pth',
        mode='max',
        metric_name='accuracy',
        max_stagnation_epochs=5  # 示例：设置最大停滞次数为5
    )

    # 训练循环中
    current_metric = 0.92  # 当前评估指标
    from torchvision.models import resnet18
    model = resnet18(False)    # 你的模型实例
    should_stop = checkpoint.save(current_metric, model, epoch=1)  # 示例调用
    if should_stop:
        print("训练早停：达到最大停滞epoch次数")
