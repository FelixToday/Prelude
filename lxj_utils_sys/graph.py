import os

from matplotlib import pyplot as plt


def save_plot(save_path, dpi=300):
    """
    将当前matplotlib图形保存为指定格式的图片

    参数:
        save_path (str): 图片保存路径（包含文件名和扩展名）
        dpi (int, 可选): 图片分辨率，默认为300

    支持格式:
        pdf, eps, svg, png, jpg

    示例:
        save_plot('output/figure.png', dpi=600)
    """
    if save_path is None:
        print("保存路径为空，图片未保存")

    supported_formats = ['pdf', 'eps', 'svg', 'png', 'jpg']  # 支持的图片格式
    file_ext = save_path.split('.')[-1].lower()  # 获取文件扩展名

    if file_ext not in supported_formats:
        raise ValueError(f"支持格式: {supported_formats}，当前格式：{file_ext}")

    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', format=file_ext, dpi=dpi)
    print(f"图片已保存至: {save_path}")