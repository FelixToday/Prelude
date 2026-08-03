# -*- coding: utf-8 -*-
"""整理防御模拟的输出目录到 ``dataset/{defense_name}/``。

防御模拟脚本（defense/wtfpad、defense/front、defense/tamaraw 等）会把
防御后的原始 trace 写入 ``defense/results/{method}_<时间戳>/``，
而 ``convert_to_npz.py`` 需要从 ``dataset/{defense_name}/`` 读取。
本脚本负责把二者衔接起来。

用法（在仓库根目录执行）::

    # --src 可以是具体目录，也可以是前缀（自动选择最近生成的匹配目录）
    python data_process/collect_defense.py \
        --src defense/results/wtfpad_ \
        --dst dataset/wtfpad_CW

之后即可执行::

    cd data_process
    python convert_to_npz.py --dataset wtfpad_CW
    python dataset_split.py --dataset wtfpad_CW
"""
import argparse
import glob
import os
import shutil


def resolve_src(src):
    """返回防御输出目录。若 src 不是已存在的目录，则按前缀匹配最近生成的目录。"""
    if os.path.isdir(src):
        return src
    parent = os.path.dirname(src) or "."
    prefix = os.path.basename(src)
    candidates = [d for d in glob.glob(os.path.join(parent, "*"))
                  if os.path.isdir(d) and os.path.basename(d).startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(
            f"Source directory not found and no directory matches prefix: {src}")
    return max(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(
        description="Collect defense simulation outputs into dataset/{name}/")
    parser.add_argument('--src', type=str, required=True,
                        help='Source directory (or prefix) of the defended raw traces')
    parser.add_argument('--dst', type=str, required=True,
                        help='Destination directory, e.g. dataset/wtfpad_CW')
    args = parser.parse_args()

    src_dir = resolve_src(args.src)

    os.makedirs(args.dst, exist_ok=True)

    files = [f for f in glob.glob(os.path.join(src_dir, '*')) if os.path.isfile(f)]
    if len(files) == 0:
        raise RuntimeError(f"No trace files found under {src_dir}")

    for f in files:
        shutil.copy2(f, os.path.join(args.dst, os.path.basename(f)))

    print(f"Copied {len(files)} trace files from {src_dir} to {args.dst}")


if __name__ == '__main__':
    main()
