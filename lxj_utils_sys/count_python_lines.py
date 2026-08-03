import datetime
import os.path
from pathlib import Path
from typing import Union, Any


def count_python_lines(
        path: Union[str, Path],
        *ignore_paths: Any,  # 支持单个路径、多个路径或列表
        save_path: str = "./code_count.txt",
        verbose: bool = False
) -> int:
    """
    递归统计指定路径下所有 .py 文件的代码总行数，支持忽略特定文件或目录。

    Args:
        path: 要统计的根路径（可以是文件或目录）
        *ignore_paths: 需要忽略的文件或目录路径，支持以下格式：
            - 多个独立参数：'path1', 'path2'
            - 单个列表参数：['path1', 'path2']
            - 混合使用：['path1'], 'path2'
        output_file: 保存结果的文件路径
        verbose: 是否打印每个文件的处理详情，默认为 False

    Returns:
        int: 所有 .py 文件的行数总和

    Raises:
        FileNotFoundError: 当根路径不存在时抛出
        PermissionError: 当无权限访问路径时抛出

    Example:
        # 统计项目代码，忽略虚拟环境和测试目录
        total = count_python_lines(
            '.',
            'venv',
            'tests',
            'build',
            save_filename='my_project_stats',
            verbose=True
        )
        print(f"\n项目总行数: {total}")
    """
    root = Path(path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"路径不存在: {root}")

    # 规范化所有忽略路径为绝对路径集合
    # 支持列表和单个参数的混合输入
    ignore_list = []
    for item in ignore_paths:
        if isinstance(item, (list, tuple)):
            ignore_list.extend(item)
        else:
            ignore_list.append(item)

    ignored = {Path(p).resolve() for p in ignore_list}

    def should_ignore(file_path: Path) -> bool:
        """检查文件是否应该被忽略（包括父目录检查）"""
        return file_path in ignored or any(parent in ignored for parent in file_path.parents)

    total_lines = 0
    processed_files = 0
    results = []  # 存储结果用于文件输出

    # 统一处理文件和目录场景
    files_to_check = [root] if root.is_file() else root.rglob("*.py")

    # 确保输出目录存在
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)

    for py_file in files_to_check:
        try:
            # 获取相对路径用于显示（关键修复：转为字符串）
            try:
                rel_path = str(py_file.relative_to(root))
            except ValueError:
                rel_path = py_file.name

            if should_ignore(py_file):
                status = "已忽略"
                if verbose:
                    print(f"忽略: {rel_path}")
            else:
                # 使用生成器表达式高效统计行数，自动处理编码异常
                with py_file.open('r', encoding='utf-8', errors='ignore') as f:
                    file_lines = sum(1 for _ in f)
                    total_lines += file_lines
                    processed_files += 1
                    status = f"{file_lines} 行"
                    if verbose:
                        print(f"{rel_path}: {file_lines} 行")

            # 关键修复：rel_path 已经是字符串，可以直接格式化
            results.append(f"{rel_path:<50} {status}")

        except Exception as e:
            error_msg = f"读取失败: {e}"
            # 关键修复：确保 rel_path 是字符串
            try:
                rel_path = str(py_file.relative_to(root))
            except ValueError:
                rel_path = py_file.name
            results.append(f"{rel_path:<50} {error_msg}")
            if verbose:
                print(f"{py_file}: {error_msg}")

    # 写入结果到文件
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"Python代码统计报告\n")
            f.write(f"源目录: {root}\n")
            f.write(f"统计时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

            for line in results:
                f.write(line + '\n')

            f.write("\n" + "=" * 70 + "\n")
            f.write(f"总计: {processed_files} 个文件, {total_lines} 行代码\n")
            f.write("=" * 70 + "\n")

        if verbose:
            print(f"\n详细报告已保存到: {save_path}")
    except Exception as e:
        print(f"警告: 无法保存报告文件: {e}")

    if verbose:
        print(f"\n总计: {processed_files} 个文件, {total_lines} 行代码")

    return total_lines


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例 1: 统计项目代码，忽略常见目录（独立参数方式）
    print("示例 1: 统计项目代码（独立参数）")
    total = count_python_lines(
        '../lxj_utils_sys',
        [
         '../.idea'],  # 忽略 Git 仓库
        save_path='../cache/count_lines.txt',
        verbose=True
    )
    print(f"最终行数: {total}\n")