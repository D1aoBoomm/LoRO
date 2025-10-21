import os
import subprocess
from pathlib import Path

def convert_ipynb_to_py():
    current_dir = Path.cwd()
    
    # 检查jupyter命令是否可用
    try:
        subprocess.run(['jupyter', '--version'], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print("Error: Jupyter is not installed or not in PATH.")
        print("Please install Jupyter Notebook using:")
        print("    pip install jupyter")
        return

    # 遍历三级目录结构
    for first_level in current_dir.iterdir():
        if first_level.is_dir():
            for second_level in first_level.iterdir():
                if second_level.is_dir():
                    for third_level in second_level.iterdir():
                        if third_level.is_dir() and third_level.name == 'ipynb':
                            parent_dir = third_level.parent
                            py_dir = parent_dir / 'py'
                            
                            # 创建py目录（如果不存在）
                            try:
                                py_dir.mkdir(exist_ok=True)
                                print(f"已创建目录: {py_dir}")
                            except Exception as e:
                                print(f"创建目录失败 {py_dir}: {e}")
                                continue
                            
                            # 转换所有.ipynb文件
                            for ipynb_file in third_level.glob('*.ipynb'):
                                cmd = [
                                    'jupyter',
                                    'nbconvert',
                                    '--to', 'script',
                                    '--output-dir', str(py_dir),
                                    str(ipynb_file)
                                ]
                                try:
                                    subprocess.run(cmd, check=True)
                                    py_filename = ipynb_file.stem + '.py'
                                    print(f"转换成功: {ipynb_file} → {py_dir/py_filename}")
                                except subprocess.CalledProcessError as e:
                                    print(f"转换失败 {ipynb_file}: {e}")

if __name__ == '__main__':
    convert_ipynb_to_py()