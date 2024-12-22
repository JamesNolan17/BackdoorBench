import os
import shutil

# 根目录
current_dir = '/mnt/hdd1/chenyuwang/Trojan2/victim_models/s5_epoch'

# 存储所有的操作映射
operations = []

# 获取所有子目录
subdirectories = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]

# 遍历每个子目录
for subdir in subdirectories:
    subdir_path = os.path.join(current_dir, subdir)
    
    # 列出该子目录中的所有文件
    files = os.listdir(subdir_path)
    
    for file in files:
        if file.startswith('final_checkpoint_epoch_'):
            try:
                # 提取 epoch 后的数字
                epoch_num = int(file.split('_')[-1])
            except ValueError:
                print(f"跳过文件 {file}，因为它的格式不符合预期。")
                continue

            epoch_num_str = str(epoch_num)

            # 查找并替换目录名中的 '20'
            replace_target = '20'
            if replace_target in subdir:
                # 仅替换第一个找到的 '20'
                new_dir_name = subdir.replace(replace_target, epoch_num_str, 1)
            else:
                print(f"目录名 {subdir} 中不包含 '{replace_target}'，无法替换。")
                continue

            new_dir_path = os.path.join(current_dir, new_dir_name)

            # 准备操作映射
            old_file_path = os.path.join(subdir_path, file)
            new_file_path = os.path.join(new_dir_path, 'final_checkpoint')
            
            # 将操作添加到列表中
            operations.append({
                'old_dir': subdir_path,
                'new_dir': new_dir_path,
                'old_file': old_file_path,
                'new_file': new_file_path
            })

# 显示所有计划的操作
print("以下是即将执行的文件和目录操作：")
for op in operations:
    print(f"\n将文件：\n  {op['old_file']}\n移动并重命名为：\n  {op['new_file']}")
    print(f"新目录路径为：\n  {op['new_dir']}")

# 询问用户是否继续
confirm = input("\n是否继续执行这些操作？请输入 'yes' 或 'no'： ")

if confirm.lower() == 'yes':
    # 执行操作
    for op in operations:
        # 创建新目录（如果不存在）
        if not os.path.exists(op['new_dir']):
            os.makedirs(op['new_dir'])
        
        # 移动并重命名文件
        shutil.move(op['old_file'], op['new_file'])
        print(f"已移动并重命名：{op['old_file']} -> {op['new_file']}")

    # 删除可能的空目录
    for subdir in subdirectories:
        subdir_path = os.path.join(current_dir, subdir)
        if not os.listdir(subdir_path):
            os.rmdir(subdir_path)
            print(f"已删除空目录：{subdir_path}")
    print("\n所有操作已完成。")
else:
    print("\n操作已取消。")