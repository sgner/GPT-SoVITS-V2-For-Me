import os
import shutil

def delete_all_files_in_folder(folder_path):
    # 确保文件夹存在
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # 删除文件
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹及其内容
                print(f"已删除: {file_path}")
            except Exception as e:
                print(f"删除 {file_path} 失败: {e}")

        # 删除文件夹本身
        try:
            os.rmdir(folder_path)  # 删除空文件夹
            print(f"已删除空文件夹: {folder_path}")
        except Exception as e:
            print(f"删除文件夹 {folder_path} 失败: {e}")
    else:
        print(f"文件夹 {folder_path} 不存在或不是一个有效的文件夹")

