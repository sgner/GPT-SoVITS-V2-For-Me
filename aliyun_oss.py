# -*- coding: utf-8 -*-
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from itertools import islice
from tools import my_utils
import os
import yaml
import logging
import time
import random
from tools.my_utils import clean_path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 检查环境变量是否已设置
required_env_vars = ['OSS_ACCESS_KEY_ID', 'OSS_ACCESS_KEY_SECRET']
for var in required_env_vars:
    if var not in os.environ:
        logging.error(f"Environment variable {var} is not set.")
        exit(1)

# 从环境变量中获取访问凭证
auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())

# 设置Endpoint和Region
endpoint = "https://oss-cn-beijing.aliyuncs.com"
region = "cn-beijing"
bucket = oss2.Bucket(auth, endpoint, "demo-1735970696-2165", region=region)


def generate_unique_bucket_name():
    # 获取当前时间戳
    timestamp = int(time.time())
    # 生成0到9999之间的随机数
    random_number = random.randint(0, 9999)
    # 构建唯一的Bucket名称
    bucket_name = f"demo-{timestamp}-{random_number}"
    return bucket_name


# 生成唯一的Bucket名称


def create_bucket_oss():
    bucket_name = generate_unique_bucket_name()
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
    try:
        bucket.create_bucket(oss2.models.BUCKET_ACL_PRIVATE)
        logging.info("Bucket created successfully")
    except oss2.exceptions.OssError as e:
        logging.error(f"Failed to create bucket: {e}")


def upload_file(object_name, data):
    try:
        # 判断data是否是字节串（bytes）
        if isinstance(data, bytes):
            # 如果是字节串，直接上传
            result = bucket.put_object(object_name, data)
            logging.info(f"File uploaded successfully, status code: {result.status}")

        # 判断data是否是路径（文件路径）
        elif isinstance(data, str) and os.path.isfile(data):
            # 如果是文件路径，读取文件内容
            with open(data, 'rb') as file:
                file_data = file.read()  # 以二进制读取文件内容
                result = bucket.put_object(object_name, file_data)
                logging.info(f"File uploaded successfully from path: {data}, status code: {result.status}")

        else:
            logging.error("Data is neither a valid file path nor byte data.")

    except oss2.exceptions.OssError as e:
        logging.error(f"Failed to upload file: {e}")


def upload_list_file(file_path: str):
    try:
        files_folders = os.listdir(file_path)
        files = [file for file in files_folders if os.path.isfile(os.path.join(file_path, file))]
        prefix = find_second_backslash(file_path)
        prefix = my_utils.replace_last_part_of_path(prefix, 'result')
        prefix = prefix.replace(os.sep, '/')
        for file in files:
            print(prefix + "/" + file + "::::::::" + file_path + os.sep + file)
            upload_file(prefix + "/" + file, file_path + os.sep + file)
    except Exception as e:
        print(e)


def upload_directory(local_directory,user_id):
    """
    上传本地目录到阿里云 OSS，并修改路径格式
    local_directory: 本地文件夹路径
    oss_base_path: 阿里云 OSS 中的基准路径，修改后的路径应为 `user_id_test/workspace/`
    """
    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            print(local_file_path)
            # 计算相对路径，删除 E:\\gptov\\GPT-SoVITS\\GPT-SoVITS-v2\\ 部分
            relative_path = os.path.relpath(local_file_path, local_directory)
            print(relative_path)

            # 修改路径，替换掉 E:\\gptov\\GPT-SoVITS\\GPT-SoVITS-v2\\ 部分
            oss_object_name = os.path.join(user_id+'/workspace', 'logs', relative_path).replace(os.sep, '/')
            print(oss_object_name)
            # 调用 upload_file 上传文件
            upload_file(oss_object_name, local_file_path)


def download_directory_4_1Ba(oss_base_path, local_directory):
    """
    从阿里云 OSS 下载整个目录并恢复原始路径
    oss_base_path: 阿里云 OSS 中的基准路径，修改后的路径应为 `user_id_test/workspace/`
    local_directory: 本地文件夹路径，下载文件保存到此目录
    """
    list_objects = bucket.list_objects_v2(prefix=oss_base_path, delimiter='/')
    print(vars(list_objects))
    for object_info in list_objects.object_list:
        last_index = object_info.key.rfind("/")
        name = object_info.key[last_index + 1:]
        print(object_info.key)
        print(name)
        download_file(object_info.key, local_directory + os.sep + name)
    for prefix_info in list_objects.prefix_list:
        next_list_objects = bucket.list_objects_v2(prefix=prefix_info, delimiter='/')
        for next_object_info in next_list_objects.object_list:
            last_index = next_object_info.key.rfind("/")
            before_last_index = next_object_info.key.rfind("/", 0, last_index)
            name = next_object_info.key[before_last_index + 1:]
            print(next_object_info.key)
            print(name)
            download_file(next_object_info.key,local_directory+os.sep+name)

    # for object_info in bucket.list_objects_v2(prefix=oss_base_path,delimiter='/').object_list:
    #     # 获取文件在 OSS 中的路径
    #     oss_object_name = object_info.key
    #     print(oss_object_name)
    #     # 计算相对路径，删除 "user_id_test/workspace/" 部分
    #     relative_path = oss_object_name.replace(f"{oss_base_path}/", "")
    #     print(relative_path)
    #     # 拼接成本地保存路径
    #     target_path = os.path.join(local_directory, relative_path)
    #     print(target_path)
    # 下载文件
    # download_file(oss_object_name, target_path)


def download_file(object_name, target_path):
    try:
        file_obj = bucket.get_object(object_name)
        content = file_obj.read()
        logging.info("File content:")
        logging.info(content)
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(target_path, 'wb') as f:
            f.write(content)
    except oss2.exceptions.OssError as e:
        logging.error(f"Failed to download file: {e}")


def download_list_file(prefix):
    try:
        target_path = ""
        result = bucket.list_objects_v2(prefix=prefix, delimiter="/")
        with open('conf.yaml', 'r') as f:
            conf = yaml.safe_load(f)
        for object in result.object_list:
            target_path = conf['tmp_path'] + object.key.replace("/", os.sep)
            target_path = my_utils.clean_path(target_path)
            print(vars(object))
            download_file(object.key, target_path)
            print(f"download success :{object.key}")
        return os.path.dirname(target_path)
    except oss2.exceptions.OssError as e:
        logging.error(f"Failed to download file: {e}")


def list_objects_oss():
    try:
        objects = list(islice(oss2.ObjectIterator(bucket), 10))
        for obj in objects:
            logging.info(obj.key)
    except oss2.exceptions.OssError as e:
        logging.error(f"Failed to list objects: {e}")


def list_file_objects(prefix):
    objects = []
    try:
        result = bucket.list_objects_v2(prefix=prefix, delimiter='/')
        for object in result.object_list:
            objects.append(object.key)
        print(objects)
        return objects
    except oss2.exceptions.OssError as e:
        logging.error(f"Failed to list objects: {e}")


def delete_objects():
    try:
        objects = list(islice(oss2.ObjectIterator(bucket), 100))
        if objects:
            for obj in objects:
                bucket.delete_object(obj.key)
                logging.info(f"Deleted object: {obj.key}")
        else:
            logging.info("No objects to delete")
    except oss2.exceptions.OssError as e:
        logging.error(f"Failed to delete objects: {e}")


def delete_bucket():
    try:
        bucket.delete_bucket()
        logging.info("Bucket deleted successfully")
    except oss2.exceptions.OssError as e:
        logging.error(f"Failed to delete bucket: {e}")


def find_second_backslash(path):
    # 找到第一个反斜杠的位置
    first_index = path.find('\\')
    if first_index == -1:
        return -1  # 如果没有反斜杠，返回 -1
    # 从第一个反斜杠之后开始找第二个反斜杠的位置
    second_index = path.find('\\', first_index + 1)
    path = path[second_index + 1:]
    return path


def upload_model(local_path, target_path):
    local_models = os.listdir(local_path)

    for model in local_models:
        # 过滤 .ckpt 文件
        if model.endswith('.ckpt'):
            # 检查文件名中最后一个 - 后的部分是否为 e15
            if model.split('-')[-1] == 'e15.ckpt':
                upload_file(target_path + '/' + model, local_path + os.sep + model)
                print(f"upload success : {model}")

        # 过滤 .pth 文件
        elif model.endswith('.pth'):
            # 检查文件名中倒数第二个 - 后的部分是否为 e8
            parts = model.split('_')
            if len(parts) > 1 and parts[-2] == 'e8':
                print(model)
                upload_file(target_path + '/' + model, local_path + os.sep + model)
                print(f"upload success : {model}")



if __name__ == '__main__':
    # with open("E:\\test-oss\\user_id_test\\workspace\\asr_workspace\\process_id\\result\\denoise.list","r",encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         print(line)
    # upload_directory('E:\\gptov\\GPT-SoVITS\\GPT-SoVITS-v2\\logs\\12345')
    # 2. 上传文件
    upload_model("E:\\gptov\\GPT-SoVITS\\GPT-SoVITS-v2\\SoVITS_weights_v2\\12345\\test02","user_id_test/model/Sovits_weight/test02")
    # download_directory_4_1Ba("user_id_test/workspace/logs/test02/", "E:\\gptov\\GPT-SoVITS\\GPT-SoVITS-v2\\logs\\12345\\test02")
    # list_objects_oss()
    # list_file_objects("test-string-file/file/")
    # download_dir = download_list_file("user_id_test/workspace/slice_workspace/process_id/upload/")
    # "user_id_test/workspace/slice_workspace/process_id/upload/"
    # print("download_dir:" + clean_path(download_dir))
    # upload_file('user_id_test/workspace/asr_workspace/process_id/upload/test.wav', "E:\\gptov\\chat\\12.mp3")
    # upload_list_file("E:\\test-oss\\user_id_test\\workspace\\uvr5_workspace\\process_id\\vocal")
    # 3. 下载文件txt')
    # #     # 4. 列出Bucket中的对象
#     # 5. 删除Bucket中的对象
#     delete_objects()
#     # 6. 删除Bucket
#     delete_bucket()
