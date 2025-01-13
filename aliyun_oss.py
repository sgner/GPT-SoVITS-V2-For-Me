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
bucket = oss2.Bucket(auth,endpoint,"demo-1735970696-2165",region=region)

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


def download_file(object_name,target_path):
    try:
        file_obj = bucket.get_object(object_name)
        content = file_obj.read()
        logging.info("File content:")
        logging.info(content)
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        with open(target_path,'wb') as f:
            f.write(content)
    except oss2.exceptions.OssError as e:
        logging.error(f"Failed to download file: {e}")
def download_list_file(prefix):
     try:
       target_path = ""
       result = bucket.list_objects_v2(prefix=prefix,delimiter="/")
       with open('conf.yaml','r') as f:
           conf = yaml.safe_load(f)
       for object in result.object_list:
           target_path = conf['tmp_path'] + object.key.replace("/", "\\")
           target_path= my_utils.clean_path(target_path)
           print(vars(object))
           download_file(object.key,target_path)
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
       result = bucket.list_objects_v2(prefix=prefix,delimiter='/')
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


if __name__ == '__main__':

    # 2. 上传文件
    # list_objects_oss()
    # list_file_objects("test-string-file/file/")
    # download_dir = download_list_file("user_id_test/workspace/slice_workspace/process_id/upload/")
    # "user_id_test/workspace/slice_workspace/process_id/upload/"
    # print("download_dir:" + clean_path(download_dir))
    upload_file('user_id_test/workspace/asr_workspace/process_id/upload/test.wav', "E:\\gptov\\chat\\12.mp3")
    # 3. 下载文件txt')
    # #     # 4. 列出Bucket中的对象
#     # 5. 删除Bucket中的对象
#     delete_objects()
#     # 6. 删除Bucket
#     delete_bucket()
