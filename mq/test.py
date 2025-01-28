import pika
import json


def startUvr5Emit(requestData):
    # 建立连接和频道
    # 这里使用默认的配置
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    # 声明交换机
    channel.exchange_declare(exchange='uvr5', exchange_type='direct')

    # 发送消息，转换 requestData 为 JSON 格式的字符串
    channel.basic_publish(exchange='uvr5', routing_key="start", body=json.dumps(requestData))
    print("请求uvr5已发送:", json.dumps(requestData))

    # 关闭连接
    connection.close()


startUvr5Emit({"model_name": "HP5_only_main_vocal",
               "input_root": "user_id_test/workspace/uvr5_workspace/process_id/upload/",
               "format0": "wav",
               "agg": 10,
               "paths": []})
