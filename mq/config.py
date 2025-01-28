import time
import pika


# 配置 RabbitMQ 连接
def create_connection():
    connection_params = pika.ConnectionParameters(
        host='localhost',  # 主机地址
        port=5672,  # 端口
        virtual_host='/',  # 虚拟主机
        credentials=pika.PlainCredentials('litian', '123321'),  # 用户名和密码
        connection_attempts=3,  # 最大重试次数
        retry_delay=1,  # 重试延迟（秒）
        socket_timeout=1,  # 连接超时
        blocked_connection_timeout=1,  # 阻塞连接超时
        heartbeat=0,  # 心跳检测
    )
    return pika.BlockingConnection(connection_params)


# 重试逻辑，最多重试 3 次
def publish_with_retry(channel, exchange, routing_key, body, max_attempts=3, interval=1):
    attempts = 0
    while attempts < max_attempts:
        try:
            # 尝试发送消息
            channel.basic_publish(exchange=exchange, routing_key=routing_key, body=body)
            print(f"消息发送成功: {body}")
            break
        except Exception as e:
            print(f"消息发送失败: {e}, 重试中...")
            time.sleep(interval)
            attempts += 1
            interval *= 1  # 延迟倍增
    if attempts == max_attempts:
        print("达到最大重试次数，消息发送失败")
