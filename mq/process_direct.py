import json
import threading
import time
import mq.config as mq_config
import math
from uvr5_api import uvr_remote
from gpt_soVITS_api import open_slice_remote, open_denoise_remote, open_asr_remote, open1abc_remote, open1Ba, open1Bb, \
    close1abc, one_click
import concurrent.futures


# 接收者：监听来自交换机的消息


def startUvr5Receive():
    # 建立连接和频道
    connection = mq_config.create_connection()
    channel = connection.channel()

    # 声明交换机和队列
    channel.exchange_declare(exchange='uvr5', exchange_type='direct', durable=True)
    channel.queue_declare(queue='start_uvr5_queue')  # 创建一个队列
    channel.queue_bind(exchange='uvr5', queue='start_uvr5_queue', routing_key="start")  # 绑定队列到交换机

    # 回调函数来处理接收到的消息
    def callback(ch, method, properties, body):
        # 反序列化消息（JSON 字符串转换为 Python 对象）
        def process_request(body):
            session_id = ""
            try:
                requestData = json.loads(body)
                print("接收到的请求数据:", requestData)
                print("开始执行uvr5")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                uvr_remote(
                    requestData['model_name'],
                    requestData['input_root'],
                    requestData['format0'],
                    requestData['agg'],
                    requestData['paths']
                )
                print("uvr5执行完成")
                session_id = requestData['session_id']
                endUvr5Emit(
                    json.dumps({"session_id": session_id,
                                "status": "success"
                                }
                               ))

            except Exception as e:
                endUvr5Emit(
                    json.dumps({"session_id": session_id,
                                "status": "error"
                                })
                )
                print(e)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(process_request, body)
        # 设置消费者

    channel.basic_consume(queue='start_uvr5_queue', on_message_callback=callback, auto_ack=False)

    # 开始消费消息
    print("等待接收消息(uvr5)...")
    channel.start_consuming()


def endUvr5Emit(message):
    connection = mq_config.create_connection()
    channel = connection.channel()
    channel.exchange_declare(exchange="uvr5", exchange_type='direct', durable=True)
    body = json.dumps(message)
    mq_config.publish_with_retry(channel=channel, exchange='uvr5', routing_key='end', body=body)
    print("结束信号(uvr5)已发送:", message)
    connection.close()


def startSliceReceive():
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='slice', exchange_type='direct', durable=True)
    channel.queue_declare(queue='start_slice_queue')
    channel.queue_bind(exchange='slice', queue='start_slice_queue', routing_key='start')

    def callback(ch, method, properties, body):
        def process_request(body):
            session_id = ""
            try:
                requestData = json.loads(body)
                session_id = requestData['session_id']
                print("接收到的请求数据:", requestData)
                print("开始执行slice")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                start_time = time.time()
                result = open_slice_remote(
                    requestData['inp'],
                    requestData['threshold'],
                    requestData['min_length'],
                    requestData['min_interval'],
                    requestData['hop_size'],
                    requestData['max_sil_kept'],
                    requestData['max'],
                    requestData['alpha'],
                    requestData['n_parts'],
                    requestData['session_id'],
                )
                for err in result:
                    if err is not None:
                        endSliceEmit(
                            err
                        )
                        return
                end_time = time.time()
                time_cost = end_time - start_time
                print("slice 已完成")
                endSliceEmit(
                    json.dumps({
                        "type": "end_message",
                        "model": "slice",
                        "status": "completed",
                        "message": "slice阶段完成",
                        "session_id": session_id,
                        "dur": math.floor(time_cost)
                    })
                )
            except Exception as e:
                endSliceEmit(
                    json.dumps({
                        "type": "end_message",
                        "session_id": session_id,
                        "status": "failed",
                        "model": "slice",
                        "message": "slice阶段失败",
                    })
                )
                print(e)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(process_request, body)

    channel.basic_consume(queue='start_slice_queue', on_message_callback=callback, auto_ack=False)

    # 开始消费消息
    print("等待接收消息(slice)...")
    channel.start_consuming()


def endSliceEmit(message):
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='slice', exchange_type='direct', durable=True)
    body = json.dumps(message)
    mq_config.publish_with_retry(channel=channel, exchange='slice', routing_key='end', body=body)
    print("结束信号(slice)已发送:", message)


def startDenoiseReceive():
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='denoise', exchange_type='direct', durable=True)
    channel.queue_declare(queue='start_denoise_queue')
    channel.queue_bind(exchange='denoise', queue='start_denoise_queue', routing_key='start')

    def callback(ch, method, properties, body):
        def process_request(body):
            session_id = ""
            try:
                requestData = json.loads(body)
                session_id = requestData['session_id']
                print("接收到的请求数据:", requestData)
                print("开始执行denoise")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                start_time = time.time()
                result = open_denoise_remote(
                    requestData['denoise_inp_dir'],
                    session_id
                )
                for err in result:
                    if (err != None):
                        endDenoiseEmit(
                            err
                        )
                        print("denoise 异常")
                        return
                end_time = time.time()
                time_cost = end_time - start_time
                print("denoise 已完成")
                endDenoiseEmit(
                    json.dumps({
                        "type": "end_message",
                        "model": "denoise",
                        "status": "completed",
                        "message": "denoise阶段完成",
                        "session_id": session_id,
                        "dur": math.floor(time_cost)
                    })
                )
            except Exception as e:
                endDenoiseEmit(
                    json.dumps({
                        "type": "end_message",
                        "session_id": session_id,
                        "status": "failed",
                        "model": "denoise",
                        "message": "denoise阶段失败",
                    })
                )
                print(e)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(process_request, body)

    channel.basic_consume(queue='start_denoise_queue', on_message_callback=callback, auto_ack=False)

    # 开始消费消息
    print("等待接收消息(denoise)...")
    channel.start_consuming()


def endDenoiseEmit(message):
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='denoise', exchange_type='direct', durable=True)
    body = json.dumps(message)
    mq_config.publish_with_retry(channel=channel, exchange='denoise', routing_key='end', body=body)
    print("结束信号(denoise)已发送:", message)


def startAsrReceive():
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='asr', exchange_type='direct', durable=True)
    channel.queue_declare(queue='start_asr_queue')
    channel.queue_bind(exchange='asr', queue='start_asr_queue', routing_key='start')

    def callback(ch, method, properties, body):
        def process_request(body):
            session_id = ""
            try:
                requestData = json.loads(body)
                session_id = requestData['session_id']
                print("接收到的请求数据:", requestData)
                print("开始执行asr")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                start_time = time.time()
                result = open_asr_remote(
                    requestData['asr_inp_dir'],
                    requestData['asr_model'],
                    requestData['asr_model_size'],
                    requestData['asr_lang'],
                    requestData['asr_precision'],
                    requestData['session_id']
                )
                for err in result:
                    if err is not None:
                        endAsrEmit(
                            err
                        )
                        return
                end_time = time.time()
                time_cost = end_time - start_time
                print("asr 已完成")
                endAsrEmit(json.dumps({
                    "type": "end_message",
                    "model": "asr",
                    "status": "completed",
                    "message": "asr阶段完成",
                    "session_id": session_id,
                    "dur": math.floor(time_cost)
                }))
            except Exception as e:
                endAsrEmit(
                    json.dumps({
                        "type": "end_message",
                        "session_id": session_id,
                        "status": "failed",
                        "model": "asr",
                        "message": "asr阶段失败",
                    })
                )
                print(e)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(process_request, body)

    channel.basic_consume(queue='start_asr_queue', on_message_callback=callback, auto_ack=False)

    # 开始消费消息
    print("等待接收消息(asr)...")
    channel.start_consuming()


def endAsrEmit(message):
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='asr', exchange_type='direct', durable=True)
    body = json.dumps(message)
    mq_config.publish_with_retry(channel=channel, exchange='asr', routing_key='end', body=body)
    print("结束信号(asr)已发送:", message)


def start1abcReceive():
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='1abc', exchange_type='direct', durable=True)
    channel.queue_declare(queue='start_1abc_queue')
    channel.queue_bind(exchange='1abc', queue='start_1abc_queue', routing_key='start')

    def callback(ch, method, properties, body):
        def process_request(body):
            session_id = ""
            try:
                requestData = json.loads(body)
                session_id = requestData['session_id']
                print("接收到的请求数据:", requestData)
                print("开始执行1abc")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                start_time = time.time()
                result = open1abc_remote(
                    requestData['inp_text'],
                    requestData['inp_wav_dir'],
                    requestData['exp_name'],
                    requestData['gpu_numbers1a'],
                    requestData['gpu_numbers1Ba'],
                    requestData['gpu_numbers1c'],
                    requestData['bert_pretrained_dir'],
                    requestData['ssl_pretrained_dir'],
                    requestData['pretrained_s2G_path'],
                    requestData['user_id'],
                    requestData['session_id']
                )
                for err in result:
                    if err is not None:
                        end1abcEmit(
                            err
                        )
                        return
                end_time = time.time()
                time_cost = end_time - start_time
                print("1abc 已完成")
                end1abcEmit(json.dumps({
                    "type": "end_message",
                    "model": "format",
                    "status": "completed",
                    "message": "format阶段完成",
                    "session_id": session_id,
                    "dur": math.floor(time_cost)
                }))
            except Exception as e:
                close1abc()
                end1abcEmit(json.dumps({
                    "type": "end_message",
                    "session_id": session_id,
                    "status": "failed",
                    "model": "format",
                    "message": "format阶段失败",
                }))
                print(e)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(process_request, body)

    channel.basic_consume(queue='start_1abc_queue', on_message_callback=callback, auto_ack=False)

    # 开始消费消息
    print("等待接收消息(1abc)...")
    channel.start_consuming()


def end1abcEmit(message):
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='1abc', exchange_type='direct', durable=True)
    body = json.dumps(message)
    mq_config.publish_with_retry(channel=channel, exchange='1abc', routing_key='end', body=body)
    print("结束信号(1abc)已发送:", message)


def start1BaReceive():
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='1Ba', exchange_type='direct', durable=True)
    channel.queue_declare(queue='start_1Ba_queue')
    channel.queue_bind(exchange='1Ba', queue='start_1Ba_queue', routing_key='start')

    def callback(ch, method, properties, body):
        def process_request(body):
            session_id = ""
            try:
                requestData = json.loads(body)
                session_id = requestData['session_id']
                print("接收到的请求数据:", requestData)
                print("开始执行1Ba")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                start_time = time.time()
                result = open1Ba(
                    requestData['batch_size'],
                    requestData['total_epoch'],
                    requestData['exp_name'],
                    requestData['text_low_lr_rate'],
                    requestData['if_save_latest'],
                    requestData['if_save_every_weights'],
                    requestData['save_every_epoch'],
                    requestData['gpu_numbers1Ba'],
                    requestData['pretrained_s2G'],
                    requestData['pretrained_s2D'],
                    requestData['user_id'],
                    requestData['session_id'],
                    requestData['inp_path']
                )
                for err in result:
                    if err is not None:
                        end1BaEmit(
                            err
                        )
                        return
                end_time = time.time()
                time_cost = end_time - start_time
                print("1Ba 已完成")
                end1BaEmit(json.dumps({
                    "type": "end_message",
                    "model": "sovitsTrain",
                    "status": "completed",
                    "message": "sovitsTrain阶段完成",
                    "session_id": session_id,
                    "dur": math.floor(time_cost)
                }))
            except Exception as e:
                end1BaEmit(json.dumps({
                    "type": "end_message",
                    "session_id": session_id,
                    "status": "failed",
                    "model": "sovitsTrain",
                    "message": "sovitsTrain阶段失败",
                }))
                print(e)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(process_request, body)

    channel.basic_consume(queue='start_1Ba_queue', on_message_callback=callback, auto_ack=False)

    # 开始消费消息
    print("等待接收消息(1Ba)...")
    channel.start_consuming()


def end1BaEmit(message):
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='1Ba', exchange_type='direct', durable=True)
    body = json.dumps(message)
    mq_config.publish_with_retry(channel=channel, exchange='1Ba', routing_key='end', body=body)
    print("结束信号(1Ba)已发送:", message)


def start1BbReceive():
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='1Bb', exchange_type='direct', durable=True)
    channel.queue_declare(queue='start_1Bb_queue')
    channel.queue_bind(exchange='1Bb', queue='start_1Bb_queue', routing_key='start')

    def callback(ch, method, properties, body):
        def process_request(body):
            session_id = ""
            try:
                requestData = json.loads(body)
                session_id = requestData['session_id']
                print("接收到的请求数据:", requestData)
                print("开始执行1Bb")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                start_time = time.time()
                result = open1Bb(
                    requestData['batch_size'],
                    requestData['total_epoch'],
                    requestData['exp_name'],
                    requestData['if_dpo'],
                    requestData['if_save_latest'],
                    requestData['if_save_every_weights'],
                    requestData['save_every_epoch'],
                    requestData['gpu_numbers'],
                    requestData['pretrained_s1'],
                    requestData['user_id'],
                    requestData['inp_path'],
                    requestData['session_id']
                )
                for err in result:
                    if err is not None:
                        end1BbEmit(
                            err
                        )
                        return
                end_time = time.time()
                time_cost = end_time - start_time
                print("1Bb 已完成")
                end1BbEmit(json.dumps({
                    "type": "end_message",
                    "model": "gptTrain",
                    "status": "completed",
                    "message": "gptTrain阶段完成",
                    "session_id": session_id,
                    "dur": math.floor(time_cost)
                }))
            except Exception as e:
                end1BbEmit(json.dumps({
                    "type": "end_message",
                    "session_id": session_id,
                    "status": "failed",
                    "model": "gptTrain",
                    "message": "gptTrain阶段失败",
                }))
                print(e)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(process_request, body)

    channel.basic_consume(queue='start_1Bb_queue', on_message_callback=callback, auto_ack=False)

    # 开始消费消息
    print("等待接收消息(1Bb)...")
    channel.start_consuming()


def end1BbEmit(message):
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='1Bb', exchange_type='direct', durable=True)
    body = json.dumps(message)
    mq_config.publish_with_retry(channel=channel, exchange='1Bb', routing_key='end', body=body)
    print("结束信号(1Bb)已发送:", message)


def startOneClickReceive():
    # 建立连接和频道
    connection = mq_config.create_connection()
    channel = connection.channel()

    # 声明交换机和队列
    channel.exchange_declare(exchange='one_click', exchange_type='direct', durable=True)
    channel.queue_declare(queue='start_one_click_queue')  # 创建一个队列
    channel.queue_bind(exchange='one_click', queue='start_one_click_queue', routing_key="start")  # 绑定队列到交换机

    # 回调函数来处理接收到的消息
    def callback(ch, method, properties, body):
        # 反序列化消息（JSON 字符串转换为 Python 对象）
        def process_request(body):
            session_id = ""
            try:
                requestData = json.loads(body)
                session_id = requestData['session_id']
                print("接收到的请求数据:", requestData)
                print("开始执行one_click")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                for message in one_click(requestData):
                    print(message)
                    one_click_process_emit(message)
                print("one_click 已完成")
            except Exception as e:
                one_click_process_emit(json.dumps({
                    "type": "fail",
                    "session_id": session_id,
                    "status": "error"
                }))
                print(e)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(process_request, body)
        # 设置消费者

    channel.basic_consume(queue='start_one_click_queue', on_message_callback=callback, auto_ack=False)

    # 开始消费消息
    print("等待接收消息(one_click)...")
    channel.start_consuming()


def one_click_process_emit(message):
    connection = mq_config.create_connection()
    channel = connection.channel()

    channel.exchange_declare(exchange='one_click', exchange_type='direct', durable=True)
    body = json.dumps(message)
    mq_config.publish_with_retry(channel=channel, exchange='one_click', routing_key='process', body=body)
    print("进程信号(one_click_process)已发送:", message)


# 运行发送者和接收者
if __name__ == "__main__":
    # 启动接收者线程
    uvr5_thread = threading.Thread(target=startUvr5Receive)
    slice_thread = threading.Thread(target=startSliceReceive)
    denoise_thread = threading.Thread(target=startDenoiseReceive)
    asr_thread = threading.Thread(target=startAsrReceive)
    abc_thread = threading.Thread(target=start1abcReceive)
    Ba_thread = threading.Thread(target=start1BaReceive)
    Bb_thread = threading.Thread(target=start1BbReceive)
    one_click_thread = threading.Thread(target=startOneClickReceive)

    uvr5_thread.daemon = True  # 使其成为守护线程，主程序结束时自动关闭
    slice_thread.daemon = True  # 使其成为守护线程，主程序结束时自动关闭
    denoise_thread.daemon = True  # 使其成为守护线程，主程序结束时自动关闭
    asr_thread.daemon = True  # 使其成为守护线程，主程序结束时自动关闭
    abc_thread.daemon = True
    Ba_thread.daemon = True
    Bb_thread.daemon = True
    one_click_thread.daemon = True

    uvr5_thread.start()
    slice_thread.start()
    denoise_thread.start()
    asr_thread.start()
    abc_thread.start()
    Ba_thread.start()
    Bb_thread.start()
    one_click_thread.start()

    print("接收者(uvr5)线程已启动")
    print("接收者(slice)线程已启动")
    print("接收者(denoise)线程已启动")
    print("接收者(asr)线程已启动")
    print("接收者(1abc)线程已启动")
    print("接收者(1Ba)线程已启动")
    print("接收者(1Bb)线程已启动")
    print("接收者(one_click)线程已启动")

    # 等待线程完成
    uvr5_thread.join()
    slice_thread.join()
    denoise_thread.join()
    asr_thread.join()
    abc_thread.join()
    Ba_thread.join()
    Bb_thread.join()
    one_click_thread.join()
