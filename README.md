在官方：https://github.com/RVC-Boss/GPT-SoVITS 基础上修改,所有逻辑来自官方PR,如有侵权请联系删除

<h1>接口调用方式</h1>

```
gpt_sovits_api.py
```
   
# GPT-SoVITS API 文档

## 简介

本项目为 GPT-SoVITS 框架提供了一组封装 API，用于处理音频切片、降噪、语音识别（ASR）及模型训练（SoVITS 和 GPT）等任务。这些 API 通过 HTTP 请求接收任务参数，并利用 RabbitMQ（通过 Pika 库）进行异步任务处理。任务完成后，通过 RabbitMQ 发送完成消息，实现高效、解耦且可扩展的任务管理。

API 基于 FastAPI 构建，使用 Pydantic 模型进行请求验证，为音频处理和模型训练流程提供健壮的开发接口。

---

## API 接口文档

以下为各 API 端点的详细说明，包括功能描述、请求模型及参数。

### 1. `/gpt_sovits/one_click`

- **描述**: 一键启动包含多步骤的完整训练流程（如切片、降噪、ASR 和训练）。
- **HTTP 方法**: POST
- **请求模型**: `all_options`
- **参数**:
  - `inp`: `str`, **必填** - 输入参数（如音频文件路径）。
  - `session_id`: `str`, **必填** - 唯一会话标识符。
  - `exp_name`: `str`, **必填** - 实验名称。
  - `user_id`: `str`, **必填** - 用户标识符。
  - `threshold`: `Optional[int] = -34` - 音频切片阈值（单位：分贝）。
  - `min_length`: `Optional[int] = 4000` - 最小音频片段长度（单位：毫秒）。
  - `min_interval`: `Optional[int] = 300` - 片段间最小间隔（单位：毫秒）。
  - `hop_size`: `Optional[int] = 10` - 音频处理跳跃步长。
  - `max_sil_kept`: `Optional[int] = 500` - 保留的最大静音时长（单位：毫秒）。
  - `max`: `Optional[float] = 0.9` - 最大振幅缩放因子。
  - `alpha`: `Optional[float] = 0.25` - 音频处理的 alpha 参数。
  - `n_parts`: `Optional[int] = 4` - 音频分割份数。
  - `asr_model`: `Optional[str] = "达摩 ASR (中文)"` - ASR 模型名称。
  - `asr_model_size`: `Optional[str] = "large"` - ASR 模型尺寸。
  - `asr_lang`: `Optional[str] = "zh"` - ASR 语言（如 "zh" 表示中文）。
  - `asr_precision`: `Optional[str] = "float32"` - ASR 计算精度类型。
  - `gpu_numbers1a`: `Optional[str] = "0-0"` - 步骤 1a 使用的 GPU 编号。
  - `gpu_numbers1Ba`: `Optional[str] = "0-0"` - 步骤 1Ba 使用的 GPU 编号。
  - `gpu_numbers1c`: `Optional[str] = "0-0"` - 步骤 1c 使用的 GPU 编号。
  - `bert_pretrained_dir`: `Optional[str] = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"` - 预训练 BERT 模型路径。
  - `ssl_pretrained_dir`: `Optional[str] = "GPT_SoVITS/pretrained_models/chinese-hubert-base"` - 预训练 SSL 模型路径。
  - `pretrained_s2G_path`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"` - 预训练 SoVITS s2G 模型路径。
  - `inp_path`: `Optional[str] = ""` - 附加输入路径（可选）。
  - `if_save_latest`: `Optional[bool] = True` - 是否保存最新模型。
  - `if_save_every_weights`: `Optional[bool] = True` - 是否每轮保存权重。
  - `batch_size`: `Optional[int] = 3` - 训练批次大小。
  - `total_epoch`: `Optional[int] = 8` - 总训练轮数。
  - `text_low_lr_rate`: `Optional[float] = 0.4` - 文本相关训练的低学习率比例。
  - `save_every_epoch`: `Optional[int] = 4` - 每 N 轮保存模型。
  - `gpu_numbers1Ba_Ba`: `Optional[str] = "0"` - 另一训练步骤的 GPU 编号。
  - `pretrained_s2G`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"` - 预训练 s2G 模型路径。
  - `pretrained_s2D`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"` - 预训练 s2D 模型路径。
  - `batch_size_Bb`: `Optional[int] = 3` - 另一训练步骤的批次大小。
  - `total_epoch_Bb`: `Optional[int] = 15` - 另一训练步骤的总轮数。
  - `if_dpo`: `Optional[bool] = True` - 是否使用数据并行优化（DPO）。
  - `save_every_epoch_Bb`: `Optional[int] = 5` - 另一训练步骤的保存频率。
  - `gpu_numbers`: `Optional[str] = "0"` - 通用 GPU 编号。
  - `pretrained_s1`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"` - 预训练 s1 模型路径。
- **响应**: 
  - 成功: `{"status": "success", "message": "开始一键训练"}`
  - 错误: HTTP 500 并返回错误详情。

### 2. `/gpt_sovits/open_slice`

- **描述**: 根据参数执行音频切片。
- **HTTP 方法**: POST
- **请求模型**: `sliceRequestRemote`
- **参数**:
  - `inp`: `str`, **必填** - 输入音频文件路径。
  - `session_id`: `str`, **必填** - 唯一会话标识符。
  - `threshold`: `Optional[int] = -34` - 音频切片阈值（单位：分贝）。
  - `min_length`: `Optional[int] = 4000` - 最小片段长度（单位：毫秒）。
  - `min_interval`: `Optional[int] = 300` - 片段间最小间隔（单位：毫秒）。
  - `hop_size`: `Optional[int] = 10` - 处理跳跃步长。
  - `max_sil_kept`: `Optional[int] = 500` - 保留的最大静音时长（单位：毫秒）。
  - `max`: `Optional[float] = 0.9` - 最大振幅缩放因子。
  - `alpha`: `Optional[float] = 0.25` - 处理 alpha 参数。
  - `n_parts`: `Optional[int] = 4` - 音频分割份数。
  - `tool`: `Optional[str] = ""` - 工具标识符（可选）。
- **响应**: 
  - 成功: `{"status": "success", "message": "start"}`
  - 错误: HTTP 500 并返回错误详情。

### 3. `/gpt_sovits/open_denoise`

- **描述**: 对输入音频文件执行降噪处理。
- **HTTP 方法**: POST
- **请求模型**: `denoiseRequestRemote`
- **参数**:
  - `denoise_inp_dir`: `str`, **必填** - 输入音频文件目录。
  - `session_id`: `str`, **必填** - 唯一会话标识符。
  - `tool`: `Optional[str] = ""` - 工具标识符（可选）。
- **响应**: 
  - 成功: `{"status": "success", "message": "start"}`
  - 错误: HTTP 500 并返回错误详情。

### 4. `/gpt_sovits/open_asr`

- **描述**: 对输入音频文件执行语音识别（ASR）。
- **HTTP 方法**: POST
- **请求模型**: `asrRequestRemote`
- **参数**:
  - `asr_inp_dir`: `str`, **必填** - 输入音频文件目录。
  - `session_id`: `str`, **必填** - 唯一会话标识符。
  - `asr_model`: `Optional[str] = "达摩 ASR (中文)"` - ASR 模型名称。
  - `asr_model_size`: `Optional[str] = "large"` - ASR 模型尺寸。
  - `asr_lang`: `Optional[str] = "zh"` - ASR 语言（如 "zh" 表示中文）。
  - `asr_precision`: `Optional[str] = "float32"` - ASR 计算精度类型。
  - `tool`: `Optional[str] = ""` - 工具标识符（可选）。
- **响应**: 
  - 成功: `{"status": "success", "message": "start"}`
  - 错误: HTTP 500 并返回错误详情。

### 5. `/gpt_sovits/open1abc`

- **描述**: 执行"一键三连"预处理（功能描述不明确，推测为组合三个步骤）。
- **HTTP 方法**: POST
- **请求模型**: `abcRequest`
- **参数**:
  - `inp_text`: `str`, **必填** - 输入文本。
  - `inp_wav_dir`: `str`, **必填** - 输入音频文件目录。
  - `exp_name`: `str`, **必填** - 实验名称。
  - `user_id`: `str`, **必填** - 用户标识符。
  - `session_id`: `str`, **必填** - 唯一会话标识符。
  - `gpu_numbers1a`: `Optional[str] = "0-0"` - 步骤 1a 的 GPU 编号。
  - `gpu_numbers1Ba`: `Optional[str] = "0-0"` - 步骤 1Ba 的 GPU 编号。
  - `gpu_numbers1c`: `Optional[str] = "0-0"` - 步骤 1c 的 GPU 编号。
  - `bert_pretrained_dir`: `Optional[str] = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"` - 预训练 BERT 模型路径。
  - `ssl_pretrained_dir`: `Optional[str] = "GPT_SoVITS/pretrained_models/chinese-hubert-base"` - 预训练 SSL 模型路径。
  - `pretrained_s2G_path`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"` - 预训练 SoVITS s2G 模型路径。
- **响应**: 
  - 成功: `{"status": "success", "message": "start"}`
  - 错误: HTTP 500 并返回错误详情。

### 6. `/gpt_sovits/sovits_train`

- **描述**: 使用指定参数训练 SoVITS 模型。
- **HTTP 方法**: POST
- **请求模型**: `sovitsTrainRequest`
- **参数**:
  - `exp_name`: `str`, **必填** - 实验名称。
  - `user_id`: `str`, **必填** - 用户标识符。
  - `session_id`: `str`, **必填** - 唯一会话标识符。
  - `inp_path`: `Optional[str] = ""` - 输入路径（可选）。
  - `if_save_latest`: `Optional[bool] = True` - 是否保存最新模型。
  - `if_save_every_weights`: `Optional[bool] = True` - 是否每轮保存权重。
  - `batch_size`: `Optional[int] = 3` - 训练批次大小。
  - `total_epoch`: `Optional[int] = 8` - 总训练轮数。
  - `text_low_lr_rate`: `Optional[float] = 0.4` - 文本相关训练的低学习率比例。
  - `save_every_epoch`: `Optional[int] = 4` - 每 N 轮保存模型。
  - `gpu_numbers1Ba`: `Optional[str] = "0"` - 训练使用的 GPU 编号。
  - `pretrained_s2G`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"` - 预训练 s2G 模型路径。
  - `pretrained_s2D`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"` - 预训练 s2D 模型路径。
- **响应**: 
  - 成功: `{"status": "success", "message": "start"}`
  - 错误: HTTP 500 并返回错误详情。

### 7. `/gpt_sovits/gpt_train`

- **描述**: 使用指定参数训练 GPT 模型。
- **HTTP 方法**: POST
- **请求模型**: `gptTrainRequest`
- **参数**:
  - `exp_name`: `str`, **必填** - 实验名称。
  - `user_id`: `str`, **必填** - 用户标识符。
  - `session_id`: `str`, **必填** - 唯一会话标识符。
  - `inp_path`: `Optional[str] = ""` - 输入路径（可选）。
  - `batch_size`: `Optional[int] = 3` - 训练批次大小。
  - `total_epoch`: `Optional[int] = 15` - 总训练轮数。
  - `if_dpo`: `Optional[bool] = True` - 是否使用 DPO。
  - `if_save_latest`: `Optional[bool] = True` - 是否保存最新模型。
  - `if_save_every_weights`: `Optional[bool] = True` - 是否每轮保存权重。
  - `save_every_epoch`: `Optional[int] = 5` - 每 N 轮保存模型。
  - `gpu_numbers`: `Optional[str] = "0"` - 训练使用的 GPU 编号。
  - `pretrained_s1`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"` - 预训练 s1 模型路径。
- **响应**: 
  - 成功: `{"status": "success", "message": "start"}`
  - 错误: HTTP 500 并返回错误详情。

---

## 架构设计

### 异步任务处理

- **概述**: 系统使用 RabbitMQ 作为消息队列进行异步任务处理。每个 API 端点接收 HTTP 请求后，将任务数据发布到特定的 RabbitMQ 交换机，由消费者进行处理。
- **组件**:
  - **FastAPI**: 处理 HTTP 请求并使用 Pydantic 模型验证输入。
  - **RabbitMQ**: 通过交换机和队列管理任务排队和路由。
  - **Pika**: 与 RabbitMQ 交互的 Python 库。
  - **消费者**: 后台进程（如 `startSliceReceive`）监听队列并执行任务。

### RabbitMQ 配置

- **交换机类型**: `direct` - 确保消息根据路由键直接路由到绑定队列。
- **持久化**: 交换机和队列声明为 `durable=True`，确保消息持久化。
- **路由键**:
  - `'start'`: 用于启动任务。
  - `'end' + 工具名`（如 `'endslice'`）: 用于发送任务完成信号。
- **消费**: 消费者使用 `basic_consume` 监听队列并处理消息，采用手动确认（`auto_ack=False`）保证可靠性。

### 任务处理流程

1. **请求接收**: API 接收 HTTP POST 请求，使用对应的 Pydantic 模型验证参数。
2. **任务发布**: 验证后的请求数据序列化为 JSON，通过 `startEmit` 函数使用 `'start'` 路由键发布到 RabbitMQ。
3. **任务执行**: 消费者（如 `startSliceReceive`）从队列获取消息，执行任务（如 `open_slice_remote`）并记录执行时间。
4. **完成通知**: 任务完成或失败后，消费者通过 `endSliceEmit` 向 `'end' + 工具名` 路由键发送完成消息。
5. **客户端反馈**: 其他服务或客户端可监听完成队列获取任务状态。

### 示例消费者：音频切片

- **队列**: `start_slice_queue`
- **交换机**: `slice`
- **流程**:
  - 从 `'start'` 路由键接收消息。
  - 使用参数执行 `open_slice_remote`。
  - 向 `'endslice'` 路由键发送完成消息（状态为 `completed` 或 `failed`）。

---

## 优势

- **异步处理**: 提升系统响应能力，通过后台任务处理支持高并发。
- **解耦**: 将 API 请求处理与任务执行分离，提高可维护性和可扩展性。
- **可扩展性**: 可部署更多消费者以应对负载增长，无需修改 API 层。
- **可靠性**: RabbitMQ 的持久化队列和消息确认机制确保任务不丢失。
- **灵活性**: 通过独立交换机和队列管理不同任务类型，便于模块化扩展。

---

## 示例：音频切片流程

### API 请求

```bash
curl -X POST "http://localhost:8000/gpt_sovits/open_slice" \
-H "Content-Type: application/json" \
-d '{
  "inp": "/path/to/audio.wav",
  "session_id": "12345",
  "threshold": -34,
  "min_length": 4000,
  "min_interval": 300,
  "hop_size": 10,
  "max_sil_kept": 500,
  "max": 0.9,
  "alpha": 0.25,
  "n_parts": 4,
  "tool": "slice"
}'
```
<div align="center">

   
```
python3 api_v2.py
```

```
访问接口地址：
http://127.0.0.1:9880/?text=怎么了，亲爱的&text_lang=zh&ref_audio_path=./参考音频/[jok老师]说得好像您带我以来我考好过几次一样.wav&prompt_lang=zh&prompt_text=说得好像您带我以来我考好过几次一样&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

```
字幕接口地址：
http://127.0.0.1:9880/srt?text=怎么了，亲爱的&text_lang=zh&ref_audio_path=./参考音频/[jok老师]说得好像您带我以来我考好过几次一样.wav&prompt_lang=zh&prompt_text=说得好像您带我以来我考好过几次一样&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```


<h1>GPT-SoVITS-WebUI</h1>
A Powerful Few-shot Voice Conversion and Text-to-Speech WebUI.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Boss/GPT-SoVITS)

<a href="https://trendshift.io/repositories/7033" target="_blank"><img src="https://trendshift.io/api/badge/repositories/7033" alt="RVC-Boss%2FGPT-SoVITS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<!-- img src="https://counter.seku.su/cmoe?name=gptsovits&theme=r34" /><br> -->

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Boss/GPT-SoVITS/blob/main/colab_webui.ipynb)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-online%20demo-yellow.svg?style=for-the-badge)](https://huggingface.co/spaces/lj1995/GPT-SoVITS-v2)
[![Discord](https://img.shields.io/discord/1198701940511617164?color=%23738ADB&label=Discord&style=for-the-badge)](https://discord.gg/dnrgs5GHfG)

**English** | [**中文简体**](./docs/cn/README.md) | [**日本語**](./docs/ja/README.md) | [**한국어**](./docs/ko/README.md) | [**Türkçe**](./docs/tr/README.md)

</div>

---

## Features:

1. **Zero-shot TTS:** Input a 5-second vocal sample and experience instant text-to-speech conversion.

2. **Few-shot TTS:** Fine-tune the model with just 1 minute of training data for improved voice similarity and realism.

3. **Cross-lingual Support:** Inference in languages different from the training dataset, currently supporting English, Japanese, Korean, Cantonese and Chinese.

4. **WebUI Tools:** Integrated tools include voice accompaniment separation, automatic training set segmentation, Chinese ASR, and text labeling, assisting beginners in creating training datasets and GPT/SoVITS models.

**Check out our [demo video](https://www.bilibili.com/video/BV12g4y1m7Uw) here!**

Unseen speakers few-shot fine-tuning demo:

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

**User guide: [简体中文](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e) | [English](https://rentry.co/GPT-SoVITS-guide#/)**

## Installation

For users in China, you can [click here](https://www.codewithgpu.com/i/RVC-Boss/GPT-SoVITS/GPT-SoVITS-Official) to use AutoDL Cloud Docker to experience the full functionality online.

### Tested Environments

- Python 3.9, PyTorch 2.0.1, CUDA 11
- Python 3.10.13, PyTorch 2.1.2, CUDA 12.3
- Python 3.9, PyTorch 2.2.2, macOS 14.4.1 (Apple silicon)
- Python 3.9, PyTorch 2.2.2, CPU devices

_Note: numba==0.56.4 requires py<3.11_

### Windows

If you are a Windows user (tested with win>=10), you can [download the integrated package](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-beta.7z?download=true) and double-click on _go-webui.bat_ to start GPT-SoVITS-WebUI.

**Users in China can [download the package here](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/dkxgpiy9zb96hob4#KTvnO).**

### Linux

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```

### macOS

**Note: The models trained with GPUs on Macs result in significantly lower quality compared to those trained on other devices, so we are temporarily using CPUs instead.**

1. Install Xcode command-line tools by running `xcode-select --install`.
2. Install FFmpeg by running `brew install ffmpeg`.
3. Install the program by running the following commands:

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
pip install -r requirements.txt
```

### Install Manually

#### Install FFmpeg

##### Conda Users

```bash
conda install ffmpeg
```

##### Ubuntu/Debian Users

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

##### Windows Users

Download and place [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) and [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) in the GPT-SoVITS root.

Install [Visual Studio 2017](https://aka.ms/vs/17/release/vc_redist.x86.exe) (Korean TTS Only)

##### MacOS Users
```bash
brew install ffmpeg
```

#### Install Dependences

```bash
pip install -r requirements.txt
```

### Using Docker

#### docker-compose.yaml configuration

0. Regarding image tags: Due to rapid updates in the codebase and the slow process of packaging and testing images, please check [Docker Hub](https://hub.docker.com/r/breakstring/gpt-sovits) for the currently packaged latest images and select as per your situation, or alternatively, build locally using a Dockerfile according to your own needs.
1. Environment Variables：

- is_half: Controls half-precision/double-precision. This is typically the cause if the content under the directories 4-cnhubert/5-wav32k is not generated correctly during the "SSL extracting" step. Adjust to True or False based on your actual situation.

2. Volumes Configuration，The application's root directory inside the container is set to /workspace. The default docker-compose.yaml lists some practical examples for uploading/downloading content.
3. shm_size： The default available memory for Docker Desktop on Windows is too small, which can cause abnormal operations. Adjust according to your own situation.
4. Under the deploy section, GPU-related settings should be adjusted cautiously according to your system and actual circumstances.

#### Running with docker compose

```
docker compose -f "docker-compose.yaml" up -d
```

#### Running with docker command

As above, modify the corresponding parameters based on your actual situation, then run the following command:

```
docker run --rm -it --gpus=all --env=is_half=False --volume=G:\GPT-SoVITS-DockerTest\output:/workspace/output --volume=G:\GPT-SoVITS-DockerTest\logs:/workspace/logs --volume=G:\GPT-SoVITS-DockerTest\SoVITS_weights:/workspace/SoVITS_weights --workdir=/workspace -p 9880:9880 -p 9871:9871 -p 9872:9872 -p 9873:9873 -p 9874:9874 --shm-size="16G" -d breakstring/gpt-sovits:xxxxx
```

## Pretrained Models

**Users in China can [download all these models here](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/dkxgpiy9zb96hob4#nVNhX).**

1. Download pretrained models from [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) and place them in `GPT_SoVITS/pretrained_models`.

2. Download G2PW models from [G2PWModel_1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip), unzip and rename to `G2PWModel`, and then place them in `GPT_SoVITS/text`.(Chinese TTS Only)

3. For UVR5 (Vocals/Accompaniment Separation & Reverberation Removal, additionally), download models from [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) and place them in `tools/uvr5/uvr5_weights`.

4. For Chinese ASR (additionally), download models from [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files), and [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) and place them in `tools/asr/models`.

5. For English or Japanese ASR (additionally), download models from [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) and place them in `tools/asr/models`. Also, [other models](https://huggingface.co/Systran) may have the similar effect with smaller disk footprint. 

## Dataset Format

The TTS annotation .list file format:

```
vocal_path|speaker_name|language|text
```

Language dictionary:

- 'zh': Chinese
- 'ja': Japanese
- 'en': English
- 'ko': Korean
- 'yue': Cantonese
  
Example:

```
D:\GPT-SoVITS\xxx/xxx.wav|xxx|en|I like playing Genshin.
```

## Finetune and inference

 ### Open WebUI

 #### Integrated Package Users

 Double-click `go-webui.bat`or use `go-webui.ps`
 if you want to switch to V1,then double-click`go-webui-v1.bat` or use `go-webui-v1.ps`

 #### Others

 ```bash
 python webui.py <language(optional)>
 ```

 if you want to switch to V1,then

 ```bash
 python webui.py v1 <language(optional)>
 ```
Or maunally switch version in WebUI

 ### Finetune

 #### Path Auto-filling is now supported

     1.Fill in the audio path

     2.Slice the audio into small chunks

     3.Denoise(optinal)

     4.ASR

     5.Proofreading ASR transcriptions

     6.Go to the next Tab, then finetune the model

 ### Open Inference WebUI

 #### Integrated Package Users

 Double-click `go-webui-v2.bat` or use `go-webui-v2.ps` ,then open the inference webui at  `1-GPT-SoVITS-TTS/1C-inference` 

 #### Others

 ```bash
 python GPT_SoVITS/inference_webui.py <language(optional)>
 ```
 OR

 ```bash
 python webui.py
 ```
then open the inference webui at `1-GPT-SoVITS-TTS/1C-inference`

 ## V2 Release Notes

New Features:

1. Support Korean and Cantonese

2. An optimized text frontend

3. Pre-trained model extended from 2k hours to 5k hours

4. Improved synthesis quality for low-quality reference audio 

    [more details](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7) ) 

Use v2 from v1 environment: 

1. `pip install -r requirements.txt` to update some packages

2. Clone the latest codes from github.

3. Download v2 pretrained models from [huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v2final-pretrained) and put them into `GPT_SoVITS\pretrained_models\gsv-v2final-pretrained`.

    Chinese v2 additional: [G2PWModel_1.1.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip)（Download G2PW models,  unzip and rename to `G2PWModel`, and then place them in `GPT_SoVITS/text`.
     
## Todo List

- [x] **High Priority:**

  - [x] Localization in Japanese and English.
  - [x] User guide.
  - [x] Japanese and English dataset fine tune training.

- [ ] **Features:**
  - [x] Zero-shot voice conversion (5s) / few-shot voice conversion (1min).
  - [x] TTS speaking speed control.
  - [ ] ~~Enhanced TTS emotion control.~~
  - [ ] Experiment with changing SoVITS token inputs to probability distribution of GPT vocabs (transformer latent).
  - [x] Improve English and Japanese text frontend.
  - [ ] Develop tiny and larger-sized TTS models.
  - [x] Colab scripts.
  - [ ] Try expand training dataset (2k hours -> 10k hours).
  - [x] better sovits base model (enhanced audio quality)
  - [ ] model mix

## (Additional) Method for running from the command line
Use the command line to open the WebUI for UVR5
```
python tools/uvr5/webui.py "<infer_device>" <is_half> <webui_port_uvr5>
```
<!-- If you can't open a browser, follow the format below for UVR processing,This is using mdxnet for audio processing
```
python mdxnet.py --model --input_root --output_vocal --output_ins --agg_level --format --device --is_half_precision 
``` -->
This is how the audio segmentation of the dataset is done using the command line
```
python audio_slicer.py \
    --input_path "<path_to_original_audio_file_or_directory>" \
    --output_root "<directory_where_subdivided_audio_clips_will_be_saved>" \
    --threshold <volume_threshold> \
    --min_length <minimum_duration_of_each_subclip> \
    --min_interval <shortest_time_gap_between_adjacent_subclips> 
    --hop_size <step_size_for_computing_volume_curve>
```
This is how dataset ASR processing is done using the command line(Only Chinese)
```
python tools/asr/funasr_asr.py -i <input> -o <output>
```
ASR processing is performed through Faster_Whisper(ASR marking except Chinese)

(No progress bars, GPU performance may cause time delays)
```
python ./tools/asr/fasterwhisper_asr.py -i <input> -o <output> -l <language> -p <precision>
```
A custom list save path is enabled

## Credits

Special thanks to the following projects and contributors:

### Theoretical Research
- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
### Pretrained Models
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
### Text Frontend for Inference
- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [LangSegment](https://github.com/juntaosun/LangSegment)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)
### WebUI Tools
- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)

Thankful to @Naozumi520 for providing the Cantonese training set and for the guidance on Cantonese-related knowledge.

## Thanks to all contributors for their efforts

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>
