```
gpt_sovits_api.py
```

# GPT-SoVITS API Documentation

## Introduction

This project provides a set of encapsulated APIs for the GPT-SoVITS framework, designed to handle tasks such as audio slicing, denoising, automatic speech recognition (ASR), and model training (SoVITS and GPT). These APIs accept HTTP requests with task parameters and leverage RabbitMQ (via the Pika library) for asynchronous task processing. Upon task completion, a completion message is sent back through RabbitMQ, ensuring efficient, decoupled, and scalable task management.

The APIs are built using FastAPI and utilize Pydantic models for request validation, offering a robust and developer-friendly interface for audio processing and model training workflows.

---

## API Interface Documentation

Below is a detailed description of each API endpoint, including its purpose, request model, and parameters.

### 1. `/gpt_sovits/one_click`

- **Description**: Initiates a one-click training process that combines multiple steps (e.g., slicing, denoising, ASR, and training).
- **HTTP Method**: POST
- **Request Model**: `all_options`
- **Parameters**:
  - `inp`: `str`, **Required** - Input parameter (e.g., audio file path).
  - `session_id`: `str`, **Required** - Unique session identifier.
  - `exp_name`: `str`, **Required** - Experiment name.
  - `user_id`: `str`, **Required** - User identifier.
  - `threshold`: `Optional[int] = -34` - Audio slicing threshold (in dB).
  - `min_length`: `Optional[int] = 4000` - Minimum audio segment length (in milliseconds).
  - `min_interval`: `Optional[int] = 300` - Minimum interval between segments (in milliseconds).
  - `hop_size`: `Optional[int] = 10` - Hop size for audio processing.
  - `max_sil_kept`: `Optional[int] = 500` - Maximum silence duration to retain (in milliseconds).
  - `max`: `Optional[float] = 0.9` - Maximum amplitude scaling factor.
  - `alpha`: `Optional[float] = 0.25` - Alpha parameter for audio processing.
  - `n_parts`: `Optional[int] = 4` - Number of parts to split the audio into.
  - `asr_model`: `Optional[str] = "达摩 ASR (中文)"` - ASR model name.
  - `asr_model_size`: `Optional[str] = "large"` - ASR model size.
  - `asr_lang`: `Optional[str] = "zh"` - ASR language (e.g., "zh" for Chinese).
  - `asr_precision`: `Optional[str] = "float32"` - ASR precision type.
  - `gpu_numbers1a`: `Optional[str] = "0-0"` - GPU numbers for step 1a.
  - `gpu_numbers1Ba`: `Optional[str] = "0-0"` - GPU numbers for step 1Ba.
  - `gpu_numbers1c`: `Optional[str] = "0-0"` - GPU numbers for step 1c.
  - `bert_pretrained_dir`: `Optional[str] = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"` - Path to pretrained BERT model.
  - `ssl_pretrained_dir`: `Optional[str] = "GPT_SoVITS/pretrained_models/chinese-hubert-base"` - Path to pretrained SSL model.
  - `pretrained_s2G_path`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"` - Path to pretrained SoVITS s2G model.
  - `inp_path`: `Optional[str] = ""` - Additional input path (if needed).
  - `if_save_latest`: `Optional[bool] = True` - Whether to save the latest model.
  - `if_save_every_weights`: `Optional[bool] = True` - Whether to save weights at every epoch.
  - `batch_size`: `Optional[int] = 3` - Training batch size.
  - `total_epoch`: `Optional[int] = 8` - Total number of training epochs.
  - `text_low_lr_rate`: `Optional[float] = 0.4` - Low learning rate for text-related training.
  - `save_every_epoch`: `Optional[int] = 4` - Save model every N epochs.
  - `gpu_numbers1Ba_Ba`: `Optional[str] = "0"` - GPU numbers for another step.
  - `pretrained_s2G`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"` - Path to pretrained s2G model.
  - `pretrained_s2D`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"` - Path to pretrained s2D model.
  - `batch_size_Bb`: `Optional[int] = 3` - Batch size for another training step.
  - `total_epoch_Bb`: `Optional[int] = 15` - Total epochs for another training step.
  - `if_dpo`: `Optional[bool] = True` - Whether to use DPO (Data Parallel Optimization).
  - `save_every_epoch_Bb`: `Optional[int] = 5` - Save frequency for another training step.
  - `gpu_numbers`: `Optional[str] = "0"` - General GPU numbers.
  - `pretrained_s1`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"` - Path to pretrained s1 model.
- **Response**: 
  - Success: `{"status": "success", "message": "开始一键训练"}`
  - Error: HTTP 500 with error details.

### 2. `/gpt_sovits/open_slice`

- **Description**: Performs audio slicing based on provided parameters.
- **HTTP Method**: POST
- **Request Model**: `sliceRequestRemote`
- **Parameters**:
  - `inp`: `str`, **Required** - Input audio file path.
  - `session_id`: `str`, **Required** - Unique session identifier.
  - `threshold`: `Optional[int] = -34` - Audio slicing threshold (in dB).
  - `min_length`: `Optional[int] = 4000` - Minimum segment length (in milliseconds).
  - `min_interval`: `Optional[int] = 300` - Minimum interval between segments (in milliseconds).
  - `hop_size`: `Optional[int] = 10` - Hop size for processing.
  - `max_sil_kept`: `Optional[int] = 500` - Maximum silence duration to retain (in milliseconds).
  - `max`: `Optional[float] = 0.9` - Maximum amplitude scaling factor.
  - `alpha`: `Optional[float] = 0.25` - Alpha parameter for processing.
  - `n_parts`: `Optional[int] = 4` - Number of parts to split the audio into.
  - `tool`: `Optional[str] = ""` - Tool identifier (optional).
- **Response**: 
  - Success: `{"status": "success", "message": "start"}`
  - Error: HTTP 500 with error details.

### 3. `/gpt_sovits/open_denoise`

- **Description**: Performs audio denoising on input audio files.
- **HTTP Method**: POST
- **Request Model**: `denoiseRequestRemote`
- **Parameters**:
  - `denoise_inp_dir`: `str`, **Required** - Directory containing input audio files.
  - `session_id`: `str`, **Required** - Unique session identifier.
  - `tool`: `Optional[str] = ""` - Tool identifier (optional).
- **Response**: 
  - Success: `{"status": "success", "message": "start"}`
  - Error: HTTP 500 with error details.

### 4. `/gpt_sovits/open_asr`

- **Description**: Performs automatic speech recognition (ASR) on input audio files.
- **HTTP Method**: POST
- **Request Model**: `asrRequestRemote`
- **Parameters**:
  - `asr_inp_dir`: `str`, **Required** - Directory containing input audio files.
  - `session_id`: `str`, **Required** - Unique session identifier.
  - `asr_model`: `Optional[str] = "达摩 ASR (中文)"` - ASR model name.
  - `asr_model_size`: `Optional[str] = "large"` - ASR model size.
  - `asr_lang`: `Optional[str] = "zh"` - ASR language (e.g., "zh" for Chinese).
  - `asr_precision`: `Optional[str] = "float32"` - ASR precision type.
  - `tool`: `Optional[str] = ""` - Tool identifier (optional).
- **Response**: 
  - Success: `{"status": "success", "message": "start"}`
  - Error: HTTP 500 with error details.

### 5. `/gpt_sovits/open1abc`

- **Description**: Executes a "one-click triple" process (specific functionality unclear; assumed to combine three steps).
- **HTTP Method**: POST
- **Request Model**: `abcRequest`
- **Parameters**:
  - `inp_text`: `str`, **Required** - Input text.
  - `inp_wav_dir`: `str`, **Required** - Directory containing input audio files.
  - `exp_name`: `str`, **Required** - Experiment name.
  - `user_id`: `str`, **Required** - User identifier.
  - `session_id`: `str`, **Required** - Unique session identifier.
  - `gpu_numbers1a`: `Optional[str] = "0-0"` - GPU numbers for step 1a.
  - `gpu_numbers1Ba`: `Optional[str] = "0-0"` - GPU numbers for step 1Ba.
  - `gpu_numbers1c`: `Optional[str] = "0-0"` - GPU numbers for step 1c.
  - `bert_pretrained_dir`: `Optional[str] = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"` - Path to pretrained BERT model.
  - `ssl_pretrained_dir`: `Optional[str] = "GPT_SoVITS/pretrained_models/chinese-hubert-base"` - Path to pretrained SSL model.
  - `pretrained_s2G_path`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"` - Path to pretrained SoVITS s2G model.
- **Response**: 
  - Success: `{"status": "success", "message": "start"}`
  - Error: HTTP 500 with error details.

### 6. `/gpt_sovits/sovits_train`

- **Description**: Trains a SoVITS model with specified parameters.
- **HTTP Method**: POST
- **Request Model**: `sovitsTrainRequest`
- **Parameters**:
  - `exp_name`: `str`, **Required** - Experiment name.
  - `user_id`: `str`, **Required** - User identifier.
  - `session_id`: `str`, **Required** - Unique session identifier.
  - `inp_path`: `Optional[str] = ""` - Input path (if needed).
  - `if_save_latest`: `Optional[bool] = True` - Whether to save the latest model.
  - `if_save_every_weights`: `Optional[bool] = True` - Whether to save weights at every epoch.
  - `batch_size`: `Optional[int] = 3` - Training batch size.
  - `total_epoch`: `Optional[int] = 8` - Total number of training epochs.
  - `text_low_lr_rate`: `Optional[float] = 0.4` - Low learning rate for text-related training.
  - `save_every_epoch`: `Optional[int] = 4` - Save model every N epochs.
  - `gpu_numbers1Ba`: `Optional[str] = "0"` - GPU numbers for training.
  - `pretrained_s2G`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"` - Path to pretrained s2G model.
  - `pretrained_s2D`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"` - Path to pretrained s2D model.
- **Response**: 
  - Success: `{"status": "success", "message": "start"}`
  - Error: HTTP 500 with error details.

### 7. `/gpt_sovits/gpt_train`

- **Description**: Trains a GPT model with specified parameters.
- **HTTP Method**: POST
- **Request Model**: `gptTrainRequest`
- **Parameters**:
  - `exp_name`: `str`, **Required** - Experiment name.
  - `user_id`: `str`, **Required** - User identifier.
  - `session_id`: `str`, **Required** - Unique session identifier.
  - `inp_path`: `Optional[str] = ""` - Input path (if needed).
  - `batch_size`: `Optional[int] = 3` - Training batch size.
  - `total_epoch`: `Optional[int] = 15` - Total number of training epochs.
  - `if_dpo`: `Optional[bool] = True` - Whether to use DPO.
  - `if_save_latest`: `Optional[bool] = True` - Whether to save the latest model.
  - `if_save_every_weights`: `Optional[bool] = True` - Whether to save weights at every epoch.
  - `save_every_epoch`: `Optional[int] = 5` - Save model every N epochs.
  - `gpu_numbers`: `Optional[str] = "0"` - GPU numbers for training.
  - `pretrained_s1`: `Optional[str] = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"` - Path to pretrained s1 model.
- **Response**: 
  - Success: `{"status": "success", "message": "start"}`
  - Error: HTTP 500 with error details.

---

## Structural Design

### Asynchronous Task Processing

- **Overview**: The system uses RabbitMQ as a message queue to handle tasks asynchronously. Each API endpoint, upon receiving an HTTP request, publishes task data to a specific RabbitMQ exchange, which is then processed by a consumer.
- **Components**:
  - **FastAPI**: Handles HTTP requests and validates input using Pydantic models.
  - **RabbitMQ**: Manages task queuing and routing via exchanges and queues.
  - **Pika**: Python library for interacting with RabbitMQ.
  - **Consumers**: Background processes (e.g., `startSliceReceive`) that listen to queues and execute tasks.

### RabbitMQ Configuration

- **Exchange Type**: `direct` - Ensures messages are routed directly to bound queues based on routing keys.
- **Durability**: Exchanges and queues are declared as `durable=True` to persist messages across restarts.
- **Routing Keys**:
  - `'start'`: Used to initiate tasks.
  - `'end' + tool` (e.g., `'endslice'`): Used to signal task completion.
- **Consumption**: Consumers use `basic_consume` to listen to queues and process messages, with manual acknowledgment (`auto_ack=False`) for reliability.

### Task Processing Workflow

1. **Request Reception**: The API receives an HTTP POST request and validates parameters using the corresponding Pydantic model.
2. **Task Publishing**: The validated request data is serialized to JSON and published to RabbitMQ using the `startEmit` function with the `'start'` routing key.
3. **Task Execution**: A consumer (e.g., `startSliceReceive`) retrieves the message from the queue, executes the task (e.g., `open_slice_remote`), and tracks execution time.
4. **Completion Notification**: Upon task completion or failure, the consumer publishes a completion message to the `'end' + tool` routing key using `endSliceEmit`.
5. **Client Feedback**: Other services or clients can listen to the completion queue to retrieve task status.

### Example Consumer: Audio Slicing

- **Queue**: `start_slice_queue`
- **Exchange**: `slice`
- **Process**: 
  - Receives message from `'start'` routing key.
  - Executes `open_slice_remote` with provided parameters.
  - Sends completion message to `'endslice'` routing key with status (`completed` or `failed`).

---

## Advantages

- **Asynchronous Processing**: Improves system responsiveness and supports high concurrency by offloading tasks to background workers.
- **Decoupling**: Separates API request handling from task execution, enhancing maintainability and scalability.
- **Scalability**: Additional consumers can be deployed to handle increased load without modifying the API layer.
- **Reliability**: RabbitMQ’s durable queues and message acknowledgment ensure tasks are not lost, even during failures.
- **Flexibility**: Different task types (e.g., slicing, training) are managed via separate exchanges and queues, allowing modular expansion.

---

## Disadvantages

- **Increased Complexity**: The use of RabbitMQ introduces additional infrastructure, requiring setup, monitoring, and maintenance.
- **Latency**: Asynchronous processing may introduce delays, making it less suitable for real-time applications.
- **Resource Overhead**: Running RabbitMQ and multiple consumers consumes extra computational resources.
- **Error Handling**: Requires robust mechanisms to handle task failures and notify clients, which adds development overhead.
- **Dependency**: Relies on RabbitMQ availability; a failure in the message queue could disrupt task processing.

---

## Example: Audio Slicing Workflow

### API Request

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
