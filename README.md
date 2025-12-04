# call-language-model： LLM API 统一调用工具

这是一个比LangChain等库更简洁方便的大语言模型 (LLM) API 统一调用工具，支持OpenAI 兼容接口和 Ollama 本地模型，可以方便地集成到现有的工作流程序中。

## 主要特点

- **支持多种模型提供商**：OpenAI、OpenAI兼容（阿里云、火山引擎）、Ollama本地模型等
- **支持多模态输入**：文本 + 图像 + 文档（PDF/Office 等）统一处理
- **支持嵌入模型调用**：提供统一的向量生成接口
- **支持批量并行调用**：可同时处理多个请求，提高处理效率
- **进度条显示**：批量处理时可实时查看处理进度和成功率
- **实时结果保存**：批量调用时支持实时保存结果到JSONL文件，每完成一个请求立即保存，避免因程序中断丢失结果
- **真正的流式调用**：支持实时输出和收集模式，您可以像使用OpenAI或ollama官方库完全一致的方式处理输出流
- **自定义配置支持**：可通过代码直接配置而无需配置文件
- **增强的错误处理**：内置重试机制、网络错误处理和详细的错误日志
- **灵活的参数传递**：支持传递任意额外的API参数，包括推理强度、输出token限制等
- **统一的调用接口**：简化集成过程，直接返回结果和信息，免去记忆多种API格式的烦恼

## 安装

下载本文件，并通过下面的命令安装依赖项。使用前，请确保安装最新版本的依赖库，特别是requests库用于HTTP请求处理：

```bash
pip install pyyaml requests tqdm
```

**注意**：本工具不依赖OpenAI或Ollama的Python库，而是直接使用requests库进行HTTP API调用，但返回格式与官方库保持一致，方便集成。

## 快速开始

1. 创建配置文件 `llm_config.yaml`：

```yaml
# 语言模型配置
all_models:
  - provider: "openai"
    model_name: ["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4o"]
    api_key: "your_openai_api_key"
    base_url: "https://api.openai.com/v1"
  
  - provider: "aliyun"
    model_name: ["qwen-max", "qwq-32b", "qwen3-235b-a22b-2507", "qwen3-4b", "qwen3-8b"]
    api_key: "your_aliyun_api_key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  
  - provider: "volcengine"
    model_name: ["deepseek-r1-250528", "deepseek-v3-250324", "doubao-seed-1-6-thinking-250615"]
    api_key: "your_volcengine_api_key"
    base_url: "https://ark.cn-beijing.volces.com/api/v3/"
  
  - provider: "ollama"
    model_name: ["llama3.1:8b", "qwen3:4b", "qwen3:8b"]
    api_key: "placeholder"  # Ollama不需要API密钥，但配置中需要存在
    base_url: "http://localhost:11434"

# 嵌入模型配置
embedding_models:
  - provider: "openai"
    model_name: ["text-embedding-3-small", "text-embedding-3-large"]
    api_key: "your_openai_api_key"
    base_url: "https://api.openai.com/v1"
  
  - provider: "ollama"
    model_name: ["nomic-embed-text", "mxbai-embed-large"]
    api_key: "placeholder"  # Ollama不需要API密钥，但配置中需要存在
    base_url: "http://localhost:11434"
```

2. 在代码中使用：

### 语言模型调用

```python
from call_language_model import call_language_model, call_embedding_model, batch_call_language_model

# 基本文本调用
response_text, tokens_used, error = call_language_model(
    model_provider='openai',
    model_name='gpt-4o',
    system_prompt="You are a helpful assistant.",
    user_prompt="请介绍一下量子计算的基本原理",
    temperature=0.7
)

# 推理模型调用（OpenAI）
response_text, tokens_used, error = call_language_model(
    model_provider='openai',
    model_name='gpt-5',
    system_prompt="You are a helpful assistant.",
    user_prompt="Give a matrix multiplication algorithm as fast as possible. Think hard about this.",
    reasoning={ # 推理配置
        "effort": "high",  # 推理强度：low, medium, high
        "summary": "auto", # 推理内容总结，注意OpenAI不支持完整推理内容返回
    },
    max_output_tokens=32768
)

# 多模态调用
response_text, tokens_used, error = call_language_model(
    model_provider='openai',
    model_name='gpt-4o',
    system_prompt="You are a helpful assistant.",
    user_prompt="请分析这张图片",
    files=['image.jpg']
)

# 文档附件调用（自动识别非图片文件并作为文件内容发送）
response_text, tokens_used, error = call_language_model(
    model_provider='openai',
    model_name='gpt-4o-mini',
    system_prompt="You are a helpful assistant that can read documents.",
    user_prompt="请用中文总结一下这份 PDF 的核心内容",
    files=['report.pdf']
)

# 流式调用（收集模式，默认）
response_text, tokens_used, error = call_language_model(
    model_provider='aliyun',
    model_name='qwen2.5-32b-instruct',
    system_prompt="You are a helpful assistant.",
    user_prompt="请写一篇关于环保的短文",
    stream=True,
    collect=True  # 收集流式结果后一次性返回
)

# 真正的流式调用
response_stream, tokens_used, error = call_language_model(
    model_provider='aliyun',
    model_name='qwen2.5-32b-instruct',
    system_prompt="You are a helpful assistant.",
    user_prompt="请写一篇关于环保的短文",
    stream=True,
    collect=False  # 返回流对象，需要自行处理
)

# 处理真正的流式响应（OpenAI兼容接口）
if not error and response_stream:
    is_first_chunk = True
    for chunk in response_stream:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'reasoning_content'):
            if delta.reasoning_content is not None:
                if is_first_chunk:
                    print("\n<think>\n")
                    is_first_chunk = False
                print(delta.reasoning_content, end='', flush=True)
            else:
                if not is_first_chunk:
                    print("\n</think>\n")
                    is_first_chunk = True
                print(delta.content, end='', flush=True)
        elif hasattr(delta, 'content') and delta.content:
            print(delta.content, end='', flush=True)

# 处理Ollama的流式响应
if not error and response_stream:
    for chunk in response_stream:
        if hasattr(chunk, 'message') and chunk.message and hasattr(chunk.message, 'content'):
            content = chunk.message.content
            print(content, end='', flush=True)

# OpenAI推理模型流式调用（自动处理推理内容）
response_stream, tokens_used, error = call_language_model(
    model_provider='openai',
    model_name='gpt-5',
    system_prompt="You are a helpful assistant.",
    user_prompt="Give a matrix multiplication algorithm as fast as possible. Think hard about this.",
    stream=True,
    collect=False,
    reasoning={
        "effort": "medium",
        "summary": "auto"
    }
)

# 处理OpenAI推理模型的流式响应
is_first_chunk = True
if not error and response_stream:
    for chunk in response_stream:
        if chunk.type == 'response.reasoning_summary_text.delta':
            if is_first_chunk:
                print("\n<think>\n", end='', flush=True)
                is_first_chunk = False
            if chunk.delta:
                print(f"{chunk.delta}", end='', flush=True)
        elif chunk.type == 'response.output_text.delta':
            if not is_first_chunk:
                print("\n</think>\n", end='', flush=True)
                is_first_chunk = True

# 跳过模型检查（直接使用指定模型名）
response_text, tokens_used, error = call_language_model(
    model_provider='openai',
    model_name='gpt-5-nano',
    system_prompt="You are a helpful assistant.",
    user_prompt="Hello!",
    skip_model_checking=True  # 跳过配置文件中的模型名称检查
)

# 使用自定义配置（无需配置文件）
response_text, tokens_used, error = call_language_model(
    model_provider='openai',
    model_name='gpt-4o',
    system_prompt="You are a helpful assistant.",
    user_prompt="Hello!",
    custom_config={
        'api_key': 'sk-your-api-key',
        'base_url': 'https://api.openai.com/v1'
    }
)
```

### 嵌入模型调用

```python
# 单个文本嵌入
embeddings, tokens_used, error = call_embedding_model(
    model_provider='openai',
    model_name='text-embedding-3-small',
    text="这是一段需要生成嵌入向量的文本"
)

# 批量文本嵌入
embeddings, tokens_used, error = call_embedding_model(
    model_provider='openai',
    model_name='text-embedding-3-small',
    text=["文本1", "文本2", "文本3"]
)

# 多模态嵌入（仅Ollama支持）
embeddings, tokens_used, error = call_embedding_model(
    model_provider='ollama',
    model_name='nomic-embed-text',
    text="描述这张图片",
    files=['image.jpg']
)

# 使用自定义配置
embeddings, tokens_used, error = call_embedding_model(
    model_provider='openai',
    model_name='text-embedding-3-large',
    text="测试文本",
    custom_config={
        'api_key': 'sk-your-api-key',
        'base_url': 'https://api.openai.com/v1'
    }
)

# 处理嵌入结果
if not error:
    print(f"嵌入向量维度: {len(embeddings[0])}")
    print(f"向量数量: {len(embeddings)}")
    print(f"使用的token数: {tokens_used}")
```

### 批量调用语言模型

```python
from call_language_model import batch_call_language_model

# 准备批量请求
batch_requests = [
    {
        "system_prompt": "You are a helpful assistant.",
        "user_prompt": "什么是Python?",
    },
    {
        "system_prompt": "You are a math tutor.",
        "user_prompt": "2+2等于多少?",
    },
    {
        "system_prompt": "You are a travel guide.",
        "user_prompt": "介绍一下巴黎的景点。",
        "files": None  # 可选的多模态文件
    },
    {
        "system_prompt": "You are an image analyzer.",
        "user_prompt": "请分析这张图片",
        "files": ["image1.jpg", "image2.png"]  # 多模态请求
    }
]

# 批量并行调用
batch_results = batch_call_language_model(
    model_provider='openai',
    model_name='gpt-5-mini',
    requests=batch_requests,
    max_workers=3,  # 并行处理3个请求
    stream=False,   # 批量模式不支持真正的流式调用
    temperature=0.7,
    skip_model_checking=True,
    config_path="./llm_config.yaml"
)

# 带进度条和文件保存的批量调用
batch_results = batch_call_language_model(
    model_provider='openai',
    model_name='gpt-5-mini',
    requests=batch_requests,
    max_workers=3,
    output_file="batch_results.jsonl",  # 保存结果到JSONL文件
    show_progress=True,  # 显示进度条（默认为True）
    temperature=0.7,
    skip_model_checking=True,
    config_path="./llm_config.yaml"
)

# 注意：当指定 output_file 时，结果会实时保存到文件中
# 每完成一个请求就立即写入文件，而不是等所有请求完成后再保存
# 这样可以避免因程序中断而丢失已完成的结果

# 处理批量结果
print(f"批量处理完成，共处理 {len(batch_results)} 个请求")
for result in batch_results:
    print(f"请求 {result['request_index']}:")
    if result['error_msg']:
        print(f"  错误: {result['error_msg']}")
    else:
        print(f"  响应长度: {len(result['response_text'])} 字符")
        print(f"  使用Token: {result['tokens_used']}")
        print(f"  响应内容: {result['response_text'][:100]}...")  # 显示前100个字符

# 不显示进度条的静默模式
batch_results = batch_call_language_model(
    model_provider='openai',
    model_name='gpt-4o',
    requests=batch_requests,
    max_workers=5,
    show_progress=False,  # 不显示进度条
    custom_config={
        'api_key': 'sk-your-api-key',
        'base_url': 'https://api.openai.com/v1'
    }
)
```

## 参数说明

### `call_language_model` 函数参数

- `model_provider`: 模型提供商，如 "openai"（使用/responses端点）, "aliyun", "volcengine"（使用/chat/completions端点）, "ollama"
- `model_name`: 模型名称，需在配置文件中定义（除非设置skip_model_checking=True）
- `system_prompt`: 系统提示，默认为None
- `user_prompt`: 用户提示，不可为空
- `stream`: 是否流式调用（默认 False）
- `collect`: 流式调用时是否收集结果（默认 True）。设为False时返回流对象，需自行处理
- `temperature`: 温度参数，控制输出随机性（可选）
- `max_tokens`: 最大生成 token 数（可选）
- `files`: 文件路径列表，用于多模态输入（图片会自动识别为图像，其他类型按文件传输；Ollama 仅支持图片）
- `skip_model_checking`: 是否跳过模型名称检查，设为True时可使用任意模型名（默认 False）
- `config_path`: 配置文件路径（默认 "./llm_config.yaml"）
- `custom_config`: 自定义配置字典，包含api_key和base_url，优先于config_path（可选）
- `max_completion_tokens`: 最大完成tokens数，用于控制回复长度（可选）
- `**kwargs`: 其他任意API参数，会直接传递给底层API调用
- `files`: 文件路径列表，用于多模态输入（图片会自动识别为图像，其他类型按文件传输；Ollama 仅支持图片）
- `skip_model_checking`: 是否跳过模型名称检查，设为True时可使用任意模型名（默认 False）
- `config_path`: 配置文件路径（默认 "./llm_config.yaml"）
- `custom_config`: 自定义配置字典，包含api_key和base_url，优先于config_path（可选）

### `call_embedding_model` 函数参数

- `model_provider`: 模型提供商，如 "openai", "ollama"
- `model_name`: 嵌入模型名称
- `text`: 单个文本字符串或文本列表
- `files`: 图片文件路径列表，用于多模态嵌入（可选，仅Ollama支持）
- `skip_model_checking`: 是否跳过模型名称检查（默认 False）
- `config_path`: 配置文件路径（默认 "./llm_config.yaml"）
- `custom_config`: 自定义配置字典，包含api_key和base_url，优先于config_path（可选）

### `batch_call_language_model` 函数参数

- `model_provider`: 模型提供商，如 "openai"（使用/responses端点）, "aliyun", "volcengine"（使用/chat/completions端点）, "ollama"
- `model_name`: 模型名称，需在配置文件中定义（除非设置skip_model_checking=True）
- `requests`: 请求列表，每个元素为字典，包含system_prompt, user_prompt和可选的files字段
  - 格式: `[{"system_prompt": "...", "user_prompt": "...", "files": [...]}, ...]`
- `max_workers`: 最大并行工作线程数（默认 4）
- `stream`: 是否流式调用（默认 False），设为True时收集流式响应，不支持真正的流式调用
- `temperature`: 温度参数，控制输出随机性（可选）
- `max_tokens`: 最大生成 token 数（可选）
- `skip_model_checking`: 是否跳过模型名称检查（默认 False）
- `config_path`: 配置文件路径（默认 "./llm_config.yaml"）
- `custom_config`: 自定义配置字典，包含api_key和base_url，优先于config_path（可选）
- `output_file`: 输出JSONL文件路径（可选）。如果提供，将保存所有结果到指定文件
- `show_progress`: 是否显示进度条（默认 True）
- `**kwargs`: 其他任意API参数，会直接传递给底层API调用

## 返回值说明

### `call_language_model` 返回值

函数返回一个三元组 `(response, tokens_used, error_msg)`：

- `response`:
  - 普通调用或流式收集模式：返回生成的文本字符串
  - 真正流式调用（`collect=False`）：返回流对象，需要自行迭代处理
- `tokens_used`: 使用的token数量（流式调用时可能为0）
- `error_msg`: 错误信息字符串，成功时为None

### `call_embedding_model` 返回值

函数返回一个三元组 `(embeddings, tokens_used, error_msg)`：

- `embeddings`: 嵌入向量列表，每个元素是一个浮点数列表
- `tokens_used`: 使用的token数量
- `error_msg`: 错误信息字符串，成功时为None

### `batch_call_language_model` 返回值

函数返回一个结果字典列表 `List[Dict]`，每个字典包含：

- `request_index`: 请求在输入列表中的索引
- `request`: 原始请求内容（包含system_prompt, user_prompt, files等）
- `response_text`: 生成的文本响应（成功时）
- `tokens_used`: 该请求使用的token数量
- `error_msg`: 错误信息字符串，成功时为None
- `model_provider`: 使用的模型提供商
- `model_name`: 使用的模型名称
- `timestamp`: 处理完成的时间戳

### 自定义配置

可以通过代码直接配置API参数，无需配置文件：

```python
custom_config = {
    'api_key': 'your-api-key',
    'base_url': 'https://your-endpoint.com/v1'
}

response, tokens, error = call_language_model(
    model_provider='custom_provider',
    model_name='custom-model',
    system_prompt="You are a helpful assistant.",
    user_prompt="Hello!",
    custom_config=custom_config  # 优先于配置文件
)
```

### 跳过模型检查

当需要使用配置文件中未列出的模型时：

```python
response, tokens, error = call_language_model(
    model_provider='openai',
    model_name='gpt-5',  # 即使配置文件中没有也能使用
    system_prompt="You are a helpful assistant.",
    user_prompt="Hello!",
    skip_model_checking=True
)
```

## 典型的自定义参数

### OpenAI推理模型的推理努力设置：
reasoning = {
    "effort": "high",  # 推理强度：low, medium, high
    "summary": "auto"  # 推理内容总结，注意OpenAI不支持完整推理内容返回
}

### Qwen3系列模型的推理开启设置：
extra_body={
    "enable_reasoning": True,  # 开启推理
}
```

### Gemini系列模型推理设置

```python
extra_body = {
    "generationConfig": {
        "thinkingConfig": {
            "includeThoughts": True,  # 返回推理过程
            "thinkingBudget": 32768,  # 推理过程token限制
        }
    }
}
```


## 批量处理高级功能

### 进度条显示

批量调用时自动显示处理进度、成功率和失败数：

```python
# 默认显示进度条
batch_results = batch_call_language_model(
    model_provider='openai',
    model_name='gpt-4o-mini',
    requests=batch_requests,
    show_progress=True  # 默认为True
)

# 关闭进度条（适用于脚本或后台处理）
batch_results = batch_call_language_model(
    model_provider='openai',
    model_name='gpt-4o-mini',
    requests=batch_requests,
    show_progress=False
)
```

### 结果保存到JSONL文件

```python
# 保存结果到文件
batch_results = batch_call_language_model(
    model_provider='openai',
    model_name='gpt-4o-mini',
    requests=batch_requests,
    output_file="results.jsonl"  # 自动保存到指定文件
)

# JSONL文件格式示例：
# {"request_index": 0, "request": {...}, "response_text": "...", "tokens_used": 150, "error_msg": null, "model_provider": "openai", "model_name": "gpt-4o-mini", "timestamp": "2025-01-31 10:30:45"}
# {"request_index": 1, "request": {...}, "response_text": "...", "tokens_used": 200, "error_msg": null, "model_provider": "openai", "model_name": "gpt-4o-mini", "timestamp": "2025-01-31 10:30:46"}
```

### 读取和分析保存的结果

```python
import json

# 读取保存的结果
def load_batch_results(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

# 分析结果
results = load_batch_results("results.jsonl")
total_tokens = sum(r['tokens_used'] for r in results if not r['error_msg'])
success_rate = len([r for r in results if not r['error_msg']]) / len(results) * 100
print(f"成功率: {success_rate:.1f}%, 总Token使用: {total_tokens}")
```

## 注意事项

- **支持两种API端点**：OpenAI提供商使用原生/responses端点（支持推理内容），其他提供商使用/chat/completions端点
- **增强的推理支持**：OpenAI o1系列模型支持推理过程显示和推理强度控制
- **支持真正的流式调用**：设置 `stream=True, collect=False` 可获得实时输出流
- **支持批量并行调用**：使用 `batch_call_language_model` 函数可同时处理多个请求
- **增强的错误处理**：内置重试机制，支持网络错误和连接超时的自动重试
- **灵活的参数传递**：支持通过kwargs传递任意自定义API参数
- **文件/图像混合支持**：OpenAI 及兼容端点同时支持图片和通用文件（例如 PDF）；Ollama 只接受图片文件，提供其他类型将抛出错误
  特别地，请自行确认模型和API是否支持多模态输入或文件输入，调用时不会进行检查，可能导致意外的错误！
- 支持嵌入模型调用，OpenAI提供商仅支持文本嵌入，Ollama支持多模态嵌入
- 批量调用模式下不支持真正的流式调用，仅支持收集模式的流式调用
- 不支持多轮对话，每次调用都是独立的单轮对话
- 流式调用时部分模型无法准确统计token消耗
- 可通过`custom_config`参数直接配置API密钥，无需配置文件

## 日志

调用日志会记录在 `model_api.log` 文件中，包含调用时间、模型信息、token 使用量等信息。

## 自定义和扩展

您可以通过以下方式扩展功能：

1. **添加新的模型提供商**：在配置文件中添加新的provider配置，只要支持OpenAI兼容接口即可直接使用
2. **自定义模型类**：继承`BaseModel`或`BaseEmbeddingModel`类，实现特定的API调用逻辑
3. **动态配置**：使用`custom_config`参数可以在运行时动态配置不同的API端点
4. **批量处理**：可以循环调用函数处理大量文本或图片

### 示例：添加新的模型提供商

```yaml
all_models:
  - provider: "custom_provider"
    model_name: ["custom-model-1", "custom-model-2"]
    api_key: "your_custom_api_key"
    base_url: "https://your-custom-endpoint.com/v1"
```

### 示例：批量文本处理

```python
from call_language_model import batch_call_language_model

# 准备批量文本处理请求
texts = ["文本1", "文本2", "文本3"]
batch_requests = []

for text in texts:
    batch_requests.append({
        "system_prompt": "请总结以下内容",
        "user_prompt": text
    })

# 使用批量调用功能（带进度条和文件保存）
results = batch_call_language_model(
    model_provider='openai',
    model_name='gpt-5-mini',
    requests=batch_requests,
    max_workers=3,  # 并行处理3个请求
    temperature=0.7,
    output_file="text_summaries.jsonl",  # 保存结果
    show_progress=True  # 显示进度条
)

# 处理结果
successful_results = []
for result in results:
    if not result['error_msg']:
        successful_results.append(result['response_text'])
        print(f"文本 {result['request_index'] + 1} 处理完成，使用 {result['tokens_used']} tokens")
    else:
        print(f"文本 {result['request_index'] + 1} 处理失败: {result['error_msg']}")

print(f"成功处理 {len(successful_results)} 个文本")
```
