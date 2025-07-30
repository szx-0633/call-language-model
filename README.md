# call-language-model： LLM API 统一调用工具

这是一个比LangChain等库更简洁方便的大语言模型 (LLM) API 统一调用工具，支持OpenAI 兼容接口和 Ollama 本地模型，可以方便地集成到现有的工作流程序中。

## 主要特点

- 通过OpenAI兼容接口支持多种模型提供商（OpenAI、阿里云、火山引擎等）
- 支持 Ollama 本地模型调用
- 支持多模态输入（文本 + 图像）
- **支持嵌入模型调用**，提供统一的向量生成接口
- **真正的流式调用**，支持实时输出和收集模式
- **自定义配置支持**，可通过代码直接配置而无需配置文件
- 统一的调用接口，简化集成过程，直接返回结果和信息，免去记忆多种API格式的烦恼
- 详细的日志记录，内置错误处理和重试机制

## 安装
下载本文件，并通过下面的命令安装依赖项
```bash
pip install pyyaml openai ollama
```

## 快速开始

1. 创建配置文件 `llm_config.yaml`：

```yaml
# 语言模型配置
all_models:
  - provider: "openai"
    model_name: ["gpt-4o", "gpt-4o-mini"]
    api_key: "your_openai_api_key"
    base_url: "https://api.openai.com/v1"
  
  - provider: "aliyun"
    model_name: ["qwen2.5-32b-instruct", "qwen-turbo", "qwen-max", "qwq-32b", "qwen3-7b-instruct", "qwen3-14b-instruct"]
    api_key: "your_aliyun_api_key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  
  - provider: "volcengine"
    model_name: ["deepseek-r1-250120", "deepseek-v3-241226", "doubao-1-5-pro-256k-250115"]
    api_key: "your_volcengine_api_key"
    base_url: "https://ark.cn-beijing.volces.com/api/v3/"
  
  - provider: "ollama"
    model_name: ["llama3.1:8b", "mistral:7b", "qwen3:7b"]
    base_url: "http://localhost:11434"

# 嵌入模型配置
embedding_models:
  - provider: "openai"
    model_name: ["text-embedding-3-small", "text-embedding-3-large"]
    api_key: "your_openai_api_key"
    base_url: "https://api.openai.com/v1"
  
  - provider: "ollama"
    model_name: ["nomic-embed-text", "mxbai-embed-large"]
    base_url: "http://localhost:11434"
```

2. 在代码中使用：

### 语言模型调用

```python
from call_language_model import call_language_model, call_embedding_model

# 基本文本调用
response_text, tokens_used, error = call_language_model(
    model_provider='openai',
    model_name='gpt-4o',
    system_prompt="You are a helpful assistant.",
    user_prompt="请介绍一下量子计算的基本原理",
    temperature=0.7
)

# 多模态调用
response_text, tokens_used, error = call_language_model(
    model_provider='openai',
    model_name='gpt-4o',
    system_prompt="You are a helpful assistant.",
    user_prompt="请分析这张图片",
    files=['image.jpg']
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

# 处理真正的流式响应（Ollama）
if not error and response_stream:
    for chunk in response_stream:
        if hasattr(chunk, 'message') and chunk.message and hasattr(chunk.message, 'content'):
            content = chunk.message.content
            print(content, end='', flush=True)

# 启用推理模式（仅对Qwen3系列模型有效）
response_text, tokens_used, error = call_language_model(
    model_provider='aliyun',
    model_name='qwen3-7b-instruct',
    system_prompt="You are a helpful assistant.",
    user_prompt="解决这个数学问题：2x + 5 = 15",
    enable_thinking=True  # 启用推理过程显示
)

# 跳过模型检查（直接使用指定模型名）
response_text, tokens_used, error = call_language_model(
    model_provider='zxshen',
    model_name='openai/gpt-4.1-nano',
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

## 参数说明

### `call_language_model` 函数参数

- `model_provider`: 模型提供商，如 "openai", "aliyun", "volcengine", "ollama"
- `model_name`: 模型名称，需在配置文件中定义（除非设置skip_model_checking=True）
- `system_prompt`: 系统提示
- `user_prompt`: 用户提示
- `stream`: 是否流式调用（默认 False）
- `collect`: 流式调用时是否收集结果（默认 True）。设为False时返回流对象，需自行处理
- `enable_thinking`: 是否启用推理模式，仅对Qwen3系列模型有效（默认 True）
- `temperature`: 温度参数，控制输出随机性（可选）
- `max_tokens`: 最大生成 token 数（可选）
- `files`: 图片文件路径列表，用于多模态输入（可选）
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

## 高级功能

### 推理模式（Thinking Mode）

对于Qwen3系列模型，支持显示推理过程：

```python
response, tokens, error = call_language_model(
    model_provider='aliyun',
    model_name='qwen3-7b-instruct',
    system_prompt="You are a helpful assistant.",
    user_prompt="解决这个复杂的数学问题",
    enable_thinking=True  # 启用推理模式
)
```

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

## 注意事项

- 支持 OpenAI 兼容接口和 Ollama 两种主流调用方式
- **支持真正的流式调用**，设置 `stream=True, collect=False` 可获得实时输出流
- 支持嵌入模型调用，OpenAI提供商仅支持文本嵌入，Ollama支持多模态嵌入
- 不支持多轮对话
- Qwen3系列模型支持推理模式（`enable_thinking`参数）
- 流式调用时无法准确统计token消耗
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
texts = ["文本1", "文本2", "文本3"]
results = []

for text in texts:
    response, tokens, error = call_language_model(
        model_provider='openai',
        model_name='gpt-4o-mini',
        system_prompt="请总结以下内容",
        user_prompt=text
    )
    if not error:
        results.append(response)
```
