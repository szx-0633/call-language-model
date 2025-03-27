# call-language-model： LLM API 统一调用工具

这是一个比LangChain等库更简洁方便的大语言模型 (LLM) API 统一调用工具，支持OpenAI 兼容接口和 Ollama 本地模型，可以方便地集成到现有的工作流程序中。

## 主要特点

- 通过OpenAI兼容接口支持多种模型提供商（OpenAI、阿里云、火山引擎等）
- 支持 Ollama 本地模型调用
- 支持多模态输入（文本 + 图像）
- 统一的调用接口，简化集成过程，直接返回结果和信息，免去记忆多种API格式的烦恼
- 详细的日志记录，内置错误处理和重试机制

## 安装

```bash
pip install pyyaml openai ollama
```

## 快速开始

1. 创建配置文件 `llm_config.yaml`：

```yaml
all_models:
  - provider: "openai"
    model_name: ["gpt-4o", "gpt-4o-mini"]
    api_key: "your_openai_api_key"
    base_url: "https://api.openai.com/v1"
  
  - provider: "aliyun"
    model_name: ["qwen2.5-32b-instruct", "qwen-turbo", "qwen-max", "qwq-32b"]
    api_key: "your_aliyun_api_key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  
  - provider: "volcengine"
    model_name: ["deepseek-r1-250120", "deepseek-v3-241226", "doubao-1-5-pro-256k-250115"]
    api_key: "your_volcengine_api_key"
    base_url: "https://ark.cn-beijing.volces.com/api/v3/"
  
  - provider: "ollama"
    model_name: ["llama3.1:8b", "mistral:7b"]
```

2. 在代码中使用：

```python
from call_language_model import call_language_model

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

# 流式调用
response_text, tokens_used, error = call_language_model(
    model_provider='aliyun',
    model_name='qwen2.5-32b-instruct',
    system_prompt="You are a helpful assistant.",
    user_prompt="请写一篇关于环保的短文",
    stream=True
)

# 处理结果
print(f"Response: {response_text}")
print(f"Tokens used: {tokens_used}")
if error:
    print(f"Error: {error}")
```

## 参数说明

`call_language_model` 函数接受以下参数：

- `model_provider`: 模型提供商，如 "openai", "aliyun", "volcengine", "ollama"，需在配置文件中定义
- `model_name`: 模型名称，需在配置文件中定义
- `system_prompt`: 系统提示
- `user_prompt`: 用户提示
- `stream`: 是否流式调用（默认 False，设置为True可支持qwq-32b等仅支持流式调用的模型）
- `temperature`: 温度参数，控制输出随机性（可选）
- `max_tokens`: 最大生成 token 数（可选）
- `files`: 图片文件路径列表，用于多模态输入（可选）
- `config_path`: 配置文件路径（默认为 "./llm_config.yaml"）

## 注意事项

- 目前仅支持 OpenAI 兼容接口和 Ollama 两种主流调用方式
- Ollama 模型不支持流式调用
- 不支持多轮对话
- 对于必须采用流式调用的模型（如 QwQ-32B），会将流式调用的结果收集后一次性返回，而不是真正的流式调用

## 日志

调用日志会记录在 `model_api.log` 文件中，包含调用时间、模型信息、token 使用量等信息。

## 自定义和扩展

您可以通过修改配置文件添加更多模型提供商和模型。只要提供商支持 OpenAI 兼容的接口，就可以直接使用。您也可以自行添加新的模型类以支持非OpenAI兼容接口。
