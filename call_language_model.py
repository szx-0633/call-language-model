#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Language model API unified calling tool.

This module provides a unified interface for calling various language models
and embedding models through OpenAI-compatible APIs and Ollama.

@File    : call_language_model.py
@Author  : Zhangxiao Shen
@Date    : 2025/8/14
@Description: Call language models and embedding models using OpenAI or Ollama APIs.
"""


import base64
import json
import logging
import os
import time
import types
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union, Any, Iterator

import yaml
import requests
from tqdm import tqdm

# NOTE:不需要OpenAI和Ollama库，使用requests直接与底层端点通信，但是后续使用方法与官方库调用一致

# 配置文件格式：llm_config.yaml，需要放在检查本文件所在路径内或者指定其路径
# 当前支持多种模型提供商，也可自行添加提供商和模型名称，但仅支持openai（包含openai官方接入点和兼容接入点）和ollama两种渠道调用模型
# 如果您使用第三方提供的OpenAI模型，请确保其支持/responses端点，否则应设置provider!="OpenAI"以使用兼容的/chat/completions端点
# 支持流式调用，设置参数collect=True会将流式调用的结果收集后返回，False会将整个流返回
# 流式调用时部分模型不支持统计token消耗
# 使用大语言模型的入口函数为call_language_model
# 支持使用嵌入模型，需使用call_embedding_model函数调用，暂不支持多模态嵌入
# 支持批量并行调用大语言模型，使用batch_call_language_model函数，该模式下不支持真正的流式调用
# 批量调用支持tqdm进度条显示和结果保存到JSONL文件

# 典型的自定义参数
# OpenAI推理模型的推理努力设置：
# reasoning = {
#     "effort": "high",  # 推理强度：low, medium, high
#     "summary": "auto"  # 推理内容总结，注意OpenAI不支持完整推理内容返回
# }
# Qwen3系列模型的推理开启设置：
# extra_body={
#     "enable_reasoning": True,  # 开启推理
# }
# Gemini系列模型推理设置：
# extra_body = {
#     "generationConfig":{
#         "thinkingConfig":
#         {
#             "includeThoughts": True, # 返回推理过程
#             "thinkingBudget": 32768, # 推理过程token限制
#         }
#     }
# }

# 示例自定义配置（优先于配置文件）：
# custom_config = {
#     'api_key': 'sk-xxx',
#     'base_url': 'https://api.openai.com/v1'
# }

# 示例配置文件：
# all_models:
#   - provider: "openai"
#     model_name: ["gpt-4o","gpt-4o-mini"]
#     api_key: "xxx"
#     base_url: "https://api.openai.com/v1"
#   - provider: "volcengine"
#     model_name: ["deepseek-r1-250120","deepseek-v3-241226","doubao-1-5-pro-256k-250115"]
#     api_key: "xxx"
#     base_url: "https://ark.cn-beijing.volces.com/api/v3/"
# embedding_models:
#   - provider: "openai"
#     model_name: ["text-embedding-3-small", "text-embedding-3-large"]
#     api_key: "xxx"
#     base_url: "https://api.openai.com/v1"
#   - provider: "ollama"
#     model_name: ["nomic-embed-text", "mxbai-embed-large"]
#     base_url: "http://localhost:11434"

# Constants
DEFAULT_CONFIG_PATH = './llm_config.yaml'
DEFAULT_LOG_FILE = './model_api.log'
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 10
DEFAULT_TIMEOUT_TIME = 300

# Logging configuration
logging.basicConfig(
    filename=DEFAULT_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def _dict_to_namespace(data: Any) -> Any:
    """
    Recursively converts a dictionary and its contents to types.SimpleNamespace objects,
    allowing attribute-style access.
    """
    if isinstance(data, dict):
        namespace = types.SimpleNamespace()
        for key, value in data.items():
            setattr(namespace, key, _dict_to_namespace(value))
        return namespace
    elif isinstance(data, list):
        return [_dict_to_namespace(item) for item in data]
    return data


class OpenAIStreamWrapper:
    """
    Wraps a requests. Response stream to emulate the behavior of the OpenAI
    stream object. It parses Server-Sent Events (SSE) and yields
    structured objects that allow attribute-style access.
    """
    def __init__(self, response: requests.Response):
        self._iterator = response.iter_lines()

    def __iter__(self) -> Iterator[types.SimpleNamespace]:
        """
        Yields structured data chunks from the SSE stream.
        """
        for line in self._iterator:
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    json_str = decoded_line[len('data: '):]
                    if json_str.strip() == '[DONE]':
                        break
                    try:
                        chunk_dict = json.loads(json_str)
                        yield _dict_to_namespace(chunk_dict)
                    except json.JSONDecodeError:
                        logging.debug(f"Skipping non-JSON line in stream: {decoded_line}")
                        continue
                        
    def __next__(self) -> types.SimpleNamespace:
        return next(self.__iter__())


class OllamaStreamWrapper:
    """
    Wraps a requests.Response stream from Ollama's API to emulate the
    behavior of the official library's stream object. It parses line-delimited
    JSON and yields structured objects that allow attribute-style access.
    """
    def __init__(self, response: requests.Response):
        self._iterator = response.iter_lines()

    def __iter__(self) -> Iterator[types.SimpleNamespace]:
        """
        Yields structured data chunks from the line-delimited JSON stream.
        """
        for line in self._iterator:
            if line:
                try:
                    chunk_dict = json.loads(line.decode('utf-8'))
                    yield _dict_to_namespace(chunk_dict)
                except json.JSONDecodeError:
                    logging.debug(f"Skipping non-JSON line in stream: {line.decode('utf-8')}")
                    continue
                
                        
    def __next__(self) -> types.SimpleNamespace:
        return next(self.__iter__())


class ModelConfig:
    """Model configuration manager.
    
    Manages loading and accessing model configurations from YAML files.
    Supports both language models and embedding models with provider validation.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize ModelConfig with configuration file path.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            AttributeError: If configuration file format is invalid.
        """
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            path: Path to the YAML configuration file.
            
        Returns:
            Dictionary containing the loaded configuration.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            AttributeError: If configuration file format is invalid.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.error(f"Config file not found: {path}")
            raise FileNotFoundError(f"Config file not found: {path}")
        except Exception as e:
            logging.error(f"Failed to load config: {str(e)}")
            raise AttributeError(f"Config file in wrong format: {path}")

    def get_credentials(self, model_provider: str, model_name: str, skip_checking: bool = False) -> Dict:
        """Get model credentials and configuration.
        
        Args:
            model_provider: Provider name (e.g., 'openai', 'ollama').
            model_name: Name of the model to get credentials for.
            skip_checking: If True, skip model name validation.
            
        Returns:
            Dictionary containing model configuration including credentials.
        """
        try:
            # Extract all_models list
            all_models = self.config.get('all_models', [])

            # If skip checking, find first matching provider and return
            if skip_checking:
                for model_info in all_models:
                    if model_info.get('provider') == model_provider:
                        model_info = model_info.copy()  # Create copy to avoid modifying original config
                        model_info['model_name'] = model_name
                        return model_info
                logging.warning(f"No configuration found for provider '{model_provider}'")
                return {}

            else:
                # Iterate through all_models list, check if provider and model_name are valid
                for model_info in all_models:
                    if model_info.get('provider') == model_provider:
                        if model_name in model_info.get('model_name', []):
                            model_info = model_info.copy()  # Create copy to avoid modifying original config
                            model_info['model_name'] = model_name
                            return model_info  # Return matching model configuration
                # If no matching provider or model_name found, log warning and return empty dict
                logging.warning(f"No valid configuration found for provider '{model_provider}' and model name '{model_name}'")
                return {}

        except Exception as e:
            logging.error(f"Error in get_credentials: {str(e)}")
            return {}
            
    def get_embedding_credentials(self, model_provider: str, model_name: str, skip_checking: bool = False) -> Dict:
        """Get embedding model credentials and configuration.
        
        Args:
            model_provider: Provider name (e.g., 'openai', 'ollama').
            model_name: Name of the embedding model to get credentials for.
            skip_checking: If True, skip model name validation.
            
        Returns:
            Dictionary containing embedding model configuration including credentials.
        """
        try:
            # Extract embedding_models list
            embedding_models = self.config.get('embedding_models', [])

            # If skip checking, find first matching provider and return
            if skip_checking:
                for model_info in embedding_models:
                    if model_info.get('provider') == model_provider:
                        model_info = model_info.copy()  # Create copy to avoid modifying original config
                        model_info['model_name'] = model_name
                        return model_info
                logging.warning(f"No embedding configuration found for provider '{model_provider}'")
                return {}

            # Iterate through embedding_models list, check if provider and model_name are valid
            for model_info in embedding_models:
                if model_info.get('provider') == model_provider:
                    if model_name in model_info.get('model_name', []):
                        model_info = model_info.copy()  # Create copy to avoid modifying original config
                        model_info['model_name'] = model_name
                        return model_info  # Return matching model configuration

            # If no matching provider or model_name found, log warning and return empty dict
            logging.warning(f"No valid embedding configuration found for provider '{model_provider}' and model name '{model_name}'")
            return {}

        except Exception as e:
            logging.error(f"Error in get_embedding_credentials: {str(e)}")
            return {}

class BaseModel:
    """Base class for all model implementations.
    
    Provides common interface for language model generation across different providers.
    All model implementations should inherit from this class and implement the required methods.
    """

    def __init__(self, credentials: Dict) -> None:
        """Initialize BaseModel with credentials.
        
        Args:
            credentials: Dictionary containing API credentials and configuration.
        """
        self.credentials = credentials

    def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            files: Optional[List[str]] = None
    ) -> tuple[str, int, str]:
        """Generate text response from the model.
        
        Args:
            system_prompt: System instruction for the model.
            user_prompt: User input text.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in response.
            files: List of file paths for multimodal input.
            
        Returns:
            Tuple containing (response_text, status_code, error_message).
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class OpenAIModel(BaseModel):
    """
    OpenAI model (or other models that supports /responses endpoint) handler.
    THIS IS ONLY for OpenAI /responses endpoint. Specify model_provider="openai" for this method.
    NOTE: Most third-party providers (including those transfers OpenAI models) do NOT support this endpoint.
    """

    def __init__(self, credentials: Dict) -> None:
        """
        Initialize the model handler with credentials.
        
        Args:
            credentials: Dictionary containing API credentials and configuration.
                         Expected keys: 'api_key', 'base_url'.
        """
        super().__init__(credentials)
        api_key = credentials.get('api_key')
        base_url = credentials.get('base_url')
        
        if base_url.endswith('/'):
            base_url = base_url[:-1]
        endpoint_url = f"{base_url}/responses"

        if not api_key or not base_url:
            raise ValueError("Credentials must include 'api_key' and 'base_url'.")

        self.endpoint_url = endpoint_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to a base64 data URL.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded data URL string.
        """
        with open(image_path, "rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode('utf-8')
            img_format = image_path.split('.')[-1]
            return f"data:image/{img_format};base64,{base64_str}"

    def _prepare_messages(self, **kwargs) -> list:
        """
        Prepare the message structure for the API call.
        
        Args:
            **kwargs: Keyword arguments containing system_prompt, user_prompt, and optional files.
            
        Returns:
            A list of message dictionaries formatted for the API.
        """
        messages = [{"role": "system", "content": kwargs['system_prompt']}]
        if kwargs.get('files'):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": kwargs['user_prompt']},
                    *[{"type": "image_url", "image_url": {"url": self._encode_image(f)}} for f in kwargs['files']]
                ]
            })
        else:
            messages.append({"role": "user", "content": kwargs['user_prompt']})
        return messages

    def _prepare_params_for_responses_endpoint(self, messages, **kwargs) -> dict:
        """
        Prepare API parameters for the /responses endpoint.
        
        Handles model-specific configurations and filters out None values.
        
        Args:
            messages: List of message dictionaries.
            **kwargs: Additional parameters like temperature, max_output_tokens, etc.
            
        Returns:
            Dictionary of API parameters with None values filtered out.
        """
        keys_to_remove = ["model_provider", "model_name", "system_prompt", "user_prompt", "stream", "max_tokens",
                          "skip_model_checking", "config_path", "custom_config"]
        params = {
            "model": self.credentials.get('model_name', 'gpt-5'),
            "input": messages,
            "max_output_tokens": kwargs.get("max_tokens"),
            **kwargs
        }
        for key in keys_to_remove:
            params.pop(key, None)
        return {k: v for k, v in params.items() if v is not None}

    def generate(self, **kwargs) -> tuple[str, int, str]:
        """
        Generate a non-streaming response from the model.
        
        Args:
            **kwargs: Keyword arguments for the generation request.
            
        Returns:
            A tuple containing (response_text, token_count, error_message).
        """
        messages = self._prepare_messages(**kwargs)
        params = self._prepare_params_for_responses_endpoint(messages, **kwargs)
        params.pop('collect', None)

        max_retries = 3
        retry_delay = 10
        for attempt in range(max_retries):
            try:
                response = self.session.post(self.endpoint_url, json=params, timeout=DEFAULT_TIMEOUT_TIME)
                response.raise_for_status()
                response_data = response.json()
                return self._parse_response(response_data)
            except requests.exceptions.RequestException as e:
                error_msg = f"API request failed: {e}"
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    if attempt < max_retries - 1:
                        logging.warning(f"Network error ({error_msg}), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        final_error = f"API request failed after {max_retries} attempts: {e}"
                        logging.error(final_error)
                        return "", 0, final_error
                else:
                    logging.error(f"{error_msg}. Response: {response.text if 'response' in locals() else 'N/A'}")
                    return "", 0, f"{error_msg}"
            except Exception as e:
                error_msg = f"An unexpected error occurred: {e}"
                if "Empty response received" in str(e):
                     if attempt < max_retries - 1:
                        logging.warning(f"Empty response received, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                logging.error(error_msg, exc_info=True)
                return "", 0, error_msg
        
        return "", 0, f"API request failed after {max_retries} retries."

    def generate_stream(self, **kwargs) -> tuple[Any, int, str]:
        """
        Generate a streaming response from the model.
        
        Args:
            **kwargs: Keyword arguments for the generation request.
            
        Returns:
            If collect=False, returns a stream iterator. 
            If collect=True, returns a tuple of (response_text, token_count, error_message).
        """
        messages = self._prepare_messages(**kwargs)
        params = self._prepare_params_for_responses_endpoint(messages, **kwargs)
        params["stream"] = True
        
        collect_stream_answer = kwargs.get('collect', True)
        params.pop('collect', None)

        try:
            response = self.session.post(self.endpoint_url, json=params, stream=True, timeout=DEFAULT_TIMEOUT_TIME)
            response.raise_for_status()
            stream = OpenAIStreamWrapper(response)

            if not collect_stream_answer:
                return stream, 0, None
            else:
                return self._parse_streaming_response(stream)

        except requests.exceptions.RequestException as e:
            error_msg = f"API stream request failed: {e}"
            logging.error(error_msg)
            return "", 0, error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred during streaming: {e}"
            logging.error(error_msg, exc_info=True)
            return "", 0, error_msg

    def _parse_response(self, response_data: dict) -> tuple[str, int, str]:
        """
        Parse a non-streaming API response dictionary from a /responses endpoint.
        
        Args:
            response_data: The JSON response from the API as a dictionary.
            
        Returns:
            A tuple containing (response_text, token_count, error_message).
        """
        try:
            output = response_data.get('output', [])
            if len(output) > 1:
                # Assumes reasoning models have text in the second output element
                complete_response = output[1].get('content', [{}])[0].get('text', '')
            else:
                complete_response = output[0].get('content', [{}])[0].get('text', '')
            
            reasoning_content = ""
            if output and 'summary' in output[0]:
                summary_texts = []
                for item in output[0]['summary']:
                    if 'summary' in item and item['summary']:
                        summary_texts.extend([s.get('text', '') for s in item['summary'] if 'text' in s])
                reasoning_content = "\n".join(summary_texts)

            if reasoning_content:
                complete_response = f"<think>\n{reasoning_content}\n</think>\n\n{complete_response}"

            if not complete_response or not complete_response.strip():
                raise ValueError("Empty response content received from API")
            
            tokens = response_data.get('usage', {}).get('total_tokens', 0)
            return complete_response, tokens, None

        except (KeyError, IndexError, ValueError) as e:
            error_msg = f"Failed to parse API response: {e}. Data: {response_data}"
            logging.error(error_msg)
            return "", 0, error_msg

    def _parse_streaming_response(self, stream: Iterator[types.SimpleNamespace]) -> tuple[str, int, str]:
        """
        Parse a streaming API response from a /responses endpoint by consuming the stream iterator.
        
        Args:
            stream: An iterator that yields structured chunk objects.
            
        Returns:
            A tuple containing (final_response_text, token_count, error_message).
        """
        complete_response = ""
        reasoning_content = ""
        tokens = 0
        last_chunk = None
        
        try:
            for chunk in stream:
                last_chunk = chunk
                if chunk.type == 'response.reasoning_summary_text.delta' and hasattr(chunk, 'delta'):
                    reasoning_content += chunk.delta
                elif chunk.type =='response.reasoning_summary_part.done':
                    reasoning_content += '\n\n'
                elif chunk.type == 'response.output_text.delta' and hasattr(chunk, 'delta'):
                    complete_response += chunk.delta
                # else:
                #     print(chunk)
            
            if last_chunk and hasattr(last_chunk, 'response') and hasattr(last_chunk.response, 'usage'):
                tokens = last_chunk.response.usage.total_tokens or 0

            if reasoning_content:
                complete_response = f"<think>\n{reasoning_content}\n</think>\n\n{complete_response}"

            if not complete_response or not complete_response.strip():
                raise ValueError("Empty response received from streaming API")

            return complete_response, tokens, None

        except Exception as e:
            error_msg = f"Failed to process stream: {e}"
            logging.error(error_msg, exc_info=True)
            return "", 0, error_msg


class OpenAICompatibleModel(BaseModel):
    """
    OpenAI-compatible handler.
    This uses an endpoint compatible with the OpenAI /chat/completions API. 
    Specify model_provider!="openai" for this method.
    NOTE: Most third-party providers ONLY support this endpoint.
    You CANNOT get reasoning content for OpenAI Models from this endpoint.
    """

    def __init__(self, credentials: Dict) -> None:
        """
        Initialize the model handler with credentials.
        
        Args:
            credentials: Dictionary containing API credentials and configuration.
                         Expected keys: 'api_key', 'base_url'.
        """
        super().__init__(credentials)
        api_key = credentials.get('api_key')
        base_url = credentials.get('base_url')
        
        if base_url.endswith('/'):
            base_url = base_url[:-1]
        endpoint_url = f"{base_url}/chat/completions"

        if not api_key or not base_url:
            raise ValueError("Credentials must include 'api_key' and 'base_url'.")

        self.endpoint_url = endpoint_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to a base64 data URL.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded data URL string.
        """
        with open(image_path, "rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode('utf-8')
            img_format = image_path.split('.')[-1]
            return f"data:image/{img_format};base64,{base64_str}"

    def _prepare_messages(self, **kwargs) -> list:
        """
        Prepare the message structure for the API call.
        
        Args:
            **kwargs: Keyword arguments containing system_prompt, user_prompt, and optional files.
            
        Returns:
            A list of message dictionaries formatted for the API.
        """
        messages = [{"role": "system", "content": kwargs['system_prompt']}]
        if kwargs.get('files'):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": kwargs['user_prompt']},
                    *[{"type": "image_url", "image_url": {"url": self._encode_image(f)}} for f in kwargs['files']]
                ]
            })
        else:
            messages.append({"role": "user", "content": kwargs['user_prompt']})
        return messages

    def _prepare_params_for_completions_endpoint(self, messages, **kwargs) -> dict:
        """
        Prepare the final dictionary of parameters for the API call.
        
        Args:
            messages: List of message dictionaries.
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            A dictionary of API parameters with irrelevant keys removed and None values filtered out.
        """
        keys_to_remove = ["model_provider", "model_name", "system_prompt", "user_prompt",
                          "skip_model_checking", "config_path", "custom_config", "endpoint_url"]
        params = {
            "model": self.credentials.get('model_name', 'gpt-5'),
            "messages": messages,
            **kwargs
        }
        for key in keys_to_remove:
            params.pop(key, None)
        return {k: v for k, v in params.items() if v is not None}

    def generate(self, **kwargs) -> tuple[str, int, str]:
        """
        Generate a non-streaming response from the model.
        
        Args:
            **kwargs: Keyword arguments for the generation request.
            
        Returns:
            A tuple containing (response_text, token_count, error_message).
        """
        messages = self._prepare_messages(**kwargs)
        params = self._prepare_params_for_completions_endpoint(messages, **kwargs)
        params.pop('stream', None)
        params.pop('collect', None)

        max_retries = 3
        retry_delay = 10
        for attempt in range(max_retries):
            try:
                response = self.session.post(self.endpoint_url, json=params, timeout=DEFAULT_TIMEOUT_TIME)
                response.raise_for_status()
                response_data = response.json()
                return self._parse_response(response_data)
            except requests.exceptions.RequestException as e:
                error_msg = f"API request failed: {e}"
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    if attempt < max_retries - 1:
                        logging.warning(f"Network error ({error_msg}), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        final_error = f"API request failed after {max_retries} attempts: {e}"
                        logging.error(final_error)
                        return "", 0, final_error
                else:
                    logging.error(f"{error_msg}. Response: {response.text if 'response' in locals() else 'N/A'}")
                    return "", 0, f"{error_msg}"
            except Exception as e:
                error_msg = f"An unexpected error occurred: {e}"
                logging.error(error_msg, exc_info=True)
                return "", 0, error_msg
        
        return "", 0, f"API request failed after {max_retries} retries."


    def generate_stream(self, **kwargs) -> tuple[Any, int, str]:
        """
        Generate a streaming response from the model.
        
        Args:
            **kwargs: Keyword arguments for the generation request.
            
        Returns:
            If collect=False, returns a stream iterator. 
            If collect=True, returns a tuple of (response_text, token_count, error_message).
        """
        messages = self._prepare_messages(**kwargs)
        params = self._prepare_params_for_completions_endpoint(messages, **kwargs)
        params["stream"] = True
        
        collect_stream_answer = kwargs.get('collect', True)
        params.pop('collect', None)

        try:
            response = self.session.post(self.endpoint_url, json=params, stream=True, timeout=DEFAULT_TIMEOUT_TIME)
            response.raise_for_status()
            stream = OpenAIStreamWrapper(response)

            if not collect_stream_answer:
                return stream, 0, None
            else:
                return self._parse_streaming_response(stream)

        except requests.exceptions.RequestException as e:
            error_msg = f"API stream request failed: {e}"
            logging.error(error_msg)
            return "", 0, error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred during streaming: {e}"
            logging.error(error_msg, exc_info=True)
            return "", 0, error_msg

    def _parse_response(self, response_data: dict) -> tuple[str, int, str]:
        """
        Parse a non-streaming API response dictionary.
        
        Args:
            response_data: The JSON response from the API as a dictionary.
            
        Returns:
            A tuple containing (response_text, token_count, error_message).
        """
        try:
            message = response_data['choices'][0]['message']
            complete_response = message.get('content', '') or ''
            
            if 'reasoning_content' in message and message['reasoning_content']:
                complete_response = f"<think>\n{message['reasoning_content']}\n</think>\n\n{complete_response}"
            
            if not complete_response.strip():
                raise ValueError("Empty response content received from API")
            
            tokens = response_data.get('usage', {}).get('total_tokens', 0)
            return complete_response, tokens, None
        
        except (KeyError, IndexError, ValueError) as e:
            error_msg = f"Failed to parse API response: {e}. Data: {response_data}"
            logging.error(error_msg)
            return "", 0, error_msg

    def _parse_streaming_response(self, stream: Iterator[types.SimpleNamespace]) -> tuple[str, int, str]:
        """
        Parse a streaming API response by consuming the stream iterator.
        
        Args:
            stream: An iterator that yields structured chunk objects.
            
        Returns:
            A tuple containing (final_response_text, token_count, error_message).
        """
        complete_response = ""
        reasoning_content = ""
        returned_tokens = 0
        
        try:
            for chunk in stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        complete_response += delta.content
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                        reasoning_content += delta.reasoning_content
                
                if hasattr(chunk, 'usage') and hasattr(chunk.usage, 'total_tokens') and chunk.usage.total_tokens is not None:
                    returned_tokens = chunk.usage.total_tokens

            if reasoning_content:
                complete_response = f"<think>\n{reasoning_content}\n</think>\n\n{complete_response}"

            if not complete_response.strip():
                raise ValueError("Empty response received from streaming API")

            return complete_response, returned_tokens, None
        
        except Exception as e:
            error_msg = f"Failed to process stream: {e}"
            logging.error(error_msg, exc_info=True)
            return "", 0, error_msg


class OllamaModel(BaseModel):
    """
    Ollama local model handler.
    
    Handles interactions with locally hosted Ollama models including multimodal support.
    """

    def __init__(self, credentials: Dict) -> None:
        """
        Initialize the model handler with credentials.
        
        Args:
            credentials: Dictionary containing API credentials and configuration.
                         Expected keys: 'api_key'(placeholder, not used), 'base_url(e.g., 'http://localhost:11434')'.
        """
        super().__init__(credentials)
        api_key = None
        base_url = credentials.get('base_url')
        
        if base_url.endswith('/'):
            base_url = base_url[:-1]
        endpoint_url = f"{base_url}/api/chat"

        if not api_key or not base_url:
            raise ValueError("Credentials must include 'api_key' and 'base_url'.")

        self.endpoint_url = endpoint_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to a base64 string for the Ollama API.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded string of the image.
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def _prepare_payload(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare the JSON payload for an Ollama API request.
        
        Args:
            **kwargs: Keyword arguments containing prompts, images, and options.
            
        Returns:
            A dictionary representing the request payload.
        """
        messages = [{"role": "system", "content": kwargs.get('system_prompt', '')}]
        user_message = {"role": "user", "content": kwargs.get('user_prompt', '')}

        if kwargs.get('files'):
            user_message["images"] = [self._encode_image(f) for f in kwargs['files']]
        
        messages.append(user_message)

        options = {
            "temperature": kwargs.get('temperature'),
            "num_predict": kwargs.get('max_tokens')
        }
        options = {k: v for k, v in options.items() if v is not None}

        payload = {
            "model": self.credentials.get('model_name', 'llama3.1:8b'),
            "messages": messages,
            "options": options,
        }
        return payload

    def generate(self, **kwargs) -> tuple[str, int, str]:
        """
        Generate a non-streaming response from the Ollama model.
        
        Args:
            **kwargs: Keyword arguments for the generation request.
            
        Returns:
            A tuple containing (response_text, token_count, error_message).
        """
        payload = self._prepare_payload(**kwargs)

        try:
            response = self.session.post(self.endpoint_url, json=payload, timeout=DEFAULT_TIMEOUT_TIME)
            response.raise_for_status()
            response_data = response.json()
            return self._parse_response(response_data)
        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama API request failed: {e}"
            logging.error(f"{error_msg}. Response: {response.text if 'response' in locals() else 'N/A'}")
            return "", 0, str(e)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            logging.error(error_msg, exc_info=True)
            return "", 0, str(e)

    def generate_stream(self, **kwargs) -> tuple[Any, int, str]:
        """
        Generate a streaming response from the Ollama model.
        
        Args:
            **kwargs: Keyword arguments for the generation request.
            
        Returns:
            If collect=False, returns a stream iterator.
            If collect=True, returns a tuple of (response_text, token_count, error_message).
        """
        payload = self._prepare_payload(**kwargs)
        payload['stream'] = True

        collect_stream_answer = kwargs.get('collect', True)

        try:
            response = self.session.post(self.endpoint_url, json=payload, stream=True, timeout=300)
            response.raise_for_status()
            stream = OllamaStreamWrapper(response)

            if not collect_stream_answer:
                return stream, 0, None
            else:
                return self._parse_streaming_response(stream)

        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama API stream request failed: {e}"
            logging.error(error_msg)
            return "", 0, str(e)
        except Exception as e:
            error_msg = f"An unexpected error occurred during streaming: {e}"
            logging.error(error_msg, exc_info=True)
            return "", 0, str(e)

    def _parse_response(self, response_data: Dict[str, Any]) -> tuple[str, int, str]:
        """
        Parse a non-streaming API response dictionary from the Ollama API.
        
        Args:
            response_data: The JSON response from the API as a dictionary.
            
        Returns:
            A tuple containing (response_text, token_count, error_message).
        """
        try:
            complete_response = response_data.get('message', {}).get('content', '')
            if not complete_response or not complete_response.strip():
                raise ValueError("Empty response content received from API")
            
            prompt_tokens = response_data.get('prompt_eval_count', 0)
            eval_tokens = response_data.get('eval_count', 0)
            tokens_used = prompt_tokens + eval_tokens
            
            return complete_response, tokens_used, None
        
        except (ValueError, KeyError) as e:
            error_msg = f"Failed to parse API response: {e}. Data: {response_data}"
            logging.error(error_msg)
            return "", 0, error_msg

    def _parse_streaming_response(self, stream: Iterator[types.SimpleNamespace]) -> tuple[str, int, str]:
        """
        Parse a streaming API response by consuming the stream iterator.
        
        Args:
            stream: An iterator that yields structured chunk objects.
            
        Returns:
            A tuple containing (final_response_text, token_count, error_message).
        """
        complete_response = ""
        final_chunk = None
        
        try:
            for chunk in stream:
                if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                    complete_response += chunk.message.content
                if hasattr(chunk, 'done') and chunk.done:
                    final_chunk = chunk
            
            if not complete_response or not complete_response.strip():
                raise ValueError("Empty response received from streaming API")
            
            tokens = 0
            if final_chunk:
                prompt_tokens = getattr(final_chunk, 'prompt_eval_count', 0)
                eval_tokens = getattr(final_chunk, 'eval_count', 0)
                tokens = prompt_tokens + eval_tokens
                
            return complete_response, tokens, None
            
        except Exception as e:
            error_msg = f"Failed to process stream: {e}"
            logging.error(error_msg, exc_info=True)
            return "", 0, error_msg


class BaseEmbeddingModel:
    """Base class for embedding model implementations.
    
    Provides common interface for embedding generation across different providers.
    All embedding model implementations should inherit from this class.
    """
    
    def __init__(self, credentials: Dict) -> None:
        self.credentials = credentials
        
    def generate_embeddings(
            self,
            text: Union[str, List[str]],
            files: Optional[List[str]] = None
    ) -> tuple[List[List[float]], int, str]:
        """Generate embedding vectors.
        
        Args:
            text: Input text(s) to generate embeddings for.
            files: Optional list of file paths for multimodal embeddings.
            
        Returns:
            Tuple containing (embeddings, status_code, error_message).
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    OpenAI embedding model handler.
    
    Handles text embedding generation using OpenAI's embedding models
    or other compatible endpoints.
    NOTE: Now only supports text embeddings, not multimodal embeddings.
    """
    
    def __init__(self, credentials: Dict) -> None:
        """
        Initialize the model handler with credentials.
        
        Args:
            credentials: Dictionary containing API credentials and configuration.
                         Expected keys: 'api_key', 'base_url'.
        """
        super().__init__(credentials)
        api_key = credentials.get('api_key')
        base_url = credentials.get('base_url')
        
        if base_url.endswith('/'):
            base_url = base_url[:-1]
        endpoint_url = f"{base_url}/embeddings"

        if not api_key or not base_url:
            raise ValueError("Credentials must include 'api_key' and 'base_url'.")

        self.endpoint_url = endpoint_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to a base64 data URL.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded data URL string.
        """
        with open(image_path, "rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode('utf-8')
            img_format = image_path.split('.')[-1]
            return f"data:image/{img_format};base64,{base64_str}"
        
    def generate_embeddings(
            self,
            text: Union[str, List[str]],
            files: Optional[List[str]] = None
    ) -> tuple[List[List[float]], int, str]:
        """
        Generate embedding vectors for text input.
        
        Note: The 'files' parameter is not used as only text embeddings are supported.
        
        Args:
            text: A single text string or a list of text strings to be embedded.
            files: An optional list of image file paths (not supported).
            
        Returns:
            A tuple containing (list_of_embeddings, tokens_used, error_message).
        """
        max_retries = 3
        retry_delay = 10
        
        if isinstance(text, str):
            input_texts = [text]
        else:
            input_texts = text
        
        payload = {
            "model": self.credentials.get('model_name'),
            "input": input_texts,
        }
            
        for attempt in range(max_retries):
            try:
                response = self.session.post(self.endpoint_url, json=payload, timeout=DEFAULT_TIMEOUT_TIME)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                
                response_data = response.json()
                
                # Extract embedding vectors and usage data
                embeddings = [item['embedding'] for item in response_data.get('data', [])]
                tokens_used = response_data.get('usage', {}).get('total_tokens', 0)
                
                if not embeddings:
                    raise ValueError("API returned no embeddings in the 'data' field.")

                return embeddings, tokens_used, None
                
            except requests.exceptions.RequestException as e:
                # Handle network-related errors (timeout, connection error)
                error_msg = f"API request failed: {e}"
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    if attempt < max_retries - 1:
                        logging.warning(f"Network error ({error_msg}), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        final_error = f"API request failed after {max_retries} attempts: {e}"
                        logging.error(final_error)
                        return [], 0, final_error
                else:
                    # Handle other request exceptions (e.g., HTTP 4xx/5xx errors)
                    logging.error(f"{error_msg}. Response: {response.text if 'response' in locals() else 'N/A'}")
                    return [], 0, f"{error_msg}"
            
            except (ValueError, KeyError) as e:
                # Handle errors from parsing the response JSON
                error_msg = f"Failed to parse API response: {e}"
                logging.error(f"{error_msg}. Response data: {response.text if 'response' in locals() else 'N/A'}")
                return [], 0, error_msg

        # This line is reached only if all retries fail
        return [], 0, f"API request failed after {max_retries} retries."


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """
    Ollama embedding model handler.
    
    Handles text and multimodal embedding generation using Ollama models.
    """
    
    def __init__(self, credentials: Dict) -> None:
        """
        Initialize the model handler with credentials.
        
        Args:
            credentials: Dictionary containing API credentials and configuration.
                         Expected keys: 'api_key'(placeholder, not used), 'base_url(e.g., 'http://localhost:11434')'.
        """
        super().__init__(credentials)
        api_key = None
        base_url = credentials.get('base_url')
        
        if base_url.endswith('/'):
            base_url = base_url[:-1]
        endpoint_url = f"{base_url}/api/embeddings"

        if not api_key or not base_url:
            raise ValueError("Credentials must include 'api_key' and 'base_url'.")

        self.endpoint_url = endpoint_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })
        
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to a base64 string.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded string of the image.
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
        
    def generate_embeddings(
            self,
            text: Union[str, List[str]],
            files: Optional[List[str]] = None
    ) -> tuple[List[List[float]], int, str]:
        """
        Generate text or multimodal embedding vectors.
        
        Args:
            text: A single text string or a list of text strings.
            files: An optional list of image file paths for multimodal embeddings.
            
        Returns:
            A tuple containing (list_of_embeddings, tokens_used, error_message).
        """
        max_retries = 3
        retry_delay = 10
        
        if isinstance(text, str):
            input_texts = [text]
        else:
            input_texts = text
            
        all_embeddings = []
        total_tokens = 0
        
        # Pre-encode images once if they are provided
        encoded_images = [self._encode_image(f) for f in files] if files else None
        
        for t in input_texts:
            payload = {
                "model": self.credentials.get('model_name', 'nomic-embed-text'),
                "prompt": t,
            }
            
            if encoded_images:
                payload["images"] = encoded_images
                
            for attempt in range(max_retries):
                try:
                    response = self.session.post(self.endpoint_url, json=payload, timeout=60)
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    
                    response_data = response.json()
                    
                    # Extract embedding vector and usage data
                    if 'embedding' not in response_data:
                        raise ValueError("API response did not contain an 'embedding' field.")
                    
                    all_embeddings.append(response_data['embedding'])
                    total_tokens += response_data.get('eval_count', 0)
                    
                    break  # Success, exit the retry loop for this text
                    
                except requests.exceptions.RequestException as e:
                    error_msg = f"API request failed for prompt '{t[:30]}...': {e}"
                    if "timeout" in str(e).lower() or "connection" in str(e).lower():
                        if attempt < max_retries - 1:
                            logging.warning(f"Network error ({error_msg}), retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                        else:
                            final_error = f"API request failed after {max_retries} attempts: {e}"
                            logging.error(final_error)
                            return [], 0, final_error
                    else:
                        logging.error(f"{error_msg}. Response: {response.text if 'response' in locals() else 'N/A'}")
                        return [], 0, str(error_msg)
                
                except (ValueError, KeyError) as e:
                    error_msg = f"Failed to parse API response for prompt '{t[:30]}...': {e}"
                    logging.error(f"{error_msg}. Response data: {response.text if 'response' in locals() else 'N/A'}")
                    return [], 0, str(error_msg)
            else:
                # This block executes if the retry loop completes without a `break`
                return [], 0, f"All retry attempts failed for a prompt."

        return all_embeddings, total_tokens, None


def call_language_model(
        model_provider: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        stream: bool = False,
        collect: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        files: Optional[List[str]] = None,
        skip_model_checking: bool = False,
        config_path: Optional[str] = r'./llm_config.yaml',
        custom_config: Optional[Dict] = None,
        **kwargs
) -> tuple[str, int, str]:
    """Unified entry function for calling language models.
    
    Import this function into your code to use. Do not use this function to call embedding models.
    
    Args:
        model_provider: Model provider like "openai", "aliyun", "volcengine", "ollama". OpenAI uses /reponses endpoint, and other OpenAI Compatible ones use /chat/completions endpoint.
        model_name: Model name, note that some providers may include version numbers.
        system_prompt: System instruction.
        user_prompt: User input text.
        stream: Whether to use streaming mode, optional.
        collect: Whether to collect streaming results (only effective when stream=True).
                Default True for collected results, False for true streaming requiring manual collection.
        temperature: Sampling temperature, optional.
        max_tokens: Maximum tokens to generate, optional.
        files: List of image file paths, optional.
        skip_model_checking: Whether to skip model name validation (default False).
                            When True, uses provided model name directly without checking configuration.
        config_path: Configuration file path, mutually exclusive with custom_config.
        custom_config: Custom configuration dict instead of file, mutually exclusive with config_path.
                      Must contain base_url and api_key fields. Overrides config_path when valid.
        **kwargs: Any additional parameters will be passed directly to the underlying API call. For example, enable_thinking, reasoning_effort, max_output_tokens...
        
    Returns:
        Tuple containing (response_text, tokens_used, error_message).
        For streaming: returns stream object when collect=False, otherwise collected result.
    """
    # Validate parameters: config_path and custom_config cannot both be None
    if config_path is None and custom_config is None:
        error_msg = "Both config_path and custom_config cannot be None. Please provide either a config file path or custom configuration."
        print(error_msg)
        logging.error(error_msg)
        return "", 0, error_msg

    # Initialize
    if custom_config is not None:
        # Use custom configuration
        credentials = {
            'provider': model_provider,
            'model_name': model_name,
            'api_key': custom_config.get('api_key', ''),
            'base_url': custom_config.get('base_url', '')
        }
    else:
        # Use configuration file
        config = ModelConfig(config_path)
        credentials = config.get_credentials(model_provider, model_name, skip_model_checking)

    if not credentials:
        error_msg = f"Model {model_name} not found in config"
        print(error_msg)
        logging.error(error_msg)
        return "", 0, error_msg

    if model_provider.lower() == "ollama":
        model_class = OllamaModel
    elif model_provider.lower() == "openai":
        model_class = OpenAIModel
    else:
        model_class = OpenAICompatibleModel

    if not model_class:
        error_msg = f"Unsupported model provider: {model_provider}"
        logging.error(error_msg)
        print(error_msg)
        return "", 0, error_msg

    model = model_class(credentials)

    try:
        if stream:
            result = model.generate_stream(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                collect=collect,
                files=files,
                **kwargs
            )
        else:
            result = model.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                files=files,
                **kwargs
            )
        # Log successful API call
        _, tokens, _ = result
        logging.info(f"API call succeeded. Model: {model_name}, Provider: {model_provider}, Tokens used: {tokens}")
        return result
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        return "", 0, error_msg


def call_embedding_model(
        model_provider: str,
        model_name: str,
        text: Union[str, List[str]],
        files: Optional[List[str]] = None,
        skip_model_checking: bool = False,
        config_path: Optional[str] = r'./llm_config.yaml',
        custom_config: Optional[Dict] = None,
) -> tuple[List[List[float]], int, str]:
    """Unified entry function for generating embedding vectors.
    
    Import this function into your code to use. Does not support streaming calls.
    Only supports embedding models, do not pass language dialog models to this function.
    
    Args:
        model_provider: Model provider like "openai" or "ollama".
        model_name: Embedding model name.
        text: Single text string or list of text strings.
        files: List of image file paths, optional.
        skip_model_checking: Whether to skip model name validation (default False).
        config_path: Configuration file path, mutually exclusive with custom_config.
        custom_config: Custom configuration dict instead of file, mutually exclusive with config_path.
        
    Returns:
        Tuple containing (embeddings, tokens_used, error_message).
    """
    # Validate parameters: config_path and custom_config cannot both be None
    if config_path is None and custom_config is None:
        error_msg = "Both config_path and custom_config cannot be None. Please provide either a config file path or custom configuration."
        print(error_msg)
        logging.error(error_msg)
        return [], 0, error_msg

    # Initialize
    if custom_config is not None:
        # Use custom configuration
        credentials = {
            'provider': model_provider,
            'model_name': model_name,
            'api_key': custom_config.get('api_key', ''),
            'base_url': custom_config.get('base_url', '')
        }
    else:
        # Use configuration file
        config = ModelConfig(config_path)
        credentials = config.get_embedding_credentials(model_provider, model_name, skip_model_checking)

    if not credentials:
        error_msg = f"Embedding model {model_name} not found in config"
        print(error_msg)
        logging.error(error_msg)
        return [], 0, error_msg

    if model_provider == "ollama":
        model_class = OllamaEmbeddingModel
    else:
        model_class = OpenAIEmbeddingModel

    if not model_class:
        error_msg = f"Unsupported embedding model provider: {model_provider}"
        logging.error(error_msg)
        print(error_msg)
        return [], 0, error_msg

    model = model_class(credentials)

    try:
        result = model.generate_embeddings(
            text=text,
            files=files
        )
        # Log successful API call
        embeddings, tokens, error = result
        logging.info(f"Embedding API call succeeded. Model: {model_name}, Provider: {model_provider}, Tokens used: {tokens}")
        return result
    except Exception as e:
        error_msg = f"Unexpected error in embedding generation: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        return [], 0, error_msg


def batch_call_language_model(
        model_provider: str,
        model_name: str,
        requests: List[Dict],
        max_workers: int = 5,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        skip_model_checking: bool = False,
        config_path: Optional[str] = r'./llm_config.yaml',
        custom_config: Optional[Dict] = None,
        output_file: Optional[str] = None,
        show_progress: bool = True,
        **kwargs
) -> List[Dict]:
    """Batch parallel calling language model unified entry function.
    This is a wrap of real-time calling, not async batch calling (which will make you wait for hours before getting response),
      so it will cost the same money as real-time calling.
    This mode does not support true streaming calls.
    
    Args:
        model_provider: Model provider like "openai", "aliyun", "volcengine", "ollama". OpenAI uses /reponses endpoint, and other OpenAI Compatible ones use /chat/completions endpoint.
        model_name: Model name, note that some providers may include version numbers.
        requests: List of request dictionaries containing system_prompt, user_prompt and optional files field.
                 Format: [{"system_prompt": "...", "user_prompt": "...", "files": [...]}, ...]
        max_workers: Maximum number of parallel worker threads (default 5).
        stream: Whether to use streaming mode (default False). When True, collects and returns streaming responses.
        temperature: Sampling temperature, optional.
        max_tokens: Maximum tokens to generate, optional.
        skip_model_checking: Whether to skip model name validation (default False).
        config_path: Configuration file path, mutually exclusive with custom_config.
        custom_config: Custom configuration dict instead of file, mutually exclusive with config_path.
        output_file: Output JSONL file path, optional. If provided, saves all results to specified file.
        show_progress: Whether to show progress bar (default True).
        **kwargs: Any additional parameters will be passed directly to the underlying API call. For example, enable_thinking, reasoning_effort, max_output_tokens...
        
    Returns:
        List of result dictionaries containing request_index, response_text, tokens_used, error_msg fields.
    """
    if not requests:
        logging.warning("Empty requests list provided in batch_call_language_model.")
        return []

    # Validate request format
    for i, req in enumerate(requests):
        if not isinstance(req, dict):
            error_msg = f"Request {i} is not a dictionary"
            logging.error(error_msg)
            return [{"request_index": i, "response_text": "", "tokens_used": 0, "error_msg": error_msg} for i in range(len(requests))]
        
        if 'system_prompt' not in req or 'user_prompt' not in req:
            error_msg = f"Request {i} missing required fields: system_prompt or user_prompt"
            logging.error(error_msg)
            return [{"request_index": i, "response_text": "", "tokens_used": 0, "error_msg": error_msg} for i in range(len(requests))]

    def _process_single_request(request_data):
        """Process single request internal function."""
        index, request = request_data
        try:
            response, tokens, error = call_language_model(
                model_provider=model_provider,
                model_name=model_name,
                system_prompt=request['system_prompt'],
                user_prompt=request['user_prompt'],
                stream=stream,
                collect=True,
                temperature=temperature,
                max_tokens=max_tokens,
                files=request.get('files', None),
                skip_model_checking=skip_model_checking,
                config_path=config_path,
                custom_config=custom_config,
                **kwargs
            )
            
            result = {
                "request_index": index,
                "request": request,  # Save original request
                "response_text": response,
                "tokens_used": tokens,
                "error_msg": error,
                "model_provider": model_provider,
                "model_name": model_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error processing request {index}: {str(e)}"
            logging.error(error_msg)
            return {
                "request_index": index,
                "request": request,
                "response_text": "",
                "tokens_used": 0,
                "error_msg": error_msg,
                "model_provider": model_provider,
                "model_name": model_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

    # Use thread pool for parallel request processing
    results = [None] * len(requests)  # Pre-allocate results list
    
    # Create progress bar
    if show_progress:
        pbar = tqdm(total=len(requests), desc="Processing requests", unit="req")
    
    # Create file lock for thread-safe writing
    file_lock = threading.Lock()
    
    # Initialize output file if specified (clear existing content)
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                pass  # Create/clear the file
            logging.info(f"Initialized output file: {output_file}")
        except Exception as e:
            error_msg = f"Failed to initialize output file {output_file}: {str(e)}"
            logging.error(error_msg)
            if show_progress:
                print(f"Warning: {error_msg}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(_process_single_request, (i, req)): i 
            for i, req in enumerate(requests)
        }
        
        # Collect results
        for future in as_completed(future_to_index):
            try:
                result = future.result()
                results[result["request_index"]] = result
                
                # Real-time save to file if output_file is specified
                if output_file and result:
                    try:
                        with file_lock:  # Thread-safe file writing
                            with open(output_file, 'a', encoding='utf-8') as f:
                                json.dump(result, f, ensure_ascii=False)
                                f.write('\n')
                                f.flush()  # Ensure immediate write to disk
                    except Exception as e:
                        error_msg = f"Failed to save result {result['request_index']} to {output_file}: {str(e)}"
                        logging.error(error_msg)
                        # Don't stop processing for file write errors, just log
                
                # Update progress bar
                if show_progress:
                    pbar.update(1)
                    # Display current status information
                    successful = sum(1 for r in results if r and not r.get("error_msg"))
                    failed = sum(1 for r in results if r and r.get("error_msg"))
                    pbar.set_postfix({"Success": successful, "Failed": failed})
                    
            except Exception as e:
                index = future_to_index[future]
                error_msg = f"Thread execution error for request {index}: {str(e)}"
                logging.error(error_msg)
                error_result = {
                    "request_index": index,
                    "request": requests[index] if index < len(requests) else {},
                    "response_text": "",
                    "tokens_used": 0,
                    "error_msg": error_msg,
                    "model_provider": model_provider,
                    "model_name": model_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                results[index] = error_result
                
                # Real-time save error result to file if output_file is specified
                if output_file:
                    try:
                        with file_lock:  # Thread-safe file writing
                            with open(output_file, 'a', encoding='utf-8') as f:
                                json.dump(error_result, f, ensure_ascii=False)
                                f.write('\n')
                                f.flush()  # Ensure immediate write to disk
                    except Exception as file_e:
                        logging.error(f"Failed to save error result {index} to {output_file}: {str(file_e)}")
                
                # Update progress bar
                if show_progress:
                    pbar.update(1)
                    successful = sum(1 for r in results if r and not r.get("error_msg"))
                    failed = sum(1 for r in results if r and r.get("error_msg"))
                    pbar.set_postfix({"Success": successful, "Failed": failed})

    # Close progress bar
    if show_progress:
        pbar.close()

    # Log batch call results
    total_tokens = sum(r.get("tokens_used", 0) for r in results if r)
    successful_requests = sum(1 for r in results if r and not r.get("error_msg"))
    
    logging.info(f"Batch API call completed. Model: {model_name}, Provider: {model_provider}, "
                f"Total requests: {len(requests)}, Successful: {successful_requests}, "
                f"Total tokens used: {total_tokens}")
    
    # File saving is now done in real-time during processing
    if output_file and show_progress:
        print(f"Results saved to {output_file} in real-time")
    
    return results