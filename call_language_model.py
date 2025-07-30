#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Language model API unified calling tool.

This module provides a unified interface for calling various language models
and embedding models through OpenAI-compatible APIs and Ollama.

@File    : call_language_model.py
@Author  : Zhangxiao Shen
@Date    : 2025/7/31
@Description: Call language models and embedding models using OpenAI or Ollama APIs.
"""

import base64
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

import ollama
import yaml
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm

# 配置文件格式：llm_config.yaml，需要放在检查本文件所在路径内或者指定其路径
# 当前支持多种模型提供商，也可自行添加提供商和模型名称，但仅支持openai和ollama两种渠道调用模型
# 支持流式调用，设置参数collect=True会将流式调用的结果收集后返回，False会将整个流返回
# 流式调用时不支持统计token消耗
# 使用大语言模型的入口函数为call_language_model
# 支持使用嵌入模型，需使用call_embedding_model函数调用，暂不支持多模态嵌入
# 支持批量并行调用大语言模型，使用batch_call_language_model函数，该模式下不支持真正的流式调用
# 批量调用支持tqdm进度条显示和结果保存到JSONL文件

# 处理OpenAI真流式响应的示例代码
#     is_first_chunk = True
#     for chunk in response:
#         delta = chunk.choices[0].delta
#         if hasattr(delta, 'reasoning_content'):
#             if delta.reasoning_content is not None:
#                 if is_first_chunk:
#                     print("\n<think>\n")
#                     is_first_chunk = False
#                 print(delta.reasoning_content, end='', flush=True)
#             else:
#                 if not is_first_chunk:
#                     print("\n</think>\n")
#                     is_first_chunk = True
#                 print(delta.content, end='', flush=True)
#         elif hasattr(delta, 'content'):
#             print(delta.content, end='', flush=True)
#         else:
#             print(delta)

# 处理Ollama真流式响应的示例代码
#     for chunk in response:
#         if hasattr(chunk, 'message') and chunk.message and hasattr(chunk.message, 'content'):
#             content = chunk.message.content
#             print(content, end='', flush=True)
#         else:
#             print(chunk)

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

# Logging configuration
logging.basicConfig(
    filename=DEFAULT_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


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
            enable_thinking: Optional[bool] = None,
            files: Optional[List[str]] = None
    ) -> tuple[str, int, str]:
        """Generate text response from the model.
        
        Args:
            system_prompt: System instruction for the model.
            user_prompt: User input text.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in response.
            enable_thinking: Whether to enable reasoning mode.
            files: List of file paths for multimodal input.
            
        Returns:
            Tuple containing (response_text, status_code, error_message).
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class OpenAIModel(BaseModel):
    """OpenAI-compatible model handler.
    
    Handles text and multimodal interactions with OpenAI-compatible APIs
    including OpenAI, Alibaba Cloud, Volcengine, and other compatible providers.
    """

    def __init__(self, credentials: Dict) -> None:
        """Initialize OpenAI model with credentials.
        
        Args:
            credentials: Dictionary containing API credentials and configuration.
        """
        super().__init__(credentials)
        self.client = OpenAI(
            api_key=credentials.get('api_key', ''),
            base_url=credentials.get('base_url', 'https://api.openai.com/v1')
        )

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string in OpenAI API format.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded string with data URL format for OpenAI API.
        """
        with open(image_path, "rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode('utf-8')
            img_format = img_file.name.split('.')[-1]
            result_str = f"data:image/{img_format};base64,{base64_str}"
            return result_str

    def _prepare_messages(self, **kwargs) -> list:
        """Prepare message format for OpenAI API calls.
        
        Handles both text-only and multimodal (text + images) requests.
        
        Args:
            **kwargs: Keyword arguments containing system_prompt, user_prompt, and optional files.
            
        Returns:
            List of message dictionaries formatted for OpenAI API.
        """
        messages = [{"role": "system", "content": kwargs['system_prompt']}]
        if kwargs.get('files'):
            # Handle multimodal requests
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": kwargs['user_prompt']},
                    *[{"type": "image_url", "image_url": {"url": f"{self._encode_image(f)}"}} for f in kwargs['files']]
                ]
            })
        else:
            messages.append({"role": "user", "content": kwargs['user_prompt']})
        return messages

    def _prepare_params(self, messages, **kwargs) -> dict:
        """Prepare API parameters for OpenAI API calls.
        
        Handles model-specific configurations and filters out None values.
        
        Args:
            messages: List of message dictionaries.
            **kwargs: Additional parameters like temperature, max_tokens, etc.
            
        Returns:
            Dictionary of API parameters with None values filtered out.
        """
        if "qwen3" in str(self.credentials.get('model_name')):
            enable_thinking = kwargs.get('enable_thinking', True)
            extra_body = {"enable_thinking": enable_thinking}
        else:
            extra_body = None
        params = {
            "model": self.credentials.get('model_name', 'gpt-4o'),
            "messages": messages,
            "temperature": kwargs.get('temperature'),
            "max_tokens": kwargs.get('max_tokens'),
            "extra_body": extra_body
        }
        return {k: v for k, v in params.items() if v is not None}

    def generate(self, **kwargs) -> tuple[str, int, str]:
        """Generate non-streaming response.
        
        Args:
            **kwargs: Keyword arguments including system_prompt, user_prompt, etc.
            
        Returns:
            Tuple containing (response_text, status_code, error_message).
        """
        messages = self._prepare_messages(**kwargs)
        params = self._prepare_params(messages, **kwargs)

        max_retries = 3
        retry_delay = 10
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                return self._parse_response(response)
            except Exception as e:
                str_e = str(e).lower()
                if "timeout" in str_e or "connection error" in str_e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Network error: {str(e)}, retrying in {retry_delay}s...")
                        print(f"Network error: {str(e)}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        error_msg = f"API request failed after {max_retries} attempts: {str(e)}"
                        print(f"API request failed after {max_retries} attempts: {str(e)}")
                        logging.error(error_msg)
                        return "", 0, error_msg
                elif "NoneType" in str_e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Error: Did not receive response object, retrying in {retry_delay}s...")
                        print(f"Network error: {str(e)}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        error_msg = f"API request failed after {max_retries} attempts: {str(e)}"
                        print(f"API request failed after {max_retries} attempts: {str(e)}")
                        logging.error(error_msg)
                        return "", 0, error_msg
                else:
                    error_msg = f"OpenAI API error: {str(e)}"
                    print(f"OpenAI API error: {str(e)}")
                    logging.error(error_msg)
                    return "", 0, error_msg

    def generate_stream(self, **kwargs) -> tuple[str, int, str]:
        """Generate streaming response.
        
        Supports both collected (returns final result) and streaming modes.
        When collect=True, returns format same as non-streaming, otherwise returns stream object.
        
        Args:
            **kwargs: Keyword arguments including system_prompt, user_prompt, collect, etc.
            
        Returns:
            Tuple containing (response_text, status_code, error_message).
        """
        messages = self._prepare_messages(**kwargs)
        params = self._prepare_params(messages, **kwargs)
        params["stream"] = True

        complete_response = ""
        reasoning_content = ""
        estimated_tokens = 0

        max_retries = 3
        retry_delay = 10

        collect_stream_answer = kwargs.get('collect', True)

        # 删除collect参数，避免OpenAI API报错
        if 'collect' in kwargs:
            del kwargs['collect']

        for attempt in range(max_retries):
            try:
                stream = self.client.chat.completions.create(**params)

                if not collect_stream_answer:
                    # 如果不收集流式结果，直接返回stream对象
                    return stream, 0, None
                else:
                    for chunk in stream:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta

                            # 处理文本内容
                            if hasattr(delta, 'content') and delta.content is not None:
                                content = delta.content
                                complete_response += content

                            # 处理reasoning_content（如果有）
                            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                                reasoning_content += delta.reasoning_content

                    # 注意流式调用不支持token消耗统计
                    tokens = 0

                    # 流结束后，如果有reasoning_content，将其添加到完整响应中
                    if reasoning_content:
                        complete_response = f"<think>\n{reasoning_content}\n</think>\n\n{complete_response}"

                    return complete_response, tokens, None

            except Exception as e:
                str_e = str(e).lower()
                if "timeout" in str_e or "connection error" in str_e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Network error: {str(e)}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        error_msg = f"API request failed after {max_retries} attempts: {str(e)}"
                        logging.error(error_msg)
                        return complete_response, int(estimated_tokens), error_msg
                else:
                    error_msg = f"OpenAI API error: {str(e)}"
                    logging.error(error_msg)
                    return complete_response, int(estimated_tokens), error_msg

    def _parse_response(self, response: ChatCompletion) -> tuple[str, int, str]:
        """Parse OpenAI API response.
        
        Args:
            response: ChatCompletion response object from OpenAI API.
            
        Returns:
            Tuple containing (response_text, token_count, error_message).
        """
        complete_response = response.choices[0].message.content
        if hasattr(response.choices[0].message, 'reasoning_content'):
            complete_response = "<think>\n" + str(response.choices[0].message.reasoning_content) + "\n</think>\n\n" + \
                                response.choices[0].message.content
        return (
            complete_response,
            response.usage.total_tokens if response.usage else 0,
            None
        )


class OllamaModel(BaseModel):
    """Ollama local model handler.
    
    Handles interactions with locally hosted Ollama models including multimodal support.
    """

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string for Ollama API.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded string of the image.
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def generate(self, **kwargs) -> tuple[str, int, str]:
        """Generate non-streaming response from Ollama model.
        
        Args:
            **kwargs: Keyword arguments including system_prompt, user_prompt, etc.
            
        Returns:
            Tuple containing (response_text, status_code, error_message).
        """
        # Construct messages
        messages = [{"role": "system", "content": kwargs.get('system_prompt')}]
        user_prompt_content = kwargs.get('user_prompt')
        if "qwen3" in str(self.credentials.get('model_name')):
            enable_thinking = kwargs.get('enable_thinking', True)
            if not enable_thinking:
                user_prompt_content += " /no_think"
        else:
            enable_thinking = None

        if kwargs.get('files'):
            image_encoded = [self._encode_image(f) for f in kwargs['files']]
            # Handle multimodal requests
            messages.append({
                "role": "user",
                "content": user_prompt_content,
                "images": kwargs['files'],
            })
        else:
            messages.append({"role": "user", "content": user_prompt_content})

        options = {
            "temperature": kwargs.get('temperature'),
            "num_predict": kwargs.get('max_tokens')
        }

        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}

        try:
            response = ollama.chat(
                model=self.credentials.get('model_name', 'llama3.1:8b'),
                messages=messages,
                options=options,
            )
            return self._parse_response(response, enable_thinking)
        except Exception as e:
            logging.error(f"Ollama API error: {str(e)}")
            print(f"Ollama API error: {str(e)}")
            return "", 0, str(e)

    def generate_stream(self, **kwargs) -> tuple[str, int, str]:
        """Generate streaming response from Ollama model.
        
        Supports both collected (returns final result) and streaming modes.
        When collect=True, returns format same as non-streaming, otherwise returns stream object.
        
        Args:
            **kwargs: Keyword arguments including system_prompt, user_prompt, collect, etc.
            
        Returns:
            Tuple containing (response_text, status_code, error_message).
        """
        # Construct messages
        messages = [{"role": "system", "content": kwargs.get('system_prompt')}]
        if "qwen3" in str(self.credentials.get('model_name')):
            enable_thinking = kwargs.get('enable_thinking', True)
            if not enable_thinking:
                user_prompt_content += " /no_think"
        else:
            enable_thinking = None

        if kwargs.get('files'):
            image_encoded = [self._encode_image(f) for f in kwargs['files']]
            # Handle multimodal requests
            messages.append({
                "role": "user",
                "content": kwargs.get('user_prompt'),
                "images": kwargs['files'],
            })
        else:
            messages.append({"role": "user", "content": kwargs.get('user_prompt')})

        options = {
            "temperature": kwargs.get('temperature'),
            "num_predict": kwargs.get('max_tokens'),
        }

        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}

        complete_response = ""
        estimated_tokens = 0

        max_retries = 3
        retry_delay = 10

        collect_stream_answer = kwargs.get('collect', True)

        # Remove collect parameter to avoid Ollama API error
        if 'collect' in kwargs:
            del kwargs['collect']

        for attempt in range(max_retries):
            try:
                stream = ollama.chat(
                    model=self.credentials.get('model_name', 'llama3.1:8b'),
                    messages=messages,
                    options=options,
                    stream=True,
                )

                if not collect_stream_answer:
                    # If not collecting stream results, return stream object directly
                    return stream, 0, None
                else:
                    # Collect streaming results
                    for chunk in stream:
                        if hasattr(chunk, 'message') and chunk.message and hasattr(chunk.message, 'content'):
                            content = chunk.message.content
                            complete_response += content

                    # Note: streaming calls don't support precise token consumption statistics
                    tokens = 0
                    if hasattr(stream, 'eval_count') and hasattr(stream, 'prompt_eval_count'):
                        tokens = stream.eval_count + stream.prompt_eval_count
                    
                    if not enable_thinking:
                        complete_response = complete_response.replace("<think>\n", "").replace("\n</think>\n\n", "")

                    return complete_response, tokens, None

            except Exception as e:
                str_e = str(e).lower()
                if "timeout" in str_e or "connection error" in str_e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Network error: {str(e)}, retrying in {retry_delay}s...")
                        print(f"Network error: {str(e)}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        error_msg = f"API request failed after {max_retries} attempts: {str(e)}"
                        print(f"API request failed after {max_retries} attempts: {str(e)}")
                        logging.error(error_msg)
                        return complete_response, int(estimated_tokens), error_msg
                else:
                    error_msg = f"Ollama API error: {str(e)}"
                    print(f"Ollama API error: {str(e)}")
                    logging.error(error_msg)
                    return complete_response, int(estimated_tokens), error_msg

    def _parse_response(self, response, enable_thinking) -> tuple[str, int, str]:
        tokens_used = response.eval_count + response.prompt_eval_count
        complete_response = response.message.content
        if not enable_thinking:
            complete_response = complete_response.replace("<think>\n", "").replace("\n</think>\n\n", "")
        return (
            complete_response,
            tokens_used,
            None
        )


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
    """OpenAI embedding model handler.
    
    Handles text embedding generation using OpenAI's embedding models.
    Note: Only supports text embeddings, not multimodal embeddings.
    """
    
    def __init__(self, credentials: Dict) -> None:
        super().__init__(credentials)
        self.client = OpenAI(
            api_key=credentials.get('api_key'),
            base_url=credentials.get('base_url')
        )
        
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string in OpenAI API format.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64 encoded string with data URL format.
        """
        with open(image_path, "rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode('utf-8')
            img_format = img_file.name.split('.')[-1]
            result_str = f"data:image/{img_format};base64,{base64_str}"
            return result_str
        
    def generate_embeddings(
            self,
            text: Union[str, List[str]],
            files: Optional[List[str]] = None
    ) -> tuple[List[List[float]], int, str]:
        """Generate embedding vectors for text input.
        
        Note: Only supports text embeddings, not multimodal embeddings.
        
        Args:
            text: Single text string or list of text strings.
            files: List of image file paths, optional (not supported by OpenAI).
            
        Returns:
            Tuple containing (embeddings, tokens_used, error_message).
        """
        max_retries = 3
        retry_delay = 10
        
        # Ensure text is in list format
        if isinstance(text, str):
            input_texts = [text]
        else:
            input_texts = text
            
        input_data = input_texts
            
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.credentials.get('model_name'),
                    input=input_data,
                )
                
                # 提取嵌入向量
                embeddings = [data.embedding for data in response.data]
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
                
                return embeddings, tokens_used, None
                
            except Exception as e:
                str_e = str(e).lower()
                if "timeout" in str_e or "connection error" in str_e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Network error: {str(e)}, retrying in {retry_delay}s...")
                        print(f"Network error: {str(e)}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        error_msg = f"API request failed after {max_retries} attempts: {str(e)}"
                        print(f"API request failed after {max_retries} attempts: {str(e)}")
                        logging.error(error_msg)
                        return [], 0, error_msg
                else:
                    error_msg = f"OpenAI Embedding API error: {str(e)}"
                    print(f"OpenAI Embedding API error: {str(e)}")
                    logging.error(error_msg)
                    return [], 0, error_msg


class OllamaEmbeddingModel(BaseEmbeddingModel):
    """Ollama embedding model handler.
    
    Handles text and multimodal embedding generation using Ollama models.
    """
    
    def __init__(self, credentials: Dict) -> None:
        super().__init__(credentials)
        
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string.
        
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
        """Generate text or multimodal embedding vectors.
        
        Args:
            text: Single text string or list of text strings.
            files: List of image file paths, optional.
            
        Returns:
            Tuple containing (embeddings, tokens_used, error_message).
        """
        max_retries = 3
        retry_delay = 10
        
        # Ensure text is in list format
        if isinstance(text, str):
            input_texts = [text]
        else:
            input_texts = text
            
        all_embeddings = []
        total_tokens = 0
        
        for t in input_texts:
            # Prepare request parameters
            params = {
                "model": self.credentials.get('model_name', 'nomic-embed-text'),
                "prompt": t,
            }
            
            # Add images (if any)
            if files:
                params["images"] = files
                
            for attempt in range(max_retries):
                try:
                    # Call Ollama API to generate embedding vectors
                    response = ollama.embeddings(**params)
                    
                    # Extract embedding vectors
                    if hasattr(response, 'embedding'):
                        all_embeddings.append(response.embedding)
                    else:
                        all_embeddings.append(response)
                        
                    # Ollama may not provide token usage
                    if hasattr(response, 'eval_count'):
                        total_tokens += response.eval_count
                        
                    break  # Successfully obtained embedding vectors, exit retry loop
                    
                except Exception as e:
                    str_e = str(e).lower()
                    if "timeout" in str_e or "connection error" in str_e:
                        if attempt < max_retries - 1:
                            logging.warning(f"Network error: {str(e)}, retrying in {retry_delay}s...")
                            print(f"Network error: {str(e)}, retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                        else:
                            error_msg = f"API request failed after {max_retries} attempts: {str(e)}"
                            print(f"API request failed after {max_retries} attempts: {str(e)}")
                            logging.error(error_msg)
                            return [], 0, error_msg
                    else:
                        error_msg = f"Ollama Embedding API error: {str(e)}"
                        print(f"Ollama Embedding API error: {str(e)}")
                        logging.error(error_msg)
                        return [], 0, error_msg
        
        return all_embeddings, total_tokens, None


def call_language_model(
        model_provider: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        stream: bool = False,
        collect: bool = True,
        enable_thinking: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        files: Optional[List[str]] = None,
        skip_model_checking: bool = False,
        config_path: Optional[str] = r'./llm_config.yaml',
        custom_config: Optional[Dict] = None,
) -> tuple[str, int, str]:
    """Unified entry function for calling language models.
    
    Import this function into your code to use. Do not use this function to call embedding models.
    
    Args:
        model_provider: Model provider like "openai", "aliyun", "volcengine", "ollama".
        model_name: Model name, note that some providers may include version numbers.
        system_prompt: System instruction.
        user_prompt: User input text.
        stream: Whether to use streaming mode, optional.
        collect: Whether to collect streaming results (only effective when stream=True).
                Default True for collected results, False for true streaming requiring manual collection.
        enable_thinking: Whether to enable reasoning mode (only effective for Qwen3 series, default True).
        temperature: Sampling temperature, optional.
        max_tokens: Maximum tokens to generate, optional.
        files: List of image file paths, optional.
        skip_model_checking: Whether to skip model name validation (default False).
                            When True, uses provided model name directly without checking configuration.
        config_path: Configuration file path, mutually exclusive with custom_config.
        custom_config: Custom configuration dict instead of file, mutually exclusive with config_path.
                      Must contain base_url and api_key fields. Overrides config_path when valid.
        
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

    if model_provider == "ollama":
        model_class = OllamaModel
    else:
        model_class = OpenAIModel

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
                enable_thinking=enable_thinking,
                collect=collect,
                files=files
            )
        else:
            result = model.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                files=files
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
        enable_thinking: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        skip_model_checking: bool = False,
        config_path: Optional[str] = r'./llm_config.yaml',
        custom_config: Optional[Dict] = None,
        output_file: Optional[str] = None,
        show_progress: bool = True,
) -> List[Dict]:
    """Batch parallel calling language model unified entry function.
    
    This mode does not support true streaming calls.
    
    Args:
        model_provider: Model provider like "openai", "aliyun", "volcengine", "ollama".
        model_name: Model name, note that some providers may include version numbers.
        requests: List of request dictionaries containing system_prompt, user_prompt and optional files field.
                 Format: [{"system_prompt": "...", "user_prompt": "...", "files": [...]}, ...]
        max_workers: Maximum number of parallel worker threads (default 5).
        stream: Whether to use streaming mode (default False). When True, collects and returns streaming responses.
        enable_thinking: Whether to enable reasoning mode (only effective for Qwen3 series, default True).
        temperature: Sampling temperature, optional.
        max_tokens: Maximum tokens to generate, optional.
        skip_model_checking: Whether to skip model name validation (default False).
        config_path: Configuration file path, mutually exclusive with custom_config.
        custom_config: Custom configuration dict instead of file, mutually exclusive with config_path.
        output_file: Output JSONL file path, optional. If provided, saves all results to specified file.
        show_progress: Whether to show progress bar (default True).
        
    Returns:
        List of result dictionaries containing request_index, response_text, tokens_used, error_msg fields.
    """
    if not requests:
        logging.warning("Empty requests list provided in batch_call_language_model.")
        return []

    from tqdm import tqdm

    # Validate request format
    for i, req in enumerate(requests):
        if not isinstance(req, dict):
            error_msg = f"Request {i} is not a dictionary"
            logging.error(error_msg)
            return [{"request_index": i, "response_text": "", "tokens_used": 0, "error_msg": error_msg} for i in range(len(requests))]
        
        if 'system_prompt' not in req or 'user_prompt' not in req:
            error_msg = f"Request {i} missing required fields: system_prompt and user_prompt"
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
                enable_thinking=enable_thinking,
                temperature=temperature,
                max_tokens=max_tokens,
                files=request.get('files', None),
                skip_model_checking=skip_model_checking,
                config_path=config_path,
                custom_config=custom_config,
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
                results[index] = {
                    "request_index": index,
                    "request": requests[index] if index < len(requests) else {},
                    "response_text": "",
                    "tokens_used": 0,
                    "error_msg": error_msg,
                    "model_provider": model_provider,
                    "model_name": model_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
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
    
    # Save results to JSONL file (if output file is specified)
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    if result:  # Ensure result is not None
                        json.dump(result, f, ensure_ascii=False)
                        f.write('\n')
            logging.info(f"Results saved to {output_file}")
            if show_progress:
                print(f"Results saved to {output_file}")
        except Exception as e:
            error_msg = f"Failed to save results to {output_file}: {str(e)}"
            logging.error(error_msg)
            if show_progress:
                print(f"Warning: {error_msg}")
    
    return results


if __name__ == "__main__":
    # 示例使用
    # 1. 调用语言模型示例
    # response, tokens_used, error = call_language_model(
    #     model_provider='openai',
    #     model_name='gpt-4.1-mini',
    #     system_prompt="You are a helpful assistant.",
    #     user_prompt="Introduce GEE", #非多模态
    #     enable_thinking=False, 
    #     stream=False,
    #     # collect=False,
    #     skip_model_checking=True,
    #     config_path="./llm_config.yaml",
    #     # custom_config={
    #     #     'api_key': 'sk-xxx',
    #     #     'base_url': 'https://api.openai.com/v1/'
    #     # }
    #     # user_prompt="Try to solve this problem with Python",
    #     # files=['1.png'] #多模态
    # )

    # print(f"\nResponse: {response}")
    # print(f"Tokens used: {tokens_used}")
    # if error:
    #     print(f"Error: {error}")

    # 处理真流式响应
    # is_first_chunk = True
    # for chunk in response:
    #     delta = chunk.choices[0].delta
    #     if hasattr(delta, 'reasoning_content'):
    #         if delta.reasoning_content is not None:
    #             if is_first_chunk:
    #                 print("\n<think>\n")
    #                 is_first_chunk = False
    #             print(delta.reasoning_content, end='', flush=True)
    #         else:
    #             if not is_first_chunk:
    #                 print("\n</think>\n")
    #                 is_first_chunk = True
    #             print(delta.content, end='', flush=True)
    #     elif hasattr(delta, 'content'):
    #         print(delta.content, end='', flush=True)
    #     else:
    #         print(delta)

    # for chunk in response:
    #     if hasattr(chunk, 'message') and chunk.message and hasattr(chunk.message, 'content'):
    #         content = chunk.message.content
    #         print(content, end='', flush=True)
    #     else:
    #         print(chunk)

    # 2. 嵌入模型使用示例
    # embeddings, tokens_used, error = call_embedding_model(
    #     model_provider='aliyun',
    #     model_name='text-embedding-v3',
    #     # text=["这是一段测试文本，用于生成嵌入向量", "This is a test text for generating embedding vectors"],
    #     config_path="llm_config.yaml",
    #     text= "这是一段测试文本，用于生成嵌入向量",
    #     files=['1.png'] # 多模态嵌入
    # )
    
    # print(f"\nEmbeddings dimension: {len(embeddings[0]) if embeddings else 0}")
    # print(f"Tokens used: {tokens_used}")
    # if error:
    #     print(f"Error: {error}")

    # 3. 批量调用语言模型示例
    # batch_requests = [
    #     {
    #         "system_prompt": "You are a helpful assistant.",
    #         "user_prompt": "What is Python?",
    #     },
    #     {
    #         "system_prompt": "You are a math tutor.",
    #         "user_prompt": "What is 2+2?",
    #     },
    #     {
    #         "system_prompt": "You are a travel guide.",
    #         "user_prompt": "Tell me about Paris.",
    #         "files": None
    #     }
    # ]
    
    # 批量调用
    # batch_results = batch_call_language_model(
    #     model_provider='openai',
    #     model_name='gpt-4.1-mini',
    #     requests=batch_requests,
    #     max_workers=3,
    #     output_file="batch_results.jsonl",
    #     show_progress=True,
    #     enable_thinking=False,
    #     skip_model_checking=True,
    #     config_path="./llm_config.yaml"
    # )
    
    # print(f"\nBatch processing completed:")
    # for result in batch_results:
    #     print(f"Request {result['request_index']}: "
    #           f"Response length: {len(result['response_text'])} chars, "
    #           f"Tokens: {result['tokens_used']}, "
    #           f"Error: {result['error_msg'] or 'None'}")

    print(1)
