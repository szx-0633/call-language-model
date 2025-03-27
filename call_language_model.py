import yaml
import logging
import time
from typing import Optional, Dict, List
from openai import OpenAI
from openai.types.chat import ChatCompletion
import base64
import ollama

# Author: Zhangxiao Shen, Claude-3.7-Sonnet, Qwen2.5-max, DeepSeek-R1
# 配置文件格式：llm_config.yaml，需要放在检查本文件所在路径内或者指定其路径
# 当前支持多种模型提供商，也可自行添加提供商和模型名称，但仅支持openai和ollama两种渠道调用模型
# 不支持多轮对话，也不支持真正的流式调用，对于必须采用流式调用的模型如QwQ-32B，会将流式调用的结果收集后返回
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

# 日志配置
logging.basicConfig(
    filename='model_api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ModelConfig:
    """模型配置管理器"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config: {str(e)}")
            raise AttributeError(f"Config file not found or wrong format: {path}")

    def get_credentials(self, model_provider: str, model_name: str) -> Dict:
        """获取模型凭证"""
        try:
            # 提取 all_models 列表
            all_models = self.config.get('all_models', [])

            # 遍历 all_models 列表，检查 provider 和 model_name 是否合法
            for model_info in all_models:
                if model_info.get('provider') == model_provider:
                    if model_name in model_info.get('model_name', []):
                        model_info['model_name'] = model_name
                        return model_info  # 返回匹配的模型配置

            # 如果没有找到匹配的 provider 或 model_name，记录警告并返回空字典
            logging.warning(f"No valid configuration found for provider '{model_provider}' and model name '{model_name}'")
            return {}

        except Exception as e:
            logging.error(f"Error in get_credentials: {str(e)}")
            return {}

class BaseModel:
    """模型基类"""

    def __init__(self, credentials: Dict):
        self.credentials = credentials

    def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            files: Optional[List[str]] = None
    ) -> (str, int, str):
        raise NotImplementedError


class OpenAIModel(BaseModel):
    """OpenAI 在线模型处理"""

    def __init__(self, credentials: Dict):
        super().__init__(credentials)
        self.client = OpenAI(
            api_key=credentials.get('api_key', ''),
            base_url=credentials.get('base_url', 'https://api.openai.com/v1')
        )

    def _encode_image(self, image_path: str) -> str:
        # 将图片编码为base64字符串，并且直接以openai api需要的格式返回
        with open(image_path, "rb") as img_file:
            base64_str = base64.b64encode(img_file.read()).decode('utf-8')
            img_format = img_file.name.split('.')[-1]
            result_str = f"data:image/{img_format};base64,{base64_str}"
            return result_str

    def _prepare_messages(self, **kwargs) -> list:
        """准备消息格式，供普通和流式调用共用"""
        messages = [{"role": "system", "content": kwargs['system_prompt']}]
        if kwargs.get('files'):
            # 处理多模态请求
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
        """准备API参数，供普通和流式调用共用"""
        params = {
            "model": self.credentials.get('model_name', 'gpt-4o'),
            "messages": messages,
            "temperature": kwargs.get('temperature'),
            "max_tokens": kwargs.get('max_tokens')
        }
        return {k: v for k, v in params.items() if v is not None}

    def generate(self, **kwargs) -> (str, int, str):
        """非流式生成回复"""
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

    def generate_stream(self, **kwargs) -> (str, int, str):
        """流式生成回复，但返回格式与非流式相同

        内部使用流式API，但会收集完整响应后返回，与generate()方法返回格式一致
        返回: (完整响应文本, token数量, 错误信息(如果有))
        """
        messages = self._prepare_messages(**kwargs)
        params = self._prepare_params(messages, **kwargs)
        params["stream"] = True

        complete_response = ""
        reasoning_content = ""
        estimated_tokens = 0

        max_retries = 3
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                stream = self.client.chat.completions.create(**params)

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

    def _parse_response(self, response: ChatCompletion) -> (str, int, str):
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
    """Ollama 本地模型处理，暂不支持流式调用"""

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def generate(self, **kwargs) -> (str, int, str):
        # 构造消息
        messages = [{"role": "system", "content": kwargs.get('system_prompt')}]
        if kwargs.get('files'):
            image_encoded = [self._encode_image(f) for f in kwargs['files']]
            # 处理多模态请求
            messages.append({
                "role": "user",
                "content": kwargs.get('user_prompt'),
                "images": kwargs['files'],
            })
        else:
            messages.append({"role": "user", "content": kwargs.get('user_prompt')})

        options = {
            "temperature": kwargs.get('temperature'),
            "num_predict": kwargs.get('max_tokens')
        }

        # 移除None值
        options = {k: v for k, v in options.items() if v is not None}

        try:
            # response = requests.post(url, json=payload)
            response = ollama.chat(
                model = self.credentials.get('model_name', 'llama3.1:8b'),
                messages = messages,
                options = options,
            )
            return self._parse_response(response)
        except Exception as e:
            logging.error(f"Ollama API error: {str(e)}")
            print(f"Ollama API error: {str(e)}")
            return "", 0, str(e)

    def _parse_response(self, response) -> (str, int, str):
        tokens_used = response.eval_count + response.prompt_eval_count
        return (
            response.message.content,
            tokens_used,
            None
        )

def call_language_model(
        model_provider: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        files: Optional[List[str]] = None,
        config_path: str = r'./llm_config.yaml'
) -> (str, int, str):
    """
    调用模型的统一入口函数，将此函数import到代码中即可使用
    :param model_provider: 模型提供商，如"openai","aliyun","volcengine","ollama"，不区分在线和本地
    :param model_name: 模型名称，注意部分提供商的模型名称可能包含版本号
    :param system_prompt: 系统提示
    :param user_prompt: 用户提示
    :param stream: 是否流式调用，可选
    :param temperature: 温度参数，可选
    :param max_tokens: 最大生成token数，可选
    :param files: 图片文件路径列表，可选
    :param config_path: 配置文件路径
    :return: (response_text, tokens_used, error_msg)
    """
    # 初始化
    config = ModelConfig(config_path)
    credentials = config.get_credentials(model_provider, model_name)

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
            if model_provider == "ollama":
                error_msg = "Ollama does not support stream=True with current function"
                logging.error(error_msg)
                raise AttributeError("Ollama does not support stream=True with current function")
            result = model.generate_stream(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                files=files
            )
        else:
            result = model.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                files=files
            )
        # 记录成功日志
        content, tokens, error = result
        logging.info(f"API call succeeded. Model: {model_name}, Provider: {model_provider}, Tokens used: {tokens}")
        return result
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        logging.error(error_msg)
        return "", 0, error_msg

if __name__ == "__main__":
    # 示例使用
    response_text, tokens_used, error = call_language_model(
        model_provider='aliyun',
        model_name='qwen2.5-32b-instruct',
        system_prompt="You are a helpful assistant.",
        user_prompt="介绍一下你自己", #非多模态
        stream=True
        # user_prompt="Try to solve this problem with Python",
        # files=['1.png'] #多模态
    )
    print(f"Response: {response_text}")
    print(f"Tokens used: {tokens_used}")
    if error:
        print(f"Error: {error}")

    print(1)
