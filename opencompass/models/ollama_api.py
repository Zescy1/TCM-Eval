import os
import json
import requests
import re
import time
from typing import Dict, List, Optional, Union
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .base_api import BaseAPIModel
from opencompass.utils import PromptList
from opencompass.utils.logging import get_logger
from opencompass.registry import MODELS
PromptType = Union[PromptList, str]
# 默认Ollama API地址
OLLAMA_API_BASE = os.environ.get('OLLAMA_API_BASE', 'http://localhost:11435/api/chat')

@MODELS.register_module()
class Ollama(BaseAPIModel):
    """Model class for Ollama API interaction"""
    
    is_api: bool = True

    def __init__(self,
                 path: str,
                 model: str = "llama2",
                 api_base: str = OLLAMA_API_BASE,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 generation_kwargs: Dict = dict(),
                 mode: str = 'none',
                 tokenizer_path: Optional[str] = None,
                 verbose: bool = False):
        """
        Args:
            path (str): Path to model (保留参数，实际未使用)
            model (str): Ollama模型名称 (如 "llama2", "mistral", "gemma")
            api_base (str): Ollama API基础地址
        """
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
            query_per_second=query_per_second,
            rpm_verbose=rpm_verbose,
            retry=retry,
            verbose=verbose,
        )
        print(f"Initializing Ollama model with API base: {api_base}")
        import tiktoken

        self.tiktoken = tiktoken
        self.model_name = model
        self.api_base = api_base
        self.headers = {"Content-Type": "application/json"}
        self.logger = get_logger()
        self.mode = mode
        self.tokenizer_path = tokenizer_path
        self.generation_kwargs = generation_kwargs

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 8192,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[str]:
        """生成结果"""
        # 使用类内temperature设置（如果提供）
        temperature = self.generation_kwargs.get('temperature', temperature)
        print(f"Using temperature: {temperature}")
        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(
                        self._generate,
                        inputs,
                        [max_out_len] * len(inputs),
                        [temperature] * len(inputs),
                    ),
                    total=len(inputs),
                    desc='Ollama Inferencing',
                ))
        #print(results)
        return results

    def _generate(self, input: PromptType, max_out_len: int,
                  temperature: float) -> str:
        # 预处理消息
        messages, adjusted_max_out_len = self._preprocess_messages(
            input, max_out_len, self.max_seq_len, self.mode,
            self.get_token_len)
        
        # 将消息转换为单一提示字符串
        prompt_text = self._convert_messages_to_prompt(messages)
        
        # 构建请求体（使用generate端点格式）
        payload = {
            "model": self.model_name,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                "num_predict": adjusted_max_out_len,
                "temperature": temperature,
                **self.generation_kwargs
            }
        }
        
        # 发送请求
        response = self._send_request(payload)
        
        # 处理响应
        if response and 'response' in response:
            
            return self.process_content(response['response'].strip())
        else:
            error = response.get('error', 'Unknown error') if isinstance(response, dict) else 'Invalid response'
            self.logger.error(f"API error: {error}")
            return "调用失败"

    def _convert_messages_to_prompt(self, messages: List[Dict]) -> str:
        """将消息列表转换为单一提示字符串"""
        prompt_lines = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            # 角色映射
            if role == 'system':
                prefix = "<|system|>"
            elif role == 'user':
                prefix = "<|user|>"
            elif role == 'assistant':
                prefix = "<|assistant|>"
            else:
                prefix = f"<|{role}|>"
            
            prompt_lines.append(f"{prefix}\n{content}")
        
        # 添加响应前缀
        prompt_lines.append("<|assistant|>")
        return "\n".join(prompt_lines)

    def _send_request(self, payload: Dict) -> Optional[Dict]:
        """发送API请求（带重试机制）"""
        for attempt in range(self.retry + 1):
            try:
                self.acquire()  # 速率控制
                
                response = requests.post(
                    self.api_base,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=120
                )
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed (attempt {attempt+1}/{self.retry+1}): {str(e)}")
                if attempt < self.retry:
                    time.sleep(2 ** attempt)  # 指数退避
            except json.JSONDecodeError:
                self.logger.error(f"JSON decode error: {response.text}")
            finally:
                self.release()
                
        return None

    def get_token_len(self, prompt: str) -> int:
        """估计token数量（简化版）"""
        # 简单中英文token估算
        english_parts = re.findall(r'[A-Za-z0-9]+', prompt)
        chinese_parts = re.findall(r'[\u4e00-\u9FFF]+', prompt)
        english_count = sum(len(part.split()) for part in english_parts)
        chinese_count = sum(len(part) for part in chinese_parts)
        print(f"English tokens: {english_count}, Chinese characters: {chinese_count}")
        return english_count + chinese_count
        
    def process_content(self,content: str) -> str:
            """辅助函数：提取Final Response 后面的内容"""
            final_marker = "## Final Response"
            if final_marker in content:
                content = content.split(final_marker, 1)[1].strip()
            return content

    def _preprocess_messages(
        self,
        input: Union[str, PromptList],
        max_out_len: int,
        max_seq_len: int,
        mode: str,
        get_token_len_func,
    ) -> tuple[List[Dict], int]:
        """预处理输入消息"""
        # 空输入检查
        if not input:
            self.logger.warning("Received empty input, returning empty messages")
            return [], 0
    
        messages = []
        input_len = 0

    # 记录原始输入
        self.logger.debug(f"Raw input type: {type(input)}, content: {input}")

    # 处理字符串输入
        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
            input_len = get_token_len_func(input)
    # 处理PromptList输入
        else:
            for i, item in enumerate(input):
            # 验证输入格式
                if not isinstance(item, dict):
                    self.logger.error(f"Item {i} is not a dict: {item}")
                    continue
                
                if 'role' not in item:
                    self.logger.error(f"Missing 'role' in item {i}: {item}")
                    continue
                
                if 'prompt' not in item:
                    self.logger.error(f"Missing 'prompt' in item {i}: {item}")
                    continue
                
                input_content = item['prompt']
                if not input_content:
                    self.logger.warning(f"Empty prompt in item {i}, skipping")
                    continue
                
            # 角色映射
                role_map = {
                'HUMAN': 'user',
                'BOT': 'assistant',
                'SYSTEM': 'system'
                }
                role = role_map.get(item['role'], 'user')  # 默认user
            
                messages.append({'role': role, 'content': input_content})
                input_len += get_token_len_func(input_content)

    # 空消息检查
        if not messages:
            self.logger.error("No valid messages processed from input")
            return [], 0

    # 检查输入长度（当模式为'none'时）
        if mode == 'none':
            if input_len > max_seq_len:
                raise ValueError(
                f'Input length ({input_len}) exceeds max_seq_len '
                f'({max_seq_len}) in "none" mode.')

    # 调整最大输出长度
        adjusted_max_out_len = max_out_len
        if max_out_len is not None:
            buffer_tokens = 100  # 安全缓冲区
            adjusted_max_out_len = min(max_out_len, max_seq_len - input_len - buffer_tokens)
        
            if adjusted_max_out_len <= 0:
                raise ValueError(
                f'Adjusted max_out_len <= 0 '
                f'({adjusted_max_out_len}), input too long.')
            
            if adjusted_max_out_len < max_out_len:
                self.logger.warning(
                f'Truncated max_out_len: {max_out_len} -> {adjusted_max_out_len}')

    # 调试日志
        self.logger.debug(f"Final messages ({len(messages)}): {messages}")
        self.logger.debug(f"Input tokens: {input_len}, Max output tokens: {adjusted_max_out_len}")
    
        return messages, adjusted_max_out_len  