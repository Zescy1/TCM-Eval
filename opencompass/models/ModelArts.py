import json
import time
import requests
from typing import List, Union, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging

# 假设你已有 BaseAPIModel 和相关类型定义
from .base_api import BaseAPIModel  # 请根据你的项目结构调整导入路径
from opencompass.utils import PromptList
from opencompass.utils.logging import get_logger

PromptType = Union[PromptList, str]

# 创建一个禁用的日志器（用于静默模式）
DISABLED_LOGGER = logging.getLogger('disabled')
DISABLED_LOGGER.disabled = True

logger = get_logger()


class HuaweiModel(BaseAPIModel):
    """Model class for Huawei Model API interaction"""

    is_api: bool = True

    def __init__(self,
                 path: str,
                 model: str = "Qibo",
                 api_base: str = "http://localhost:5000/tohuawei",
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 3,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 generation_kwargs: Dict = dict(),
                 mode: str = 'none',
                 tokenizer_path: Optional[str] = None,
                 verbose: bool = True):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
            query_per_second=query_per_second,
            rpm_verbose=rpm_verbose,
            retry=retry,
            verbose=verbose,
        )

        self.model_name = model
        self.api_base = api_base
        self.generation_kwargs = generation_kwargs
        self.headers = {"Content-Type": "application/json"}
        self.logger = get_logger(__name__) if verbose else DISABLED_LOGGER

        self._log_init_info()

    def _log_init_info(self):
        """打印模型初始化信息"""
        self.logger.info(" HuaweiModel 初始化成功 ".center(60, "-"))
        self.logger.info(f"🔌 API 地址: {self.api_base}")
        self.logger.info(f"🤖 模型名称: {self.model_name}")
        self.logger.info(f"🔁 最大重试次数: {self.retry}")
        self.logger.info(f"⏱️ 每秒请求数 (QPS): {self.query_per_second}")
        self.logger.info("-" * 60)

    def _build_payload(self, prompt: str, max_out_len: int, temperature: float) -> dict:
        """
        构建请求体，默认只传入 query 字段。
        你可以根据实际接口文档扩展参数。
        """
        return {
            "query": prompt
        }

    def _send_request(self, payload: dict) -> Optional[dict]:
        """发送 HTTP 请求并兼容性修复单引号问题"""
        for attempt in range(1, self.retry + 1):
            try:
                time.sleep(1 / self.query_per_second)
                response = requests.post(
                    self.api_base,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=120
                )
                print(f"[DEBUG] Status code: {response.status_code}")
                print(f"[DEBUG] Raw response text: {response.text}")

                # 尝试修复不合法 JSON（如单引号）
                content = response.text.replace("'", '"')
                return json.loads(content)

            except Exception as e:
                print(f"[ERROR] Request failed (attempt {attempt}): {e}")
                time.sleep(2 ** attempt)
        return None

    def _extract_text_from_response(self, response: dict) -> str:
        """
        安全提取 response 中的生成文本内容

        支持格式：
          - response['choices'][0]['text']['conntent']
          - response['choices'][0]['text']
          - response['text']
          - response['output']
        """
        if not isinstance(response, dict):
            return ""

        # 方法一：尝试从 choices 提取（注意处理 text 是 dict 的情况）
        try:
            content = response['choices'][0]['text'] # 多层安全获取
            if isinstance(content, str):
                return content.strip()
        except Exception as e:
            pass

        # 如果都没有找到，返回空字符串
        self.logger.warning("⚠️ 无法从 response 中提取生成文本")
        return "失败"

    def _process_response(self, response: dict) -> str:
        """解析响应内容并提取生成文本"""
        generated_text = self._extract_text_from_response(response)
        self.logger.info(f"📥 接收到响应内容: {generated_text}")
        print("🔍 原始 response 内容:", response)
        return generated_text

    def _generate_single(self, prompt: str, max_out_len: int, temperature: float) -> str:
        """单条生成逻辑"""
        payload = self._build_payload(prompt, max_out_len, temperature)
        raw_response = self._send_request(payload)
        if raw_response:
            return self._process_response(raw_response)
        return "调用失败"

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 8192,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[str]:
        """批量生成接口"""
        temperature = self.generation_kwargs.get('temperature', temperature)
        self.logger.info(f"🔥 开始批量生成，共 {len(inputs)} 条输入，温度: {temperature}")

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(
                    lambda x: self._generate_single(x, max_out_len, temperature),
                    inputs
                ),
                total=len(inputs),
                desc="HuaweiModel Inferencing"
            ))

        return results

    def get_token_len(self, prompt: str) -> int:
        """估算 token 长度（简化版）"""
        return len(prompt) // 4  # 粗略估算