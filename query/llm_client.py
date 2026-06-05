"""
LLM 统一调用客户端
支持所有 OpenAI 兼容 API（DeepSeek、GPT 等）
"""

import time
import json
from openai import OpenAI
from config import MODELS, REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY


class LLMClient:
    """统一的 LLM 调用客户端"""

    def __init__(self, model_name: str):
        if model_name not in MODELS:
            raise ValueError(
                f"未知模型: {model_name}，可用模型: {list(MODELS.keys())}"
            )
        self.model_name = model_name
        self.cfg = MODELS[model_name]
        self.client = OpenAI(
            api_key=self.cfg["api_key"],
            base_url=self.cfg["base_url"],
            timeout=REQUEST_TIMEOUT,
        )

    def query(self, prompt: str, system_prompt: str = None) -> dict:
        """
        向 LLM 发送查询，返回结构化结果。

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词（可选）

        Returns:
            dict: {
                "model": 模型名称,
                "prompt": 原始提示词,
                "response_text": 模型原始文本回复,
                "response_json": 解析后的 JSON（如果模型返回了 JSON），否则为 None,
                "usage": token 使用统计,
                "error": 错误信息（如果有）,
                "elapsed_seconds": 请求耗时
            }
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        result = {
            "model": self.model_name,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response_text": None,
            "response_json": None,
            "usage": None,
            "error": None,
            "elapsed_seconds": 0,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.cfg["model"],
                    messages=messages,
                    max_tokens=self.cfg["max_tokens"],
                    temperature=self.cfg["temperature"],
                )
                elapsed = time.time() - start

                raw_text = response.choices[0].message.content
                result["response_text"] = raw_text
                result["usage"] = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                result["elapsed_seconds"] = round(elapsed, 2)

                # 尝试从回复中提取 JSON
                result["response_json"] = self._extract_json(raw_text)
                return result

            except Exception as e:
                error_msg = f"[尝试 {attempt}/{MAX_RETRIES}] {type(e).__name__}: {e}"
                result["error"] = error_msg
                print(f"  ⚠ {error_msg}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        return result

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """尝试从文本中提取 JSON 块（支持 ```json ... ``` 或直接 JSON）"""
        if not text:
            return None

        # 尝试提取 ```json ... ``` 代码块
        import re
        json_block = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取 ``` ... ``` 代码块中的 JSON
        code_block = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试直接解析整个文本为 JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        return None


def extract_code(response_text: str) -> str | None:
    """从模型回复中提取 Python 代码块"""
    if not response_text:
        return None
    import re
    # 匹配 ```python ... ```
    match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1)
    # 匹配 ``` ... ```
    match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        code = match.group(1)
        if "def " in code or "import " in code or "print(" in code:
            return code
    return None
