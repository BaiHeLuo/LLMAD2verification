# ============================================================
# LLM API 配置文件 - 在此处修改 API Key、Base URL 和模型名称
# ============================================================

# 所有可用的模型配置，每个模型包含：
#   - api_key: API 密钥
#   - base_url: API 基础 URL（OpenAI 兼容格式）
#   - model: 模型标识符
#   - max_tokens: 最大生成 token 数
#   - temperature: 生成温度（0 为确定性输出，越高越随机）

MODELS = {
    "deepseek-chat": {
        "api_key": "sk-your-deepseek-api-key",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    "deepseek-reasoner": {
        "api_key": "sk-your-deepseek-api-key",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-reasoner",
        "max_tokens": 8192,
        "temperature": 0.0,
    },
    "gpt-4o": {
        "api_key": "sk-your-openai-api-key",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    "gpt-4o-mini": {
        "api_key": "sk-your-openai-api-key",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    # 添加更多模型...
    # "custom-model": {
    #     "api_key": "your-key",
    #     "base_url": "https://your-api-endpoint/v1",
    #     "model": "model-name",
    #     "max_tokens": 4096,
    #     "temperature": 0.0,
    # },
}

# 默认使用的模型列表（run_query.py 中可通过参数覆盖）
DEFAULT_MODELS = ["deepseek-chat", "gpt-4o-mini"]

# 请求超时时间（秒）
REQUEST_TIMEOUT = 120

# 失败重试次数
MAX_RETRIES = 3

# 重试间隔（秒）
RETRY_DELAY = 5
