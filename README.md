# TSPclass — LLM 算法设计能力评估框架

基于TSP的十层评测框架，用于系统性探究大语言模型在算法自动化设计中是依赖记忆还是理解。

## 核心研究问题

> 在算法自动化设计中，LLM 是依赖对已知问题的模式匹配（记忆），还是能真正理解问题本质并创造/适配解决方案（推理）？

## 项目结构

```
TSPclass/
├── query/                      # 模块一：LLM 查询
│   ├── config.py               # API Key / Base URL 配置（顶部修改）
│   ├── llm_client.py           # 统一 LLM 调用客户端（OpenAI 兼容）
│   └── run_query.py            # 主程序：读取测试集 → 调用 LLM → 保存结果
├── evaluate/                   # 模块二：评估
│   ├── code_runner.py          # 执行 LLM 生成的代码，捕获输出与性能
│   ├── metrics.py              # 全部评估指标（识别/性能/元认知/迁移）
│   └── run_evaluate.py         # 主程序：读取响应 → 运行评估 → 输出报告
├── data/
│   ├── test_set.json           # 测试集（32 个用例，覆盖 10 层）
│   ├── responses/              # LLM 查询结果
│   └── evaluations/            # 评估结果
├── requirements.txt            # Python 依赖
└── requirement.txt             # 原始需求文档
```

## 快速开始

```bash
# 0. 安装依赖
pip install -r requirements.txt

# 1. 配置 API（编辑 query/config.py，填入 api_key 和 base_url）

# 2. 运行查询
cd query
python run_query.py
python run_query.py --models deepseek-chat gpt-4o-mini   # 指定模型
python run_query.py --input path/to/test_set.json        # 指定测试集

# 3. 运行评估
cd evaluate
python run_evaluate.py --input ../data/responses/responses_xxx.json

# 4. 跨模型对比
python run_evaluate.py --compare

# 5. 仅评估识别（跳过代码执行）
python run_evaluate.py --input ../data/responses/responses_xxx.json --no-code
```

## 十层评测框架

| 层级 | 名称 | 测试目标 | 用例数 |
|:----:|------|----------|:------:|
| L1 | 显式识别 | 标准/数学/应用场景下能否识别 TSP | 3 |
| L2 | 表述不变性 | 改变领域术语后能否识别同构问题 | 3 |
| L3 | 消融测试 | 模糊关键约束后的边界识别能力 | 3 |
| L4 | 反事实干扰 | 区分 TSP 变体（TSPTW/CVRP/mTSP） | 3 |
| L5 | 解法先验 | 不提问题名称，能否自发设计正确算法 | 3 |
| L6 | 性能 Probe | 生成代码在 benchmark 上的实际求解质量 | 3 |
| L7 | 延迟识别 | 多轮渐进式揭示信息，记录识别转折点 | 3 |
| L8 | 置信度校准 | 判断准确性 + 置信度是否与信息完整度匹配 | 3 |
| L9 | 元认知 | 推理链追溯、知识边界自评、失败模式分析 | 2 |
| L10 | 迁移与类比 | 跨问题解法迁移、类比构造、核心思想抽象 | 3 |

### 问题分类

| 类别 | 描述 | 测试目的 |
|:----:|------|----------|
| **A** | 标准 TSP | 高暴露度问题的记忆程度 |
| **B** | 伪装 TSP（数学同构） | 抽象类比能力 |
| **C** | TSP 变体（相似但不等价） | 边界识别和区分能力 |
| **D** | 复杂新问题 | 深层推理和创新能力 |

## 评估指标

### 核心对比指标

| 指标 | 公式 / 说明 | 含义 |
|------|-------------|------|
| **Δ_识别** | `accuracy(标准TSP) − accuracy(伪装TSP)` | 越大 → 越依赖表面特征记忆 |
| **Δ_性能** | `performance(伪装TSP) / performance(标准TSP)` | 越接近 1 → 推理能力越强 |
| **泛化衰减曲线** | A → B → C → D 各类别性能变化 | 下降越平缓 → 推理能力越强 |
| **近似比** | `obtained_distance / optimal_distance` | 越接近 1 越好 |

### 层级专属指标

| 层级 | 专属指标 | 评估方式 |
|------|----------|----------|
| L7 | 首次正确识别轮次、收敛质量 | 多轮对话中每轮独立检测 |
| L8 | 置信度提取、判断正确性、校准分 | 正则提取 `%` 数值 + Yes/No 判断 |
| L9 | 推理链步骤数、不确定性/边界/失败模式检测 | 关键词 + 结构化检测 |
| L10 | 解释/映射/适配/对比/代码 五维评分 | 关键词覆盖度 |

### 综合评分

根据层级动态加权（归一化权重）：

| 层级范围 | 识别权重 | 算法权重 | 专属权重 |
|:--------:|:--------:|:--------:|:--------:|
| L1–L4 | 0.8 | 0.2 | 0.0 |
| L5–L6 | 0.3 | 0.7 | 0.0 |
| L7–L8 | 0.2 | 0.3 | 0.5 |
| L9–L10 | 0.1 | 0.2 | 0.7 |

## 测试集格式

测试集为 JSON 文件，包含 `test_cases` 数组。每个用例结构：

```json
{
  "id": "L1_01_standard_tsp",
  "layer": 1,
  "category": "A",
  "prompt": "问题描述文本...",
  "distance_matrix": [[0,10,...],[10,0,...]],
  "expected_output": {
    "problem_type": "Traveling Salesman Problem (TSP)",
    "optimal_distance": 80,
    "optimal_route": [0, 1, 3, 4, 2]
  }
}
```

**Layer 7 多轮对话**使用 `prompts` 数组 + `"multi_turn": true`：

```json
{
  "id": "L7_01_gradual_tsp",
  "layer": 7,
  "multi_turn": true,
  "prompts": [
    "Turn 1: 部分信息...",
    "Turn 2: 更多约束...",
    "Turn 3: 关键信息...",
    "Turn 4: 最终问题..."
  ]
}
```

## 添加新模型

编辑 `query/config.py`，在 `MODELS` 字典中添加配置：

```python
MODELS = {
    "your-model": {
        "api_key": "sk-xxx",
        "base_url": "https://your-api-endpoint/v1",
        "model": "model-name",
        "max_tokens": 4096,
        "temperature": 0.0,
    },
}
```

所有 OpenAI 兼容 API 均可直接使用（DeepSeek、GPT、本地部署模型等）。

## 输出示例

评估完成后生成结构化 JSON，包含：

```json
{
  "summary": {
    "model": "gpt-4o",
    "avg_total_score": 73.08,
    "id_accuracy_by_category": {"A": 0.625, "B": 1.0, "C": 1.0, "D": 0.0},
    "avg_score_by_layer": {"1": 96.58, "2": 96.58, "..."},
    "code_execution_stats": {"total_with_code": 12, "successful_runs": 4}
  },
  "evaluations": [
    {
      "test_case_id": "L7_01_gradual_tsp",
      "scores": {
        "total_score": 58.42,
        "identification_score": 66.7,
        "algorithm_score": null,
        "layer_specific_score": 43.5
      },
      "delayed_identification": {
        "first_correct_turn": 3,
        "convergence_score": 0.725
      }
    }
  ]
}
```
