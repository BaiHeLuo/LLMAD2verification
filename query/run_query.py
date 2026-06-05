"""
LLM 查询主程序
用法:
    python run_query.py                           # 使用默认配置运行
    python run_query.py --input test_set.json     # 指定输入文件
    python run_query.py --models deepseek-chat    # 指定模型
    python run_query.py --models deepseek-chat gpt-4o-mini  # 多模型
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from llm_client import LLMClient, extract_code
from config import DEFAULT_MODELS

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DEFAULT_INPUT = os.path.join(PROJECT_DIR, "data", "test_set.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "responses")

# 系统提示词：引导模型以结构化方式回答
SYSTEM_PROMPT = (
    "You are an expert in algorithm design and optimization. "
    "When answering, please structure your response as follows:\n"
    "1. **Problem Identification**: State what optimization problem this is and why.\n"
    "2. **Mathematical Formulation**: Provide a formal mathematical model.\n"
    "3. **Algorithm Design**: Describe your algorithm with pseudocode.\n"
    "4. **Complexity Analysis**: Analyze time and space complexity.\n"
    "5. **Implementation**: Provide complete, runnable Python code in a ```python ... ``` block.\n"
    "   The code should include a `solve(distance_matrix)` function that takes a 2D list "
    "(distance matrix) and returns a dict with keys 'route' (list of city indices) and "
    "'total_distance' (number).\n"
    "If the question only asks for identification, you may omit sections 3-5."
)


def load_test_set(path: str) -> list[dict]:
    """加载测试集 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "test_cases" in data:
        return data["test_cases"]
    raise ValueError("无法解析测试集格式，期望 list 或 {test_cases: [...]}")


def build_prompt(test_case: dict) -> str:
    """根据测试用例构建完整提示词"""
    prompt = test_case["prompt"]

    # 如果有附加信息（如距离矩阵），附加到提示词后
    if test_case.get("distance_matrix") is not None:
        prompt += "\n\nDistance Matrix:\n"
        prompt += json.dumps(test_case["distance_matrix"], indent=2)

    if "expected_output" in test_case:
        opt = test_case['expected_output'].get('optimal_distance', 'N/A')
        prompt += f"\n\nFor reference, the known optimal route length is: {opt}"

    return prompt


def query_multi_turn(client: LLMClient, test_case: dict) -> dict:
    """
    多轮对话查询（用于 Layer 7 延迟识别等）。
    按顺序发送 prompts 列表中的每个提示词，
    将前一轮的回答作为上下文传入下一轮。
    """
    prompts = test_case.get("prompts", [])
    if not prompts:
        return query_single(client, test_case)

    all_turns = []
    messages = []
    if test_case.get("system_prompt"):
        messages.append({"role": "system", "content": test_case["system_prompt"]})

    total_start = time.time()
    error = None

    for i, prompt in enumerate(prompts):
        print(f"  -> Turn {i+1}/{len(prompts)}...", end="", flush=True)
        messages.append({"role": "user", "content": prompt})

        try:
            start = time.time()
            from openai import OpenAI
            response = client.client.chat.completions.create(
                model=client.cfg["model"],
                messages=messages,
                max_tokens=client.cfg["max_tokens"],
                temperature=client.cfg["temperature"],
            )
            elapsed = time.time() - start
            raw_text = response.choices[0].message.content

            messages.append({"role": "assistant", "content": raw_text})
            all_turns.append({
                "prompt": prompt,
                "response": raw_text,
                "elapsed_seconds": round(elapsed, 2),
            })
            print(f" ✓ ({elapsed:.1f}s)")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            all_turns.append({
                "prompt": prompt,
                "response": None,
                "error": error_msg,
            })
            error = error_msg
            print(f" ✗ ({error_msg[:50]})")
            break

        if i < len(prompts) - 1:
            time.sleep(1)

    total_elapsed = time.time() - total_start
    full_text = "\n\n".join(
        t.get("response", "") for t in all_turns if t.get("response")
    )

    return {
        "model": client.model_name,
        "test_case_id": test_case["id"],
        "layer": test_case.get("layer"),
        "category": test_case.get("category"),
        "prompt": json.dumps(prompts),  # 保存完整 prompts
        "system_prompt": test_case.get("system_prompt"),
        "response_text": full_text,
        "turns": all_turns,
        "multi_turn": True,
        "generated_code": extract_code(full_text),
        "response_json": None,
        "usage": None,
        "error": error,
        "elapsed_seconds": round(total_elapsed, 2),
        "timestamp": datetime.now().isoformat(),
    }


def query_single(client: LLMClient, test_case: dict) -> dict:
    """对单个测试用例进行查询"""
    prompt = build_prompt(test_case)
    print(f"  -> 查询中...", end="", flush=True)

    result = client.query(prompt, system_prompt=SYSTEM_PROMPT)

    # 附加元数据
    result["test_case_id"] = test_case["id"]
    result["layer"] = test_case.get("layer")
    result["category"] = test_case.get("category")
    result["timestamp"] = datetime.now().isoformat()

    # 提取代码
    result["generated_code"] = extract_code(result["response_text"])

    status = "✓" if result["error"] is None else f"✗ ({result['error'][:50]})"
    print(f" {status} ({result['elapsed_seconds']}s)")

    return result


def run(test_set_path: str, model_names: list[str], output_dir: str):
    """主运行流程"""
    os.makedirs(output_dir, exist_ok=True)

    test_cases = load_test_set(test_set_path)
    print(f"加载测试集: {test_set_path} ({len(test_cases)} 个用例)")
    print(f"使用模型: {model_names}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"{'='*60}")

        client = LLMClient(model_name)
        responses = []

        for i, tc in enumerate(test_cases):
            print(f"[{i+1}/{len(test_cases)}] Layer {tc.get('layer','?')} | {tc['id']}")

            # 判断是否为多轮对话（Layer 7 等）
            if tc.get("multi_turn") or "prompts" in tc:
                resp = query_multi_turn(client, tc)
            else:
                resp = query_single(client, tc)
            responses.append(resp)

            # 请求间隔，避免 rate limit
            if i < len(test_cases) - 1:
                time.sleep(1)

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(output_dir, f"responses_{model_name}_{timestamp}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": model_name,
                    "timestamp": timestamp,
                    "test_set": test_set_path,
                    "total_cases": len(responses),
                    "responses": responses,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n结果已保存: {out_file}")

        # 统计
        errors = sum(1 for r in responses if r["error"] is not None)
        with_code = sum(1 for r in responses if r["generated_code"] is not None)
        print(f"统计: 成功 {len(responses)-errors}/{len(responses)}, "
              f"含代码 {with_code}/{len(responses)}")


def main():
    parser = argparse.ArgumentParser(description="LLM TSP 查询工具")
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"测试集 JSON 文件路径 (默认: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"要使用的模型 (默认: {DEFAULT_MODELS})"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 测试集文件不存在: {args.input}")
        print(f"请将测试集 JSON 文件放在: {DEFAULT_INPUT}")
        sys.exit(1)

    run(args.input, args.models, args.output)


if __name__ == "__main__":
    main()
