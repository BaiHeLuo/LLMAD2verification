"""
评估主程序
用法:
    python run_evaluate.py                                    # 评估所有 responses
    python run_evaluate.py --input responses_xxx.json         # 评估单个文件
    python run_evaluate.py --compare                         # 跨模型对比模式
"""

import argparse
import json
import os
import sys
import glob
from datetime import datetime

from code_runner import run_generated_code, validate_route, compute_route_distance
from metrics import (
    score_identification,
    score_algorithm_performance,
    compute_overall_score,
    compute_cross_model_metrics,
)

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DEFAULT_RESPONSES_DIR = os.path.join(PROJECT_DIR, "data", "responses")
DEFAULT_TEST_SET = os.path.join(PROJECT_DIR, "data", "test_set.json")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "evaluations")


def load_test_set(path: str) -> dict:
    """加载测试集，返回 {id: test_case} 字典"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data if isinstance(data, list) else data.get("test_cases", [])
    return {tc["id"]: tc for tc in cases}


def load_responses(path: str) -> dict:
    """加载响应文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_response(
    response: dict,
    test_cases: dict,
    run_code: bool = True,
) -> dict:
    """
    评估单个 LLM 响应。

    Args:
        response: 来自 run_query.py 的响应记录
        test_cases: {id: test_case} 字典
        run_code: 是否执行生成的代码

    Returns:
        dict: 完整评估结果
    """
    tc_id = response.get("test_case_id", "unknown")
    tc = test_cases.get(tc_id, {})
    category = tc.get("category", response.get("category", "A"))
    layer = tc.get("layer", response.get("layer", 1))

    eval_result = {
        "test_case_id": tc_id,
        "model": response.get("model", "unknown"),
        "layer": layer,
        "category": category,
        "has_error": response.get("error") is not None,
        "error": response.get("error"),
    }

    # ---- 1. 识别准确率评估 ----
    id_result = score_identification(
        response.get("response_text", ""),
        category,
    )
    eval_result["identification"] = id_result

    # ---- 2. 算法性能评估 ----
    alg_result = None
    code_result = None

    generated_code = response.get("generated_code")
    distance_matrix = tc.get("distance_matrix")
    expected = tc.get("expected_output", {})
    optimal_distance = expected.get("optimal_distance")

    if run_code and generated_code and distance_matrix:
        n_cities = len(distance_matrix)
        print(f"    -> 运行生成代码 (n={n_cities})...", end="", flush=True)

        code_result = run_generated_code(generated_code, distance_matrix)
        print(f" {'✓' if code_result['success'] else '✗'} "
              f"({code_result['elapsed_seconds']}s)")

        if code_result["success"]:
            route = code_result["route"]
            obtained_dist = code_result["total_distance"]

            # 验证路径
            route_validation = validate_route(route, n_cities)

            # 如果代码返回的距离不对，用路径重新计算
            if obtained_dist is None and route and route_validation["valid"]:
                obtained_dist = compute_route_distance(route, distance_matrix)

            alg_result = score_algorithm_performance(
                obtained_distance=obtained_dist,
                optimal_distance=optimal_distance,
                route_valid=route_validation["valid"],
                execution_time=code_result["elapsed_seconds"],
                n_cities=n_cities,
            )
            alg_result["route_validation"] = route_validation
            alg_result["obtained_distance"] = obtained_dist
            alg_result["optimal_distance"] = optimal_distance
        else:
            alg_result = {
                "success": False,
                "error": code_result["error"],
                "stderr": code_result["stderr"][:300],
                "performance_score": 0.0,
                "approximation_ratio": None,
                "optimality_gap_pct": None,
                "route_valid": False,
                "execution_time_s": code_result["elapsed_seconds"],
            }
    elif run_code and not generated_code:
        code_result = {"error": "无生成代码"}
        alg_result = None

    eval_result["code_execution"] = code_result
    eval_result["algorithm_performance"] = alg_result

    # ---- 3. 综合评分 ----
    eval_result["scores"] = compute_overall_score(
        identification=id_result,
        algorithm_perf=alg_result,
        layer=layer,
    )

    return eval_result


def run_evaluation(
    responses_path: str,
    test_set_path: str,
    output_dir: str,
    run_code: bool = True,
) -> dict:
    """评估单个响应文件"""
    os.makedirs(output_dir, exist_ok=True)

    test_cases = load_test_set(test_set_path)
    responses_data = load_responses(responses_path)
    model_name = responses_data.get("model", "unknown")
    responses = responses_data.get("responses", [])

    print(f"评估模型: {model_name}")
    print(f"响应文件: {responses_path}")
    print(f"测试用例数: {len(responses)}")
    print("=" * 60)

    evaluations = []
    for i, resp in enumerate(responses):
        tc_id = resp.get("test_case_id", "unknown")
        layer = resp.get("layer", "?")
        print(f"[{i+1}/{len(responses)}] Layer {layer} | {tc_id}")

        ev = evaluate_response(resp, test_cases, run_code=run_code)
        evaluations.append(ev)

        score = ev["scores"]["total_score"]
        print(f"    -> 综合评分: {score}/100")

    # 汇总统计
    summary = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "responses_file": responses_path,
        "total_cases": len(evaluations),
        "avg_total_score": round(
            sum(ev["scores"]["total_score"] for ev in evaluations) / len(evaluations), 2
        ) if evaluations else 0,
        "avg_identification_score": round(
            sum(ev["scores"]["identification_score"] for ev in evaluations) / len(evaluations), 2
        ) if evaluations else 0,
        "id_accuracy_by_category": {},
        "code_execution_stats": {
            "total_with_code": sum(1 for ev in evaluations if ev.get("code_execution")),
            "successful_runs": sum(
                1 for ev in evaluations
                if ev.get("code_execution") and ev["code_execution"].get("success")
            ),
        },
    }

    # 按类别统计识别准确率
    for cat in ["A", "B", "C", "D"]:
        cat_evals = [ev for ev in evaluations if ev["category"] == cat]
        if cat_evals:
            correct = sum(1 for ev in cat_evals if ev["scores"]["identification_score"] > 50)
            summary["id_accuracy_by_category"][cat] = round(correct / len(cat_evals), 3)

    result = {
        "summary": summary,
        "evaluations": evaluations,
    }

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"evaluation_{model_name}_{timestamp}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"评估完成！结果已保存: {out_file}")
    print(f"平均综合评分: {summary['avg_total_score']}/100")
    print(f"识别准确率 (按类别): {summary['id_accuracy_by_category']}")
    print(f"代码执行: {summary['code_execution_stats']}")

    return result


def run_cross_model_comparison(
    responses_dir: str,
    test_set_path: str,
    output_dir: str,
    run_code: bool = True,
):
    """跨模型对比模式"""
    os.makedirs(output_dir, exist_ok=True)

    # 找到所有响应文件
    pattern = os.path.join(responses_dir, "responses_*.json")
    files = glob.glob(pattern)

    if not files:
        print(f"未找到响应文件: {pattern}")
        print(f"请先运行 query/run_query.py 生成响应")
        return

    print(f"找到 {len(files)} 个响应文件")
    print("=" * 60)

    all_evaluations = {}

    for fpath in files:
        data = load_responses(fpath)
        model_name = data.get("model", "unknown")
        test_cases = load_test_set(test_set_path)
        responses = data.get("responses", [])

        print(f"\n评估模型: {model_name} ({len(responses)} 个用例)")
        evals = []
        for resp in responses:
            ev = evaluate_response(resp, test_cases, run_code=run_code)
            evals.append(ev)
            print(f"  [{ev['test_case_id']}] score={ev['scores']['total_score']}/100")

        all_evaluations[model_name] = evals

    # 计算跨模型对比指标
    cross_metrics = compute_cross_model_metrics(all_evaluations)

    # 保存对比结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"cross_model_comparison_{timestamp}.json")

    # 保存完整的对比数据
    comparison_data = {
        "timestamp": timestamp,
        "models_compared": list(all_evaluations.keys()),
        "cross_metrics": cross_metrics,
        "per_model_evaluations": {
            model: evals for model, evals in all_evaluations.items()
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)

    # 打印关键发现
    print(f"\n{'='*60}")
    print("跨模型对比结果")
    print(f"{'='*60}")
    print(f"\nΔ_识别 (标准-伪装，越大越依赖记忆):")
    for model, delta in cross_metrics["delta_identification"].items():
        print(f"  {model}: {delta:.3f}")

    print(f"\nΔ_性能 (伪装/标准，越接近1推理越强):")
    for model, ratio in cross_metrics["delta_performance"].items():
        print(f"  {model}: {ratio:.3f}")

    print(f"\n泛化衰减 (A→D):")
    for model, curve in cross_metrics["generalization_decay"].items():
        scores = [f"{c['category']}:{c['performance']}" for c in curve]
        print(f"  {model}: {' → '.join(scores)}")

    print(f"\n结果已保存: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="LLM TSP 评估工具")
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="单个响应文件路径（评估单个模型）"
    )
    parser.add_argument(
        "--responses-dir",
        default=DEFAULT_RESPONSES_DIR,
        help=f"响应文件目录 (默认: {DEFAULT_RESPONSES_DIR})"
    )
    parser.add_argument(
        "--test-set", "-t",
        default=DEFAULT_TEST_SET,
        help=f"测试集 JSON 路径 (默认: {DEFAULT_TEST_SET})"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="启用跨模型对比模式"
    )
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="跳过代码执行评估（仅评估识别）"
    )
    args = parser.parse_args()

    if not os.path.exists(args.test_set):
        print(f"错误: 测试集文件不存在: {args.test_set}")
        sys.exit(1)

    run_code = not args.no_code

    if args.compare:
        run_cross_model_comparison(
            args.responses_dir, args.test_set, args.output, run_code
        )
    elif args.input:
        if not os.path.exists(args.input):
            print(f"错误: 响应文件不存在: {args.input}")
            sys.exit(1)
        run_evaluation(args.input, args.test_set, args.output, run_code)
    else:
        # 默认：评估 responses_dir 中所有文件
        pattern = os.path.join(args.responses_dir, "responses_*.json")
        files = glob.glob(pattern)
        if not files:
            print(f"未找到响应文件: {pattern}")
            print("用法:")
            print(f"  python run_evaluate.py --input <响应文件>       # 评估单个模型")
            print(f"  python run_evaluate.py --compare               # 跨模型对比")
            sys.exit(1)

        for fpath in files:
            print(f"\n{'#'*60}")
            run_evaluation(fpath, args.test_set, args.output, run_code)


if __name__ == "__main__":
    main()
