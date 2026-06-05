"""
评估指标模块
包含：识别准确率、算法性能、综合评分等评估指标
"""

import re
import math


# ============================================================
# 1. 识别准确率（Identification Accuracy）
# ============================================================

# 各测试层类别的标准答案关键词（不区分大小写）
IDENTIFICATION_KEYWORDS = {
    "A": [
        "traveling salesman", "travelling salesman", "tsp",
        "旅行商", "旅行推销员", "货郎担",
        "hamiltonian cycle", "hamiltonian circuit", "哈密顿回路",
    ],
    "B": [  # 伪装 TSP，应与 TSP 数学等价
        "traveling salesman", "travelling salesman", "tsp",
        "旅行商", "同构", "isomorphic", "mathematically equivalent",
        "数学等价", "哈密顿回路", "hamiltonian",
    ],
    "C": [  # TSP 变体，应识别为变体而非标准 TSP
        "vehicle routing", "vrp", "车辆路径",
        "tsp with time window", "tsptw", "带时间窗",
        "capacitated", "容量约束",
        "multiple traveling salesman", "mtsp", "多旅行商",
        "变体", "variant", "extension", "扩展",
    ],
    "D": [  # 复杂新问题，无固定答案
        # 对于 D 类问题，我们评估回答的合理性而非关键词匹配
    ],
}


def score_identification(response_text: str, category: str) -> dict:
    """
    评估模型对问题类型的识别准确率。

    Args:
        response_text: 模型的文本回复
        category: 问题类别 ("A", "B", "C", "D")

    Returns:
        dict: {
            "identified_correctly": bool,
            "confidence_score": float (0-1),
            "matched_keywords": list[str],
            "category": str
        }
    """
    if not response_text:
        return {
            "identified_correctly": False,
            "confidence_score": 0.0,
            "matched_keywords": [],
            "category": category,
        }

    text_lower = response_text.lower()
    keywords = IDENTIFICATION_KEYWORDS.get(category, [])

    matched = [kw for kw in keywords if kw.lower() in text_lower]
    identified = len(matched) > 0

    # 置信度基于匹配关键词数量（最多匹配 3 个即满分）
    confidence = min(len(matched) / 3.0, 1.0) if identified else 0.0

    # 对 C 类（变体）：如果模型错误地认为是标准 TSP 且未提及变体，则扣分
    if category == "C":
        has_tsp = any(kw in text_lower for kw in ["tsp", "旅行商", "travelling salesman"])
        has_variant = any(kw in text_lower for kw in [
            "variant", "变体", "extension", "扩展", "different", "不同",
            "time window", "时间窗", "capacitat", "容量", "multiple", "多",
        ])
        if has_tsp and not has_variant:
            # 识别为标准 TSP 但未注意到变体特征
            identified = False
            confidence = 0.3

    return {
        "identified_correctly": identified,
        "confidence_score": round(confidence, 3),
        "matched_keywords": matched,
        "category": category,
    }


# ============================================================
# 2. 算法性能评估（Algorithm Performance）
# ============================================================

def score_algorithm_performance(
    obtained_distance: float,
    optimal_distance: float,
    route_valid: bool,
    execution_time: float,
    n_cities: int,
) -> dict:
    """
    评估生成算法的求解性能。

    Args:
        obtained_distance: 模型算法得到的路径总距离
        optimal_distance: 已知最优解距离
        route_valid: 路径是否合法
        execution_time: 执行时间（秒）
        n_cities: 城市数量

    Returns:
        dict: {
            "approximation_ratio": float,     # 近似比（越接近 1 越好）
            "optimality_gap_pct": float,       # 与最优解的差距百分比
            "route_valid": bool,               # 路径合法性
            "execution_time_s": float,         # 执行时间
            "performance_score": float,        # 性能综合分（0-1）
            "timeout": bool                    # 是否超时
        }
    """
    # 路径不合法则性能分为 0
    if not route_valid or obtained_distance is None:
        return {
            "approximation_ratio": float("inf"),
            "optimality_gap_pct": float("inf"),
            "route_valid": route_valid,
            "execution_time_s": execution_time,
            "performance_score": 0.0,
            "timeout": execution_time >= 59,
        }

    if optimal_distance is None or optimal_distance <= 0:
        # 无最优解参考，仅根据路径合法性和执行时间评分
        return {
            "approximation_ratio": None,
            "optimality_gap_pct": None,
            "route_valid": True,
            "execution_time_s": execution_time,
            "performance_score": 0.5,  # 无法评估求解质量
            "timeout": False,
        }

    approx_ratio = obtained_distance / optimal_distance
    gap_pct = (approx_ratio - 1.0) * 100

    # 性能综合分计算：
    # - 近似比 = 1.0 → 1.0 分
    # - 近似比 = 1.1 → ~0.8 分（差距 10%）
    # - 近似比 = 1.5 → ~0.4 分（差距 50%）
    # - 近似比 >= 2.0 → 0.0 分
    # 使用指数衰减: score = exp(-k * (ratio - 1))，k=3 时 ratio=1.23 → score≈0.5
    k = 3.0
    perf_score = math.exp(-k * max(approx_ratio - 1.0, 0))

    # 时间惩罚：超过合理时间则扣分
    # 小问题（n<=20）应在 1s 内完成，大问题（n=50）在 10s 内
    time_limit = 1.0 if n_cities <= 20 else 10.0
    if execution_time > time_limit:
        time_penalty = min((execution_time - time_limit) / time_limit, 0.3)
        perf_score *= (1 - time_penalty)

    return {
        "approximation_ratio": round(approx_ratio, 4),
        "optimality_gap_pct": round(gap_pct, 2),
        "route_valid": True,
        "execution_time_s": execution_time,
        "performance_score": round(perf_score, 4),
        "timeout": execution_time >= 59,
    }


# ============================================================
# 3. 综合评分（Overall Scoring）
# ============================================================

def compute_overall_score(
    identification: dict,
    algorithm_perf: dict | None,
    layer: int,
) -> dict:
    """
    计算单个测试用例的综合评分。

    评分维度（各维度权重根据层级不同而调整）：
    - 识别准确率（identification_score）
    - 算法性能（algorithm_score）
    - 回答完整性（completeness_score）

    Args:
        identification: score_identification() 的返回值
        algorithm_perf: score_algorithm_performance() 的返回值（可为 None）
        layer: 测试层级 (1-10)

    Returns:
        dict: {
            "total_score": float (0-100),
            "identification_score": float (0-100),
            "algorithm_score": float (0-100),
            "breakdown": {...}
        }
    """
    id_score = identification["confidence_score"] * 100 if identification["identified_correctly"] else 0.0
    alg_score = (algorithm_perf["performance_score"] * 100) if algorithm_perf else None

    # 根据层级调整权重
    # 低层级（1-4）：以识别为主
    # 高层级（5-8）：以算法性能为主
    # 超高层级（9-10）：两者兼顾
    if layer <= 4:
        id_weight, alg_weight = 0.8, 0.2
    elif layer <= 8:
        id_weight, alg_weight = 0.3, 0.7
    else:
        id_weight, alg_weight = 0.5, 0.5

    # 如果没有算法评估，则识别分数权重为 1
    if alg_score is None:
        total = id_score
    else:
        total = id_weight * id_score + alg_weight * alg_score

    return {
        "total_score": round(total, 2),
        "identification_score": round(id_score, 2),
        "algorithm_score": round(alg_score, 2) if alg_score is not None else None,
        "breakdown": {
            "identified_correctly": identification["identified_correctly"],
            "id_confidence": identification["confidence_score"],
            "matched_keywords": identification["matched_keywords"],
            "approximation_ratio": algorithm_perf.get("approximation_ratio") if algorithm_perf else None,
            "optimality_gap_pct": algorithm_perf.get("optimality_gap_pct") if algorithm_perf else None,
            "route_valid": algorithm_perf.get("route_valid") if algorithm_perf else None,
            "execution_time_s": algorithm_perf.get("execution_time_s") if algorithm_perf else None,
        },
    }


# ============================================================
# 4. 跨模型对比指标（Cross-Model Metrics）
# ============================================================

def compute_cross_model_metrics(all_evaluations: dict) -> dict:
    """
    计算跨模型对比的核心指标。

    Args:
        all_evaluations: {
            "model_name": [eval_result_1, eval_result_2, ...],
            ...
        }

    Returns:
        dict: {
            "models": {...},        # 各模型汇总指标
            "delta_identification": {...},  # 识别准确率差距
            "delta_performance": {...},     # 算法性能差距
            "generalization_decay": {...},  # 泛化衰减曲线数据
        }
    """
    model_summaries = {}

    for model_name, evals in all_evaluations.items():
        # 按类别分组
        by_category = {"A": [], "B": [], "C": [], "D": []}
        for ev in evals:
            cat = ev.get("category", "A")
            by_category.setdefault(cat, []).append(ev)

        # 识别准确率（按类别）
        id_accuracy = {}
        for cat, items in by_category.items():
            if items:
                correct = sum(1 for it in items if it["scores"]["identification_score"] > 50)
                id_accuracy[cat] = round(correct / len(items), 3)
            else:
                id_accuracy[cat] = None

        # 算法性能（按类别，仅统计有算法评估的）
        alg_performance = {}
        for cat, items in by_category.items():
            perf_items = [
                it for it in items
                if it["scores"].get("algorithm_score") is not None
            ]
            if perf_items:
                avg_perf = sum(it["scores"]["algorithm_score"] for it in perf_items) / len(perf_items)
                alg_performance[cat] = round(avg_perf, 2)
            else:
                alg_performance[cat] = None

        model_summaries[model_name] = {
            "id_accuracy_by_category": id_accuracy,
            "alg_performance_by_category": alg_performance,
            "avg_total_score": round(
                sum(ev["scores"]["total_score"] for ev in evals) / len(evals), 2
            ) if evals else 0,
            "total_cases": len(evals),
        }

    # 识别准确率差距：Δ_识别 = accuracy(标准TSP) - accuracy(伪装TSP)
    delta_id = {}
    for model_name, summary in model_summaries.items():
        a_acc = summary["id_accuracy_by_category"].get("A") or 0
        b_acc = summary["id_accuracy_by_category"].get("B") or 0
        delta_id[model_name] = round(a_acc - b_acc, 3)

    # 算法性能差距：Δ_性能 = performance(伪装TSP) / performance(标准TSP)
    delta_perf = {}
    for model_name, summary in model_summaries.items():
        a_perf = summary["alg_performance_by_category"].get("A") or 0
        b_perf = summary["alg_performance_by_category"].get("B") or 0
        ratio = b_perf / a_perf if a_perf > 0 else 0
        delta_perf[model_name] = round(ratio, 3)

    # 泛化衰减数据：按类别（A→B→C→D）的性能变化
    decay_curves = {}
    for model_name, summary in model_summaries.items():
        curve = []
        for cat in ["A", "B", "C", "D"]:
            perf = summary["alg_performance_by_category"].get(cat)
            curve.append({"category": cat, "performance": perf})
        decay_curves[model_name] = curve

    return {
        "models": model_summaries,
        "delta_identification": delta_id,
        "delta_performance": delta_perf,
        "generalization_decay": decay_curves,
        "interpretation": {
            "delta_identification": "Δ_识别 = accuracy(标准) - accuracy(伪装)，越大说明越依赖表面特征记忆",
            "delta_performance": "Δ_性能 = performance(伪装) / performance(标准)，越接近1说明推理能力越强",
            "generalization_decay": "泛化衰减曲线：A→B→C→D 性能下降越平缓，推理能力越强",
        },
    }
