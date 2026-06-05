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
    extra_scores: dict | None = None,
) -> dict:
    """
    计算单个测试用例的综合评分。

    评分维度（各维度权重根据层级不同而调整）：
    - 识别准确率（identification_score）
    - 算法性能（algorithm_score）
    - Layer 7-10 的额外评分

    Args:
        identification: score_identification() 的返回值
        algorithm_perf: score_algorithm_performance() 的返回值（可为 None）
        layer: 测试层级 (1-10)
        extra_scores: 额外评分 (delayed_id, confidence_cal, metacognition, transfer)

    Returns:
        dict: {
            "total_score": float (0-100),
            "identification_score": float (0-100),
            "algorithm_score": float (0-100),
            "layer_specific_score": float (0-100) | None,
            "breakdown": {...}
        }
    """
    id_score = identification["confidence_score"] * 100 if identification["identified_correctly"] else 0.0
    alg_score = (algorithm_perf["performance_score"] * 100) if algorithm_perf else None
    extra = extra_scores or {}

    # 层级专属分数
    layer_specific = None
    if layer == 7 and "delayed_id" in extra:
        d = extra["delayed_id"]
        layer_specific = (d["convergence_score"] * 60 + d["early_guess_score"] * 40)
    elif layer == 8 and "confidence_cal" in extra:
        layer_specific = extra["confidence_cal"]["calibration_score"] * 100
    elif layer == 9 and "metacognition" in extra:
        layer_specific = extra["metacognition"]["metacognition_score"] * 100
    elif layer == 10 and "transfer" in extra:
        layer_specific = extra["transfer"]["transfer_score"] * 100

    # 根据层级调整权重
    if layer <= 4:
        id_weight, alg_weight, extra_weight = 0.8, 0.2, 0.0
    elif layer <= 6:
        id_weight, alg_weight, extra_weight = 0.3, 0.7, 0.0
    elif layer <= 8:
        id_weight, alg_weight, extra_weight = 0.2, 0.3, 0.5
    else:
        id_weight, alg_weight, extra_weight = 0.1, 0.2, 0.7

    # 计算加权总分
    components = []
    weights = []

    components.append(id_score)
    weights.append(id_weight)

    if alg_score is not None:
        components.append(alg_score)
        weights.append(alg_weight)
    else:
        # 没有算法分数时，将其权重分配给其他组件
        if layer_specific is not None:
            extra_weight += alg_weight
        else:
            id_weight += alg_weight
        weights[-1] = id_weight if alg_score is None else weights[-1]

    if layer_specific is not None:
        components.append(layer_specific)
        weights.append(extra_weight)
    else:
        # 没有额外分数时，将其权重分配给识别
        weights[0] += extra_weight

    # 归一化权重并计算总分
    total_w = sum(weights)
    total = sum(c * w / total_w for c, w in zip(components, weights))

    return {
        "total_score": round(total, 2),
        "identification_score": round(id_score, 2),
        "algorithm_score": round(alg_score, 2) if alg_score is not None else None,
        "layer_specific_score": round(layer_specific, 2) if layer_specific is not None else None,
        "breakdown": {
            "identified_correctly": identification["identified_correctly"],
            "id_confidence": identification["confidence_score"],
            "matched_keywords": identification["matched_keywords"],
            "approximation_ratio": algorithm_perf.get("approximation_ratio") if algorithm_perf else None,
            "optimality_gap_pct": algorithm_perf.get("optimality_gap_pct") if algorithm_perf else None,
            "route_valid": algorithm_perf.get("route_valid") if algorithm_perf else None,
            "execution_time_s": algorithm_perf.get("execution_time_s") if algorithm_perf else None,
            "extra": {k: v for k, v in extra.items()},
        },
    }


# ============================================================
# 5. Layer 7：延迟识别评估（Delayed Identification）
# ============================================================

def score_delayed_identification(turns: list[dict], category: str) -> dict:
    """
    评估多轮对话中的延迟识别能力。
    记录模型在哪一轮首次正确识别出问题类型。

    Args:
        turns: [{"prompt": ..., "response": ...}, ...] 多轮对话记录
        category: 问题类别 ("A", "B", "C")

    Returns:
        dict: {
            "first_correct_turn": int | None,  # 首次正确识别的轮次（1-based）
            "total_turns": int,
            "per_turn_identification": [...],   # 每轮的识别结果
            "early_guess_score": float,         # 早期猜测质量（0-1）
            "convergence_score": float,         # 最终收敛质量（0-1）
        }
    """
    per_turn = []
    first_correct = None

    for i, turn in enumerate(turns):
        resp = turn.get("response", "") or ""
        id_result = score_identification(resp, category)
        id_result["turn"] = i + 1
        per_turn.append(id_result)

        if id_result["identified_correctly"] and first_correct is None:
            first_correct = i + 1

    total = len(turns)

    # 早期猜测质量：前几轮猜得越接近越好
    early_score = 0.0
    if per_turn:
        confidences = [t["confidence_score"] for t in per_turn]
        # 越早的轮次权重越高（线性衰减）
        weights = [total - i for i in range(total)]
        total_weight = sum(weights)
        early_score = sum(c * w for c, w in zip(confidences, weights)) / total_weight

    # 最终收敛质量：最后一轮的置信度
    convergence = per_turn[-1]["confidence_score"] if per_turn else 0.0

    return {
        "first_correct_turn": first_correct,
        "total_turns": total,
        "per_turn_identification": per_turn,
        "early_guess_score": round(early_score, 3),
        "convergence_score": round(convergence, 3),
        "identified": first_correct is not None,
    }


# ============================================================
# 6. Layer 8：置信度校准评估（Confidence Calibration）
# ============================================================

def _extract_confidence(text: str) -> float | None:
    """从文本中提取置信度百分比 (0-100)"""
    if not text:
        return None
    import re
    # 匹配 "X%" 或 "X percent" 或 "confidence: X"
    patterns = [
        r'(\d{1,3})\s*%',
        r'(\d{1,3})\s*percent',
        r'confidence[^:]*:\s*(\d{1,3})',
        r'(\d{1,3})\s*/\s*100',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 100:
                return val
            if 0 <= val <= 1:
                return val * 100
    return None


def _extract_judgment(text: str) -> bool | None:
    """从文本中提取 Yes/No 判断"""
    if not text:
        return None
    text_lower = text.lower()
    # 优先查找明确的判断
    if re.search(r'\b(yes|是|equivalent|等价|同构)\b', text_lower):
        return True
    if re.search(r'\b(no|否|not\s+(the\s+)?(same|standard|tsp)|不同|不是)\b', text_lower):
        return False
    return None


def score_confidence_calibration(response_text: str, expected: dict) -> dict:
    """
    评估置信度校准质量。
    一个“诚实”的模型应在信息完整时给高置信度，在信息不全时给低置信度。

    Args:
        response_text: 模型回复
        expected: expected_output 字段，包含 judgment 和 expected_confidence

    Returns:
        dict: {
            "extracted_confidence": float | None,
            "extracted_judgment": bool | None,
            "judgment_correct": bool | None,
            "confidence_gap": float,  # |extracted - expected|
            "calibration_score": float (0-1),
        }
    """
    extracted_conf = _extract_confidence(response_text)
    extracted_judg = _extract_judgment(response_text)
    expected_judg = expected.get("judgment")
    expected_conf = expected.get("expected_confidence", 50)

    judgment_correct = None
    if extracted_judg is not None and expected_judg is not None:
        judgment_correct = (extracted_judg == expected_judg)

    conf_gap = abs(extracted_conf - expected_conf) if extracted_conf is not None else 100
    # 校准分数：置信度越接近期望越好，判断正确额外加分
    cal_score = max(0, 1 - conf_gap / 50)  # 差距 50% 则 0 分
    if judgment_correct is True:
        cal_score = min(cal_score + 0.2, 1.0)
    elif judgment_correct is False:
        cal_score *= 0.5

    return {
        "extracted_confidence": extracted_conf,
        "extracted_judgment": extracted_judg,
        "judgment_correct": judgment_correct,
        "confidence_gap": round(conf_gap, 2),
        "calibration_score": round(cal_score, 3),
    }


# ============================================================
# 7. Layer 9：元认知评估（Metacognition）
# ============================================================

def score_metacognition(response_text: str) -> dict:
    """
    评估元认知能力：推理链追溯、知识边界自评、不确定性分析。
    使用结构化检测来评估回答的深度和组织性。

    Args:
        response_text: 模型回复

    Returns:
        dict: {
            "has_reasoning_chain": bool,
            "reasoning_steps": int,
            "has_uncertainty_analysis": bool,
            "has_knowledge_boundary": bool,
            "has_failure_modes": bool,
            "structural_completeness": float (0-1),
            "metacognition_score": float (0-1),
        }
    """
    if not response_text:
        return {
            "has_reasoning_chain": False, "reasoning_steps": 0,
            "has_uncertainty_analysis": False, "has_knowledge_boundary": False,
            "has_failure_modes": False, "structural_completeness": 0.0,
            "metacognition_score": 0.0,
        }

    text_lower = response_text.lower()
    text_len = len(response_text)

    # 检测推理链
    step_markers = re.findall(
        r'(?:step|步骤|point|关键|关键步骤|feature|要素)\s*\d',
        text_lower
    )
    numbered_items = re.findall(r'(?:^|\n)\s*\d+[.\)]\s', response_text)
    reasoning_steps = max(len(step_markers), len(numbered_items))
    has_chain = reasoning_steps >= 3

    # 检测不确定性分析
    uncertainty_kw = [
        "uncertain", "uncertainty", "不确定", "risk", "风险",
        "limitation", "局限", "might not", "may not", "可能不",
        "assumption", "假设", "potential issue", "潜在问题",
    ]
    has_uncertainty = any(kw in text_lower for kw in uncertainty_kw)

    # 检测知识边界
    boundary_kw = [
        "insufficient", "不足", "lack", "缺少", "need to learn",
        "需要学习", "don't know", "不知道", "beyond", "超出",
        "not confident", "不自信", "gap", "空白", "boundary", "边界",
        "would need", "需要补充",
    ]
    has_boundary = any(kw in text_lower for kw in boundary_kw)

    # 检测失败模式分析
    failure_kw = [
        "fail", "失败", "poor", "差", "worst case", "最坏",
        "edge case", "边界情况", "degenerate", "退化",
        "not work", "不适用", "struggle", "困难",
    ]
    has_failure = any(kw in text_lower for kw in failure_kw)

    # 结构完整性
    components = [has_chain, has_uncertainty, has_boundary, has_failure]
    completeness = sum(components) / len(components)

    # 深度加分：回答越长且包含更多结构化元素越好（但有上限）
    depth_bonus = min(text_len / 2000, 0.2)

    meta_score = completeness * 0.8 + depth_bonus
    meta_score = min(meta_score, 1.0)

    return {
        "has_reasoning_chain": has_chain,
        "reasoning_steps": reasoning_steps,
        "has_uncertainty_analysis": has_uncertainty,
        "has_knowledge_boundary": has_boundary,
        "has_failure_modes": has_failure,
        "structural_completeness": round(completeness, 3),
        "metacognition_score": round(meta_score, 3),
    }


# ============================================================
# 8. Layer 10：迁移与类比评估（Transfer & Analogy）
# ============================================================

def score_transfer_analogy(response_text: str, has_code: bool) -> dict:
    """
    评估迁移与类比能力：是否能将一个问题的解法迁移到新领域。

    Args:
        response_text: 模型回复
        has_code: 是否生成了代码

    Returns:
        dict: {
            "has_explanation": bool,
            "has_mapping": bool,
            "has_adaptation": bool,
            "has_code": bool,
            "has_comparison": bool,
            "transfer_score": float (0-1),
        }
    """
    if not response_text:
        return {
            "has_explanation": False, "has_mapping": False,
            "has_adaptation": False, "has_code": False,
            "has_comparison": False, "transfer_score": 0.0,
        }

    text_lower = response_text.lower()

    # 检测解释说明
    explain_kw = [
        "explain", "解释", "describe", "描述", "how", "如何",
        "approach", "方法", "strategy", "策略",
    ]
    has_explain = any(kw in text_lower for kw in explain_kw)

    # 检测映射关系
    mapping_kw = [
        "correspond", "对应", "map", "映射", "analogous", "类似",
        "equivalent", "等价", "similar to", "similar structure",
        "items", "weights", "values", "capacity",
        "item", "weight", "value",
    ]
    has_mapping = any(kw in text_lower for kw in mapping_kw)

    # 检测适配修改
    adapt_kw = [
        "modify", "修改", "adapt", "适配", "adjust", "调整",
        "change", "变化", "differ", "不同", "new constraint",
        "additional", "额外", "extension", "扩展",
        "cannot directly", "不能直接", "need to", "需要",
    ]
    has_adapt = any(kw in text_lower for kw in adapt_kw)

    # 检测对比分析
    compare_kw = [
        "differ", "不同", "compare", "比较", "contrast",
        "advantage", "advantage", "disadvantage", "劣势",
        "valid for", "适用", "not valid", "不适用",
    ]
    has_compare = any(kw in text_lower for kw in compare_kw)

    components = [has_explain, has_mapping, has_adapt, has_code, has_compare]
    transfer_score = sum(components) / len(components)

    return {
        "has_explanation": has_explain,
        "has_mapping": has_mapping,
        "has_adaptation": has_adapt,
        "has_code": has_code,
        "has_comparison": has_compare,
        "transfer_score": round(transfer_score, 3),
    }


# ============================================================
# 9. 跨模型对比指标（Cross-Model Metrics）
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
