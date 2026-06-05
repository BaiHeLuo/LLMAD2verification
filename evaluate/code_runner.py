"""
代码运行器：安全地执行 LLM 生成的 Python 代码，并捕获输出和性能数据。
"""

import json
import subprocess
import sys
import os
import tempfile
import time


# 代码执行的超时时间（秒）
CODE_TIMEOUT = 60


def run_generated_code(
    code: str,
    distance_matrix: list[list[float]],
    timeout: int = CODE_TIMEOUT
) -> dict:
    """
    执行 LLM 生成的 Python 代码，传入距离矩阵，获取求解结果。

    期望生成的代码包含 solve(distance_matrix) 函数，
    返回 {"route": [...], "total_distance": float}

    Args:
        code: Python 代码字符串
        distance_matrix: 距离矩阵（二维列表）
        timeout: 执行超时（秒）

    Returns:
        dict: {
            "success": bool,
            "route": list[int] | None,
            "total_distance": float | None,
            "stdout": str,
            "stderr": str,
            "elapsed_seconds": float,
            "error": str | None
        }
    """
    result = {
        "success": False,
        "route": None,
        "total_distance": None,
        "stdout": "",
        "stderr": "",
        "elapsed_seconds": 0,
        "error": None,
    }

    if not code or not code.strip():
        result["error"] = "无代码可执行"
        return result

    # 构造完整的执行脚本
    dm_json = json.dumps(distance_matrix)
    runner_script = f'''
import sys
import json
import time

# 注入的距离矩阵
_distance_matrix = json.loads({json.dumps(dm_json)})

# ---- LLM 生成的代码 ----
{code}
# ---- 结束 ----

# 调用 solve 函数
try:
    start = time.time()
    output = solve(_distance_matrix)
    elapsed = time.time() - start
    if isinstance(output, dict):
        output["_elapsed"] = elapsed
    print("__RESULT__" + json.dumps(output))
except Exception as e:
    print("__ERROR__" + str(e))
'''

    # 使用子进程执行
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(runner_script)
            tmp_path = tmp.name

        start = time.time()
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start

        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["elapsed_seconds"] = round(elapsed, 3)

        # 解析结果
        for line in proc.stdout.splitlines():
            if line.startswith("__RESULT__"):
                data = json.loads(line[len("__RESULT__"):])
                result["route"] = data.get("route")
                result["total_distance"] = data.get("total_distance")
                result["success"] = True
                break
            elif line.startswith("__ERROR__"):
                result["error"] = line[len("__ERROR__"):]
                break
        else:
            if proc.returncode != 0:
                result["error"] = f"Exit code {proc.returncode}: {proc.stderr[:200]}"
            else:
                result["error"] = "未找到 solve() 函数输出，代码可能未正确实现"

    except subprocess.TimeoutExpired:
        result["error"] = f"执行超时 (>{timeout}s)"
        result["elapsed_seconds"] = timeout
    except Exception as e:
        result["error"] = f"执行异常: {type(e).__name__}: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return result


def validate_route(route: list[int], n: int) -> dict:
    """
    验证路径是否合法：
    - 是否访问了所有城市
    - 每个城市是否只访问一次
    - 是否形成回路（首尾相连）

    Args:
        route: 城市索引列表
        n: 城市总数

    Returns:
        dict: {"valid": bool, "issues": [str]}
    """
    issues = []

    if not route or not isinstance(route, list):
        return {"valid": False, "issues": ["路径为空或格式错误"]}

    if len(route) != n:
        issues.append(f"路径长度 {len(route)} != 城市数 {n}")

    if len(set(route)) != len(route):
        issues.append("存在重复访问的城市")

    if any(not isinstance(c, int) or c < 0 or c >= n for c in route):
        issues.append(f"城市索引超出范围 [0, {n-1}]")

    return {"valid": len(issues) == 0, "issues": issues}


def compute_route_distance(route: list[int], distance_matrix: list[list[float]]) -> float:
    """计算给定路径的总距离（含回到起点）"""
    total = 0
    n = len(route)
    for i in range(n):
        total += distance_matrix[route[i]][route[(i + 1) % n]]
    return total
