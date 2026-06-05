使用方式：
# 1. 配置 API（编辑 query/config.py，填入你的 api_key 和 base_url）

# 2. 运行查询
cd query
python run_query.py --models deepseek-chat gpt-4o-mini

# 3. 运行评估
cd evaluate
python run_evaluate.py --input ../data/responses/responses_xxx.json

# 4. 跨模型对比
python run_evaluate.py --compare

评估指标设计（metrics.py）：
指标	说明
识别准确率差距 Δ识别 | accuracy(标准TSP) - accuracy(伪装TSP)，差值越大说明越依赖表面特征记忆 |
算法性能差距 Δ性能 | performance(伪装TSP) / performance(标准TSP)，越接近 1 说明推理能力越强 |
泛化衰减曲线	A→B→C→D 类别的性能下降速度，越平缓推理越强
近似比	obtained_distance / optimal_distance，越接近 1 越好
综合评分	根据层级动态加权（低层以识别为主，高层以算法性能为主）
测试集格式（JSON）： 见 data/test_set.json，已包含 8 个示例