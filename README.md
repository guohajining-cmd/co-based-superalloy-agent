# 钴基高温合金设计 Workflow

这个项目是一个面向钴基高温合金设计流程的工程化 workflow 原型。它把已有的性能预测、模型解释、训练范围检查和多目标优化步骤整理成统一的 Python 调用流程，并提供 Streamlit 页面用于演示和查看结果。

用户可以输入合金成分、热处理条件、测试条件和微观组织参数。系统会按照预设流程调用屈服强度预测、氧化增重预测、SHAP 解释、训练集范围检查和 NSGA-II 候选优化模块，并输出预测结果、特征贡献、训练范围提示、Pareto 候选合金和工具调用记录。

```text
自然语言或表单输入
→ 标准 AlloyInput
→ 输入检查和流程选择
→ XGBoost 屈服强度预测
→ XGBoost 氧化增重预测
→ SHAP 解释
→ 训练集范围检查
→ NSGA-II 双目标优化
→ 报告和界面展示
```

## 当前功能

| 功能 | 当前实现 |
|---|---|
| 合金输入 | 支持表单输入；自然语言输入会先解析为标准输入结构 |
| 屈服强度预测 | 调用已接入的 XGBoost 模型 |
| 氧化增重预测 | 调用已接入的 XGBoost 模型 |
| SHAP 解释 | 使用 `shap.TreeExplainer` 生成主要特征贡献 |
| 训练集范围检查 | 根据训练 CSV 提示输入是否接近或超出训练范围 |
| NSGA-II 优化 | 使用 `pymoo` 生成候选合金 |
| 结果展示 | Streamlit 页面展示预测值、解释图、范围提示、Pareto 图和候选表 |
| 命令行示例 | 提供 `demo.py` |
| 测试 | `42 tests OK` |

## 快速运行

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

启动 Streamlit 页面：

```bash
python3 -m streamlit run web_streamlit.py
```

打开：

```text
http://127.0.0.1:8501
```

运行命令行 demo：

```bash
python3 demo.py
```

运行测试：

```bash
python3 -m unittest discover -s tests -v
```

## 运行模式

| Mode | 作用 | 主要步骤 |
|---|---|---|
| `evaluate` | 评估一个已有合金 | 强度预测、氧化预测、SHAP 解释、训练范围检查、评估报告 |
| `optimize` | 在给定搜索空间内优化候选合金 | NSGA-II 搜索、候选强度/氧化评分、优化报告 |
| `full` | 先评估当前合金，再生成候选合金 | `evaluate` + 优化请求构建 + `optimize` + 完整报告 |

## 主要模块

```text
src/alloy_agent/
├── schemas.py                # 输入输出数据结构
├── agent.py                  # workflow 调度入口
├── agent_loop.py             # 规则型流程入口
├── natural_language.py       # 自然语言输入解析
├── validators.py             # 输入检查和结果检查
├── design_constraints.py     # NSGA-II 搜索空间和约束
├── tool_trace.py             # 工具调用记录
├── workflows/
│   ├── evaluate.py           # 已有合金评估流程
│   ├── optimize.py           # 候选合金优化流程
│   └── full.py               # 评估 + 优化流程
├── tools/
│   ├── strength_model.py     # 屈服强度预测
│   ├── oxidation_model.py    # 氧化增重预测
│   ├── shap_explainer.py     # SHAP 解释
│   ├── distribution_check.py # 训练范围检查
│   ├── nsga2_optimizer.py    # NSGA-II 优化
│   └── report_generator.py   # 报告整理
└── web_app.py                # 简单 HTTP 页面入口

web_streamlit.py              # Streamlit 演示页面
demo.py                       # 命令行示例
tests/                        # unittest 测试
Agent-acta/                   # workflow 运行所需模型和训练范围检查 CSV
```

## 输入数据结构

核心输入结构为 `AlloyInput`，包含四类信息：

| 字段 | 内容 |
|---|---|
| `composition` | 合金成分，如 Co、Ni、Al、Cr、Ta、Ti、W、V、Nb、Mo |
| `processing` | 热处理参数，如 aging temperature 和 aging time |
| `test_conditions` | 测试条件，如强度测试温度、氧化温度和氧化时间 |
| `microstructure` | 微观组织参数，如 `Vol` / `Vγ′` |

## NSGA-II 搜索空间

workflow 当前支持两种搜索空间：

| Profile | 说明 |
|---|---|
| `local` | 围绕当前输入合金做局部搜索 |
| `script` | 使用原始 NSGA-II 脚本中定义的搜索范围，并在 workflow 中重新实现调用 |

默认优化约束定义在 `src/alloy_agent/design_constraints.py`：

```python
DEFAULT_CONSTRAINTS = {
    "yield_strength_min": 800.0,
    "oxidation_mass_gain_min": 0.0,
    "oxidation_mass_gain_max": 3.0,
}
```

## 上传内容说明

这个仓库只保留 workflow 运行相关文件，包括源码、测试、演示入口、依赖文件，以及运行所需的模型和训练范围检查 CSV。论文稿件、临时输出、打包文件和原始分析脚本不放入仓库。
