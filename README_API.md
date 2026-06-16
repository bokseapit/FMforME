# FMforME — API 配置与使用说明 (Windows)

## 三种模型使用方法

### ① DeepSeek V4 Flash（免费/便宜，速度快）

**PowerShell（推荐）：**
```powershell
$env:DEEPSEEK_API_KEY="sk-你的DeepSeek密钥"
$env:LLM_BACKEND="deepseek"
python run.py
```

**CMD：**
```cmd
set DEEPSEEK_API_KEY=sk-你的DeepSeek密钥
set LLM_BACKEND=deepseek
python run.py
```

说明：`LLM_BACKEND=deepseek` 使用 `deepseek-chat` 模型（V4 Flash）。

---

### ② DeepSeek V4 Pro

**PowerShell：**
```powershell
$env:DEEPSEEK_API_KEY="sk-你的DeepSeek密钥"
$env:LLM_BACKEND="deepseek-pro"
python run.py
```

**CMD：**
```cmd
set DEEPSEEK_API_KEY=sk-你的DeepSeek密钥
set LLM_BACKEND=deepseek-pro
python run.py
```

---

### ③ 智谱 GLM

**PowerShell：**
```powershell
$env:OPENAI_API_KEY="你的智谱API密钥"
$env:LLM_BACKEND="openai_compatible"
$env:LLM_API_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
$env:LLM_MODEL="glm-4.5"
python run.py
```

**CMD：**
```cmd
set OPENAI_API_KEY=你的智谱API密钥
set LLM_BACKEND=openai_compatible
set LLM_API_BASE_URL=https://open.bigmodel.cn/api/paas/v4
set LLM_MODEL=glm-4.5
python run.py
```

智谱可选模型：`glm-4.5`、`glm-4-plus`、`glm-4-flash`（免费）、`glm-4-air`。

---

## 命令速查

| 命令 | 作用 |
|------|------|
| `python run.py` | 完整流水线（实验→指标→评估→归档） |
| `python run.py --quick` | 快速调试（样本极小，几秒验证流程） |
| `python run.py --experiment-only` | 只跑实验，不评估 |
| `python run.py --evaluate-only` | 已有结果，只做评估+归档 |
| `python run.py --no-archive` | 不归档结果 |

---

## 文件输出结构

每次运行后，结果会自动保存到 `results/` 下带描述性名称的文件夹：

```
FMforME/
├── results/
│   ├── 20260616_143052_deepseek-chat_nat30_rep3_se/
│   │   ├── _run_info.txt                ← 运行参数记录
│   │   ├── experiment_results.json      ← 原始详细结果
│   │   ├── experiment_summary.json      ← 汇总指标 + 95% CI
│   │   ├── evaluation_report.json       ← RQ1-RQ4 完整报告
│   │   ├── tables_for_paper.tex         ← LaTeX 表格
│   │   └── generated_specs/             ← 每个测试的 ModelSpec JSON
│   │
│   ├── 20260616_163021_deepseek-v4-pro_nat30_rep3_se/
│   │   └── ...
│   │
│   └── 20260616_190845_glm-4.5_nat30_rep3_se/
│       └── ...
│
├── run.py                   ← 一键入口
├── README_API.md            ← 本文档
├── ...（其他源码文件）
```

文件夹命名规则：`日期_时间_模型名_nat自然测试数_rep重复次数_se自我修正`

---

## 自定义实验规模

```powershell
# PowerShell 一次性设置所有参数
$env:DEEPSEEK_API_KEY="sk-xxx"
$env:NATURAL_RUNS="50"
$env:N_REPEATS="5"
$env:SELF_EXAMINE="false"
$env:LLM_TEMPERATURE="0.7"
python run.py
```

## 其他 API 平台

只要平台提供 OpenAI 兼容接口 (`/chat/completions`)，就能用：

```powershell
# OpenAI
$env:OPENAI_API_KEY="sk-xxx"
$env:LLM_BACKEND="openai_compatible"
python run.py

# Groq
$env:OPENAI_API_KEY="gsk_xxx"
$env:LLM_BACKEND="openai_compatible"
$env:LLM_API_BASE_URL="https://api.groq.com/openai/v1"
$env:LLM_MODEL="llama-3.1-70b-versatile"
python run.py

# 本地 Ollama
$env:LLM_BACKEND="openai_compatible"
$env:LLM_API_BASE_URL="http://localhost:11434/v1"
$env:LLM_MODEL="llama3"
python run.py

# 本地 vLLM
$env:LLM_BACKEND="openai_compatible"
$env:LLM_API_BASE_URL="http://localhost:8000/v1"
$env:LLM_MODEL="Qwen2.5-72B"
python run.py
```

---

## 全部环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_BACKEND` | `deepseek` / `deepseek-pro` / `gemini` / `openai_compatible` | `deepseek` |
| `LLM_MODEL` | 模型名称 | 各后端默认不同 |
| `LLM_API_BASE_URL` | API 端点地址 | 各后端默认不同 |
| `LLM_TEMPERATURE` | 采样温度 (0~2) | `1.0` |
| `LLM_MAX_TOKENS` | 最大输出 token 数 | `2048` |
| `DEEPSEEK_API_KEY` | DeepSeek 密钥 | - |
| `OPENAI_API_KEY` | OpenAI/通用密钥 | - |
| `GEMINI_API_KEY` | Gemini 密钥 | - |
| `NATURAL_RUNS` | 每个零件自然生成次数 | `30` |
| `N_REPEATS` | 每个种子测试重复次数 | `3` |
| `SELF_EXAMINE` | 是否启用自我修正 | `true` |
| `SELF_EXAMINE_RETRIES` | 自我修正重试上限 | `3` |

---

