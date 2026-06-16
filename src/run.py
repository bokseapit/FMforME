#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FMforME — Runtime Constraint Verification for AI-Assisted Structural Modeling
=============================================================================

一键入口，完整实验流水线：LLM生成 → Monitor检查 → CAD建模 → 评估报告
运行结束后自动将所有结果归档到带模型名称+时间戳的文件夹。

用法 (Windows PowerShell):
  python run.py                    # 完整流水线（使用已设置的环境变量）
  python run.py --quick            # 快速测试（小样本调试用）
  python run.py --experiment-only  # 只跑实验，跳过评估
  python run.py --evaluate-only    # 已有结果，只做评估
  python run.py --help             # 查看所有选项

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三种模型的具体用法 (Windows)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

--- ① DeepSeek V4 Pro / Flash (默认) ---
PowerShell:
  $env:DEEPSEEK_API_KEY="sk-your-deepseek-key"
  $env:LLM_BACKEND="deepseek"            # V4 Flash (deepseek-chat)
  # 或: $env:LLM_BACKEND="deepseek-pro"  # V4 Pro (deepseek-v4-pro)
  python run.py

CMD:
  set DEEPSEEK_API_KEY=sk-your-deepseek-key
  set LLM_BACKEND=deepseek
  python run.py

--- ② 智谱 GLM ---
PowerShell:
  $env:OPENAI_API_KEY="你的智谱API密钥"
  $env:LLM_BACKEND="openai_compatible"
  $env:LLM_API_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
  $env:LLM_MODEL="glm-4.5"              # 改成你用的模型名
  python run.py

CMD:
  set OPENAI_API_KEY=你的智谱API密钥
  set LLM_BACKEND=openai_compatible
  set LLM_API_BASE_URL=https://open.bigmodel.cn/api/paas/v4
  set LLM_MODEL=glm-4.5
  python run.py

--- ③ 其他 OpenAI 兼容 API ---
PowerShell:
  $env:OPENAI_API_KEY="sk-your-key"
  $env:LLM_BACKEND="openai_compatible"
  $env:LLM_API_BASE_URL="https://your-provider.com/v1"
  $env:LLM_MODEL="your-model-name"
  python run.py
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 需要归档的输出文件列表 ────────────────────────────────────────────────
ARCHIVE_FILES = [
    "experiment_results.json",
    "experiment_summary.json",
    "evaluation_report.json",
    "tables_for_paper.tex",
]

ARCHIVE_DIRS = [
    "generated_specs",
]


def get_env(key, default=""):
    """跨平台读环境变量兼容函数"""
    return os.environ.get(key, default)


def check_env():
    """检查 API key 是否已设置，否则打印详细帮助"""
    backend = get_env("LLM_BACKEND", "deepseek")
    key = (get_env("DEEPSEEK_API_KEY") or
           get_env("OPENAI_API_KEY") or
           get_env("GEMINI_API_KEY"))

    if not key:
        print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  !!  没有找到 API 密钥                                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Windows PowerShell 设置方法（推荐）：                                  ║
║                                                                        ║
║    # DeepSeek (默认)                                                   ║
║    $env:DEEPSEEK_API_KEY="sk-你的密钥"                                  ║
║    python run.py                                                       ║
║                                                                        ║
║    # 智谱 GLM                                                          ║
║    $env:OPENAI_API_KEY="你的智谱密钥"                                    ║
║    $env:LLM_BACKEND="openai_compatible"                                  ║
║    $env:LLM_API_BASE_URL="https://open.bigmodel.cn/api/paas/v4"         ║
║    $env:LLM_MODEL="glm-4.5"                                            ║
║    python run.py                                                       ║
║                                                                        ║
║  CMD 设置方法：                                                        ║
║    set DEEPSEEK_API_KEY=sk-你的密钥                                    ║
║    python run.py                                                       ║
║                                                                        ║
║  详见 README_API.md                                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
        return False
    return True


def run_cmd(cmd: list, description: str) -> bool:
    """执行一个步骤命令"""
    print(f"\n{'='*70}")
    print(f">>> {description}")
    print(f">>>   {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=PROJECT_DIR, env=os.environ.copy())
    ok = result.returncode == 0
    if not ok:
        print(f"\n!!  此步骤退出码 = {result.returncode}（非致命，继续后续步骤）")
    return ok


def build_archive_name() -> str:
    """根据当前配置生成描述性文件夹名

    格式: YYYYMMDD_HHMMSS_模型名_实验参数
    例如: 20260616_143052_deepseek-chat_nat30_rep3
    """
    now = datetime.now()
    backend = get_env("LLM_BACKEND", "deepseek")
    model = get_env("LLM_MODEL", "")
    natural = get_env("NATURAL_RUNS", "10")
    repeats = get_env("N_REPEATS", "2")
    se = "se" if get_env("SELF_EXAMINE", "true").lower() == "true" else "nose"

    # 解析出干净的模型简称
    if not model:
        if backend == "deepseek":
            model = "deepseek-chat"
        elif backend == "deepseek-pro":
            model = "deepseek-v4-pro"
        elif backend == "gemini":
            model = "gemini-2.5-flash"
        elif backend == "openai_compatible":
            model = "custom"
        else:
            model = backend

    # 简化模型名用于文件名（去掉特殊字符）
    safe_model = model.replace("/", "-").replace("\\", "-").replace(":", "-").replace(" ", "_")

    name = f"{now.strftime('%Y%m%d_%H%M%S')}_{safe_model}_nat{natural}_rep{repeats}_{se}"
    return name


def archive_results(archive_name: str):
    """将本次实验结果归档到 results/ 下的专用文件夹"""
    archive_dir = os.path.join(PROJECT_DIR, "results", archive_name)
    os.makedirs(archive_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f">>> 归档结果到: results/{archive_name}/")
    print(f"{'='*70}")

    # 复制文件
    for fname in ARCHIVE_FILES:
        src = os.path.join(PROJECT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(archive_dir, fname))
            print(f"  已归档: {fname}")

    # 复制 generated_specs 整个目录
    specs_src = os.path.join(PROJECT_DIR, "generated_specs")
    specs_dst = os.path.join(archive_dir, "generated_specs")
    if os.path.exists(specs_src):
        if os.path.exists(specs_dst):
            shutil.rmtree(specs_dst)
        shutil.copytree(specs_src, specs_dst)
        spec_count = sum(1 for f in os.listdir(specs_src) if f.endswith('.json'))
        print(f"  已归档: generated_specs/ ({spec_count} 个 JSON 文件)")

    # 写一个运行信息文件
    info_path = os.path.join(archive_dir, "_run_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"FMforME Experiment Run\n")
        f.write(f"{'='*50}\n")
        f.write(f"时间:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"后端:        {get_env('LLM_BACKEND', 'deepseek')}\n")
        f.write(f"模型:        {get_env('LLM_MODEL', '默认')}\n")
        f.write(f"API地址:     {get_env('LLM_API_BASE_URL', '默认')}\n")
        f.write(f"自然测试/件: {get_env('NATURAL_RUNS', '10')}\n")
        f.write(f"重复次数:    {get_env('N_REPEATS', '2')}\n")
        f.write(f"自我修正:    {get_env('SELF_EXAMINE', 'true')}\n")
        f.write(f"温度:        {get_env('LLM_TEMPERATURE', '1.0')}\n")
        f.write(f"最大Token:   {get_env('LLM_MAX_TOKENS', '2048')}\n")
    print(f"  已写入: _run_info.txt")

    print(f"\n  所有结果已归档到: {archive_dir}")


def print_config():
    """打印当前运行配置"""
    backend = get_env("LLM_BACKEND", "deepseek")
    model = get_env("LLM_MODEL", "(自动)")
    base_url = get_env("LLM_API_BASE_URL", "(默认)")
    natural = get_env("NATURAL_RUNS", "10")
    repeats = get_env("N_REPEATS", "2")
    se = get_env("SELF_EXAMINE", "true")

    print("运行配置:")
    print(f"  后端:           {backend}")
    print(f"  模型:           {model}")
    print(f"  API地址:        {base_url}")
    print(f"  温度:           {get_env('LLM_TEMPERATURE', '1.0')}")
    print(f"  最大Token:      {get_env('LLM_MAX_TOKENS', '2048')}")
    print(f"  自然测试/零件:  {natural} (共 {int(natural)*3} 次)")
    print(f"  种子测试重复:   {repeats} (48个用例 x {repeats} = {48*int(repeats)} 次)")
    print(f"  自我修正:       {se}")
    print(f"  总测试次数:     {48*int(repeats) + int(natural)*3}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="FMforME — 一键实验流水线（运行后自动归档结果）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例 (Windows PowerShell):

  # DeepSeek
  $env:DEEPSEEK_API_KEY="sk-xxx"
  python run.py

  # 智谱 GLM
  $env:OPENAI_API_KEY="你的智谱密钥"
  $env:LLM_BACKEND="openai_compatible"
  $env:LLM_API_BASE_URL="https://open.bigmodel.cn/api/paas/v4"
  $env:LLM_MODEL="glm-4.5"
  python run.py

  # 快速调试
  python run.py --quick

结果自动保存到: results/YYYYMMDD_HHMMSS_模型名_nat30_rep3_se/
        """
    )
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式（NATURAL_RUNS=2, N_REPEATS=1, 关闭自我修正）")
    parser.add_argument("--experiment-only", action="store_true",
                        help="只运行实验，不评估也不归档")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="评估已有的 experiment_results.json")
    parser.add_argument("--no-archive", action="store_true",
                        help="跳过结果归档")
    args = parser.parse_args()

    # ── Banner ──────────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  运行时约束验证 — AI辅助结构建模实验                       ║")
    print("║  FMforME — Runtime Constraint Verification                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Quick mode ──────────────────────────────────────────────────────
    if args.quick:
        print("⚡ 快速测试模式（小样本调试）")
        os.environ["NATURAL_RUNS"] = "2"
        os.environ["N_REPEATS"] = "1"
        os.environ["SELF_EXAMINE"] = "false"

    print_config()

    # ── Evaluate only ───────────────────────────────────────────────────
    if args.evaluate_only:
        print("📊 仅评估模式：跳过实验")
        run_cmd([sys.executable, "compute_metrics.py"], "步骤 1/2: 计算指标")
        run_cmd([sys.executable, "evaluate.py"], "步骤 2/2: 生成评估报告")

        if not args.no_archive:
            archive_name = build_archive_name() + "_evalonly"
            archive_results(archive_name)
        print("\nDONE.")
        return

    # ── Experiment only ─────────────────────────────────────────────────
    if args.experiment_only:
        if not check_env():
            sys.exit(1)
        run_cmd([sys.executable, "run_experiment.py"], "运行实验（LLM → Monitor → CAD）")
        print("\n实验完成。运行 'python run.py --evaluate-only' 进行评估。")
        return

    # ── Full pipeline ───────────────────────────────────────────────────
    if not check_env():
        sys.exit(1)

    t_start = time.time()

    ok1 = run_cmd([sys.executable, "run_experiment.py"],
                  "步骤 1/3: 运行实验（LLM → Monitor → CAD）")
    ok2 = run_cmd([sys.executable, "compute_metrics.py"],
                  "步骤 2/3: 计算详细指标（含置信区间）")
    ok3 = run_cmd([sys.executable, "evaluate.py"],
                  "步骤 3/3: 生成评估报告和LaTeX表格")

    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f} 分钟)")

    # ── 归档 ────────────────────────────────────────────────────────────
    if not args.no_archive:
        archive_name = build_archive_name()
        archive_results(archive_name)

    print("\n" + "=" * 70)
    print("全部完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
