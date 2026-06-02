"""
DataPrep 后端评估客户端：不用打开前端浏览器，直接命令行跑评估。

先启动后端：
    python main.py

示例：
    # EDIT
    python evaluate_backend.py --task edit

    # MLNClean Detection，使用原版 MLNClean 数据
    python evaluate_backend.py --task mln-det

    # MLNClean Correction，用原版 out.csv 作为 reference，对齐检查
    python evaluate_backend.py --task mln-cor
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    import websockets
except ImportError:
    print("缺依赖，请安装: pip install websockets")
    sys.exit(1)

DATASETS = {
    "flights": {
        "dirty": "datasets/flights/flights_dirty.csv",
        "clean": "datasets/flights/flights_clean.csv",
        "mask": "datasets/flights/flights_dirty_error_detection.csv",
        "rules": "datasets/flights/rules.txt",
        "evidence": "datasets/flights/rules_data.csv",
    },
    "rayyan": {
        "dirty": "datasets/rayyan/rayyan_dirty.csv",
        "clean": "datasets/rayyan/rayyan_clean.csv",
        "mask": "datasets/rayyan/rayyan_dirty_error_detection.csv",
        "rules": "datasets/rayyan/rules.txt",
        "evidence": "datasets/rayyan/rules_data.csv",
    },
    "tax-1k": {
        "dirty": "datasets/tax-1k/tax-1k_dirty.csv",
        "clean": "datasets/tax-1k/tax-1k_clean.csv",
        "mask": "datasets/tax-1k/tax-1k_dirty_error_detection.csv",
        "rules": "datasets/tax-1k/rules.txt",
        "evidence": "datasets/tax-1k/rules_data.csv",
    },
    "hospital": {
        "dirty": "datasets/hospital/hospital_dirty.csv",
        "clean": "datasets/hospital/hospital_clean.csv",
        "mask": "datasets/hospital/hospital_dirty_error_detection.csv",
        "rules": "datasets/hospital/rules.txt",
        "evidence": "datasets/hospital/rules_data.csv",
    },
    "beers": {
        "dirty": "datasets/beers/beers_dirty.csv",
        "clean": "datasets/beers/beers_clean.csv",
        "mask": "datasets/beers/beers_dirty_error_detection.csv",
        "rules": "datasets/beers/rules.txt",
        "evidence": "datasets/beers/rules_data.csv",
    },
    "hospital_2rule": {
        "dirty": "datasets/hospital/hospital_dirty.csv",
        "clean": "datasets/hospital/hospital_clean.csv",
        "mask": "datasets/hospital/hospital_dirty_error_detection.csv",
        "rules": "datasets/hospital/rules_2.txt",
        "evidence": "datasets/hospital/rules_data_2.csv",
    },
    "hospital_5rule": {
        "dirty": "datasets/hospital/hospital_dirty.csv",
        "clean": "datasets/hospital/hospital_clean.csv",
        "mask": "datasets/hospital/hospital_dirty_error_detection.csv",
        "rules": "datasets/hospital/rules_5.txt",
        "evidence": "datasets/hospital/rules_data_5.csv",
    },
}

def build_task(args):
    base = Path(args.data_root)

    if args.task == "edit":
        return {
            "method": "EDIT",
            "paths": {
                "dataPath": str(base / "imputation" / "weather_raw.csv"),
                "missingMaskPath": str(base / "imputation" / "weather_missing_mask.csv"),
                "groundTruthPath": str(base / "imputation" / "weather_ground_truth.csv"),
            },
            "params": {
                "batch_size": args.batch_size,
                "hint_rate": 0.9,
                "alpha": 10,
                "epoch": args.epoch,
                "initial_size": args.initial_size,
                "validation_size": args.validation_size,
            },
        }

    if args.task == "mln-det":
        ds = DATASETS[args.dataset]
        return {
            "method": "MLNClean_Det",
            "paths": {
                "dataPath": ds["dirty"],

                # 新增：用于同一次 MLNClean_Det 中计算 Detection -> Correction pipeline 指标
                # 不影响原有 detection 评估；如果后端不用该字段，也不会影响运行。
                "cleanDataPath": ds["clean"],

                "errorDetectionPath": ds["mask"],
                "rulesPath": ds["rules"],
                "evidencePath": ds["evidence"],
            },
            "params": {
                "mcmc_samples": args.mcmc_samples,
                "mcmc_warmup": args.mcmc_warmup,
                "partition_number": 1,
                "agp_threshold": 2,
            },
        }

    if args.task == "mln-cor":
        ds = DATASETS[args.dataset]
        return {
            "method": "MLNClean_Cor",
            "paths": {
                "dataPath": ds["dirty"],
                "cleanDataPath": ds["clean"],
                "detectionPath": ds["mask"],
                "rulesPath": ds["rules"],
                "evidencePath": ds["evidence"],
            },
            "params": {
                "mcmc_samples": args.mcmc_samples,
                "mcmc_warmup": args.mcmc_warmup,
                "partition_number": 1,
                "agp_threshold": 2,
            },
        }
    if args.task == "gain":
        return {
            "method": "GAIN",
            "paths": {
                "dataPath": str(base / "imputation" / "weather_raw.csv"),
                "missingMaskPath": str(base / "imputation" / "weather_missing_mask.csv"),
                "groundTruthPath": str(base / "imputation" / "weather_ground_truth.csv"),
            },
            "params": {
                "batch_size": 128,
                "hint_rate": 0.9,
                "alpha": 100,
                "epoch": args.gain_epoch,
            },
        }
    if args.task == "zeroed":
        ds = DATASETS[args.dataset]
        return {
            "method": "ZeroED",
            "paths": {
                "dataPath": ds["dirty"],
                "errorDetectionPath": ds["mask"],
            },
            "params": {
                "api_key": args.zeroed_api_key,
                "model_name": args.zeroed_model_name,
                "base_url": args.zeroed_base_url,
                "n_method": args.zeroed_n_method,
                "result_dir": args.zeroed_result_dir,
                "verbose": False,
            },
        }
    if args.task == "zeroec":
        ds = DATASETS[args.dataset]
        return {
            "method": "ZeroEC",
            "paths": {
                "dataPath": ds["dirty"],
                "cleanDataPath": ds["clean"],
                "detectionPath": ds["mask"],
                "embeddingModelPath": args.zeroec_embedding_model_path,
                "promptDir": args.zeroec_prompt_dir,
                "outputDir": args.zeroec_output_dir,
            },
            "params": {
                "model_name": args.zeroec_model_name,
                "openai_api_base": args.zeroec_api_base,
                "openai_api_key": args.zeroec_api_key,
                "human_repair_num": args.zeroec_human_repair_num,
            },
        }

    raise ValueError(f"Unknown task: {args.task}")


async def run(args):
    task = build_task(args)
    print(f">>> 连接 {args.ws_url} ...")
    async with websockets.connect(args.ws_url, max_size=None) as ws:
        print(f">>> 发送任务: method={task['method']}")
        await ws.send(json.dumps(task, ensure_ascii=False))

        while True:
            try:
                raw = await ws.recv()
            except websockets.exceptions.ConnectionClosed:
                print("\n>>> 连接关闭")
                break

            msg = json.loads(raw)
            if "log" in msg:
                print(msg["log"], end="", flush=True)
            elif msg.get("status") == "success":
                print("\n" + "=" * 60)
                print(" 评估指标")
                print("=" * 60)
                for k, v in msg["metrics"].items():
                    print(f"  {k:25s} {v}")
                print("=" * 60)
                break
            elif msg.get("status") == "error":
                print(f"\n>>> ERROR: {msg['detail']}")
                break


def parse_args():
    parser = argparse.ArgumentParser(description="Run DataPrep backend evaluation via WebSocket.")
    parser.add_argument(
        "--task",
        choices=["edit", "gain", "zeroed", "zeroec", "mln-det", "mln-cor"],
        default="edit"
    )
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8088/api/ws/run_task")
    parser.add_argument("--data-root", default="datasets")

    parser.add_argument("--gain-epoch", type=int, default=100)

    # EDIT params
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--initial-size", type=int, default=500)
    parser.add_argument("--validation-size", type=int, default=500)

    # MLNClean params; 先用小数值 smoke test，确认跑通后再调到 20/20
    parser.add_argument("--mcmc-samples", type=int, default=5)
    parser.add_argument("--mcmc-warmup", type=int, default=5)
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="flights")

    # ZeroED params
    parser.add_argument("--zeroed-api-key", default="sk-rnokjhjonvggoaddaprdpwgqsqwsmpnaadmziesaagxnuxxg")
    parser.add_argument("--zeroed-model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--zeroed-base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--zeroed-n-method", default="dbscan")
    parser.add_argument("--zeroed-result-dir", default="./temp_zeroed_results")

    # ZeroEC params
    parser.add_argument("--zeroec-model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--zeroec-api-base", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--zeroec-api-key", default="sk-rnokjhjonvggoaddaprdpwgqsqwsmpnaadmziesaagxnuxxg")
    parser.add_argument("--zeroec-embedding-model-path", default="tabular/correction/all-MiniLM-L6-v2")
    parser.add_argument("--zeroec-human-repair-num", type=int, default=10)
    parser.add_argument("--zeroec-output-dir", default="./runs_output")
    parser.add_argument("--zeroec-prompt-dir", default="prompt_templates")


    return parser.parse_args()
if __name__ == "__main__":
    asyncio.run(run(parse_args()))
