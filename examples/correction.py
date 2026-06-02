import pandas as pd
import numpy as np
import time

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataprep.tabular.correction.ZeroEC import ZeroEC
from dataprep.tabular.correction.MLNClean import MLNClean as MLNCleanCorrector

# 1. 评估函数：用于计算修复准确率
def evaluate_correction(name, df_clean, df_corrected, df_mask, time_cost):
    # 确保索引和列对齐
    common_idx = df_clean.index.intersection(df_corrected.index)
    common_col = df_clean.columns.intersection(df_corrected.columns)

    gt = df_clean.loc[common_idx, common_col]
    pred = df_corrected.loc[common_idx, common_col]
    mask = df_mask.loc[common_idx, common_col].astype(bool)

    # 提取错误位置的真值和预测值
    y_true_vals = gt.values[mask]
    y_pred_vals = pred.values[mask]

    if len(y_true_vals) == 0:
        return {"Model": name, "Accuracy (Repair)": "0%", "Fixed/Total": "0/0", "Time(s)": time_cost}

    # 计算修正准确率
    correct_count = 0
    total_errors = len(y_true_vals)

    for t, p in zip(y_true_vals, y_pred_vals):
        if str(t).strip() == str(p).strip():
            correct_count += 1

    accuracy = correct_count / total_errors

    return {
        "Model": name,
        "Accuracy (Repair)": f"{accuracy:.2%}",
        "Fixed/Total": f"{correct_count}/{total_errors}",
        "Time(s)": round(time_cost, 2)
    }


# 2. 主程序
if __name__ == "__main__":
    # 配置路径
    BASE_DIR = '../'
    clean_path = f'{BASE_DIR}/datasets/rayyan/rayyan_clean.csv'
    dirty_path = f'{BASE_DIR}/datasets/rayyan/rayyan_dirty.csv'
    detect_path = f'{BASE_DIR}/datasets/rayyan/rayyan_dirty_error_detection.csv'

    # 加载与预处理
    print("Loading Data...")
    df_clean = pd.read_csv(clean_path, index_col=0)
    df_dirty = pd.read_csv(dirty_path, index_col=0)
    df_mask = pd.read_csv(detect_path)

    # 移除多余列并确保索引对齐
    for df in [df_mask]:
        for col in ['index', 'Unnamed: 0']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    min_len = min(len(df_clean), len(df_dirty), len(df_mask))
    df_clean = df_clean.iloc[:min_len].reset_index(drop=True)
    df_dirty = df_dirty.iloc[:min_len].reset_index(drop=True)
    df_mask = df_mask.iloc[:min_len].reset_index(drop=True)

    # 统一格式
    df_mask = df_mask.replace({'True': True, 'False': False, 1: True, 0: False}).astype(bool)

    print(f"Data ready. Shape: {df_dirty.shape}")

    # 运行 ZeroEC
    print("\n>>> Running ZeroEC (LLM-based Correction)...")
    start_time = time.time()

    # 初始化 ZeroEC
    zeroec = ZeroEC(
        model_name="",
        openai_api_base="",
        openai_api_key="",
        embedding_model_path=f'{BASE_DIR}/all-MiniLM-L6-v2',
        human_repair_num=10,  # 每一列提供给 LLM 的样本参考数
        output_dir='./temp/runs_rayyan',
        clean_data_path=clean_path,
        dirty_data_path=dirty_path,
        detection_path=detect_path,
        prompt_dir=f'{BASE_DIR}/prompt_templates',
        max_workers=3  # 并行处理的线程数
    )

    # 执行预测并获取修复后的结果
    df_fixed_zeroec = zeroec.train_and_predict()

    cost = time.time() - start_time

    # 评估结果
    result = evaluate_correction("ZeroEC (LLM)", df_clean, df_fixed_zeroec, df_mask, cost)

    print("\n" + "=" * 60)
    print(" ZeroEC Correction Result")
    print("=" * 60)
    print(pd.DataFrame([result]).to_string(index=False))
    print("-" * 60)

    # # 1. 先用 MLNClean 检测器找错
    # det = MLNCleanDetector(
    #     rules_path='datasets/flights/rules.txt',
    #     verbose=True,
    # )
    # error_mask = det.train_and_predict(df_dirty)
    #
    # # 2. 再用 MLNClean 修复器修错 (复用同一份规则)
    # cor = MLNCleanCorrector(
    #     rules_path='datasets/flights/rules.txt',
    #     evidence_path=None,
    #     partition_number=1,
    #     mcmc_samples=20,
    #     verbose=True,
    # )
    # df_fixed = cor.train_and_predict(df_dirty, detection_mask=error_mask)
    #
    # # 也可以全量修复 (不传 mask, 相当于原版 MLNClean):
    # # df_fixed_full = cor.train_and_predict(df_dirty)
    #
    # # 评估
    # result = evaluate_correction("MLNClean", df_clean, df_fixed, df_mask, cost)
    # print(pd.DataFrame([result]).to_string(index=False))