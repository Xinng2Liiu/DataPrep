import pandas as pd
import os
import shutil

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from dataprep.tabular.detection.ZeroED import ZeroED  # 导入类
from dataprep.tabular.detection.MLNClean import MLNClean as MLNCleanDetector

# ==========================================
# 1. 准备数据
# ==========================================
def setup_data():
    # 构造一个包含 3 个明显错误的数据集
    data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'ErrorVal', 'Charlie', 'David', 'Frank', 'Grace', 'Hebe', 'Ivy', 'Jack'],
        'age': ['25', '30', '999', '35', '40', '28', '22', '45', '33', '29'],
        'city': ['NY', 'LA', 'Unknown', 'SF', 'NY', 'LA', 'SF', 'NY', 'LA', 'SF']
    })
    return data


def print_errors_nicely(df_data, df_mask):
    print("\n" + "=" * 40)
    print("      🔍 错误检测报告")
    print("=" * 40)

    error_count = 0
    # 遍历掩码矩阵，找到 True 的位置
    for col in df_mask.columns:
        for idx, is_error in df_mask[col].items():
            if is_error:  # 如果是 True，说明检测到了错误
                val = df_data.at[idx, col]
                print(f" [发现错误] 第 {idx} 行, 列 '{col}'")
                print(f"    -> 异常值: {val}")
                error_count += 1

    if error_count == 0:
        print("✅ 未发现任何错误。")
    else:
        print("-" * 40)
        print(f"总计发现 {error_count} 个错误。")
    print("=" * 40 + "\n")


# ==========================================
# 2. 主程序
# ==========================================
if __name__ == "__main__":  # 注意：这里必须是双下划线

    # 1. 准备数据
    data = setup_data()
    print(">>> 原始数据预览:")
    print(data.head(3))

    # 2. 初始化检测器
    detector = ZeroED(
        api_key='EMPTY',
        model_name="qwen2.5-7b",
        base_url="http://localhost:8000/v1",
        local_model_use=True,
        n_method='dbscan',
        result_dir='./temp',
        verbose=True
    )

    # detector = MLNCleanDetector(
    #     rules_path='datasets/flights/rules.txt',     # MLN 规则文件路径
    #     evidence_path=None,                           # 不传则用 dirty_df 自身作证据
    #     partition_number=1,                           # 一般保持为 1
    #     agp_threshold=2,                              # AGP 异常组阈值
    #     mcmc_samples=20,                              # Pyro MCMC 采样数
    #     mcmc_warmup=20,
    #     verbose=True,
    # )
    # # MLNClean 期望输入 DataFrame 含 'ID' 列
    # # 训练 + 预测
    # error_mask = detector.train_and_predict(dirty_df)   # 返回 bool DataFrame


    # 3. 训练
    print("\n>>> 正在运行")
    detector.train(data)
    res_mask = detector.predict(data)

    # 5. 打印结果
    print_errors_nicely(data, res_mask)
