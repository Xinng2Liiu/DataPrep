import numpy as np
import torch

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataprep.tabular.imputation.GAIN import GAIN
from dataprep.tabular.imputation.SCIS import SCIS
from dataprep.tabular.imputation.VAEGAIN import VAEGAIN
from dataprep.tabular.imputation.EDIT import EDIT

def generate_fake_data(N=1000, D=10, missing_rate=0.2):
    """
    生成模拟数据
    Returns:
        data_true: 完整真实数据 (Ground Truth)
        data_missing: 带 NaN 的缺失数据
        mask: 掩码矩阵 (1=观测, 0=缺失)
    """
    # 1. 生成真实数据 (正态分布)
    data_true = np.random.randn(N, D)

    # 2. 生成掩码 (随机缺失 MCAR)
    # np.random.rand 返回 [0,1) 均匀分布，> 0.2 意味着保留 80% 的数据
    mask = (np.random.rand(N, D) > missing_rate).astype(float)

    # 3. 制造缺失
    data_missing = data_true.copy()
    data_missing[mask == 0] = np.nan

    return data_true, data_missing, mask


def imputation():
    print("========================================")
    print("      Imputation            ")
    print("========================================")

    # 1. 准备数据
    N, D = 1000, 10
    data_true, data_missing, mask = generate_fake_data(N, D, missing_rate=0.2)
    print(f"Data Shape: {data_true.shape}, Missing Rate: 0.2")

    # 2. 初始化模型
    #这里以GAIN为例
    imputer = GAIN(
        batch_size=64,
        hint_rate=0.9,
        alpha=100,
        epoch=100,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 以下为SCIS的使用示例
    # imputer = SCIS(
    #     batch_size=64,
    #     epoch=100,
    #     initial_value=300,  # 初始训练集大小
    #     thre_value=0.2,
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )

    # 以下为VAEGAIN的使用示例
    # imputer = VAEGAIN(
    #     batch_size=32,
    #     epoch=100,
    #     latent_size=5,  # 隐变量维度
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )

    # # 以下为 EDIT 的使用示例 (基于影响函数选样本)
    # imputer = EDIT(
    #     batch_size=8,
    #     hint_rate=0.9,
    #     alpha=10,
    #     epoch=10,
    #     initial_size=300,        # 初始训练集大小, 论文用 6000, 小数据集应适配
    #     validation_size=200,     # 用于影响函数打分的验证集大小
    #     damping=1e-2,            # generator L2 正则系数
    #     device='cuda' if torch.cuda.is_available() else 'cpu',
    # )

    # 3. 训练并预测
    print("\n[Step 1] Training & Predicting...")
    imputed_data = imputer.train_and_predict(data_missing, mask)
    metrics = imputer.estimate(data_true, imputed_data, mask)

if __name__ == "__main__":
    imputation()
