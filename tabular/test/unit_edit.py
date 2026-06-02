"""
EDIT 单元测试 (放到 tabular/test/unit_edit.py)

仿照 unit_gain.py 写, 测试:
    1. EDIT_modules 中的网络结构和工具函数
    2. EDIT 主类的 train / predict 行为
"""
import sys
import os
import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# ==========================================
# 1. 导入路径设置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from dataprep.tabular.imputation.EDIT import EDIT
    import dataprep.tabular.imputation.EDIT_modules as em
except ImportError as e:
    raise ImportError(f"导入失败，请检查文件位置。\n详细错误: {e}")


# ==========================================
# 2. 测试 EDIT_modules
# ==========================================

class TestEDITModules(unittest.TestCase):
    """测试 EDIT_modules.py 中的底层函数和网络结构"""

    def setUp(self):
        self.data = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ])
        self.dim = 2
        self.h_dim = 4

    def test_normalization_renormalization(self):
        """归一化和反归一化应该可逆"""
        norm_data, params = em.normalization(self.data)
        self.assertTrue((norm_data >= 0).all() and (norm_data <= 1).all())
        renorm_data = em.renormalization(norm_data, params)
        np.testing.assert_array_almost_equal(self.data, renorm_data)

    def test_generator_shape(self):
        """生成器输入输出形状"""
        net = em.EditGenerator(self.dim, self.h_dim)
        bs = 5
        x = torch.randn(bs, self.dim)
        m = torch.randn(bs, self.dim)
        out = net(x, m)
        self.assertEqual(out.shape, (bs, self.dim))
        # Sigmoid 输出在 [0, 1]
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_discriminator_shape(self):
        """判别器输入输出形状"""
        net = em.EditDiscriminator(self.dim, self.h_dim)
        bs = 5
        x = torch.randn(bs, self.dim)
        h = torch.randn(bs, self.dim)
        out = net(x, h)
        self.assertEqual(out.shape, (bs, self.dim))

    def test_select_top_k_by_influence(self):
        """累积选 Top-k 的工具函数"""
        scores = np.array([5.0, 1.0, 4.0, -2.0, 3.0], dtype=np.float32)
        # sum = 11.0, 降序: idx [0, 2, 4, 1, 3] -> 5, 4, 3, 1, -2
        # 累加: 5(>11? no), 9(>11? no), 12(>11? yes, break)
        # 至少应该选了 [0, 2, 4]
        top_k = em.select_top_k_by_influence(scores)
        self.assertIn(0, top_k)
        self.assertIn(2, top_k)
        # 兜底逻辑保证至少 10%
        self.assertGreaterEqual(len(top_k), 1)


# ==========================================
# 3. 测试 EDIT 主类
# ==========================================

class TestEDITMain(unittest.TestCase):
    """测试 EDIT.py 中的主类行为"""

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)

        # 构造一份带缺失的小数据
        self.raw_data = np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
            [np.nan, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
        ])
        self.mask = 1 - np.isnan(self.raw_data).astype(float)

        self.imputer = EDIT(
            batch_size=2, epoch=1,
            initial_size=3, validation_size=2,
            device='cpu',
        )

        # Mock 掉文件系统相关方法
        self.imputer._create_temp_dir = MagicMock()
        self.imputer._save_checkpoint = MagicMock()

    @patch('dataprep.tabular.imputation.EDIT.em.train_edit_algorithm')
    def test_train_pipeline(self, mock_train_algo):
        """训练流程: mock 掉底层训练循环, 只验证主类逻辑"""
        self.imputer.train(self.raw_data, self.mask)

        # 1. 模型和归一化参数应被初始化
        self.assertIsNotNone(self.imputer.generator)
        self.assertIsNotNone(self.imputer.discriminator)
        self.assertIsNotNone(self.imputer.norm_parameters)

        # 2. 底层算法应被调用一次
        mock_train_algo.assert_called_once()

        # 3. checkpoint 应被保存
        self.imputer._save_checkpoint.assert_called_once()

    def test_predict_without_train(self):
        """未训练直接 predict 必须报错"""
        with self.assertRaises(RuntimeError):
            self.imputer.predict(self.raw_data)

    def test_predict_pipeline(self):
        """predict 形状/取值检查"""
        dim = 2
        h_dim = 2
        self.imputer.generator = em.EditGenerator(dim, h_dim)
        self.imputer.norm_parameters = {
            'min': np.array([1.0, 10.0]),
            'max': np.array([6.0, 60.0]),
            'den': np.array([5.0, 50.0]),
        }

        imputed = self.imputer.predict(self.raw_data)

        # 形状一致
        self.assertEqual(imputed.shape, self.raw_data.shape)
        # 输出无 NaN
        self.assertFalse(np.isnan(imputed).any())
        # 观测值应保持不变 (第 0 行全是观测)
        np.testing.assert_array_almost_equal(imputed[0], self.raw_data[0], decimal=4)

    def test_train_and_predict_end_to_end(self):
        """端到端 (epoch=1, 小数据): 不抛错且输出无 NaN"""
        # 用稍大数据避免太小不稳定
        np.random.seed(0)
        N, D = 60, 4
        data = np.random.randn(N, D).astype(np.float32)
        mask = (np.random.rand(N, D) > 0.2).astype(np.float32)
        data[mask == 0] = np.nan

        imp = EDIT(
            batch_size=4, epoch=1,
            initial_size=20, validation_size=10,
            device='cpu',
        )
        imp._create_temp_dir = MagicMock()
        imp._save_checkpoint = MagicMock()

        result = imp.train_and_predict(data, mask)
        self.assertEqual(result.shape, data.shape)
        self.assertFalse(np.isnan(result).any())


if __name__ == '__main__':
    unittest.main()
