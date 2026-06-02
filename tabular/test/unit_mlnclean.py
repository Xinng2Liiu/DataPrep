"""
MLNClean unit tests (放到 tabular/test/unit_mlnclean.py)

重点测:
    1. MLNClean_modules 中的纯函数 (规则解析、diff mask、apply mask)
    2. detection / correction 主类的接口契约 (规则缺失报错、未训练 predict 报错)
    3. 共享 mocked pipeline 的端到端流程

由于 pyro MCMC 较慢, 真正的训练流程在 smoke test 里跑;
这里用 mock 跳过 MCMC, 只测我们写的胶水代码.
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import dataprep.tabular.detection.MLNClean_modules as mo
    from dataprep.tabular.detection.MLNClean import MLNClean as MLNCleanDetector
    from dataprep.tabular.correction.MLNClean import MLNClean as MLNCleanCorrector
except ImportError as e:
    raise ImportError(f"导入失败, 请检查文件位置.\n详细错误: {e}")


# =============================================================================
# 1. 测试 MLNClean_modules 中的纯函数
# =============================================================================

class TestMLNCleanModules(unittest.TestCase):

    def test_parse_rules_basic(self):
        """规则解析: !A(v) v B(v) -> ([A], [B])"""
        rules = ["!flight(x) v act_arr_time(y)\n",
                 "!flight(x) v sched_dep_time(y)\n"]
        parsed = mo.parse_rules(rules)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0], [['flight'], ['act_arr_time']])
        self.assertEqual(parsed[1], [['flight'], ['sched_dep_time']])

    def test_parse_rules_skips_empty_lines(self):
        rules = ["!a(x) v b(y)", "", "  ", "!c(x) v d(y)"]
        parsed = mo.parse_rules(rules)
        self.assertEqual(len(parsed), 2)

    def test_compute_diff_mask(self):
        """diff mask: 值变了的格子标 True"""
        dirty = pd.DataFrame({
            'ID': [1, 2, 3],
            'name': ['Alice', 'Bob', 'XXX'],
            'age': [25, 99, 35],
        })
        cleaned = pd.DataFrame({
            'ID': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
        })
        mask = mo.compute_diff_mask(dirty, cleaned)
        self.assertEqual(mask.shape, dirty.shape)
        self.assertTrue(mask.values.dtype == bool)
        # ID 列在 exclude_columns 中, 即使有改也是 False
        self.assertFalse(mask['ID'].any())
        # 改动位置
        self.assertTrue(mask.at[2, 'name'])
        self.assertTrue(mask.at[1, 'age'])
        # 未改动位置
        self.assertFalse(mask.at[0, 'name'])

    def test_apply_corrections_with_mask(self):
        """只在 mask=True 的位置应用 cleaned 的值"""
        dirty = pd.DataFrame({
            'ID': [1, 2, 3],
            'val': ['a', 'BAD', 'BAD'],
        })
        cleaned = pd.DataFrame({
            'ID': [1, 2, 3],
            'val': ['x', 'b', 'c'],   # 算法把 'a' 也想改成 'x', 但 mask 不允许
        })
        mask = pd.DataFrame({
            'ID': [False, False, False],
            'val': [False, True, True],
        })
        fixed = mo.apply_corrections_with_mask(dirty, cleaned, mask)
        # mask=False 的格子保留 dirty
        self.assertEqual(fixed.at[0, 'val'], 'a')
        # mask=True 的格子用 cleaned
        self.assertEqual(fixed.at[1, 'val'], 'b')
        self.assertEqual(fixed.at[2, 'val'], 'c')

    def test_data_partition_single(self):
        """partition_number=1 时应原样返回"""
        df = pd.DataFrame({'ID': [1, 2, 3], 'x': ['a', 'b', 'c']})
        parts = mo.data_partition(df, partition_num=1)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].shape, df.shape)

    def test_binary_heap_basic(self):
        """二叉堆基本操作"""
        h = mo.BinaryHeap()
        h.insert([[3, 'c'], [1, 'a'], [2, 'b']])
        self.assertEqual(h.get_min(), 1)
        self.assertEqual(h.delete_min()[1], 'a')
        self.assertEqual(h.get_min(), 2)


# =============================================================================
# 2. 测试 MLNClean Detector 主类
# =============================================================================

class TestMLNCleanDetector(unittest.TestCase):

    def test_rules_required(self):
        """rules 和 rules_path 必须二选一"""
        with self.assertRaises(ValueError):
            MLNCleanDetector()

    def test_rules_from_list(self):
        """从字符串列表直接初始化"""
        det = MLNCleanDetector(rules=["!a(x) v b(y)"], verbose=False)
        self.assertEqual(len(det.rules), 1)

    def test_predict_without_train(self):
        """未训练 predict 必须报错"""
        det = MLNCleanDetector(rules=["!a(x) v b(y)"], verbose=False)
        with self.assertRaises(RuntimeError):
            det.predict(pd.DataFrame({'ID': [1], 'x': ['a']}))

    @patch('dataprep.tabular.detection.MLNClean.mo.run_mln_clean_pipeline')
    def test_train_then_predict(self, mock_pipeline):
        """主流程: mock 掉底层 pipeline, 看胶水代码"""
        dirty = pd.DataFrame({
            'ID': [1, 2, 3],
            'name': ['Alice', 'Bob', 'XXX'],
        })
        cleaned = pd.DataFrame({
            'ID': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
        })
        mock_pipeline.return_value = cleaned

        det = MLNCleanDetector(rules=["!a(x) v b(y)"], verbose=False)
        det._create_temp_dir = MagicMock()
        det._save_checkpoint = MagicMock()

        mask = det.train_and_predict(dirty)

        mock_pipeline.assert_called_once()
        self.assertEqual(mask.shape, dirty.shape)
        self.assertTrue(mask.at[2, 'name'])
        self.assertFalse(mask.at[0, 'name'])


# =============================================================================
# 3. 测试 MLNClean Corrector 主类
# =============================================================================

class TestMLNCleanCorrector(unittest.TestCase):

    def test_rules_required(self):
        with self.assertRaises(ValueError):
            MLNCleanCorrector()

    def test_predict_without_train(self):
        cor = MLNCleanCorrector(rules=["!a(x) v b(y)"], verbose=False)
        with self.assertRaises(RuntimeError):
            cor.predict(pd.DataFrame({'ID': [1], 'x': ['a']}))

    @patch('dataprep.tabular.correction.MLNClean.mo.run_mln_clean_pipeline')
    def test_correction_with_mask(self, mock_pipeline):
        """correction with mask: 只在 mask=True 处覆盖"""
        dirty = pd.DataFrame({
            'ID': [1, 2, 3],
            'val': ['a', 'BAD', 'c'],
        })
        cleaned = pd.DataFrame({
            'ID': [1, 2, 3],
            'val': ['z', 'b', 'z'],   # 算法想改 0 和 2, 但 mask 只允许改 1
        })
        mask = pd.DataFrame({
            'ID': [False, False, False],
            'val': [False, True, False],
        })
        mock_pipeline.return_value = cleaned

        cor = MLNCleanCorrector(rules=["!a(x) v b(y)"], verbose=False)
        cor._create_temp_dir = MagicMock()
        cor._save_checkpoint = MagicMock()

        fixed = cor.train_and_predict(dirty, detection_mask=mask)

        # 没在 mask 中的位置应保留 dirty
        self.assertEqual(fixed.at[0, 'val'], 'a')
        self.assertEqual(fixed.at[2, 'val'], 'c')
        # mask 中的位置应被改成 cleaned 的值
        self.assertEqual(fixed.at[1, 'val'], 'b')

    @patch('dataprep.tabular.correction.MLNClean.mo.run_mln_clean_pipeline')
    def test_correction_no_mask_full_replacement(self, mock_pipeline):
        """correction without mask: 直接返回 cleaned (全量替换)"""
        dirty = pd.DataFrame({'ID': [1, 2], 'val': ['a', 'BAD']})
        cleaned = pd.DataFrame({'ID': [1, 2], 'val': ['x', 'b']})
        mock_pipeline.return_value = cleaned

        cor = MLNCleanCorrector(rules=["!a(x) v b(y)"], verbose=False)
        cor._create_temp_dir = MagicMock()
        cor._save_checkpoint = MagicMock()

        fixed = cor.train_and_predict(dirty)
        self.assertEqual(fixed.at[0, 'val'], 'x')
        self.assertEqual(fixed.at[1, 'val'], 'b')


if __name__ == '__main__':
    unittest.main()
