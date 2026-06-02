# DataPrep EDIT / MLNClean 评估补丁说明

本文档说明如何在 **尽量保留 DataPrep 原项目结构** 的前提下，接入并评估 `EDIT` 与 `MLNClean`。当前评估先只覆盖：

- `EDIT`：`imputation` 任务；
- `MLNClean`：`flights` 数据集上的 `detection` 与 `correction` 任务；
- `ZeroED / ZeroEC`：作为原项目已有方法，单独运行并与 MLNClean 结果做表格对比。

> 当前暂不加入 `rayyan` 和 `tax-1k` 的 MLNClean 评估。Rayyan 的规则字段高基数、长文本多，当前 MLNClean 的 Pyro-NUTS 权重学习在该数据集上不稳定；tax-1k 的规则属于从 `tax` 迁移/裁剪，后续再单独补充。

---

## 1. 文件说明

### 1.1 后端评估文件

```text
main_patch.py
```

用途：

- 作为评估用后端，不覆盖原始 `main.py`；
- 继续使用原项目的 FastAPI + WebSocket 形式；
- 新增 `EDIT`、`MLNClean_Det`、`MLNClean_Cor` 任务分支；
- 保留原项目的 `GAIN`、`ZeroED`、`ZeroEC` 等方法；
- `ZeroED` 和 `ZeroEC` 不再嵌入 MLNClean 分支中，而是作为独立任务运行。

启动方式：

```bash
python main_patch.py
```

注意：不要再把 `main_patch.py` 覆盖到 `main.py`，也不要用：

```python
uvicorn.run("dataprep.main:app", ...)
```

否则会绕回原项目 `main.py`，导致 patch 分支不生效。`main_patch.py` 末尾应保持：

```python
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8088, reload=False)
```

---

### 1.2 命令行评估客户端

```text
evaluate_backend.py
```

用途：

- 不打开前端页面，直接通过 WebSocket 调用后端；
- 默认连接：

```text
ws://127.0.0.1:8088/api/ws/run_task
```

当前建议支持任务：

```text
edit
gain
zeroed
zeroec
mln-det
mln-cor
```

其中：

| task | 后端 method | 任务 |
|---|---|---|
| `edit` | `EDIT` | 缺失值填补 |
| `gain` | `GAIN` | 原项目 GAIN 对比 |
| `zeroed` | `ZeroED` | 原项目错误检测方法 |
| `zeroec` | `ZeroEC` | 原项目错误修复方法 |
| `mln-det` | `MLNClean_Det` | MLNClean 拆分后的错误检测 |
| `mln-cor` | `MLNClean_Cor` | MLNClean 拆分后的错误修复 |

---

## 2. 算法文件放置位置

按照 DataPrep 原项目结构放置，不另建孤立目录。

### 2.1 EDIT

```text
tabular/imputation/EDIT.py
tabular/imputation/EDIT_modules.py
```

设计要求：

- `EDIT.py` 只负责 DataPrep 风格接口封装；
- `EDIT_modules.py` 放原 EDIT/GAIN 核心训练、影响函数、样本选择与重训练逻辑；
- 不在类内部固定读取某个 CSV；
- 输入由后端传入：

```python
data_missing
missing_mask
```

其中：

```text
missing_mask = 1 表示观测值
missing_mask = 0 表示缺失值
```

---

### 2.2 MLNClean Detection

```text
tabular/detection/MLNClean.py
tabular/detection/MLNClean_modules.py
```

设计逻辑：

```text
dirty_df + rules.txt + rules_data.csv
        ↓
run_mln_clean_pipeline()
        ↓
cleaned_df
        ↓
pred_mask = dirty_df != cleaned_df
```

输出：

```text
True  = 检测为错误
False = 检测为正常
```

---

### 2.3 MLNClean Correction

```text
tabular/correction/MLNClean.py
```

设计逻辑：

```text
dirty_df + rules.txt + rules_data.csv
        ↓
run_mln_clean_pipeline()
        ↓
cleaned_df
        ↓
如果有 detection_mask：
    只在 mask=True 的位置用 cleaned_df 覆盖 dirty_df
否则：
    返回完整 cleaned_df
```

---

## 3. 当前使用的数据文件

当前只使用 DataPrep 项目自带 `flights` 数据集。

```text
datasets/flights/
├── flights_dirty.csv
├── flights_clean.csv
├── flights_dirty_error_detection.csv
├── rules.txt
└── rules_data.csv
```

各文件作用如下：

| 文件 | 用途 |
|---|---|
| `flights_dirty.csv` | 待检测/待修复数据 |
| `flights_clean.csv` | correction 评估真值 |
| `flights_dirty_error_detection.csv` | detection/correction 的真实错误位置 mask |
| `rules.txt` | MLNClean 规则结构 |
| `rules_data.csv` | MLNClean evidence，用于规则候选与权重学习 |

---

## 4. flights 的规则文件

`datasets/flights/rules.txt` 内容建议为：

```text
!flight(x) v act_dep_time(y)
!flight(x) v act_arr_time(y)
!flight(x) v sched_dep_time(y)
!flight(x) v sched_arr_time(y)
```

解释：

| rule | reason | result | 含义 |
|---|---|---|---|
| `!flight(x) v act_dep_time(y)` | `flight` | `act_dep_time` | 同一航班的实际出发时间应一致 |
| `!flight(x) v act_arr_time(y)` | `flight` | `act_arr_time` | 同一航班的实际到达时间应一致 |
| `!flight(x) v sched_dep_time(y)` | `flight` | `sched_dep_time` | 同一航班的计划出发时间应一致 |
| `!flight(x) v sched_arr_time(y)` | `flight` | `sched_arr_time` | 同一航班的计划到达时间应一致 |

MLNClean 中：

```text
带 ! 的字段 = reason 字段
不带 ! 的字段 = result 字段
```

因此：

```text
!flight(x) v act_dep_time(y)
```

会被解释为：

```text
reason = ["flight"]
result = ["act_dep_time"]
```

---

## 5. rules_data.csv 应该如何生成

`rules_data.csv` 主实验应由 dirty 数据生成，但不能直接完整复制 dirty。

原因：

- `flights_dirty.csv` 中规则相关字段存在缺失值；
- 如果 `rules_data.csv` 中保留 NaN，MLNClean 的 MCMC 权重学习阶段可能出现 `_find_location` 找不到候选值的问题；
- 因此应只保留规则涉及字段非空的记录。

### 5.1 flights 的生成脚本

在项目根目录运行：

```python
import pandas as pd

dirty = pd.read_csv("datasets/flights/flights_dirty.csv")

rule_cols = [
    "flight",
    "act_dep_time",
    "act_arr_time",
    "sched_dep_time",
    "sched_arr_time",
]

rules_data = dirty.dropna(subset=rule_cols).copy()
rules_data.to_csv("datasets/flights/rules_data.csv", index=False)

print("dirty rows:", len(dirty))
print("rules_data rows:", len(rules_data))
print(rules_data[rule_cols].isna().sum())
```

要求最后输出中规则字段缺失数全为 0：

```text
flight            0
act_dep_time      0
act_arr_time      0
sched_dep_time    0
sched_arr_time    0
```

### 5.2 dirty-derived evidence 与 clean-derived evidence 的区别

主实验使用：

```text
rules_data.csv = flights_dirty.csv 去掉规则列缺失行后生成
```

这属于：

```text
dirty-derived evidence
```

不使用 clean 真值，比较公平。

如果使用：

```text
rules_data.csv = flights_clean.csv
```

则属于：

```text
oracle evidence
```

会使用真值信息，只能作为参考上限，不能作为主实验。

---

## 6. evaluate_backend.py 中的 flights 路径映射

当前只保留 `flights`，不加入 `rayyan` 和 `tax-1k`：

```python
DATASETS = {
    "flights": {
        "dirty": "datasets/flights/flights_dirty.csv",
        "clean": "datasets/flights/flights_clean.csv",
        "mask": "datasets/flights/flights_dirty_error_detection.csv",
        "rules": "datasets/flights/rules.txt",
        "evidence": "datasets/flights/rules_data.csv",
    },
}
```


---

## 7. 运行方式

### 7.1 启动后端

在项目根目录运行：

```bash
python main_patch.py
```

如果端口正确，会看到后端监听：

```text
http://127.0.0.1:8088
```

---

### 7.2 运行 EDIT

```bash
python evaluate_backend.py --task edit
```

可调整参数：

```bash
python evaluate_backend.py --task edit --batch-size 8 --epoch 100 --initial-size 500 --validation-size 500
```

---

### 7.3 运行 GAIN

```bash
python evaluate_backend.py --task gain --gain-epoch 100
```

正式结果可使用：

```bash
python evaluate_backend.py --task gain --gain-epoch 1000
```

---

### 7.4 运行 ZeroED

```bash
python evaluate_backend.py --task zeroed --dataset flights
```

ZeroED 通过 API 调用大模型时，需要配置：

```bash
python evaluate_backend.py --task zeroed --dataset flights 
```


---

### 7.5 运行 ZeroEC

```bash
python evaluate_backend.py --task zeroec --dataset flights
```

ZeroEC 通过 API 调用大模型时，需要配置：

```bash
python evaluate_backend.py --task zeroec --dataset flights 
```

需要确保以下目录存在：

```text
tabular/correction/all-MiniLM-L6-v2
prompt_templates
```

---

### 7.6 运行 MLNClean Detection

调试阶段先用：

```bash
python evaluate_backend.py --task mln-det --dataset flights --mcmc-samples 5 --mcmc-warmup 5
```

正式实验可用：

```bash
python evaluate_backend.py --task mln-det --dataset flights --mcmc-samples 20 --mcmc-warmup 20
```

当前已跑通的 5/5 结果：

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| Isolation Forest | 0.7534 | 0.4701 | 0.5789 |
| LOF | 0.8241 | 0.5142 | 0.6332 |
| MLNClean | 0.8104 | 0.9811 | 0.8876 |

说明：

- 这里的 detection 指标是后端当前实现中的 row-level 指标；
- 即每一行只要存在一个错误单元格，就认为该行为 dirty；
- 如果需要 cell-level 指标，应额外保存预测 mask 并逐单元格计算。

---

### 7.7 运行 MLNClean Correction

调试阶段先用：

```bash
python evaluate_backend.py --task mln-cor --dataset flights --mcmc-samples 5 --mcmc-warmup 5
```

正式实验可用：

```bash
python evaluate_backend.py --task mln-cor --dataset flights --mcmc-samples 20 --mcmc-warmup 20
```

当前已跑通的 5/5 结果：

| Method | Precision | Recall | F1 | EDR |
|---|---:|---:|---:|---:|
| Mode | 1.52% | 1.52% | 1.52% | 1.52% |
| KNN | 5.44% | 5.43% | 5.43% | 5.43% |
| Iterative | 4.69% | 4.67% | 4.68% | 4.67% |
| MLNClean | 49.99% | 43.68% | 46.62% | 43.68% |

说明：

- 该 correction 评估使用 `flights_dirty_error_detection.csv` 作为 oracle mask；
- 即已知哪些位置存在错误，只评估修复器是否能修对；
- 这不是完整的 detection → correction pipeline。

---

## 9. 已知问题与处理方式

### 9.1 Rayyan 暂不纳入 MLNClean 实验

当前不建议跑：

```bash
python evaluate_backend.py --task mln-det --dataset rayyan
python evaluate_backend.py --task mln-cor --dataset rayyan
```

原因：

- Rayyan 的 `journal_title` 是长文本；
- `jounral_abbreviation`、`journal_issn`、`journal_title` 缺失较多；
- 有效 evidence 中大量组合唯一，重复支持不足；
- 当前 MLNClean 的 Pyro-NUTS 权重学习在该数据上可能在 MCMC 初始化阶段导致进程直接退出。

因此本文档当前仅保留 flights 的 MLNClean 评估。

---

### 9.2 rules_data.csv 不能含规则字段 NaN

如果出现：

```text
IndexError: list index out of range
```

并且报错位置在：

```text
_find_location(...)
```

通常说明 `rules_data.csv` 中含有规则字段缺失值。应重新按第 5 节生成 `rules_data.csv`。

---

### 9.3 Pyro MCMC 异常后需要重启后端

如果 MCMC 中途异常，可能会污染 Pyro 状态。建议：

```bash
Ctrl+C 关闭 main_patch.py
python main_patch.py
```

然后再重新运行任务。

---

### 9.4 API Key 不要写死

`evaluate_backend.py` 中不要把真实 API key 写成默认值。建议：

```python
import os

parser.add_argument("--zeroed-api-key", default=os.getenv("SILICONFLOW_API_KEY", ""))
parser.add_argument("--zeroec-api-key", default=os.getenv("SILICONFLOW_API_KEY", ""))
```

运行前：

```bash
set SILICONFLOW_API_KEY=你的key
```

---

## 10. 依赖环境

建议使用独立环境：

```bash
conda create -n dataprep python=3.10 -y
conda activate dataprep
```

安装依赖：

```bash
pip install fastapi uvicorn websockets
pip install numpy pandas scipy scikit-learn openpyxl tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pyro-ppl python-Levenshtein
pip install faiss-cpu==1.7.4
pip install sentence-transformers==2.7.0
pip install fast_sentence_transformers==0.4.1
pip install onnxruntime==1.17.1
pip install langchain==0.1.12 langchain-community==0.0.28 langchain-core==0.1.32 langchain-openai==0.0.5
pip install openai requests httpx transformers==4.40.2 accelerate
pip install tensorflow
conda install -c conda-forge h5py -y
```

安装后建议检查：

```bash
python -c "import fastapi, uvicorn, websockets; print('backend ok')"
python -c "import torch, pyro, Levenshtein; print('mlnclean ok')"
python -c "import sklearn, pandas, numpy, tqdm; print('basic ml ok')"
python -c "import openai, langchain, langchain_openai; print('llm api ok')"
python -c "import sentence_transformers, fast_sentence_transformers, onnxruntime; print('embedding ok')"
python -c "import tensorflow, h5py; print('tf/h5py ok')"
```

---

## 11. 最终说明

当前评估补丁的定位是：

```text
不覆盖原 main.py；
保留 DataPrep 原项目结构；
通过 main_patch.py 做后端评估；
通过 evaluate_backend.py 发送任务；
优先完成 EDIT 与 MLNClean 在 flights 数据集上的可运行评估。
```

当前不加入 `rayyan` 和 `tax-1k` 的 MLNClean 实验，避免由于数据特性和规则迁移问题引入额外不稳定因素。