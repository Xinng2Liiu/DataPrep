import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
import random
import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ==================== 算法与评估库引入 ====================
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataprep.tabular.imputation.GAIN import GAIN
from dataprep.tabular.imputation.VAEGAIN import VAEGAIN
from dataprep.tabular.imputation.SCIS import SCIS
from dataprep.tabular.imputation.EDIT import EDIT
from dataprep.tabular.detection.ZeroED import ZeroED
from dataprep.tabular.detection.MLNClean import MLNClean as MLNCleanDet
from dataprep.tabular.correction.ZeroEC import ZeroEC
from dataprep.tabular.correction.MLNClean import MLNClean as MLNCleanCor

app = FastAPI(title="DataPrep Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def setup_seed(seed=49):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_rmse(data_true, data_imputed, mask):
    if isinstance(data_imputed, torch.Tensor):
        data_imputed = data_imputed.cpu().detach().numpy()
    elif isinstance(data_imputed, pd.DataFrame):
        data_imputed = data_imputed.values
    if isinstance(mask, pd.DataFrame): mask = mask.values
    if isinstance(data_true, pd.DataFrame): data_true = data_true.values

    missing_mask = (mask == 0)
    if np.sum(missing_mask) == 0: return 0.0
    return float(np.sqrt(np.mean((data_true[missing_mask] - data_imputed[missing_mask]) ** 2)))


def calc_mae(data_true, data_imputed, mask):
    if isinstance(data_imputed, torch.Tensor):
        data_imputed = data_imputed.cpu().detach().numpy()
    elif isinstance(data_imputed, pd.DataFrame):
        data_imputed = data_imputed.values
    if isinstance(mask, pd.DataFrame): mask = mask.values
    if isinstance(data_true, pd.DataFrame): data_true = data_true.values

    missing_mask = (mask == 0)
    if np.sum(missing_mask) == 0: return 0.0
    return float(np.mean(np.abs(data_true[missing_mask] - data_imputed[missing_mask])))


def calc_nrmse(data_true, data_imputed, mask):
    """
    Normalized RMSE:
    对每一列单独计算 RMSE / std(column)，再对所有列取平均。
    这样可以减弱 snowfall_mm 这种大尺度列对总 RMSE 的主导影响。
    """
    if isinstance(data_imputed, torch.Tensor):
        data_imputed = data_imputed.cpu().detach().numpy()
    elif isinstance(data_imputed, pd.DataFrame):
        data_imputed = data_imputed.values

    if isinstance(data_true, pd.DataFrame):
        data_true = data_true.values
    if isinstance(mask, pd.DataFrame):
        mask = mask.values

    data_true = np.asarray(data_true, dtype=float)
    data_imputed = np.asarray(data_imputed, dtype=float)
    mask = np.asarray(mask)

    nrmse_list = []

    for j in range(data_true.shape[1]):
        missing_pos = (mask[:, j] == 0)

        if missing_pos.sum() == 0:
            continue

        col_std = np.nanstd(data_true[:, j])
        if col_std < 1e-8:
            continue

        rmse_j = np.sqrt(
            np.mean(
                (data_true[missing_pos, j] - data_imputed[missing_pos, j]) ** 2
            )
        )

        nrmse_list.append(rmse_j / col_std)

    return float(np.mean(nrmse_list)) if len(nrmse_list) > 0 else 0.0


def calc_column_metrics(data_true, data_imputed, mask, columns):
    """
    逐列计算 RMSE / MAE。
    用于检查总 RMSE 是不是被某一列，例如 snowfall_mm 拉高。
    """
    if isinstance(data_imputed, torch.Tensor):
        data_imputed = data_imputed.cpu().detach().numpy()
    elif isinstance(data_imputed, pd.DataFrame):
        data_imputed = data_imputed.values

    if isinstance(data_true, pd.DataFrame):
        data_true = data_true.values
    if isinstance(mask, pd.DataFrame):
        mask = mask.values

    data_true = np.asarray(data_true, dtype=float)
    data_imputed = np.asarray(data_imputed, dtype=float)
    mask = np.asarray(mask)

    rows = []

    for j, col in enumerate(columns):
        missing_pos = (mask[:, j] == 0)

        if missing_pos.sum() == 0:
            continue

        diff = data_true[missing_pos, j] - data_imputed[missing_pos, j]

        rmse_j = float(np.sqrt(np.mean(diff ** 2)))
        mae_j = float(np.mean(np.abs(diff)))

        rows.append({
            "column": col,
            "rmse": rmse_j,
            "mae": mae_j,
            "missing_count": int(missing_pos.sum()),
            "true_min": float(np.nanmin(data_true[:, j])),
            "true_max": float(np.nanmax(data_true[:, j])),
            "true_std": float(np.nanstd(data_true[:, j])),
        })

    return rows

def load_and_prep_detection_data(path_dirty, path_gt):
    df_dirty = pd.read_csv(path_dirty, index_col=0) if 'index_col' in str(
        pd.read_csv(path_dirty, nrows=1).columns) else pd.read_csv(path_dirty)
    df_gt = pd.read_csv(path_gt)

    if 'index' in df_gt.columns: df_gt.drop(columns=['index'], inplace=True)
    if 'index' in df_dirty.columns: df_dirty.drop(columns=['index'], inplace=True)

    if df_gt.dtypes.iloc[0] == object:
        df_gt = df_gt.replace({'True': True, 'False': False, '1': True, '0': False})

    y_true = df_gt.values.any(axis=1).astype(int)

    df_sklearn = df_dirty.copy()
    imputer = SimpleImputer(strategy='most_frequent')
    df_sklearn = pd.DataFrame(imputer.fit_transform(df_sklearn), columns=df_sklearn.columns)

    for col in df_sklearn.columns:
        if df_sklearn[col].dtype == 'object':
            df_sklearn[col] = LabelEncoder().fit_transform(df_sklearn[col].astype(str))

    return df_dirty, df_sklearn, y_true



def _drop_artifact_index_cols(df: pd.DataFrame) -> pd.DataFrame:
    """清理 CSV 保存时产生的伪索引列，但保留真正的 ID 列。"""
    df = df.copy()
    for col in ["index", "Unnamed: 0"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


def _read_table(path: str) -> pd.DataFrame:
    if not path:
        raise ValueError("CSV path is empty or None.")
    return _drop_artifact_index_cols(pd.read_csv(path))


def _to_bool_mask(df: pd.DataFrame) -> pd.DataFrame:
    df = _drop_artifact_index_cols(df)
    return df.replace({
        'True': True, 'False': False,
        'true': True, 'false': False,
        1: True, 0: False,
        '1': True, '0': False,
    }).astype(bool)


def _align_mask_to_df(mask: pd.DataFrame, df: pd.DataFrame, fill_value=False) -> pd.DataFrame:
    """让 mask 的列顺序与 df 一致；缺失列补 fill_value，多余列丢弃。"""
    mask = mask.copy()
    for col in df.columns:
        if col not in mask.columns:
            mask[col] = fill_value
    return mask[df.columns].astype(bool)


class SklearnCorrector:
    def __init__(self, strategy='most_frequent'):
        self.strategy = strategy
        self.encoders = {}

    def correct(self, df_dirty, df_mask):
        X = df_dirty.copy()
        df_mask = _align_mask_to_df(_to_bool_mask(df_mask), X, fill_value=False)
        X = X.mask(df_mask.astype(bool), np.nan)
        X_encoded = X.copy()
        for col in X.columns:
            if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
                le = LabelEncoder()
                valid_vals = X[col].dropna().unique()
                le.fit(valid_vals.astype(str))
                self.encoders[col] = le
                non_null_mask = X[col].notnull()
                X_encoded.loc[non_null_mask, col] = le.transform(X.loc[non_null_mask, col].astype(str))
                X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')

        if self.strategy == 'most_frequent':
            imputer = SimpleImputer(strategy='most_frequent')
        elif self.strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        elif self.strategy == 'iterative':
            imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
        else:
            raise ValueError(f"Unsupported SklearnCorrector strategy: {self.strategy}")

        X_imputed = imputer.fit_transform(X_encoded)
        df_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

        df_final = df_imputed.copy()
        for col, le in self.encoders.items():
            if col in df_imputed.columns:
                series_col = pd.to_numeric(df_imputed[col], errors='coerce')
                vals = series_col.round().astype(int).clip(0, len(le.classes_) - 1)
                df_final[col] = le.inverse_transform(vals)
        return df_final


def evaluate_correction_metrics(clean_data, dirty_data, corrected_data, df_mask):
    common_idx = clean_data.index.intersection(corrected_data.index).intersection(dirty_data.index)
    common_col = clean_data.columns.intersection(corrected_data.columns).intersection(dirty_data.columns)

    gt = clean_data.loc[common_idx, common_col]
    dirty = dirty_data.loc[common_idx, common_col]
    pred = corrected_data.loc[common_idx, common_col]
    mask = df_mask.loc[common_idx, common_col].astype(bool)

    def safe_str_format(df):
        return df.fillna("").astype(str).map(lambda x: x.strip())

    gt_str = safe_str_format(gt)
    dirty_str = safe_str_format(dirty)
    pred_str = safe_str_format(pred)
    mask_bool = mask.values

    actual_errors_num = mask_bool.sum()

    if actual_errors_num == 0:
        return 0.0, 0.0, 0.0

    is_correct = (pred_str == gt_str).values
    correctly_fixed_num = (is_correct & mask_bool).sum()

    is_modified = (pred_str != dirty_str).values
    total_modified_num = is_modified.sum()

    Recall = correctly_fixed_num / actual_errors_num if actual_errors_num > 0 else 0.0
    Precision = correctly_fixed_num / total_modified_num if total_modified_num > 0 else 0.0
    F1 = (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0.0

    return Precision, Recall, F1

# ========== 新增：cell-level detection 指标 ==========
def evaluate_cell_detection_metrics(true_mask: pd.DataFrame, pred_mask: pd.DataFrame):
    """
    后端原有 detection 主要是 row-level：
        一行有任意错误即为 dirty row。

    这里补充 cell-level：
        逐单元格评估 MLNClean 是否定位到具体错误格子。
    """
    true_mask = true_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    common_idx = true_mask.index.intersection(pred_mask.index)
    common_col = true_mask.columns.intersection(pred_mask.columns)

    y_true = true_mask.loc[common_idx, common_col].values.reshape(-1)
    y_pred = pred_mask.loc[common_idx, common_col].values.reshape(-1)

    return (
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
        int(y_true.sum()),
        int(y_pred.sum()),
    )


# ========== 新增：Detection -> Correction 端到端评估 ==========
def evaluate_pipeline_repair_metrics(clean_data, dirty_data, cleaned_by_mln, pred_mask, true_mask):
    """
    端到端清洗评估：
        dirty
          -> MLNClean detection 得到 pred_mask
          -> 仅在 pred_mask=True 的位置用 cleaned_by_mln 覆盖
          -> 与 clean ground truth 比较

    与 MLNClean_Cor 的 oracle correction 不同：
        Oracle correction 使用真实 mask；
        Pipeline correction 使用 detection 自己预测出来的 mask。
    """
    common_idx = (
        clean_data.index
        .intersection(dirty_data.index)
        .intersection(cleaned_by_mln.index)
        .intersection(pred_mask.index)
        .intersection(true_mask.index)
    )

    common_col = (
        clean_data.columns
        .intersection(dirty_data.columns)
        .intersection(cleaned_by_mln.columns)
        .intersection(pred_mask.columns)
        .intersection(true_mask.columns)
    )

    clean_eval = clean_data.loc[common_idx, common_col]
    dirty_eval = dirty_data.loc[common_idx, common_col]
    cleaned_eval = cleaned_by_mln.loc[common_idx, common_col]
    pred_mask_eval = pred_mask.loc[common_idx, common_col].astype(bool)
    true_mask_eval = true_mask.loc[common_idx, common_col].astype(bool)

    repaired = dirty_eval.copy()
    repaired[pred_mask_eval] = cleaned_eval[pred_mask_eval]

    p, r, f1 = evaluate_correction_metrics(
        clean_eval,
        dirty_eval,
        repaired,
        true_mask_eval,
    )
    edr = calc_edr_df(dirty_eval, repaired, clean_eval)

    return p, r, f1, edr, int(pred_mask_eval.values.sum())


# ========== 新增：内存版 EDR 计算 ==========
def calc_edr_df(dirty_df, repaired_df, clean_df):
    """
    模拟从CSV读取全字符串比较的效果，直接在内存中比较三个 DataFrame 的 EDR。
    使用 Numpy 矩阵运算，防止 Pandas 索引不对齐产生的 NaN。
    """
    # 1. 强制对齐：取三个 DataFrame 的公共行和公共列
    common_idx = clean_df.index.intersection(repaired_df.index).intersection(dirty_df.index)
    common_col = clean_df.columns.intersection(repaired_df.columns).intersection(dirty_df.columns)

    # 2. 提取对齐后的数据，处理空值并转为字符串
    dirty_align = dirty_df.loc[common_idx, common_col].fillna("").astype(str)
    rep_align = repaired_df.loc[common_idx, common_col].fillna("").astype(str)
    clean_align = clean_df.loc[common_idx, common_col].fillna("").astype(str)

    # 3. 提取底层 numpy 矩阵进行逻辑比较，绝对安全
    dirty_arr = dirty_align.values
    rep_arr = rep_align.values
    clean_arr = clean_align.values

    # 找到不相等的位置 (布尔矩阵)
    dirty_wrong = (dirty_arr != clean_arr)
    repaired_wrong = (rep_arr != clean_arr)

    # 计算各项指标
    d_w = int(dirty_wrong.sum())
    dis_r2c = int(repaired_wrong.sum())
    d_w2r = int((dirty_wrong & ~repaired_wrong).sum())
    d_r2w = int((~dirty_wrong & repaired_wrong).sum())

    if d_w == 0:
        return 0.0
    else:
        edr = (d_w - dis_r2c) / d_w
        return edr


class WsLogCatcher:
    def __init__(self, loop, queue, original_stream):
        self.loop = loop
        self.queue = queue
        self.original_stream = original_stream

    def write(self, s):
        self.original_stream.write(s)
        self.original_stream.flush()
        if s:
            asyncio.run_coroutine_threadsafe(self.queue.put({"log": s}), self.loop)
        return len(s)

    def flush(self):
        self.original_stream.flush()


def generate_result_data(df: pd.DataFrame, df_mask: pd.DataFrame, max_rows: int = 1000) -> dict:
    df_head = df.head(max_rows).copy()
    mask_head = df_mask.head(max_rows).copy()
    df_head = df_head.astype(object).where(pd.notnull(df_head), None)

    return {
        "columns": df_head.columns.tolist(),
        "rows": df_head.to_dict(orient="records"),
        "masks": mask_head.astype(bool).to_dict(orient="records")
    }


@app.get("/api/preview")
def preview_data(path: str = Query(..., description="CSV File Path")):
    try:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"找不到文件，请检查路径: {path}")

        df = pd.read_csv(path)
        preview_df = df.head(1000).replace({np.nan: None})

        stats = []
        for col in df.columns:
            col_data = df[col]
            missing_count = int(col_data.isnull().sum())

            if pd.api.types.is_numeric_dtype(col_data):
                stats.append({
                    "name": col, "type": "Numeric", "missing": missing_count,
                    "mean": round(float(col_data.mean()), 2) if not pd.isnull(col_data.mean()) else None,
                    "std": round(float(col_data.std()), 2) if not pd.isnull(col_data.std()) else None,
                    "min": round(float(col_data.min()), 2) if not pd.isnull(col_data.min()) else None,
                    "max": round(float(col_data.max()), 2) if not pd.isnull(col_data.max()) else None
                })
            else:
                stats.append({
                    "name": col, "type": "Categorical", "missing": missing_count,
                    "unique": int(col_data.nunique())
                })

        return {
            "columns": df.columns.tolist(),
            "rows": preview_df.to_dict(orient="records"),
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/api/ws/run_task")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    payload = await websocket.receive_json()

    method = payload.get("method")
    paths = payload.get("paths", {})
    params = payload.get("params", {})
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loop = asyncio.get_running_loop()
    log_queue = asyncio.Queue()

    def run_ml_task():
        setup_seed(49)
        stdout_catcher = WsLogCatcher(loop, log_queue, sys.stdout)
        stderr_catcher = WsLogCatcher(loop, log_queue, sys.stderr)

        with contextlib.redirect_stdout(stdout_catcher), contextlib.redirect_stderr(stderr_catcher):
            try:
                print(f"========== 开始执行任务: {method} ==========\n")
                result_data = None

                if method in ["GAIN", "VAEGAIN", "SCIS", "EDIT"]:
                    print("Loading data...")
                    df_missing = pd.read_csv(paths.get("dataPath"))
                    columns = df_missing.columns
                    data_missing = df_missing.values
                    data_true = pd.read_csv(paths.get("groundTruthPath")).values
                    mask = pd.read_csv(paths.get("missingMaskPath")).values

                    imp_bayes = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
                    res_bayes = imp_bayes.fit_transform(data_missing)
                    rmse_bayes = calc_rmse(data_true, res_bayes, mask)
                    mae_bayes = calc_mae(data_true, res_bayes, mask)

                    imp_rf = IterativeImputer(
                        estimator=RandomForestRegressor(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42),
                        max_iter=5, random_state=42)
                    res_rf = imp_rf.fit_transform(data_missing)
                    rmse_rf = calc_rmse(data_true, res_rf, mask)
                    mae_rf = calc_mae(data_true, res_rf, mask)

                    print(f"Running {method}...")
                    params['device'] = device
                    if method == "GAIN":
                        model = GAIN(**params)
                    elif method == "VAEGAIN":
                        model = VAEGAIN(**params)
                    elif method == "SCIS":
                        model = SCIS(**params)
                    elif method == "EDIT":
                        model = EDIT(**params)

                    res_ours = model.train_and_predict(data_missing, mask)

                    # ================= Overall metrics =================
                    rmse_ours = calc_rmse(data_true, res_ours, mask)
                    mae_ours = calc_mae(data_true, res_ours, mask)

                    nrmse_bayes = calc_nrmse(data_true, res_bayes, mask)
                    nrmse_rf = calc_nrmse(data_true, res_rf, mask)
                    nrmse_ours = calc_nrmse(data_true, res_ours, mask)

                    # ================= Per-column metrics =================
                    col_metrics_bayes = calc_column_metrics(data_true, res_bayes, mask, columns)
                    col_metrics_rf = calc_column_metrics(data_true, res_rf, mask, columns)
                    col_metrics_ours = calc_column_metrics(data_true, res_ours, mask, columns)

                    print("\n[DEBUG] Per-column metrics:")

                    for i, row in enumerate(col_metrics_ours):
                        col = row["column"]

                        b = col_metrics_bayes[i]
                        r = col_metrics_rf[i]
                        o = col_metrics_ours[i]

                        print(
                            f"  {col}: "
                            f"missing={o['missing_count']}, "
                            f"range=[{o['true_min']:.4f}, {o['true_max']:.4f}], "
                            f"std={o['true_std']:.4f}"
                        )
                        print(
                            f"      BayesianRidge: RMSE={b['rmse']:.4f}, MAE={b['mae']:.4f}"
                        )
                        print(
                            f"      RandomForest : RMSE={r['rmse']:.4f}, MAE={r['mae']:.4f}"
                        )
                        print(
                            f"      {method:<12}: RMSE={o['rmse']:.4f}, MAE={o['mae']:.4f}"
                        )

                    result = {
                        "rmse_bayes": round(rmse_bayes, 4),
                        "mae_bayes": round(mae_bayes, 4),
                        "nrmse_bayes": round(nrmse_bayes, 4),

                        "rmse_rf": round(rmse_rf, 4),
                        "mae_rf": round(mae_rf, 4),
                        "nrmse_rf": round(nrmse_rf, 4),

                        "rmse_ours": round(rmse_ours, 4),
                        "mae_ours": round(mae_ours, 4),
                        "nrmse_ours": round(nrmse_ours, 4),
                    }

                    df_res = pd.DataFrame(res_ours, columns=columns)
                    df_highlight = pd.DataFrame(mask == 0, columns=columns)
                    result_data = generate_result_data(df_res, df_highlight)

                elif method == "ZeroED":
                    df_raw, df_enc, y_true = load_and_prep_detection_data(paths.get("dataPath"),
                                                                          paths.get("errorDetectionPath"))
                    contamination_rate = max(0.01, min(0.5, np.sum(y_true) / len(y_true)))

                    iso = IsolationForest(contamination=contamination_rate, random_state=49, n_jobs=-1)
                    y_pred_iso = np.where(iso.fit_predict(df_enc.values) == -1, 1, 0)

                    lof = LocalOutlierFactor(contamination=contamination_rate, n_neighbors=20)
                    y_pred_lof = np.where(lof.fit_predict(df_enc.values) == -1, 1, 0)

                    print("Running ZeroED...")
                    detector = ZeroED(**params)
                    detector.train(df_raw)

                    df_pred_mask = detector.predict(df_raw)
                    if isinstance(df_pred_mask, np.ndarray):
                        df_pred_mask = pd.DataFrame(df_pred_mask, columns=df_raw.columns)

                    # 保证用于评估/展示的 mask 与原始 df_raw 列对齐。
                    df_pred_mask = _align_mask_to_df(df_pred_mask, df_raw, fill_value=False)

                    y_pred_ours = df_pred_mask.values.any(axis=1).astype(int)

                    result = {
                        "det_prec_iso": round(precision_score(y_true, y_pred_iso, zero_division=0), 4),
                        "det_rec_iso": round(recall_score(y_true, y_pred_iso, zero_division=0), 4),
                        "det_f1_iso": round(f1_score(y_true, y_pred_iso, zero_division=0), 4),
                        "det_prec_lof": round(precision_score(y_true, y_pred_lof, zero_division=0), 4),
                        "det_rec_lof": round(recall_score(y_true, y_pred_lof, zero_division=0), 4),
                        "det_f1_lof": round(f1_score(y_true, y_pred_lof, zero_division=0), 4),
                        "det_prec_ours": round(precision_score(y_true, y_pred_ours, zero_division=0), 4),
                        "det_rec_ours": round(recall_score(y_true, y_pred_ours, zero_division=0), 4),
                        "det_f1_ours": round(f1_score(y_true, y_pred_ours, zero_division=0), 4)
                    }
                    result_data = generate_result_data(df_raw, df_pred_mask.astype(bool))

                elif method == "ZeroEC":
                    df_clean = pd.read_csv(paths.get("cleanDataPath"), index_col=0)
                    df_dirty = pd.read_csv(paths.get("dataPath"), index_col=0)
                    df_mask = pd.read_csv(paths.get("detectionPath"))

                    if 'index' in df_mask.columns: df_mask.drop(columns=['index'], inplace=True)
                    if 'Unnamed: 0' in df_mask.columns: df_mask.drop(columns=['Unnamed: 0'])

                    min_len = min(len(df_clean), len(df_dirty), len(df_mask))
                    df_clean, df_dirty, df_mask = df_clean.iloc[:min_len].reset_index(drop=True), df_dirty.iloc[
                        :min_len].reset_index(drop=True), df_mask.iloc[:min_len].reset_index(drop=True)
                    df_mask = df_mask.replace(
                        {'True': True, 'False': False, 1: True, 0: False, '1': True, '0': False}).astype(bool)

                    print("Running Sklearn Baselines...")
                    res_mode = SklearnCorrector(strategy='most_frequent').correct(df_dirty, df_mask)
                    p_m, r_m, f1_m = evaluate_correction_metrics(df_clean, df_dirty, res_mode, df_mask)
                    edr_m = calc_edr_df(df_dirty, res_mode, df_clean)  # 新增

                    res_knn = SklearnCorrector(strategy='knn').correct(df_dirty, df_mask)
                    p_k, r_k, f1_k = evaluate_correction_metrics(df_clean, df_dirty, res_knn, df_mask)
                    edr_k = calc_edr_df(df_dirty, res_knn, df_clean)  # 新增

                    print("Running ZeroEC...")
                    params.update({
                        'clean_data_path': paths.get('cleanDataPath'),
                        'dirty_data_path': paths.get('dataPath'),
                        'detection_path': paths.get('detectionPath'),
                        'embedding_model_path': paths.get('embeddingModelPath'),
                        'output_dir': paths.get('outputDir'),
                        'prompt_dir': paths.get('promptDir')
                    })
                    zeroec = ZeroEC(**params)
                    df_corrected = zeroec.train_and_predict()

                    p_o, r_o, f1_o = evaluate_correction_metrics(df_clean, df_dirty, df_corrected, df_mask)
                    edr_o = calc_edr_df(df_dirty, df_corrected, df_clean)  # 新增

                    # 输出新增的 edr
                    result = {
                        "cor_prec_mode": f"{p_m:.2%}", "cor_rec_mode": f"{r_m:.2%}", "cor_f1_mode": f"{f1_m:.2%}",
                        "cor_edr_mode": f"{edr_m:.2%}",
                        "cor_prec_knn": f"{p_k:.2%}", "cor_rec_knn": f"{r_k:.2%}", "cor_f1_knn": f"{f1_k:.2%}",
                        "cor_edr_knn": f"{edr_k:.2%}",
                        "cor_prec_ours": f"{p_o:.2%}", "cor_rec_ours": f"{r_o:.2%}", "cor_f1_ours": f"{f1_o:.2%}",
                        "cor_edr_ours": f"{edr_o:.2%}",
                    }

                    result_data = generate_result_data(df_corrected, df_mask)

                elif method == "MLNClean_Det":
                    # MLNClean 作为 detector: 跑完整 MLN 管线, 把"清洗前 vs 清洗后"差异作为错误检测结果
                    df_raw, df_enc, y_true = load_and_prep_detection_data(
                        paths.get("dataPath"), paths.get("errorDetectionPath"))
                    contamination_rate = max(0.01, min(0.5, np.sum(y_true) / len(y_true)))

                    # Sklearn 基线
                    iso = IsolationForest(contamination=contamination_rate, random_state=49, n_jobs=-1)
                    y_pred_iso = np.where(iso.fit_predict(df_enc.values) == -1, 1, 0)

                    lof = LocalOutlierFactor(contamination=contamination_rate, n_neighbors=20)
                    y_pred_lof = np.where(lof.fit_predict(df_enc.values) == -1, 1, 0)

                    print("Running MLNClean Detection...")
                    mln_params = {
                        k: v for k, v in params.items()
                        if k in ["mcmc_samples", "mcmc_warmup", "partition_number", "agp_threshold", "verbose"]
                    }
                    mln_params["rules_path"] = paths.get("rulesPath")
                    if paths.get("evidencePath"):
                        mln_params["evidence_path"] = paths.get("evidencePath")

                    # MLNClean 要求输入有 'ID' 列, 没有就补一个
                    df_input = df_raw.copy()
                    id_col_was_added = False
                    if 'ID' not in df_input.columns:
                        df_input.insert(0, 'ID', range(len(df_input)))
                        id_col_was_added = True

                    detector = MLNCleanDet(**mln_params)
                    detector.train(df_input)
                    df_pred_mask = detector.predict(df_input)

                    # 如果补了 ID 列, 把 mask 里的 ID 列移除, 对齐 df_raw
                    if id_col_was_added and 'ID' in df_pred_mask.columns:
                        df_pred_mask = df_pred_mask.drop(columns=['ID'])

                    if isinstance(df_pred_mask, np.ndarray):
                        df_pred_mask = pd.DataFrame(df_pred_mask, columns=df_raw.columns)

                    # 保证用于评估/展示的 mask 与原始 df_raw 列对齐。
                    df_pred_mask = _align_mask_to_df(df_pred_mask, df_raw, fill_value=False)

                    y_pred_ours = df_pred_mask.values.any(axis=1).astype(int)

                    df_true_mask_cell = _to_bool_mask(pd.read_csv(paths.get("errorDetectionPath")))
                    df_true_mask_cell = df_true_mask_cell.iloc[:len(df_raw)].reset_index(drop=True)
                    df_true_mask_cell = _align_mask_to_df(df_true_mask_cell, df_raw, fill_value=False)

                    cell_p, cell_r, cell_f1, true_dirty_cells, pred_dirty_cells = evaluate_cell_detection_metrics(
                        df_true_mask_cell,
                        df_pred_mask,
                    )

                    result = {
                        "det_prec_iso":  round(precision_score(y_true, y_pred_iso, zero_division=0), 4),
                        "det_rec_iso":   round(recall_score(y_true, y_pred_iso, zero_division=0), 4),
                        "det_f1_iso":    round(f1_score(y_true, y_pred_iso, zero_division=0), 4),
                        "det_prec_lof":  round(precision_score(y_true, y_pred_lof, zero_division=0), 4),
                        "det_rec_lof":   round(recall_score(y_true, y_pred_lof, zero_division=0), 4),
                        "det_f1_lof":    round(f1_score(y_true, y_pred_lof, zero_division=0), 4),
                        "det_prec_ours": round(precision_score(y_true, y_pred_ours, zero_division=0), 4),
                        "det_rec_ours":  round(recall_score(y_true, y_pred_ours, zero_division=0), 4),
                        "det_f1_ours":   round(f1_score(y_true, y_pred_ours, zero_division=0), 4),
                        # 新增：MLNClean cell-level detection 指标
                        "det_cell_prec_ours": round(cell_p, 4),
                        "det_cell_rec_ours": round(cell_r, 4),
                        "det_cell_f1_ours": round(cell_f1, 4),
                        "det_true_dirty_cells": true_dirty_cells,
                        "det_pred_dirty_cells": pred_dirty_cells,
                    }
                    if paths.get("cleanDataPath"):
                        df_clean_for_pipe = _read_table(paths.get("cleanDataPath"))
                        df_clean_for_pipe = df_clean_for_pipe.iloc[:len(df_raw)].reset_index(drop=True)

                        # detector.cleaned_df_ 是 MLNClean 清洗后的结果表。
                        df_cleaned_by_mln = detector.cleaned_df_.copy()

                        # 如果为 MLNClean 临时补了 ID 列，评估时删除。
                        if id_col_was_added and 'ID' in df_cleaned_by_mln.columns:
                            df_cleaned_by_mln = df_cleaned_by_mln.drop(columns=['ID'])

                        df_cleaned_by_mln = _drop_artifact_index_cols(df_cleaned_by_mln)
                        df_cleaned_by_mln = df_cleaned_by_mln.iloc[:len(df_raw)].reset_index(drop=True)

                        pipe_p, pipe_r, pipe_f1, pipe_edr, pipe_modified = evaluate_pipeline_repair_metrics(
                            df_clean_for_pipe,
                            df_raw,
                            df_cleaned_by_mln,
                            df_pred_mask,
                            df_true_mask_cell,
                        )

                        result.update({
                            "pipe_prec_ours": f"{pipe_p:.2%}",
                            "pipe_rec_ours": f"{pipe_r:.2%}",
                            "pipe_f1_ours": f"{pipe_f1:.2%}",
                            "pipe_edr_ours": f"{pipe_edr:.2%}",
                            "pipe_modified_cells": pipe_modified,
                        })
                    result_data = generate_result_data(df_raw, df_pred_mask.astype(bool))

                elif method == "MLNClean_Cor":
                    # MLNClean 作为 corrector: 跑完整管线, 只在 detection_mask 标记位置覆盖
                    # 注意：这里不要用 index_col=0，否则可能把原版 MLNClean 的 ID 列吃掉。
                    df_clean = _read_table(paths.get("cleanDataPath"))
                    df_dirty = _read_table(paths.get("dataPath"))
                    df_mask = _to_bool_mask(pd.read_csv(paths.get("detectionPath")))

                    min_len = min(len(df_clean), len(df_dirty), len(df_mask))
                    df_clean = df_clean.iloc[:min_len].reset_index(drop=True)
                    df_dirty = df_dirty.iloc[:min_len].reset_index(drop=True)
                    df_mask = df_mask.iloc[:min_len].reset_index(drop=True)

                    # baseline/correction 评估用的 mask 必须与 df_dirty 列对齐。
                    df_mask = _align_mask_to_df(df_mask, df_dirty, fill_value=False)

                    # Sklearn 基线
                    print("Running Sklearn Baselines...")
                    res_mode = SklearnCorrector(strategy='most_frequent').correct(df_dirty, df_mask)
                    p_m, r_m, f1_m = evaluate_correction_metrics(df_clean, df_dirty, res_mode, df_mask)
                    edr_m = calc_edr_df(df_dirty, res_mode, df_clean)

                    res_knn = SklearnCorrector(strategy='knn').correct(df_dirty, df_mask)
                    p_k, r_k, f1_k = evaluate_correction_metrics(df_clean, df_dirty, res_knn, df_mask)
                    edr_k = calc_edr_df(df_dirty, res_knn, df_clean)

                    res_iter = SklearnCorrector(strategy='iterative').correct(df_dirty, df_mask)
                    p_i, r_i, f1_i = evaluate_correction_metrics(df_clean, df_dirty, res_iter, df_mask)
                    edr_i = calc_edr_df(df_dirty, res_iter, df_clean)


                    print("Running MLNClean Correction...")
                    mln_params = {
                        k: v for k, v in params.items()
                        if k in ["mcmc_samples", "mcmc_warmup", "partition_number", "agp_threshold", "verbose"]
                    }
                    mln_params["rules_path"] = paths.get("rulesPath")
                    if paths.get("evidencePath"):
                        mln_params["evidence_path"] = paths.get("evidencePath")

                    # MLNClean 要求输入有 'ID' 列；如果原数据没有，临时补一个。
                    df_dirty_for_mln = df_dirty.copy()
                    df_mask_for_mln = df_mask.copy()
                    id_col_was_added = False

                    if 'ID' not in df_dirty_for_mln.columns:
                        df_dirty_for_mln.insert(0, 'ID', range(len(df_dirty_for_mln)))
                        id_col_was_added = True

                    # detection_mask 也要和传给 MLNClean 的 dirty_df 对齐。
                    if 'ID' in df_dirty_for_mln.columns and 'ID' not in df_mask_for_mln.columns:
                        df_mask_for_mln.insert(0, 'ID', False)
                    df_mask_for_mln = _align_mask_to_df(df_mask_for_mln, df_dirty_for_mln, fill_value=False)

                    corrector = MLNCleanCor(**mln_params)
                    df_corrected = corrector.train_and_predict(
                        df_dirty_for_mln,
                        detection_mask=df_mask_for_mln,
                    )

                    # 如果补了临时 ID, 从输出里删除；评估仍使用原始 df_dirty/df_mask。
                    if id_col_was_added and 'ID' in df_corrected.columns:
                        df_corrected = df_corrected.drop(columns=['ID'])

                    # 评估前再次对齐列，避免 MLNClean 输出多/少列导致指标错位。
                    common_cols = df_clean.columns.intersection(df_dirty.columns).intersection(df_corrected.columns).intersection(df_mask.columns)
                    df_clean_eval = df_clean[common_cols]
                    df_dirty_eval = df_dirty[common_cols]
                    df_corrected_eval = df_corrected[common_cols]
                    df_mask_eval = df_mask[common_cols]

                    p_o, r_o, f1_o = evaluate_correction_metrics(
                        df_clean_eval, df_dirty_eval, df_corrected_eval, df_mask_eval
                    )
                    edr_o = calc_edr_df(df_dirty_eval, df_corrected_eval, df_clean_eval)

                    result = {
                        "cor_prec_mode": f"{p_m:.2%}", "cor_rec_mode": f"{r_m:.2%}",
                        "cor_f1_mode":   f"{f1_m:.2%}", "cor_edr_mode":  f"{edr_m:.2%}",

                        "cor_prec_knn":  f"{p_k:.2%}", "cor_rec_knn":  f"{r_k:.2%}",
                        "cor_f1_knn":    f"{f1_k:.2%}", "cor_edr_knn":   f"{edr_k:.2%}",

                        "cor_prec_iter": f"{p_i:.2%}", "cor_rec_iter": f"{r_i:.2%}",
                        "cor_f1_iter":   f"{f1_i:.2%}", "cor_edr_iter":  f"{edr_i:.2%}",

                        "cor_prec_ours": f"{p_o:.2%}", "cor_rec_ours": f"{r_o:.2%}",
                        "cor_f1_ours":   f"{f1_o:.2%}", "cor_edr_ours":  f"{edr_o:.2%}",
                    }
                    result_data = generate_result_data(df_corrected_eval, df_mask_eval)

                else:
                    raise Exception(f"不支持的算法: {method}")

                print("\n✅ 任务全部执行完毕！")
                asyncio.run_coroutine_threadsafe(
                    log_queue.put({"__done__": True, "metrics": result, "result_data": result_data}),
                    loop
                )

            except Exception as e:
                traceback.print_exc()
                asyncio.run_coroutine_threadsafe(log_queue.put({"__error__": str(e)}), loop)

    executor = ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, run_ml_task)

    try:
        while True:
            msg = await log_queue.get()
            if "__done__" in msg:
                await websocket.send_json({
                    "status": "success",
                    "metrics": msg["metrics"],
                    "result_data": msg.get("result_data")
                })
                break
            elif "__error__" in msg:
                await websocket.send_json({"status": "error", "detail": msg["__error__"]})
                break
            else:
                await websocket.send_json(msg)
    except Exception as e:
        print("WebSocket disconnected:", e)
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8088, reload=False)