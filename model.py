# airfoil_train_eval.py
# CSV -> 데이터 분석 -> 모델 학습(Cl: MLP, Cd: XGBoost) -> 모델 저장 -> 예측/실측 비교(정합도 + 인덱스 비교) + 학습 곡선
# 실행: python airfoil_train_eval.py

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb  # callbacks 용
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===================== 사용자 설정 =====================
CSV_PATH = "augmented.csv"            # <-- 필요시 수정
OUTDIR   = "./airfoil_outputs"        # 결과물 폴더
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE  = 0.2                       # XGBoost early-stopping용
# ======================================================

os.makedirs(OUTDIR, exist_ok=True)

def detect_columns(df: pd.DataFrame):
    """CSV 컬럼 자동 감지 (Alpha/AoA, Cl, Cd)"""
    col_map = {c.lower(): c for c in df.columns}
    def pick(cands):
        for k in cands:
            if k in col_map:
                return col_map[k]
        return None

    alpha = pick(["alpha", "aoa", "angle", "angle_of_attack"])
    cl    = pick(["cl", "c_l", "lift", "lift_coefficient"])
    cd    = pick(["cd", "c_d", "drag", "drag_coefficient"])

    if not all([alpha, cl, cd]):
        raise ValueError(
            f"필수 컬럼(Alpha/AoA, Cl, Cd)을 찾지 못했어요. 현재 컬럼: {list(df.columns)}"
        )
    return alpha, cl, cd

def describe_data(df, alpha_col, cl_col, cd_col):
    print("=== 기본 정보 ===")
    print(df[[alpha_col, cl_col, cd_col]].describe().T, "\n")

    # 간단한 산점도: AoA-CL, AoA-CD
    plt.figure()
    plt.scatter(df[alpha_col], df[cl_col], s=8)
    plt.xlabel(alpha_col); plt.ylabel(cl_col); plt.title(f"{alpha_col} vs {cl_col}")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTDIR, "scatter_aoa_cl.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.scatter(df[alpha_col], df[cd_col], s=8)
    plt.xlabel(alpha_col); plt.ylabel(cd_col); plt.title(f"{alpha_col} vs {cd_col}")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTDIR, "scatter_aoa_cd.png"), bbox_inches="tight")
    plt.close()

def metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred))
    }

def parity_plot(y_true, y_pred, title, outpath):
    plt.figure()
    plt.scatter(y_true, y_pred, s=14)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def index_compare_scatter(y_true, y_pred, ylabel, title, outpath):
    """테스트셋 순서(index) 기준으로 실제값/예측값 점 비교"""
    n = len(y_true)
    x = np.arange(n)
    plt.figure()
    plt.scatter(x, y_true, s=12, label="True")
    plt.scatter(x, y_pred, s=12, label="Pred")
    plt.xlabel("Sample index (test set)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def plot_mlp_curves(mlp_pipeline: Pipeline, outdir: str):
    """MLPRegressor 학습 곡선(Train Loss, Validation R^2) 저장"""
    mlp = mlp_pipeline.named_steps["mlp"]

    # Training loss
    if hasattr(mlp, "loss_curve_") and mlp.loss_curve_:
        plt.figure()
        plt.plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, marker='o', linewidth=1)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("MLP Training Loss Curve (CL)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(outdir, "mlp_cl_train_loss_curve.png"), bbox_inches="tight")
        plt.close()

    # Validation R^2 (버전에 따라 없을 수 있음)
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
        plt.figure()
        plt.plot(np.arange(1, len(mlp.validation_scores_) + 1), mlp.validation_scores_, marker='o', linewidth=1)
        plt.xlabel("Epoch")
        plt.ylabel("Validation R^2")
        plt.title("MLP Validation Score Curve (CL)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(outdir, "mlp_cl_val_score_curve.png"), bbox_inches="tight")
        plt.close()

def plot_xgb_curves(xgb_model: XGBRegressor, outdir: str):
    """XGBoost evals_result 기반 학습 곡선(RMSE 등) 저장"""
    evals_result = None
    if hasattr(xgb_model, "evals_result"):
        try:
            evals_result = xgb_model.evals_result()
        except Exception:
            evals_result = None
    if evals_result is None and hasattr(xgb_model, "evals_result_"):
        evals_result = xgb_model.evals_result_

    if not evals_result:
        return  # 로그가 없으면 스킵

    for eval_name, metrics_dict in evals_result.items():
        for metric_name, values in metrics_dict.items():
            try:
                y = [float(v) for v in values]
            except Exception:
                continue
            plt.figure()
            plt.plot(np.arange(1, len(y) + 1), y, marker='o', linewidth=1)
            plt.xlabel("Boosting Round")
            plt.ylabel(metric_name.upper())
            plt.title(f"XGBoost {metric_name.upper()} Curve (CD) - {eval_name}")
            plt.grid(True, alpha=0.3)
            fn = f"xgb_cd_{eval_name}_{metric_name}_curve.png".replace("/", "_")
            plt.savefig(os.path.join(outdir, fn), bbox_inches="tight")
            plt.close()

def main():
    # 1) 데이터 로드 & 전처리
    df = pd.read_csv(CSV_PATH)
    alpha_col, cl_col, cd_col = detect_columns(df)

    # 숫자형 변환 & 결측 제거
    use_cols = [alpha_col, cl_col, cd_col]
    for c in use_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[use_cols].dropna().copy()

    # 기본 분석 및 산점도 저장
    describe_data(df, alpha_col, cl_col, cd_col)

    # (필요시) 추가 특징 확장 가능. 현재는 AoA 하나만 사용.
    X = df[[alpha_col]].values
    y_cl = df[cl_col].values
    y_cd = df[cd_col].values

    # 2) 학습/검증/테스트 분리
    X_temp, X_test, ycl_temp, ycl_test = train_test_split(
        X, y_cl, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # XGB early-stopping용 검증셋 분리
    X_train, X_val, ycl_train, ycl_val = train_test_split(
        X_temp, ycl_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # Cd 쪽도 동일 분리
    X_temp2, X_test2, ycd_temp, ycd_test = train_test_split(
        X, y_cd, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train2, X_val2, ycd_train, ycd_val = train_test_split(
        X_temp2, ycd_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # 3) 모델 정의
    # 3-1) MLP (for Cl)
    mlp_cl = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            random_state=RANDOM_STATE,
            max_iter=3000,
            early_stopping=True,
            n_iter_no_change=30,
            validation_fraction=0.1
        ))
    ])

    # 3-2) XGBoost (for Cd)
    xgb_cd = XGBRegressor(
        n_estimators=600,             # 속도·성능 균형
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        tree_method="hist",
        verbosity=0                   # 로그 억제
        # device="cuda",              # GPU 사용 시
    )

    # 4) 학습
    print("학습 중: MLP (Cl)...")
    mlp_cl.fit(X_train, ycl_train)

    print("학습 중: XGBoost (Cd) with compatibility fallback...")
    # ---- XGBoost 버전 호환: 최신(callbacks) -> 중간(early_stopping_rounds) -> 구버전(없음) ----
    try:
        # 최신 계열: callbacks
        xgb_cd.set_params(eval_metric="rmse")
        xgb_cd.fit(
            X_train2, ycd_train,
            eval_set=[(X_val2, ycd_val)],
            callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True, maximize=False)]
        )
    except TypeError:
        try:
            # 중간 계열: early_stopping_rounds 지원
            xgb_cd.fit(
                X_train2, ycd_train,
                eval_set=[(X_val2, ycd_val)],
                eval_metric="rmse",
                early_stopping_rounds=50
            )
        except TypeError:
            # 구버전: early stopping 미지원 → 그냥 학습
            warnings.warn(
                "현재 XGBoost는 early stopping 인자를 지원하지 않습니다. "
                "early stopping 없이 학습합니다. (필요시 n_estimators를 줄이세요)"
            )
            xgb_cd.fit(X_train2, ycd_train)

    # ---- 학습 곡선 저장 ----
    plot_mlp_curves(mlp_cl, OUTDIR)
    plot_xgb_curves(xgb_cd, OUTDIR)

    # 5) 테스트 예측
    ycl_pred = mlp_cl.predict(X_test)
    ycd_pred = xgb_cd.predict(X_test2)

    # 6) 지표 계산
    cl_metrics = metrics(ycl_test, ycl_pred)
    cd_metrics = metrics(ycd_test, ycd_pred)
    print("\n=== 성능 지표 ===")
    print("[CL - MLP]", cl_metrics)
    print("[CD - XGB]", cd_metrics)

    # 7) 결과 저장 (모델/예측/그래프)
    cl_model_path = os.path.join(OUTDIR, "mlp_cl.joblib")
    cd_model_path = os.path.join(OUTDIR, "xgb_cd.joblib")
    joblib.dump(mlp_cl, cl_model_path)
    joblib.dump(xgb_cd, cd_model_path)

    results_df = pd.DataFrame({
        "Alpha_for_CL": X_test.flatten(),
        "CL_true": ycl_test,
        "CL_pred": ycl_pred,
        "Alpha_for_CD": X_test2.flatten(),
        "CD_true": ycd_test,
        "CD_pred": ycd_pred
    }).reset_index(drop=True)
    preds_csv_path = os.path.join(OUTDIR, "test_predictions.csv")
    results_df.to_csv(preds_csv_path, index=False)

    # 8) 정합도(Parity) 플롯
    parity_plot(ycl_test, ycl_pred, "Parity - CL (MLP)",
                os.path.join(OUTDIR, "parity_cl.png"))
    parity_plot(ycd_test, ycd_pred, "Parity - CD (XGBoost)",
                os.path.join(OUTDIR, "parity_cd.png"))

    # 9) 인덱스 기준 실측/예측 비교(요청 반영: AoA 미사용)
    index_compare_scatter(ycl_test, ycl_pred, "CL",
                          "CL: True vs Pred (scatter by index)",
                          os.path.join(OUTDIR, "cl_true_vs_pred_scatter.png"))
    index_compare_scatter(ycd_test, ycd_pred, "CD",
                          "CD: True vs Pred (scatter by index)",
                          os.path.join(OUTDIR, "cd_true_vs_pred_scatter.png"))

    # 10) 요약 저장
    summary = {
        "Detected Columns": {"Alpha": alpha_col, "CL": cl_col, "CD": cd_col},
        "Dataset size (after cleaning)": int(len(df)),
        "Split": {"test_size": TEST_SIZE, "val_size": VAL_SIZE},
        "CL Metrics (MLP)": cl_metrics,
        "CD Metrics (XGBoost)": cd_metrics,
        "Saved Models": {"CL": cl_model_path, "CD": cd_model_path},
        "Saved Predictions CSV": preds_csv_path,
        "Saved Plots": [
            "scatter_aoa_cl.png", "scatter_aoa_cd.png",
            "parity_cl.png", "parity_cd.png",
            "cl_true_vs_pred_scatter.png", "cd_true_vs_pred_scatter.png",
            "mlp_cl_train_loss_curve.png", "mlp_cl_val_score_curve.png",
            # xgb 곡선 파일명은 버전에 따라 다를 수 있음 (예: xgb_cd_validation_0_rmse_curve.png)
        ]
    }
    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 저장 완료 ===")
    print(f"- 모델: {cl_model_path}, {cd_model_path}")
    print(f"- 예측 CSV: {preds_csv_path}")
    print(f"- 요약: {os.path.join(OUTDIR, 'summary.json')}")
    print(f"- 그래프: {OUTDIR} 폴더 확인")

if __name__ == "__main__":
    main()
