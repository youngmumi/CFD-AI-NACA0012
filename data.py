# augment_alpha_interpolate.py
import numpy as np
import pandas as pd
from pathlib import Path

# === 사용자 설정 ===
CSV_IN  = "raw.csv"          # 입력 CSV (컬럼: Alpha, Cl, Cd)
CSV_OUT = "augmented.csv"    # 출력 CSV
ALPHA_STEP_DEG = 0.01         # 보간 간격(도)
SMOOTH_SAVGOL = True         # 원데이터 소폭 스무딩(있으면 깔끔)
DEC_ALPHA = 2                # Alpha(=AoA) 출력 자릿수: 둘째자리
# ======================

def _savgol(y, window=5, poly=2):
    if len(y) < window or window < poly + 2:
        return y
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")
    except Exception:
        return y

def _fit_spline_or_linear(alpha, values):
    """Scipy CubicSpline 있으면 사용, 없으면 선형보간으로 대체."""
    order = np.argsort(alpha)
    a = np.asarray(alpha)[order]
    v = np.asarray(values)[order]
    a_unique, idx = np.unique(a, return_index=True)
    v_unique = v[idx]

    try:
        from scipy.interpolate import CubicSpline
        if len(a_unique) >= 3:
            return CubicSpline(a_unique, v_unique, bc_type="natural")
    except Exception:
        pass

    # fallback: 선형보간
    from numpy import interp
    def f(x):
        return interp(x, a_unique, v_unique)
    return f

def _make_dense_alpha(a_min, a_max, step, dec=DEC_ALPHA):
    """정수 격자 방식으로 부동소수점 오차 없이 등간격 Alpha 생성."""
    scale = 10 ** dec
    a_min_i = int(np.round(a_min * scale))
    a_max_i = int(np.round(a_max * scale))
    step_i  = int(np.round(step  * scale))
    return np.arange(a_min_i, a_max_i + 1, step_i) / scale

def augment_interpolate(df, alpha_col="Alpha", cl_col="Cl", cd_col="Cd"):
    g = df[[alpha_col, cl_col, cd_col]].dropna().copy()
    g = g.sort_values(alpha_col)

    a = g[alpha_col].to_numpy().astype(float)
    cl = g[cl_col].to_numpy().astype(float)
    cd = g[cd_col].to_numpy().astype(float)

    # (선택) 약한 스무딩으로 출렁임 완화
    if SMOOTH_SAVGOL:
        win = min(7, max(3, (len(cl)//2)*2 + 1))
        cl = _savgol(cl, window=win, poly=2)
        cd = _savgol(cd, window=win, poly=2)

    a_min, a_max = a.min(), a.max()
    a_dense = _make_dense_alpha(a_min, a_max, ALPHA_STEP_DEG, dec=DEC_ALPHA)

    f_cl = _fit_spline_or_linear(a, cl)
    f_cd = _fit_spline_or_linear(a, cd)

    # 보간값
    cl_new = f_cl(a_dense)
    cd_new = f_cd(a_dense)

    # 물리 필터: Cd>=0만 유지
    keep = cd_new >= 0.0
    a_new, cl_new, cd_new = a_dense[keep], cl_new[keep], cd_new[keep]

    # Alpha는 둘째자리로 반올림
    a_new = np.round(a_new, DEC_ALPHA)

    # 원본 + 증강 병합 (같은 Alpha는 원본 우선)
    out = pd.DataFrame({"Alpha": a_new, "Cl": cl_new, "Cd": cd_new})
    merged = pd.concat([df[["Alpha","Cl","Cd"]], out], ignore_index=True)
    merged = merged.sort_values("Alpha").drop_duplicates(subset=["Alpha"], keep="first").reset_index(drop=True)

    # 저장 시 Alpha만 둘째자리 표시 (Cl, Cd 정밀도 유지)
    save_df = merged.copy()
    save_df["Alpha"] = save_df["Alpha"].map(lambda x: f"{x:.{DEC_ALPHA}f}")
    return save_df

if __name__ == "__main__":
    df = pd.read_csv(CSV_IN)
    need = {"Alpha", "Cl", "Cd"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df_aug = augment_interpolate(df, "Alpha", "Cl", "Cd")

    Path(CSV_OUT).parent.mkdir(parents=True, exist_ok=True)
    # float_format은 Cl, Cd에만 적용됨(Alpha는 문자열로 저장되어 둘째자리 고정)
    df_aug.to_csv(CSV_OUT, index=False, float_format="%.6f")
    print(f"[완료] {CSV_OUT} 저장 | 행 수: {len(df_aug)}")
