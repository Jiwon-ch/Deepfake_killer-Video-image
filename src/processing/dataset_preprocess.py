# src/processing/dataset_preprocess.py
from __future__ import annotations

import gc
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


def balance_dataframe_with_ros(
    df: pd.DataFrame,
    label_col: str = "label",
    random_state: int = 83,
) -> pd.DataFrame:
    """
    (선택) 판다스 DataFrame 기반 라벨 불균형 완화.
    - HF Dataset 으로 바로 학습할 때는 보통 필요 없음.
    - CSV 같은 외부 데이터 정리 시 사용.
    """
    y = df[[label_col]]
    X = df.drop(columns=[label_col])

    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(X, y)

    X_res[label_col] = y_res
    del y, y_res
    gc.collect()
    return X_res


__all__ = [
    "balance_dataframe_with_ros",
]