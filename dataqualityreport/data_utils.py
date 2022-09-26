# Copyright 2022 DoorDash, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


import warnings
from typing import Any, List, Optional

import pandas as pd
from pandas.api.types import is_float_dtype, is_numeric_dtype

LIKELY_NON_UNIQUE_PERC_DISTINCT = 0.9


def summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute various statistics about columns in a DataFrame."""
    num_rows = len(df)
    bool_cols = df.dtypes == "bool"
    quantiles = df.loc[:, ~bool_cols].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
    df_samp = df.sample(n=min(100000, num_rows))  # Use a sample to speed up
    IQR = quantiles.loc[0.75, :] - quantiles.loc[0.25, :]
    df_n_unique = df_samp.nunique()
    decimal_col = (df.loc[:, df.dtypes.apply(is_float_dtype)].fillna(-9999) % 1 == 0).all()
    df_count = df.count()
    continuous_cols = df.columns[(df.dtypes.apply(is_numeric_dtype)) & (df_n_unique > 30)]
    perc_distinct = df_n_unique * 1.0 / df_samp.count()
    df_mode = df_samp.loc[:, perc_distinct < LIKELY_NON_UNIQUE_PERC_DISTINCT].mode()
    if len(df_mode):
        val_most_freq = df_mode.iloc[0, :]
        # filter by perc_distinct as this is an expensive operation
        num_most_freq = (df_samp.loc[:, perc_distinct < 0.9] == val_most_freq[perc_distinct < 0.9]).sum()
    else:
        val_most_freq = pd.Series()
        num_most_freq = pd.Series()
    coerce_to_numeric_cols = []
    for col in df.columns:
        try:
            df[col].astype(float)
        except Exception:
            continue
        coerce_to_numeric_cols.append(col)

    num_negative = (df[coerce_to_numeric_cols].astype(float) < 0).sum()
    perc_missing = 1 - (df_count / num_rows)
    num_zeros = (df[coerce_to_numeric_cols] == 0).sum()
    perc_zeros = num_zeros / df_count[coerce_to_numeric_cols]
    perc_negative = num_negative / df_count[coerce_to_numeric_cols]
    perc_distinct = df_n_unique / df_samp.count()
    df_cont, quantiles_df = df[continuous_cols].align(quantiles, axis=1, copy=True)
    num_low_3x_IQR_outliers = (df_cont < (quantiles_df.loc[0.05, :] - 3 * IQR)).sum()
    num_high_3x_IQR_outliers = (df_cont > (quantiles_df.loc[0.95, :] + 3 * IQR)).sum()
    num_low_10x_IQR_outliers = (df_cont < (quantiles_df.loc[0.05, :] - 10 * IQR)).sum()
    num_high_10x_IQR_outliers = (df_cont > (quantiles_df.loc[0.95, :] + 10 * IQR)).sum()
    perc_most_freq = num_most_freq / df_samp.count()
    p0 = quantiles.loc[0, :]
    p05 = quantiles.loc[0.05, :]
    p25 = quantiles.loc[0.25, :]
    median = quantiles.loc[0.5, :]
    mean = df.mean(numeric_only=True)
    p75 = quantiles.loc[0.75, :]
    p95 = quantiles.loc[0.95, :]
    p1 = quantiles.loc[1, :]
    dtype = df.dtypes.map(lambda s: f"<div title='{s}'>{str(s)[0].upper()}</div>")
    skew = df_samp[continuous_cols].skew(axis=0, numeric_only=True)

    summary_df = pd.DataFrame(
        {
            "perc_missing": perc_missing,
            "perc_zeros": perc_zeros,
            "num_negative": num_negative,
            "num_zeros": num_zeros,
            "perc_negative": perc_negative,
            "perc_distinct": perc_distinct,
            "num_low_3x_IQR_outliers": num_low_3x_IQR_outliers,
            "num_high_3x_IQR_outliers": num_high_3x_IQR_outliers,
            "num_low_10x_IQR_outliers": num_low_10x_IQR_outliers,
            "num_high_10x_IQR_outliers": num_high_10x_IQR_outliers,
            "count": df_count,
            "n_unique": df_n_unique,
            "decimal_col": decimal_col,
            "perc_most_freq": perc_most_freq,
            "val_most_freq": val_most_freq,
            "min": p0,
            "p05": p05,
            "p25": p25,
            "median": median,
            "mean": mean,
            "p75": p75,
            "p95": p95,
            "max": p1,
            "dtype": dtype,
            "skew": skew,
        }
    )
    summary_df.loc[:, "decimal_col"] = summary_df.decimal_col.fillna(False).astype(bool)
    summary_df = summary_df.loc[df.columns, :]
    return summary_df


def is_outlier_iqr(
    df: pd.DataFrame, IQR_p: float = 0.25, IQR_multiple: int = 10, numeric_only: bool = True
) -> pd.DataFrame:
    """Get a mask for cells that contain outliers."""
    QLo = df.quantile(IQR_p, numeric_only=numeric_only)
    QHi = df.quantile(1 - IQR_p, numeric_only=numeric_only)
    IQR = QHi - QLo
    df_n_unique = df.nunique()
    continuous_cols = df.columns[(df.dtypes.apply(is_numeric_dtype)) & (df_n_unique > 30)]
    return (df[continuous_cols] < (QLo[continuous_cols] - IQR_multiple * IQR[continuous_cols])) | (
        df[continuous_cols] > (QHi[continuous_cols] + IQR_multiple * IQR[continuous_cols])
    )


def drop_outliers_iqr(
    df: pd.DataFrame, subset: Optional[List[str]] = None, axis: int = 1, verbose: int = 0, **kwargs: Any
) -> pd.DataFrame:
    """Drop rows in DataFrame that contain outliers using IQR method."""
    if subset is None:
        subset = df.columns[df.dtypes != "bool"]
    mask = is_outlier_iqr(df.loc[:, subset], **kwargs)
    if verbose:
        warnings.warn(
            f"Dropping {mask.any(axis=axis).sum()} ({mask.any(axis=axis).sum() * 1.0 / len(df):.2%}) of {len(df)}"
        )
    return df.loc[~mask.any(axis=axis), :]


def drop_outliers_iqr_ser(ser: pd.Series, **kwargs: Any) -> pd.Series:
    """Drop rows in Series that contain outliers using IQR method."""
    return drop_outliers_iqr(ser.to_frame(), **kwargs).iloc[:, 0]


def is_outlier_iqr_ser(ser: pd.Series, **kwargs: Any) -> pd.Series:
    """Get a mask for entries that contain outliers."""
    return is_outlier_iqr(ser.to_frame(), **kwargs).iloc[:, 0]


pd.DataFrame.drop_outliers_iqr = drop_outliers_iqr
pd.Series.drop_outliers_iqr = drop_outliers_iqr_ser

pd.DataFrame.is_outlier_iqr = is_outlier_iqr
pd.Series.is_outlier_iqr = is_outlier_iqr_ser
