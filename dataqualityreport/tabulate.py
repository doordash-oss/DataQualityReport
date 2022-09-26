# Copyright 2022 DoorDash, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from functools import partial
from typing import List, Optional

import pandas as pd
from joblib import Parallel, delayed
from matplotlib import cm

from .viz import millify, spark_box, spark_donut, spark_hist, spark_missing_bar, spark_missing_heatmap


def _add_caption(s: str, caption: str) -> str:
    return f"<div title='{caption}'>{s}</div>"


PERC_MISSING_LABEL = "%<br>Missing"
MISSING_HEATMAP_LABEL = "% Missing<br>Heatmap"
MISSING_PARTITION_LABEL = "% Missing<br>Partition"
CARDINALITY_LABEL = _add_caption(
    "Card<br>*Unique", "Cardinality (# of unique elements)\n* -> Unique (Cardinality = Row Count)"
)
PERC_NEGATIVE_LABEL = "%<br>Negative"
PERC_ZEROS_LABEL = "%<br>Zeros"


def _dqr_table_raw(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    n_jobs: int = -1,
    missing_by: str = "active_date",
    missing_by_bar: bool = True,
    box: bool = True,
    hist: bool = True,
    ref_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = raw_df
    display_df = summary_df.copy().rename(
        columns={
            "perc_missing": PERC_MISSING_LABEL,
            "perc_zeros": "% zeros",
            "perc_distinct": "% distinct",
            "perc_negative": "% negative",
        }
    )

    missing_cols = display_df.index[display_df[PERC_MISSING_LABEL] > 0].values
    df_missing_hm_samp = df.sample(n=min(2000, len(df)), random_state=0)
    display_df.loc[missing_cols, MISSING_HEATMAP_LABEL] = Parallel(n_jobs=n_jobs)(
        delayed(partial(spark_missing_heatmap, figsize=(1, 0.25)))(df_missing_hm_samp[col]) for col in missing_cols
    )

    def card_calc(ser: pd.Series) -> str:
        return f"{millify(ser['n_unique'])}{'*' if ser['% distinct'] == 1.0 else ''}"

    display_df[CARDINALITY_LABEL] = display_df.apply(card_calc, axis=1)

    display_df[PERC_MISSING_LABEL] = Parallel(n_jobs=n_jobs)(
        delayed(
            partial(spark_donut, figsize=(0.3, 0.3), caption=f"{perc:.02%} missing", color=cm.Reds((perc + 0.1) / 1.1))
        )(perc)
        for perc in display_df[PERC_MISSING_LABEL].fillna(-1).values
    )

    display_df[PERC_ZEROS_LABEL] = Parallel(n_jobs=n_jobs)(
        delayed(partial(spark_donut, figsize=(0.3, 0.3), caption=f"{perc:.02%} zeros", color=cm.tab10.colors[0]))(perc)
        for perc in display_df["% zeros"].fillna(-1).values
    )

    display_df[PERC_NEGATIVE_LABEL] = Parallel(n_jobs=n_jobs)(
        delayed(partial(spark_donut, figsize=(0.3, 0.3), caption=f"{perc:.02%} negative", color=cm.tab10.colors[1]))(
            perc
        )
        for perc in display_df["% negative"].fillna(-1).values
    )

    def stats_caption(col: str, stats: List[str]) -> str:
        return "\n".join([f"{stat}: {display_df.loc[col, stat]}" for stat in stats])

    if box:
        print("Constructing box plots...")

        box_stats = [
            "min",
            "num_low_10x_IQR_outliers",
            "num_low_3x_IQR_outliers",
            "p05",
            "p25",
            "median",
            "mean",
            "p75",
            "p95",
            "num_high_3x_IQR_outliers",
            "num_high_10x_IQR_outliers",
            "max",
        ]
        display_df["Box Plot"] = Parallel(n_jobs=n_jobs)(
            delayed(partial(spark_box, figsize=(1, 0.25), caption=stats_caption(col, stats=box_stats)))(
                df[col], ref_ser=ref_df[col] if ref_df is not None else None
            )
            for col in df.columns
        )

    if hist:
        print("Spreading hist plots...")

        hist_stats = ["val_most_freq", "perc_most_freq"]

        display_df["Robust Histogram"] = Parallel(n_jobs=n_jobs)(
            delayed(partial(spark_hist, figsize=(2, 0.25), caption=stats_caption(col, stats=hist_stats)))(
                df[col].dropna(), ref_ser=ref_df[col].dropna() if ref_df is not None else None
            )
            for col in df.columns
        )

    if missing_by_bar and (missing_by is not None) and (missing_by in df.columns):
        print("Building missing_by plots...")
        df_perc_missing_by = 1 - (df.groupby(missing_by).count() * 1.0 / df.groupby(missing_by).agg(lambda x: len(x)))

        display_df.loc[df_perc_missing_by.columns, MISSING_PARTITION_LABEL] = Parallel(n_jobs=n_jobs)(
            delayed(partial(spark_missing_bar, figsize=(1, 0.25)))(df_perc_missing_by[col])
            for col in df_perc_missing_by.columns
        )

    return (
        display_df.loc[
            :,
            [
                col
                for col in [
                    "dtype",
                    CARDINALITY_LABEL,
                    MISSING_PARTITION_LABEL,
                    MISSING_HEATMAP_LABEL,
                    PERC_MISSING_LABEL,
                    PERC_ZEROS_LABEL,
                    PERC_NEGATIVE_LABEL,
                    "Box Plot",
                    "Robust Histogram",
                ]
                if col in display_df.columns
            ],
        ]
        .reset_index(drop=False)
        .rename(columns={"index": "Column", "dtype": "Type"})
        .set_index(["Column"])
    )


def _style_dqr_table(display_df: pd.DataFrame, num_rows: Optional[int] = None) -> pd.DataFrame.style:
    """Stylize display_table."""
    html = (
        display_df.style.format(
            {
                MISSING_HEATMAP_LABEL: "{}",
                MISSING_PARTITION_LABEL: "{}",
            },
            na_rep="",
        )
        .set_properties(
            **{
                "text-align": "center",
                "width": "fit-content",
                "height": "fit-content",
                "padding-top": "2px",
                "padding-bottom": "2px",
            }
        )
        .set_table_styles(
            [
                {"selector": "th", "props": [("font-weight", "normal")]},
                {"selector": ".col_heading", "props": [("text-align", "center")]},
                {"selector": ".level0.index_name", "props": [("text-align", "left")]},
                {"selector": ".level0.row_heading", "props": [("text-align", "left")]},
                {"selector": "tr", "props": [("background-color", "white")]},
                {"selector": "tbody tr:hover", "props": [("background-color", "#ededed")]},
            ]
        )
        .set_caption("{:,} rows by {} columns".format(num_rows, len(display_df)) if num_rows is not None else "")
    )
    return html
