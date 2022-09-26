# Copyright 2022 DoorDash, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


import bisect
import numbers
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data_utils import summarize_df
from .tabulate import _dqr_table_raw, _style_dqr_table


@dataclass
class DataQualityWarning:
    level: int
    field: str
    msg: str


@dataclass
class DataQualityRule:
    level: int
    condition: str
    fields: List[str]
    msg: Optional[str] = None

    def __post_init__(self) -> None:
        """Resolve msg default."""
        if self.msg is None:
            self.msg = ""


TARGET_RULES = [
    DataQualityRule(0, "perc_missing > 0", ["perc_missing"]),
    DataQualityRule(0, "dtype == 'object'", ["dtype"]),
    DataQualityRule(0, "n_unique==1", ["n_unique", "mean"]),
    DataQualityRule(
        0, "(num_missing_partitions > 0)", ["num_missing_partitions", "min_missing_partition", "max_missing_partition"]
    ),
    DataQualityRule(1, "(perc_negative > 0) & (perc_negative < 0.05)", ["perc_negative", "num_negative", "min"]),
    DataQualityRule(
        1,
        "((n_unique > 30) | decimal_col) & (perc_most_freq > 0.4)",
        ["perc_most_freq", "val_most_freq"],
        "High percentage of a single value",
    ),
    DataQualityRule(2, "perc_zeros > 0.5", ["perc_zeros"]),
    DataQualityRule(
        2, "num_low_3x_IQR_outliers > 0", ["num_low_3x_IQR_outliers", "num_low_10x_IQR_outliers", "min", "p05", "p25"]
    ),
    DataQualityRule(
        2,
        "num_high_3x_IQR_outliers > 0",
        ["num_high_3x_IQR_outliers", "num_high_10x_IQR_outliers", "p75", "p95", "max"],
    ),
    DataQualityRule(3, "abs(skew) > 0.5", ["skew"], "Data is skewed - maybe try a transformation (log?)"),
]

FEATURE_RULES = [
    DataQualityRule(0, "perc_missing > 0.95", ["perc_missing"]),
    DataQualityRule(0, "n_unique==1", ["n_unique", "mean"]),
    DataQualityRule(
        0, "(num_missing_partitions > 0)", ["num_missing_partitions", "min_missing_partition", "max_missing_partition"]
    ),
    DataQualityRule(1, "(perc_distinct > 0.99) & (perc_distinct < 1)", ["perc_distinct"]),
    DataQualityRule(1, "(perc_negative > 0) & (perc_negative < 0.05)", ["perc_negative", "num_negative", "min"]),
    DataQualityRule(2, "(perc_zeros > 0.5)", ["perc_zeros"]),
    DataQualityRule(2, "dtype == 'object'", ["dtype"]),
    DataQualityRule(2, "(perc_missing > 0.5) & (perc_missing <= 0.95)", ["perc_missing"]),
    DataQualityRule(
        2,
        "((n_unique > 30) | decimal_col) & (perc_most_freq > 0.4)",
        ["perc_most_freq", "val_most_freq"],
        "High percentage of a single value",
    ),
    DataQualityRule(3, "(num_low_10x_IQR_outliers > 0)", ["min", "p05", "p25", "num_low_10x_IQR_outliers"]),
    DataQualityRule(3, "(num_high_10x_IQR_outliers > 0)", ["num_high_10x_IQR_outliers", "p75", "p95", "max"]),
    DataQualityRule(3, "(perc_missing > 0) & (perc_missing <= 0.5)", ["perc_missing"]),
]


class DataQualityReport:
    """Class to assess data quality of pandas DataFrames for machine learning applications.

    `DataQualityReport`s generate multiple types of reporting:
    * `.warnings_summary_str` - One-line text summary of data quality issues found (Ex: S1:10, S2:40)
    * `.warnings_detail_str` - List of all warnings found, sorted by severity
    * `.display_table` - html table showing metrics for all fields and, optionally, inline spark charts

    Rules:
    `DataQualityReport`s use rules to define what is acceptable in terms of data quality and
    how severe data quality issues are. Rules provide a 'severity' level:

    - 0 is most severe,
    - 1 moderately severe,
    - ...

    Usage:

    ```
    myDataQualityReport = DataQualityReport(myPandasDf)

    print(myDataQualityReport) # prints warnings summary and detail
    myDataQualityReport.warnings_report_str() # ^ same as above

    myDataQualityReport.display_table() # renders HTML table in notebooks
    ```

    """

    def __init__(
        self,
        df: pd.DataFrame,
        missing_by: str = "active_date",
        rules: Optional[List[DataQualityRule]] = None,
        max_rows: Optional[int] = 100000,
        n_jobs: int = -1,
    ) -> None:
        """Create a new DataQualityReport. Also computes summary_df on initialization.

        Args:
            df: pd.DataFrame to quantify data quality issues
            missing_by: column in pd.DataFrame to compute missing rates over
                Commonly this should be 'active_date' or similar field which was used for
                filtering which days to compute features for.
            rules: List of DataQualityRules to enforce - see discussion in class documentation
            max_rows: maximum number of rows to process
            n_jobs: Number of processes to use - set n_jobs=1 (no parallelization) for debugging
                defaults to n_jobs=-1 - set parallelism to number of cores available in OS (may not work in containerized environments)
        """
        num_rows = len(df)
        if (max_rows is not None) and (len(df) > max_rows):
            print(
                f"DataFrame has {num_rows} rows, sampling {max_rows} to reduce latency. Specify `max_rows=None` to disable."
            )
            self.df = df.sample(max_rows)
        else:
            self.df = df
        self.missing_by = missing_by
        if rules is None:
            self.rules = FEATURE_RULES
        else:
            self.rules = rules
        self._summary_df = None
        self.n_jobs = n_jobs

    @property
    def summary_df(self) -> pd.DataFrame:
        """Lazily generate summary_df."""
        if self._summary_df is None:
            self.build_summary_df()
        return self._summary_df

    def build_summary_df(self, col_chunk_size: int = 3) -> pd.DataFrame:
        """Generate a summary of data quality statistics for the table by field."""
        print("Building summary df...")

        def chunks(lst: List[Any], n: int) -> Iterable[Any]:
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        summary_df_no_missing = pd.concat(
            Parallel(n_jobs=self.n_jobs)(
                delayed(summarize_df)(self.df.loc[:, cols]) for cols in chunks(self.df.columns, col_chunk_size)
            )
        )

        if self.missing_by is not None and self.missing_by in self.df.columns:
            missing_ds = self.df.groupby(self.missing_by).count() == 0
        else:
            missing_ds = pd.DataFrame()
        missing_df_summary = pd.DataFrame(
            {
                "min_missing_partition": missing_ds.aggregate(lambda x: x.index[x].min()),
                "max_missing_partition": missing_ds.aggregate(lambda x: x.index[x].max()),
                "num_missing_partitions": missing_ds.sum(),
            }
        )

        self._summary_df = pd.concat([summary_df_no_missing, missing_df_summary], axis=1)
        return self._summary_df

    def warnings(self, min_dq_level: float = np.inf) -> List[DataQualityWarning]:
        """Fetch all warnings with severity <= min_dq_level, all by default."""
        warns = list()
        for rule in filter(lambda x: x.level <= min_dq_level, self.rules):
            warns.extend(self._warns_from_rule(rule))

        return warns

    def _warns_from_rule(self, rule: Any) -> List[DataQualityWarning]:
        warns = list()

        def get_formatted_str(r: Tuple[int, str, List[str], str], f: str) -> str:
            if f.startswith("perc"):
                return f"{f}: {getattr(row, f)}"
            if isinstance(getattr(row, f), numbers.Number):
                return f"{f}: {getattr(row, f):g}"
            else:
                return f"{f}: {getattr(row, f)}"

        for row in self.summary_df.query(rule.condition).itertuples():
            warns.append(
                DataQualityWarning(
                    field=row.Index,
                    level=rule.level,
                    msg="{}{}".format(
                        rule.msg + " " if len(rule.msg) > 0 else "",
                        ", ".join([get_formatted_str(row, f) for f in rule.fields]),
                    ),
                )
            )
        return warns

    def get_severities_for_table(self) -> pd.DataFrame:
        """Get a DataFrame of field warning summaries."""
        warns = self.warnings()
        max_warns: Dict[str, List[str]] = defaultdict(list)
        for w in warns:
            bisect.insort_left(max_warns[w.field], f"S{w.level} {w.msg}")

        severity_colors = {"S0": "maroon", "S1": "red"}
        other_color = "orange"

        if len(max_warns):
            return pd.DataFrame.from_records(
                [
                    {
                        "field": k,
                        "Sev": "<div title='{}'><font color='{}'>".format(
                            "\n".join(v), severity_colors.get(v[0][0:2], other_color)
                        )
                        + v[0][0:2]
                        + "</font></div>",
                        "msg": "\n".join(v),
                    }
                    for k, v in max_warns.items()
                ],
                index="field",
            )
        else:
            return pd.DataFrame()

    def warnings_summary_str(self, **kwargs: Any) -> str:
        """Generate a one-line summary of data quality warnings."""
        warns = self.warnings(**kwargs)
        if len(warns):
            warn_level_count = Counter(w.level for w in warns)
            return ", ".join(f"S{s}:{warn_level_count[s]}" for s in sorted(warn_level_count))
        else:
            return "No Warnings"

    def warnings_detail_str(self, **kwargs: Any) -> str:
        """Generate a detailed text string of data quality issues."""
        warns = self.warnings(**kwargs)
        return "\n".join([str(x) for x in sorted(warns, key=lambda w: str(w.level) + str(w.field))])

    def warnings_report_str(self, **kwargs: Any) -> str:
        """Generate a text report of data quality issues (summary & detail)."""
        return "Data Quality Report\n{}\n{}".format(
            self.warnings_summary_str(**kwargs), self.warnings_detail_str(**kwargs)
        )

    def __repr__(self) -> str:
        """Generate a text representation of data quality."""
        return self.warnings_report_str()

    def _display_table_raw(self, **kwargs: Any) -> pd.DataFrame:
        return _dqr_table_raw(self.df, self.summary_df, n_jobs=self.n_jobs, missing_by=self.missing_by, **kwargs)

    def display_table(self, **kwargs: Any) -> pd.DataFrame.style:
        """Create a rich HTML table summarizing data quality.

        Note - plotting uses multiprocessing, which may not behave well in containerized environments.

        Args:
            missing_by_bar: True to display a bar representing missing data by the missing_by column (usually by day)
                Partitions (e.g. days) that are entirely missing are Red. Partitions that are partially available (as indicated
                by the height of the bar) are Orange. Partitions that have no missing values are blue.
            box: True to display a box plot for each field - useful for understanding outliers
            hist: True to display a 'robust' histogram showing the distribution of only non-outlier values
            n_jobs: Number of processes to run in display generation

        Returns:
            pd.styler object that will render in Jupyter Notebooks, or can be converted to HTML with `.render()`
        """
        display_df = self._display_table_raw(**kwargs)
        html = _style_dqr_table(display_df, num_rows=len(self.df))
        return html


def dqr_compare(
    dfs: List[pd.DataFrame],
    suffixes: Optional[List[str]] = None,
    rules: Optional[List[DataQualityRule]] = None,
    n_jobs: int = -1,
    **kwargs: Any,
) -> pd.DataFrame.style:
    """Compare two dataframes for data quality - box and hist axes are the same across pairs of fields."""
    if len(dfs) == 0:
        raise RuntimeError("You must include at least one DataFrame")
    if suffixes is None:
        suffixes = [str(i) for i in range(len(dfs))]

    display_dfs = []
    ref_df = pd.concat(dfs)

    for i, df in enumerate(dfs):
        display_dfs.append(
            DataQualityReport(
                df.rename(columns={c: f"{c}-{suffixes[i]}" for c in df.columns}),
                rules=rules,
                n_jobs=n_jobs,
            )._display_table_raw(ref_df=ref_df.rename(columns={c: f"{c}-{suffixes[i]}" for c in df.columns}), **kwargs)
        )
    display_table = pd.concat(display_dfs)
    display_table.sort_index(inplace=True)

    return _style_dqr_table(display_table)


def dqr_table(
    df: pd.DataFrame, missing_by: str = "active_date", n_jobs: int = -1, max_rows: int = 100000, **kwargs: Any
) -> pd.DataFrame.style:
    """Shortcut for making a dqr_table."""
    this_dqr = DataQualityReport(df, n_jobs=n_jobs, max_rows=max_rows, missing_by=missing_by)
    return this_dqr.display_table(**kwargs)
