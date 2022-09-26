# Copyright 2022 DoorDash, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


import numpy as np
import pandas as pd
import pytest

from dataqualityreport.dataqualityreport import DataQualityReport, DataQualityRule, DataQualityWarning, dqr_compare

SALARIES_DF = pd.read_csv("tests/ds_salaries.csv", index_col=0)

MISSING_BY_COL = "missing_by_test"
TEST_RULES = [
    DataQualityRule(0, "perc_missing > 0.95", ["perc_missing"]),
    DataQualityRule(
        0, "(num_missing_partitions > 0)", ["num_missing_partitions", "min_missing_partition", "max_missing_partition"]
    ),
    DataQualityRule(1, "(perc_distinct > 0.99) & (perc_distinct < 1)", ["perc_distinct"]),
    DataQualityRule(1, "(perc_negative > 0) & (perc_negative < 0.05)", ["perc_negative", "num_negative", "min"]),
    DataQualityRule(2, "(perc_zeros > 0.5)", ["perc_zeros"]),
    DataQualityRule(2, "dtype == 'object'", ["dtype"]),
    DataQualityRule(2, "(perc_missing > 0.5) & (perc_missing <= 0.95)", ["perc_missing"]),
    DataQualityRule(3, "(num_low_10x_IQR_outliers > 0)", ["num_low_10x_IQR_outliers", "p05", "min"]),
    DataQualityRule(3, "(num_high_10x_IQR_outliers > 0)", ["num_high_10x_IQR_outliers", "p95", "max"]),
    DataQualityRule(3, "(perc_missing > 0) & (perc_missing <= 0.5)", ["perc_missing"]),
]

NUM_FIELDS = len(SALARIES_DF.columns)
NUM_WARNINGS = 1
NUM_WARNINGS_GTE_2 = 0
NUM_FIELDS_WITH_WARNINGS = 1
NUM_NON_OBJECT_FIELDS = len([ty for ty in SALARIES_DF.dtypes if str(ty) != "object"])


class TestDataQualityReport:
    @pytest.fixture
    def data_quality_report(self):
        data_quality_report = DataQualityReport(SALARIES_DF, missing_by=MISSING_BY_COL, rules=TEST_RULES, n_jobs=1)
        return data_quality_report

    @pytest.fixture
    def data_quality_report_short(self):
        data_quality_report = DataQualityReport(
            SALARIES_DF.iloc[:, 0:5], missing_by=MISSING_BY_COL, rules=TEST_RULES, n_jobs=1
        )
        return data_quality_report

    def test_init(self, data_quality_report):
        pd.testing.assert_frame_equal(SALARIES_DF, data_quality_report.df)
        assert len(data_quality_report.rules) == len(TEST_RULES)
        assert all([a == b for a, b in zip(data_quality_report.rules, TEST_RULES)])

    def test_summary_df(self, data_quality_report):
        summary_df = data_quality_report.summary_df
        assert len(summary_df) == NUM_FIELDS, "Summary DataFrame is wrong length"
        assert summary_df.perc_missing.count() == NUM_FIELDS, "Missing percent not computed for all fields"
        assert (
            summary_df.perc_zeros.count() == NUM_NON_OBJECT_FIELDS
        ), "Zero percent not computed for all non-object fields"
        assert summary_df.perc_distinct.count() == NUM_FIELDS, "Distinct not computed for all fields"
        summary_cols = set(
            [
                "perc_missing",
                "perc_zeros",
                "num_negative",
                "num_zeros",
                "perc_negative",
                "perc_distinct",
                "num_low_3x_IQR_outliers",
                "num_high_3x_IQR_outliers",
                "num_low_10x_IQR_outliers",
                "num_high_10x_IQR_outliers",
                "count",
                "n_unique",
                "decimal_col",
                "perc_most_freq",
                "val_most_freq",
                "min",
                "p05",
                "p25",
                "median",
                "mean",
                "p75",
                "p95",
                "max",
                "dtype",
                "skew",
                "min_missing_partition",
                "max_missing_partition",
                "num_missing_partitions",
            ]
        )
        assert set(summary_df.columns) == summary_cols, "Summary Columns don't match expectation"

    def test_warnings(self, data_quality_report):
        warns = data_quality_report.warnings()
        assert (
            len(warns) == NUM_WARNINGS
        ), f"Number of warnings, {len(warns)}, doesn't match expectation, {NUM_WARNINGS}"
        warns = data_quality_report.warnings(min_dq_level=2)
        assert len(warns) == NUM_WARNINGS_GTE_2, "Number of warnings at Severity >= 2 doesn't match expectation"
        assert all([type(w) == DataQualityWarning for w in warns])

    def test_get_severities_for_table(self, data_quality_report):
        severities_df = data_quality_report.get_severities_for_table()
        assert len(severities_df) == NUM_FIELDS_WITH_WARNINGS
        assert severities_df.index.name == "field"
        assert set(severities_df.columns) == set(["Sev", "msg"])

    def test_warnings_summary_str(self, data_quality_report):
        warnings_summary_str = data_quality_report.warnings_summary_str()
        assert warnings_summary_str == "S3:1"

    def test_warnings_report_str(self, data_quality_report):
        warnings_report_str = data_quality_report.warnings_report_str()
        assert len(warnings_report_str.splitlines()) == NUM_WARNINGS + 2

    def test_repr(self, data_quality_report):
        repr_str = data_quality_report.__repr__()
        assert len(repr_str.splitlines()) == NUM_WARNINGS + 2

    def test_display_table_raw(self, data_quality_report_short):
        display_df = data_quality_report_short._display_table_raw()
        assert len(display_df.columns) > 0

        assert len(display_df) == len(data_quality_report_short.df.columns)

    def test_style_table(self, data_quality_report_short):
        display_html = data_quality_report_short.display_table()
        assert "<table" in display_html.render()

    def test_missing_by(self):
        salaries_missing_part_df = SALARIES_DF.copy()
        salaries_missing_part_df.loc[:, "active_date"] = np.random.choice(a=range(5), size=len(SALARIES_DF))

        test_field = salaries_missing_part_df.columns[3]
        # Data missing for entirety of one day
        salaries_missing_part_df.loc[salaries_missing_part_df.active_date == 1, test_field] = np.nan

        # Data missing for partial day
        salaries_missing_part_df.loc[
            (salaries_missing_part_df.active_date == 3) & (np.random.choice(a=range(3), size=len(SALARIES_DF)) == 0),
            test_field,
        ] = np.nan

        table_df = DataQualityReport(
            salaries_missing_part_df.iloc[:, 0:4],
            missing_by="active_date",
            n_jobs=1,
        )._display_table_raw(missing_by_bar=True)

        assert len(table_df) == 4


def test_dqr_compare():
    df_train = SALARIES_DF.sample(frac=0.8, random_state=200).iloc[:, 0:3]
    df_eval = SALARIES_DF.drop(df_train.index).iloc[:, 0:3]

    html = dqr_compare([df_train, df_eval], suffixes=["train", "eval"], n_jobs=1)
    assert "<table" in html.render()
