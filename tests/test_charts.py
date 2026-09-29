import unittest

import pandas as pd

from charts import VIVID_COLOURWAY, create_cashflow_chart, create_percentile_paths_chart


class ChartTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "ui_language": "🇬🇧 English",
            "start_financial_year": 2027,
            "household_mode": "One Person",
            "person1_name": "Alex",
            "person1_current_age": 60,
            "person1_retirement_age": 65,
            "person1_pension_start_age": 65,
        }

    def test_percentile_paths_use_high_contrast_colours(self):
        percentile_df = pd.DataFrame({
            "financial_year_end": [2027, 2028],
            "p10": [80, 70],
            "p50": [100, 105],
            "p90": [120, 140],
        })
        fig = create_percentile_paths_chart(percentile_df, self.inputs, "Paths")
        self.assertEqual([trace.line.color for trace in fig.data], ["#D62728", "#1F77B4", "#2CA02C"])
        self.assertGreaterEqual(len(VIVID_COLOURWAY), 10)

    def test_cashflow_chart_stacks_inflows_outflows_and_adds_net_line(self):
        det_df = pd.DataFrame({
            "financial_year_end": [2027, 2028],
            "household_net_income": [100.0, 80.0],
            "total_minimum_pension_drawdown": [20.0, 30.0],
            "non_super_withdrawal": [10.0, 5.0],
            "total_extra_super_withdrawal": [5.0, 0.0],
            "spending": [80.0, 90.0],
            "total_cash_contributions": [15.0, 5.0],
            "non_super_tax_paid": [3.0, 2.0],
            "total_super_withdrawal_cgt_tax": [2.0, 1.0],
        })
        fig = create_cashflow_chart(det_df, self.inputs, "Cash Flow")

        bar_traces = [trace for trace in fig.data if trace.type == "bar"]
        net_trace = next(trace for trace in fig.data if trace.type == "scatter")
        self.assertTrue(any(float(value) > 0 for trace in bar_traces for value in trace.y))
        self.assertTrue(any(float(value) < 0 for trace in bar_traces for value in trace.y))
        self.assertEqual(list(net_trace.y), [35.0, 17.0])
        self.assertEqual(fig.layout.barmode, "relative")


if __name__ == "__main__":
    unittest.main()
