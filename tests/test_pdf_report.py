import unittest
from unittest.mock import patch

import pandas as pd

import pdf_report
from pdf_report import PDF_CHART_KEYS, build_key_milestones, build_pdf_report_bytes


class PdfReportTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "ui_language": "🇬🇧 English",
            "report_title": "Test Projection",
            "household_mode": "One Person",
            "person1_name": "Alex",
            "person1_current_age": 60,
            "person1_retirement_age": 62,
            "person1_pension_start_age": 63,
        }
        self.det_df = pd.DataFrame({
            "financial_year_end": [2027, 2028, 2029, 2030],
            "total_wealth": [1_000_000, 1_050_000, 990_000, 920_000],
            "person1_net_income": [90_000, 90_000, 20_000, 10_000],
            "person1_min_pension_drawdown": [0, 0, 35_000, 38_000],
            "spending": [80_000, 82_000, 84_000, 86_000],
            "total_tax_paid": [25_000, 26_000, 8_000, 6_000],
            "person1_income_tax": [20_000, 21_000, 5_000, 3_000],
            "person1_medicare_levy": [2_000, 2_100, 500, 300],
            "person1_total_super_earnings_tax": [3_000, 2_900, 2_500, 2_700],
            "unmet_shortfall": [0, 0, 0, 5_000],
        })
        self.summary_df = pd.DataFrame({
            "simulation_id": range(10),
            "success": [True] * 8 + [False] * 2,
            "final_wealth": [400_000, 500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000, 1_100_000, 1_200_000, 1_300_000],
        })
        self.percentile_df = pd.DataFrame({
            "financial_year_end": [2027, 2028, 2029, 2030],
            "p10": [900_000, 850_000, 750_000, 650_000],
            "p50": [1_000_000, 1_020_000, 980_000, 920_000],
            "p90": [1_100_000, 1_200_000, 1_250_000, 1_300_000],
        })
        self.failure_df = pd.DataFrame({
            "financial_year_end": [2027, 2028, 2029, 2030],
            "failure_probability": [0.0, 0.0, 0.1, 0.2],
        })
        self.result = {
            "inputs": self.inputs,
            "det_df": self.det_df,
            "summary_df": self.summary_df,
            "percentile_df": self.percentile_df,
            "failure_prob_df": self.failure_df,
            "success_rate": 0.8,
        }

    def test_key_milestones_include_retirement_pension_and_shortfall(self):
        milestones = build_key_milestones(self.det_df, self.inputs)
        events = {item["event"] for item in milestones}
        self.assertIn("Alex retires", events)
        self.assertIn("Alex pension starts", events)
        self.assertIn("First deterministic shortfall", events)

    def test_pdf_report_builds_with_every_selectable_chart(self):
        pdf_bytes = build_pdf_report_bytes(
            selected_result=self.result,
            comparison_results={"Base Case": self.result},
            selected_scenario="Base Case",
            selected_chart_keys=PDF_CHART_KEYS,
            input_warnings=["Review input assumption"],
            output_warnings=["Review output risk"],
        )
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 10_000)

    def test_pdf_report_supports_chinese_text(self):
        result_with_old_english_language = {**self.result, "inputs": {**self.inputs, "ui_language": "🇬🇧 English", "report_title": "退休财务预测"}}
        with patch("pdf_report._styles", wraps=pdf_report._styles) as style_spy:
            pdf_bytes = build_pdf_report_bytes(
                selected_result=result_with_old_english_language,
                comparison_results={"Base Case": result_with_old_english_language},
                selected_scenario="Base Case",
                selected_chart_keys=["wealth_projection"],
                input_warnings=["Non-super cost base is lower than market value, so future withdrawals may crystallise capital gains."],
                report_language="🇨🇳 中文",
            )
            style_spy.assert_called_once_with(True)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 5_000)

    def test_technical_appendix_report_includes_strategy_support_sections(self):
        strategy_result = {
            **self.result,
            "inputs": {
                **self.inputs,
                "withdrawal_order": ["cash", "pension", "non_super", "property"],
                "cash_reserve_floor": 10_000,
                "non_super_estate_reserve": 100_000,
                "property_estate_reserve": 0,
            },
            "success_rate": 0.9,
            "p10_final_wealth": 200_000,
        }
        pdf_bytes = build_pdf_report_bytes(
            selected_result=self.result,
            comparison_results={"Base Case": self.result, "Strategy A": strategy_result},
            selected_scenario="Base Case",
            selected_chart_keys=[],
            report_detail="Technical Appendix",
            adviser_notes="Compare tax and estate outcomes before implementation.",
        )
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 10_000)

    def test_english_report_body_uses_12_point_arial_family(self):
        font_name, styles = pdf_report._styles(False)
        self.assertEqual(styles["body"].fontSize, 12)
        self.assertIn(font_name, {"ArialReport", "Helvetica"})

    def test_cover_uses_household_names_in_report_language(self):
        two_people = {
            **self.inputs,
            "household_mode": "Two People",
            "person2_name": "Taylor",
        }
        self.assertEqual(pdf_report._household_display_name(two_people, False), "Alex & Taylor")
        self.assertEqual(pdf_report._household_display_name(two_people, True), "Alex、Taylor")

    def test_warning_text_is_localised_for_chinese_report(self):
        translated = pdf_report._localise_warning(
            "The deterministic projection realises capital gains on non-super withdrawals in at least one year.",
            True,
        )
        self.assertIn("确定性预测", translated)
        self.assertNotIn("deterministic projection", translated)


if __name__ == "__main__":
    unittest.main()
