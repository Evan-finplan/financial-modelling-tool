import unittest

from model import (
    calculate_division_293_tax,
    calculate_progressive_income_tax,
    calculate_super_guarantee_contribution,
    get_tax_schedule_key_for_financial_year,
    run_deterministic_projection,
)
from policy import (
    get_concessional_contributions_cap,
    get_general_transfer_balance_cap,
    get_non_concessional_contributions_cap,
    get_super_guarantee_maximum_earnings_base,
)


class PolicySettingsTests(unittest.TestCase):
    def test_2026_27_super_thresholds(self):
        self.assertEqual(get_concessional_contributions_cap(2027), 32_500.0)
        self.assertEqual(get_non_concessional_contributions_cap(2027), 130_000.0)
        self.assertEqual(get_general_transfer_balance_cap(2027), 2_100_000.0)
        self.assertEqual(get_super_guarantee_maximum_earnings_base(2027), 270_830.0)

    def test_latest_published_threshold_is_carried_forward(self):
        self.assertEqual(get_concessional_contributions_cap(2030), 32_500.0)
        self.assertEqual(get_super_guarantee_maximum_earnings_base(2030), 270_830.0)

    def test_tax_schedule_transition(self):
        self.assertEqual(get_tax_schedule_key_for_financial_year(2026), 2026)
        self.assertEqual(get_tax_schedule_key_for_financial_year(2027), 2027)
        self.assertEqual(get_tax_schedule_key_for_financial_year(2028), "2028_PLUS")
        self.assertAlmostEqual(calculate_progressive_income_tax(45_000, 2026), 4_288.0)
        self.assertAlmostEqual(calculate_progressive_income_tax(45_000, 2027), 4_020.0)
        self.assertAlmostEqual(calculate_progressive_income_tax(45_000, "2028_PLUS"), 3_752.0)


class HighIncomeTaxTests(unittest.TestCase):
    def test_sg_is_capped_at_2026_27_maximum_earnings_base(self):
        result = calculate_super_guarantee_contribution(300_000, 2027)
        self.assertEqual(result["sg_earnings_base"], 270_830.0)
        self.assertEqual(result["income_above_sg_base"], 29_170.0)
        self.assertAlmostEqual(result["sg_contribution"], 32_499.60)

    def test_sg_below_maximum_earnings_base_is_unchanged(self):
        result = calculate_super_guarantee_contribution(200_000, 2027)
        self.assertEqual(result["sg_earnings_base"], 200_000.0)
        self.assertAlmostEqual(result["sg_contribution"], 24_000.0)

    def test_division_293_partial_threshold_excess(self):
        result = calculate_division_293_tax(240_000, 15_000, 2027)
        self.assertEqual(result["division_293_taxable_contributions"], 5_000.0)
        self.assertEqual(result["division_293_tax"], 750.0)

    def test_division_293_full_contribution_amount(self):
        result = calculate_division_293_tax(300_000, 32_500, 2027)
        self.assertEqual(result["division_293_taxable_contributions"], 32_500.0)
        self.assertEqual(result["division_293_tax"], 4_875.0)

    def test_division_293_ignores_contributions_above_general_cap(self):
        result = calculate_division_293_tax(300_000, 40_000, 2027)
        self.assertEqual(result["division_293_super_contributions"], 32_500.0)
        self.assertEqual(result["division_293_tax"], 4_875.0)

    def test_division_293_below_threshold_is_zero(self):
        result = calculate_division_293_tax(200_000, 32_500, 2027)
        self.assertEqual(result["division_293_taxable_contributions"], 0.0)
        self.assertEqual(result["division_293_tax"], 0.0)


class ProjectionIntegrationTests(unittest.TestCase):
    def test_high_income_projection_reports_sg_cap_and_division_293(self):
        inputs = {
            "start_financial_year": 2027,
            "projection_years": 1,
            "retirement_spending_trigger": "Either Retired",
            "household_mode": "One Person",
            "person1_current_age": 45,
            "person2_current_age": 0,
            "person1_retirement_age": 60,
            "person2_retirement_age": 0,
            "person1_pension_start_age": 67,
            "person2_pension_start_age": 0,
            "person1_accum_super_balance": 500_000.0,
            "person1_pension_super_balance": 0.0,
            "person2_accum_super_balance": 0.0,
            "person2_pension_super_balance": 0.0,
            "person1_accum_super_cost_base": 500_000.0,
            "person1_pension_super_cost_base": 0.0,
            "person2_accum_super_cost_base": 0.0,
            "person2_pension_super_cost_base": 0.0,
            "person1_transfer_balance_cap": 2_100_000.0,
            "person2_transfer_balance_cap": 0.0,
            "non_super_balance": 0.0,
            "non_super_cost_base": 0.0,
            "cgt_discount_rate": 0.50,
            "person1_annual_income": 300_000.0,
            "person2_annual_income": 0.0,
            "annual_living_expenses": 100_000.0,
            "retirement_spending": 100_000.0,
            "non_super_ownership_person1": 1.0,
            "inflation_rate": 0.03,
            "super_income_return_mean": 0.02,
            "super_income_return_std": 0.02,
            "super_capital_return_mean": 0.04,
            "super_capital_return_std": 0.09,
            "non_super_income_return_mean": 0.02,
            "non_super_income_return_std": 0.02,
            "non_super_capital_return_mean": 0.03,
            "non_super_capital_return_std": 0.08,
            "number_of_simulations": 1000,
            "assumption_preset": "Custom",
            "contribution_events": [],
        }

        result = run_deterministic_projection(inputs).iloc[0]

        self.assertAlmostEqual(result["person1_sg_contribution"], 32_499.60)
        self.assertAlmostEqual(result["person1_division_293_tax"], 4_874.94)
        self.assertAlmostEqual(
            result["total_personal_tax"],
            result["person1_personal_tax_total"] + result["person1_division_293_tax"],
        )
        self.assertEqual(result["policy_concessional_contributions_cap"], 32_500.0)
        self.assertEqual(result["policy_general_transfer_balance_cap"], 2_100_000.0)
        self.assertEqual(
            result["policy_version"],
            "2026-27 Budget phase 3 - CGT, residential property and trust minimum tax",
        )


if __name__ == "__main__":
    unittest.main()
