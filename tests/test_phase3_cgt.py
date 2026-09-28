import unittest

from model import (
    calculate_budget_cgt_on_sale,
    calculate_cgt_minimum_tax_gap,
    run_deterministic_projection,
)


class BudgetCgtCalculationTests(unittest.TestCase):
    def test_transition_split_protects_old_gain_and_indexes_new_gain(self):
        result = calculate_budget_cgt_on_sale(
            sale_proceeds=200_000,
            pool_market_value=200_000,
            pool_cost_base=100_000,
            pool_indexed_cost_base=150_000,
            pool_deferred_pre_2027_gain=50_000,
            opening_capital_losses=0,
            financial_year_end=2028,
            indexation_rate=0.10,
        )
        self.assertEqual(result["deferred_pre_2027_gain"], 50_000)
        self.assertAlmostEqual(result["post_2027_real_gain"], 35_000)
        self.assertAlmostEqual(result["discounted_taxable_capital_gain"], 60_000)
        self.assertAlmostEqual(result["minimum_tax_capital_gain"], 35_000)
        self.assertEqual(result["calculation_method"], "Transition split + indexed real gain")

    def test_carried_losses_are_applied_to_minimum_tax_gain_first(self):
        result = calculate_budget_cgt_on_sale(
            sale_proceeds=200_000,
            pool_market_value=200_000,
            pool_cost_base=100_000,
            pool_indexed_cost_base=150_000,
            pool_deferred_pre_2027_gain=50_000,
            opening_capital_losses=20_000,
            financial_year_end=2028,
            indexation_rate=0.10,
        )
        self.assertAlmostEqual(result["post_2027_real_gain"], 15_000)
        self.assertAlmostEqual(result["deferred_pre_2027_gain"], 50_000)
        self.assertAlmostEqual(result["minimum_tax_capital_gain"], 15_000)
        self.assertEqual(result["remaining_capital_losses"], 0)

    def test_new_residential_discount_choice_retains_50_percent_discount(self):
        result = calculate_budget_cgt_on_sale(
            sale_proceeds=200_000,
            pool_market_value=200_000,
            pool_cost_base=100_000,
            pool_indexed_cost_base=150_000,
            pool_deferred_pre_2027_gain=50_000,
            opening_capital_losses=0,
            financial_year_end=2028,
            indexation_rate=0.10,
            asset_category="New residential dwelling",
            new_residential_method="50% discount",
        )
        self.assertEqual(result["discounted_taxable_capital_gain"], 50_000)
        self.assertEqual(result["minimum_tax_capital_gain"], 0)
        self.assertEqual(result["calculation_method"], "New/affordable residential 50% discount")

    def test_pre_reform_disposal_uses_existing_discount(self):
        result = calculate_budget_cgt_on_sale(
            sale_proceeds=200_000,
            pool_market_value=200_000,
            pool_cost_base=100_000,
            pool_indexed_cost_base=150_000,
            pool_deferred_pre_2027_gain=50_000,
            opening_capital_losses=0,
            financial_year_end=2027,
            indexation_rate=0.10,
        )
        self.assertFalse(result["reform_applies"])
        self.assertEqual(result["discounted_taxable_capital_gain"], 50_000)
        self.assertEqual(result["minimum_tax_capital_gain"], 0)


class CgtMinimumTaxTests(unittest.TestCase):
    def test_gap_tops_basic_tax_up_to_30_percent(self):
        result = calculate_cgt_minimum_tax_gap(
            taxable_income=20_000,
            minimum_tax_capital_gain=10_000,
            tax_schedule_key="2028_PLUS",
        )
        self.assertEqual(result["cgt_minimum_tax_target"], 3_000)
        self.assertAlmostEqual(result["basic_tax_attributable_to_gain"], 252)
        self.assertEqual(result["cgt_minimum_tax_gap"], 2_748)

    def test_high_marginal_taxpayer_has_no_top_up(self):
        result = calculate_cgt_minimum_tax_gap(
            taxable_income=300_000,
            minimum_tax_capital_gain=50_000,
            tax_schedule_key="2028_PLUS",
        )
        self.assertEqual(result["cgt_minimum_tax_gap"], 0)


class CgtProjectionIntegrationTests(unittest.TestCase):
    def test_projection_reports_reformed_cgt_and_minimum_tax(self):
        inputs = {
            "start_financial_year": 2028,
            "projection_years": 1,
            "retirement_spending_trigger": "Either Retired",
            "household_mode": "One Person",
            "person1_current_age": 70,
            "person2_current_age": 0,
            "person1_retirement_age": 60,
            "person2_retirement_age": 0,
            "person1_pension_start_age": 67,
            "person2_pension_start_age": 0,
            "person1_accum_super_balance": 0.0,
            "person1_pension_super_balance": 0.0,
            "person2_accum_super_balance": 0.0,
            "person2_pension_super_balance": 0.0,
            "person1_accum_super_cost_base": 0.0,
            "person1_pension_super_cost_base": 0.0,
            "person2_accum_super_cost_base": 0.0,
            "person2_pension_super_cost_base": 0.0,
            "person1_transfer_balance_cap": 2_100_000.0,
            "person2_transfer_balance_cap": 0.0,
            "non_super_balance": 200_000.0,
            "non_super_cost_base": 100_000.0,
            "non_super_transition_value_2027": 150_000.0,
            "non_super_opening_capital_losses": 0.0,
            "cgt_reform_enabled": True,
            "cgt_asset_acquired_before_2027": True,
            "cgt_indexation_rate": 0.10,
            "cgt_asset_category": "Other",
            "cgt_new_residential_method": "Indexation and 30% minimum tax",
            "cgt_held_at_least_12_months": True,
            "cgt_minimum_tax_exempt": False,
            "cgt_discount_rate": 0.50,
            "person1_annual_income": 0.0,
            "person2_annual_income": 0.0,
            "annual_living_expenses": 50_000.0,
            "retirement_spending": 50_000.0,
            "non_super_ownership_person1": 1.0,
            "inflation_rate": 0.03,
            "super_income_return_mean": 0.0,
            "super_income_return_std": 0.0,
            "super_capital_return_mean": 0.0,
            "super_capital_return_std": 0.0,
            "non_super_income_return_mean": 0.0,
            "non_super_income_return_std": 0.0,
            "non_super_capital_return_mean": 0.0,
            "non_super_capital_return_std": 0.0,
            "number_of_simulations": 10,
            "assumption_preset": "Custom",
            "contribution_events": [],
        }
        result = run_deterministic_projection(inputs).iloc[0]
        self.assertTrue(result["non_super_cgt_reform_applies"])
        self.assertEqual(result["non_super_cgt_calculation_method"], "Transition split + indexed real gain")
        self.assertGreater(result["non_super_deferred_pre_2027_gain"], 0)
        self.assertGreater(result["non_super_post_2027_real_gain"], 0)
        self.assertGreater(result["non_super_minimum_tax_capital_gain"], 0)
        self.assertGreater(result["total_cgt_minimum_tax"], 0)
        self.assertEqual(result["cgt_core_policy_status"], "Enacted - Tax Reform No. 1 Act 2026")

    def test_confirmed_exemption_disables_top_up(self):
        result = calculate_cgt_minimum_tax_gap(
            taxable_income=20_000,
            minimum_tax_capital_gain=10_000,
            tax_schedule_key="2028_PLUS",
            exempt_from_minimum_tax=True,
        )
        self.assertEqual(result["cgt_minimum_tax_gap"], 0)


if __name__ == "__main__":
    unittest.main()
