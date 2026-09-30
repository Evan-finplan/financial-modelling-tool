import unittest

import pandas as pd

from model import allocate_shortfall_by_asset_order
from strategy_analysis import apply_strategy_profile, build_strategy_comparison_df


class AssetDrawdownTests(unittest.TestCase):
    def test_cash_then_non_super_respects_reserve_floors(self):
        result = allocate_shortfall_by_asset_order(
            required_amount=100_000,
            withdrawal_order=["cash", "non_super", "pension", "accumulation", "property"],
            cash_balance=50_000,
            cash_floor=10_000,
            non_super_balance=100_000,
            non_super_floor=60_000,
            person1_accum_balance=50_000,
            person2_accum_balance=0,
            person1_pension_balance=50_000,
            person2_pension_balance=0,
        )
        self.assertEqual(result["cash_withdrawal"], 40_000)
        self.assertEqual(result["non_super_withdrawal"], 40_000)
        self.assertEqual(result["person1_extra_pension_withdrawal"], 20_000)
        self.assertEqual(result["unfunded_after_assets"], 0)

    def test_property_source_supports_partial_disposal(self):
        result = allocate_shortfall_by_asset_order(
            required_amount=190_000,
            withdrawal_order=["property"],
            cash_balance=0,
            cash_floor=0,
            non_super_balance=0,
            non_super_floor=0,
            person1_accum_balance=0,
            person2_accum_balance=0,
            person1_pension_balance=0,
            person2_pension_balance=0,
            property_value=1_000_000,
            property_loan_balance=400_000,
            property_sale_cost_rate=0.05,
        )
        self.assertAlmostEqual(result["residential_property_sale_proceeds"], 190_000)
        self.assertAlmostEqual(result["residential_property_disposal_fraction"], 190_000 / 550_000)
        self.assertLess(result["remaining_residential_property_value"], 1_000_000)
        self.assertLess(result["remaining_residential_property_loan_balance"], 400_000)


class StrategyComparisonTests(unittest.TestCase):
    def test_profiles_only_change_strategy_inputs(self):
        base = {"annual_living_expenses": 100_000, "non_super_estate_reserve": 250_000}
        strategy = apply_strategy_profile(base, "Strategy A")
        self.assertEqual(strategy["annual_living_expenses"], 100_000)
        self.assertEqual(strategy["withdrawal_order"][1], "pension")
        self.assertEqual(strategy["non_super_estate_reserve"], 0.0)

    def test_comparison_reports_advantage_and_tax_delta(self):
        base_df = pd.DataFrame({
            "financial_year_end": [2027, 2028],
            "person1_age": [59, 60],
            "total_wealth": [1_000_000, 900_000],
            "total_tax_paid": [10_000, 10_000],
            "household_cash_available_before_extra_withdrawals": [100_000, 100_000],
            "unmet_shortfall": [0, 0],
        })
        strategy_df = base_df.copy()
        strategy_df["total_wealth"] = [990_000, 950_000]
        strategy_df["total_tax_paid"] = [9_000, 9_000]
        results = {
            "Base Case": {"det_df": base_df, "inputs": {"person1_retirement_age": 60}, "success_rate": 0.9, "p10_final_wealth": 100_000},
            "Strategy A": {"det_df": strategy_df, "inputs": {"person1_retirement_age": 60}, "success_rate": 0.95, "p10_final_wealth": 150_000},
        }
        comparison = build_strategy_comparison_df(results)
        row = comparison.loc[comparison["scenario"] == "Strategy A"].iloc[0]
        self.assertEqual(row["final_wealth_delta"], 50_000)
        self.assertEqual(row["cumulative_tax_delta"], -2_000)
        self.assertEqual(row["first_advantage_year"], 2028)
        self.assertEqual(row["break_even_year"], 2028)


if __name__ == "__main__":
    unittest.main()
