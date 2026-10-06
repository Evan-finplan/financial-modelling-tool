import unittest

import pandas as pd

from debt_analysis import (
    allocate_cash_surplus,
    apply_debt_strategy_profile,
    build_debt_strategy_comparison_df,
)
from model import calculate_loan_year_from_annual_repayment, run_deterministic_projection


class DebtAllocationTests(unittest.TestCase):
    def test_entered_annual_repayment_pays_interest_then_principal(self):
        result = calculate_loan_year_from_annual_repayment(
            balance=500_000,
            annual_interest_rate=0.06,
            annual_repayment=50_000,
            offset_balance=100_000,
        )
        self.assertEqual(result["interest"], 24_000)
        self.assertEqual(result["principal"], 26_000)
        self.assertEqual(result["ending_balance"], 474_000)
    def test_surplus_stays_in_cash_when_non_super_module_is_disabled(self):
        result = allocate_cash_surplus(
            surplus=50_000,
            allocation_order=["cash_reserve", "non_super"],
            cash_reserve_balance=10_000,
            cash_reserve_target=20_000,
            allow_non_super_investment=False,
        )
        self.assertEqual(result["ending_cash_reserve_balance"], 60_000)
        self.assertEqual(result["surplus_cash_to_non_super"], 0)

    def test_cash_destination_retains_all_surplus_without_a_target_cap(self):
        result = allocate_cash_surplus(
            surplus=50_000,
            allocation_order=["cash"],
            cash_reserve_balance=10_000,
            cash_reserve_target=20_000,
            allow_non_super_investment=True,
        )
        self.assertEqual(result["ending_cash_reserve_balance"], 60_000)
        self.assertEqual(result["surplus_cash_to_non_super"], 0)

    def test_non_deductible_offset_is_filled_before_repayment(self):
        result = allocate_cash_surplus(
            surplus=80_000,
            allocation_order=["non_deductible_offset", "non_deductible_repayment", "non_super"],
            non_deductible_debt_balance=100_000,
            non_deductible_offset_balance=40_000,
        )
        self.assertEqual(result["non_deductible_offset_contribution"], 60_000)
        self.assertEqual(result["non_deductible_principal_repayment"], 0)
        self.assertEqual(result["surplus_cash_to_non_super"], 20_000)
        self.assertEqual(result["ending_non_deductible_offset_balance"], 100_000)

    def test_direct_repayment_preserves_existing_offset(self):
        result = allocate_cash_surplus(
            surplus=90_000,
            allocation_order=["non_deductible_repayment", "deductible_repayment", "non_super"],
            non_deductible_debt_balance=100_000,
            non_deductible_offset_balance=25_000,
            deductible_debt_balance=50_000,
            deductible_offset_balance=10_000,
        )
        self.assertEqual(result["non_deductible_principal_repayment"], 75_000)
        self.assertEqual(result["ending_non_deductible_debt_balance"], 25_000)
        self.assertEqual(result["deductible_principal_repayment"], 15_000)
        self.assertEqual(result["ending_non_deductible_offset_balance"], 25_000)

    def test_profile_changes_only_surplus_order_fields(self):
        base = {"annual_living_expenses": 100_000, "surplus_allocation_order": ["non_super"]}
        strategy = apply_debt_strategy_profile(base, "Offset First")
        self.assertEqual(strategy["annual_living_expenses"], 100_000)
        self.assertEqual(strategy["surplus_allocation_order"][0], "main_residence_offset")
        self.assertEqual(strategy["debt_strategy_name"], "Offset First")

    def test_surplus_can_be_split_across_home_property_and_investment_debt(self):
        result = allocate_cash_surplus(
            surplus=150_000,
            allocation_order=[
                "main_residence_offset",
                "property_loan_repayment",
                "investment_deductible_offset",
            ],
            main_residence_debt_balance=200_000,
            main_residence_offset_balance=150_000,
            deductible_debt_balance=300_000,
            deductible_offset_balance=0,
            investment_deductible_debt_balance=100_000,
            investment_deductible_offset_balance=50_000,
        )
        self.assertEqual(result["main_residence_offset_contribution"], 50_000)
        self.assertEqual(result["deductible_principal_repayment"], 100_000)
        self.assertEqual(result["investment_deductible_offset_contribution"], 0)
        self.assertEqual(result["ending_main_residence_offset_balance"], 200_000)
        self.assertEqual(result["ending_deductible_debt_balance"], 200_000)


class DebtComparisonTests(unittest.TestCase):
    def test_comparison_calculates_interest_saving_and_debt_free_year(self):
        base_df = pd.DataFrame({
            "financial_year_end": [2027, 2028],
            "total_wealth": [900_000, 950_000],
            "total_tax_paid": [20_000, 20_000],
            "total_debt_interest": [30_000, 25_000],
            "non_deductible_debt_balance": [50_000, 25_000],
            "non_deductible_offset_balance": [0, 0],
            "residential_property_loan_balance": [400_000, 390_000],
            "deductible_offset_balance": [0, 0],
            "main_residence_loan_balance": [500_000, 480_000],
            "main_residence_offset_balance": [20_000, 30_000],
            "investment_deductible_debt_balance": [80_000, 70_000],
            "investment_deductible_offset_balance": [5_000, 10_000],
        })
        strategy_df = base_df.copy()
        strategy_df["total_wealth"] = [905_000, 970_000]
        strategy_df["total_debt_interest"] = [25_000, 20_000]
        strategy_df["non_deductible_debt_balance"] = [25_000, 0]
        results = {
            "Current Strategy": {"det_df": base_df, "inputs": {"surplus_allocation_order": ["non_super"]}, "success_rate": 0.9},
            "Non-deductible First": {"det_df": strategy_df, "inputs": {"surplus_allocation_order": ["non_deductible_repayment"]}, "success_rate": 0.95},
        }
        comparison = build_debt_strategy_comparison_df(results)
        row = comparison.loc[comparison["scenario"] == "Non-deductible First"].iloc[0]
        self.assertEqual(row["interest_saved_vs_base"], 10_000)
        self.assertEqual(row["non_deductible_debt_free_year"], 2028)
        self.assertEqual(row["final_wealth_delta"], 20_000)
        self.assertEqual(row["ending_home_loan"], 480_000)
        self.assertEqual(row["ending_property_loan"], 390_000)
        self.assertEqual(row["ending_deductible_debt"], 70_000)
        self.assertEqual(row["ending_offset_balance"], 40_000)


class DebtProjectionIntegrationTests(unittest.TestCase):
    def test_projection_charges_non_deductible_interest_and_reduces_principal(self):
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
            "person1_annual_income": 300_000.0,
            "person2_annual_income": 0.0,
            "non_super_balance": 0.0,
            "non_super_cost_base": 0.0,
            "non_super_ownership_person1": 1.0,
            "annual_living_expenses": 80_000.0,
            "retirement_spending": 80_000.0,
            "inflation_rate": 0.03,
            "super_income_return_mean": 0.02,
            "super_income_return_std": 0.02,
            "super_capital_return_mean": 0.04,
            "super_capital_return_std": 0.09,
            "non_super_income_return_mean": 0.02,
            "non_super_income_return_std": 0.02,
            "non_super_capital_return_mean": 0.03,
            "non_super_capital_return_std": 0.08,
            "number_of_simulations": 10,
            "assumption_preset": "Custom",
            "contribution_events": [],
            "cash_reserve_balance": 0.0,
            "cash_reserve_floor": 0.0,
            "cash_reserve_target": 0.0,
            "withdrawal_order": ["cash", "non_super", "pension", "accumulation", "property"],
            "surplus_allocation_order": ["cash"],
            "non_deductible_debt_balance": 100_000.0,
            "non_deductible_offset_balance": 20_000.0,
            "non_deductible_interest_rate": 0.10,
            "non_deductible_annual_repayment": 20_000.0,
            "main_residence_value": 1_000_000.0,
            "main_residence_loan_balance": 500_000.0,
            "main_residence_interest_rate": 0.06,
            "main_residence_annual_loan_repayment": 50_000.0,
            "main_residence_offset_balance": 100_000.0,
            "investment_deductible_debt_balance": 200_000.0,
            "investment_deductible_interest_rate": 0.06,
            "investment_deductible_annual_repayment": 30_000.0,
            "investment_deductible_offset_balance": 0.0,
            "residential_property_enabled": False,
        }
        result = run_deterministic_projection(inputs).iloc[0]
        self.assertEqual(result["non_deductible_debt_interest"], 8_000)
        self.assertEqual(result["non_deductible_scheduled_principal"], 12_000)
        self.assertEqual(result["non_deductible_debt_balance"], 88_000)
        self.assertEqual(result["non_deductible_offset_balance"], 20_000)
        self.assertEqual(result["main_residence_loan_interest"], 24_000)
        self.assertEqual(result["main_residence_scheduled_principal"], 26_000)
        self.assertEqual(result["main_residence_loan_balance"], 474_000)
        self.assertEqual(result["investment_deductible_debt_interest"], 12_000)
        self.assertEqual(result["investment_deductible_scheduled_principal"], 18_000)
        self.assertEqual(result["investment_deductible_debt_balance"], 182_000)


if __name__ == "__main__":
    unittest.main()
