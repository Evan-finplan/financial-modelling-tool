import unittest

from model import (
    calculate_discretionary_trust_minimum_tax,
    calculate_incremental_budget_tax,
    calculate_residential_property_year,
    run_deterministic_projection,
    validate_inputs,
)


class ResidentialPropertyTaxTests(unittest.TestCase):
    def test_affected_established_property_loss_is_quarantined_from_2028(self):
        result = calculate_residential_property_year(
            gross_rent=30_000,
            deductible_operating_expenses=10_000,
            loan_interest=35_000,
            opening_quarantined_loss=0,
            financial_year_end=2028,
        )
        self.assertTrue(result["restriction_applies"])
        self.assertEqual(result["net_cashflow"], -15_000)
        self.assertEqual(result["taxable_rental_income"], 0)
        self.assertEqual(result["closing_quarantined_loss"], 15_000)

    def test_grandfathered_property_keeps_negative_gearing(self):
        result = calculate_residential_property_year(
            gross_rent=30_000,
            deductible_operating_expenses=10_000,
            loan_interest=35_000,
            opening_quarantined_loss=0,
            financial_year_end=2028,
            acquired_before_budget_time=True,
        )
        self.assertFalse(result["restriction_applies"])
        self.assertEqual(result["taxable_rental_income"], -15_000)

    def test_new_build_keeps_negative_gearing(self):
        result = calculate_residential_property_year(
            gross_rent=30_000,
            deductible_operating_expenses=10_000,
            loan_interest=35_000,
            opening_quarantined_loss=0,
            financial_year_end=2028,
            is_new_build=True,
        )
        self.assertFalse(result["restriction_applies"])
        self.assertEqual(result["taxable_rental_income"], -15_000)

    def test_carried_loss_offsets_future_residential_income(self):
        result = calculate_residential_property_year(
            gross_rent=50_000,
            deductible_operating_expenses=10_000,
            loan_interest=20_000,
            opening_quarantined_loss=12_000,
            financial_year_end=2029,
        )
        self.assertEqual(result["quarantined_loss_used"], 12_000)
        self.assertEqual(result["taxable_rental_income"], 8_000)
        self.assertEqual(result["closing_quarantined_loss"], 0)


class DiscretionaryTrustMinimumTaxTests(unittest.TestCase):
    def test_minimum_tax_starts_in_2029_fy(self):
        before = calculate_discretionary_trust_minimum_tax(100_000, 0, 2028)
        after = calculate_discretionary_trust_minimum_tax(100_000, 0, 2029)
        self.assertFalse(before["minimum_tax_applies"])
        self.assertEqual(before["trustee_minimum_tax"], 0)
        self.assertTrue(after["minimum_tax_applies"])
        self.assertEqual(after["trustee_minimum_tax"], 30_000)

    def test_excluded_income_is_removed_from_minimum_tax_base(self):
        result = calculate_discretionary_trust_minimum_tax(100_000, 25_000, 2029)
        self.assertEqual(result["minimum_tax_income"], 75_000)
        self.assertEqual(result["trustee_minimum_tax"], 22_500)

    def test_non_refundable_credit_is_capped_at_tax_on_trust_income(self):
        result = calculate_incremental_budget_tax(
            base_taxable_income=0,
            residential_taxable_income=0,
            trust_taxable_income=10_000,
            trust_tax_credit=3_000,
            tax_schedule_key="2028_PLUS",
        )
        self.assertLess(result["trust_tax_credit"], 3_000)
        self.assertEqual(result["trust_tax_after_credit"], 0)


def one_person_projection_inputs():
    return {
        "start_financial_year": 2029,
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
        "number_of_simulations": 10,
        "assumption_preset": "Custom",
        "contribution_events": [],
        "residential_property_enabled": True,
        "residential_property_value": 1_000_000.0,
        "residential_property_loan_balance": 600_000.0,
        "residential_property_gross_rent": 30_000.0,
        "residential_property_operating_expenses": 10_000.0,
        "residential_property_interest_rate": 0.06,
        "residential_property_capital_growth_rate": 0.03,
        "residential_property_rent_growth_rate": 0.0,
        "residential_property_expense_growth_rate": 0.0,
        "residential_property_opening_quarantined_loss": 0.0,
        "residential_property_acquired_before_budget_time": False,
        "residential_property_is_new_build": False,
        "residential_property_is_exempt_housing": False,
        "residential_property_ownership_person1": 1.0,
        "discretionary_trust_enabled": True,
        "discretionary_trust_net_income": 100_000.0,
        "discretionary_trust_excluded_income": 20_000.0,
        "discretionary_trust_income_growth_rate": 0.0,
        "discretionary_trust_subject_to_minimum_tax": True,
        "discretionary_trust_ownership_person1": 1.0,
    }


class Phase2ProjectionIntegrationTests(unittest.TestCase):
    def test_projection_reports_property_loss_pool_and_trust_tax(self):
        result = run_deterministic_projection(one_person_projection_inputs()).iloc[0]
        self.assertEqual(result["residential_property_current_year_quarantined_loss"], 16_000)
        self.assertEqual(result["closing_residential_property_quarantined_loss"], 16_000)
        self.assertEqual(result["discretionary_trust_minimum_tax_income"], 80_000)
        self.assertEqual(result["discretionary_trust_trustee_minimum_tax"], 24_000)
        self.assertEqual(result["ending_residential_property_value"], 1_030_000)
        self.assertEqual(result["residential_property_net_equity"], 430_000)
        self.assertIn("Exposure draft", result["discretionary_trust_policy_status"])

    def test_property_loan_uses_principal_and_interest_repayments(self):
        inputs = one_person_projection_inputs()
        inputs["residential_property_annual_loan_repayment"] = 60_000
        result = run_deterministic_projection(inputs).iloc[0]
        self.assertEqual(result["residential_property_loan_interest"], 36_000)
        self.assertEqual(result["residential_property_scheduled_principal"], 24_000)
        self.assertLessEqual(result["residential_property_loan_balance"], 576_000)

    def test_trust_asset_pool_derives_income_and_tracks_balance_and_cost_base(self):
        inputs = one_person_projection_inputs()
        inputs.update({
            "discretionary_trust_balance": 1_000_000.0,
            "discretionary_trust_cost_base": 600_000.0,
            "discretionary_trust_income_return_mean": 0.04,
            "discretionary_trust_income_return_std": 0.0,
            "discretionary_trust_capital_return_mean": 0.05,
            "discretionary_trust_capital_return_std": 0.0,
            "discretionary_trust_excluded_income_pct": 0.25,
        })
        result = run_deterministic_projection(inputs).iloc[0]
        self.assertEqual(result["discretionary_trust_net_income"], 40_000)
        self.assertEqual(result["discretionary_trust_excluded_income"], 10_000)
        self.assertEqual(result["discretionary_trust_trustee_minimum_tax"], 9_000)
        self.assertEqual(result["ending_discretionary_trust_balance"], 1_050_000)
        self.assertEqual(result["ending_discretionary_trust_cost_base"], 600_000)

    def test_negative_gearing_cannot_create_negative_personal_or_total_tax(self):
        inputs = one_person_projection_inputs()
        inputs.update({
            "start_financial_year": 2027,
            "person1_annual_income": 0.0,
            "person1_accum_super_balance": 0.0,
            "person1_accum_super_cost_base": 0.0,
            "non_super_balance": 250_000.0,
            "non_super_cost_base": 250_000.0,
            "residential_property_value": 1_500_000.0,
            "residential_property_loan_balance": 1_200_000.0,
            "residential_property_gross_rent": 45_000.0,
            "residential_property_operating_expenses": 18_000.0,
            "residential_property_interest_rate": 0.065,
            "residential_property_annual_loan_repayment": 95_000.0,
            "residential_property_acquired_before_budget_time": True,
            "discretionary_trust_enabled": False,
        })
        result = run_deterministic_projection(inputs).iloc[0]
        self.assertEqual(result["person1_total_taxable_income"], 0.0)
        self.assertLess(result["person1_property_tax_adjustment"], 0.0)
        self.assertGreaterEqual(result["person1_personal_tax_total"], 0.0)
        self.assertGreaterEqual(result["total_tax_paid"], 0.0)

    def test_validation_rejects_unreasonable_age_and_super_cost_base(self):
        inputs = one_person_projection_inputs()
        inputs["person1_current_age"] = 130
        inputs["person1_accum_super_cost_base"] = 2_000_000.0
        errors = validate_inputs(inputs)
        self.assertIn("person1_current_age must be between 18 and 100.", errors)
        self.assertIn(
            "person1_accum_super_cost_base cannot exceed person1_accum_super_balance under the current pooled cost-base setup.",
            errors,
        )

    def test_validation_allows_already_retired_client(self):
        inputs = one_person_projection_inputs()
        inputs["person1_current_age"] = 70
        inputs["person1_retirement_age"] = 65
        self.assertEqual(validate_inputs(inputs), [])

    def test_legacy_disabled_pension_flag_still_transfers_at_start_age(self):
        inputs = one_person_projection_inputs()
        inputs.update({
            "start_financial_year": 2027,
            "projection_years": 3,
            "person1_current_age": 65,
            "person1_retirement_age": 67,
            "person1_pension_start_age": 67,
            "module_pension_enabled": False,
            "residential_property_enabled": False,
            "discretionary_trust_enabled": False,
        })
        result = run_deterministic_projection(inputs)
        pension_start = result.loc[result["person1_started_pension_this_year"]]
        self.assertEqual(len(pension_start), 1)
        self.assertEqual(int(pension_start.iloc[0]["person1_age"]), 67)
        self.assertGreater(float(pension_start.iloc[0]["person1_transfer_to_pension"]), 0.0)


if __name__ == "__main__":
    unittest.main()
