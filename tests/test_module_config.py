import unittest

from module_config import apply_module_scope, module_flags_from_inputs


class ModuleConfigTests(unittest.TestCase):
    def test_disabled_property_and_trust_are_removed_from_calculation_scope(self):
        scoped = apply_module_scope({
            "module_property_enabled": False,
            "module_trust_enabled": False,
            "module_deductible_debt_enabled": True,
            "residential_property_enabled": True,
            "residential_property_value": 1_000_000,
            "residential_property_loan_balance": 600_000,
            "deductible_offset_balance": 100_000,
            "discretionary_trust_enabled": True,
            "discretionary_trust_net_income": 200_000,
            "withdrawal_order": ["cash", "property", "non_super"],
            "surplus_allocation_order": ["deductible_repayment", "non_super"],
        })
        self.assertFalse(scoped["residential_property_enabled"])
        self.assertEqual(scoped["residential_property_value"], 0)
        self.assertEqual(scoped["residential_property_loan_balance"], 0)
        self.assertEqual(scoped["deductible_offset_balance"], 0)
        self.assertFalse(scoped["discretionary_trust_enabled"])
        self.assertEqual(scoped["discretionary_trust_net_income"], 0)
        self.assertNotIn("property", scoped["withdrawal_order"])
        self.assertNotIn("deductible_repayment", scoped["surplus_allocation_order"])

    def test_disabled_super_and_non_super_neutralise_balances_and_orders(self):
        scoped = apply_module_scope({
            "module_super_enabled": False,
            "module_pension_enabled": True,
            "module_non_super_enabled": False,
            "person1_accum_super_balance": 500_000,
            "person1_pension_super_balance": 200_000,
            "person1_transfer_balance_cap": 2_000_000,
            "non_super_balance": 700_000,
            "non_super_cost_base": 400_000,
            "contribution_events": [{"amount": 20_000}],
            "withdrawal_order": ["cash", "non_super", "pension", "accumulation"],
            "surplus_allocation_order": ["non_super"],
        })
        self.assertEqual(scoped["person1_accum_super_balance"], 0)
        self.assertEqual(scoped["person1_pension_super_balance"], 0)
        self.assertEqual(scoped["contribution_events"], [])
        self.assertFalse(scoped["module_pension_enabled"])
        self.assertEqual(scoped["non_super_balance"], 0)
        self.assertEqual(scoped["withdrawal_order"], ["cash"])
        self.assertEqual(scoped["surplus_allocation_order"], ["cash_reserve"])

    def test_legacy_inputs_infer_existing_optional_modules(self):
        flags = module_flags_from_inputs({
            "household_mode": "One Person",
            "residential_property_enabled": True,
            "residential_property_loan_balance": 500_000,
            "discretionary_trust_enabled": True,
            "non_deductible_debt_balance": 100_000,
        })
        self.assertFalse(flags["module_second_person_enabled"])
        self.assertTrue(flags["module_property_enabled"])
        self.assertTrue(flags["module_deductible_debt_enabled"])
        self.assertTrue(flags["module_trust_enabled"])
        self.assertTrue(flags["module_non_deductible_debt_enabled"])

    def test_new_modules_are_scoped_independently(self):
        scoped = apply_module_scope({
            "module_cash_surplus_enabled": False,
            "module_investment_debt_enabled": False,
            "module_property_enabled": True,
            "main_residence_loan_balance": 800_000,
            "residential_property_loan_balance": 600_000,
            "investment_deductible_debt_balance": 300_000,
            "non_deductible_debt_balance": 100_000,
            "surplus_allocation_order": ["investment_deductible_repayment", "property_loan_repayment"],
        })
        self.assertEqual(scoped["main_residence_loan_balance"], 800_000)
        self.assertEqual(scoped["residential_property_loan_balance"], 600_000)
        self.assertEqual(scoped["investment_deductible_debt_balance"], 0)
        self.assertEqual(scoped["non_deductible_debt_balance"], 0)
        self.assertEqual(scoped["surplus_allocation_order"], ["cash"])


if __name__ == "__main__":
    unittest.main()
