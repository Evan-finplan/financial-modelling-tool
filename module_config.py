"""Client module selection and calculation-scope normalisation."""

from __future__ import annotations

import copy


MODULE_DEFAULTS = {
    "module_second_person_enabled": True,
    "module_super_enabled": True,
    "module_pension_enabled": True,
    "module_non_super_enabled": True,
    "module_property_enabled": False,
    "module_trust_enabled": False,
    "module_cash_surplus_enabled": True,
    "module_investment_debt_enabled": False,
    "module_non_deductible_debt_enabled": False,
    "module_deductible_debt_enabled": False,
}


MODULE_LABELS = {
    "module_second_person_enabled": ("Second household member", "第二位家庭成员"),
    "module_super_enabled": ("Super accumulation", "Super 累积账户"),
    "module_pension_enabled": ("Pension accounts", "Pension 账户"),
    "module_non_super_enabled": ("Non-super investments", "非 Super 投资"),
    "module_property_enabled": ("Residential investment property", "住宅投资物业"),
    "module_trust_enabled": ("Discretionary trust", "Discretionary Trust"),
    "module_cash_surplus_enabled": ("Cash surplus strategy", "现金盈余策略"),
    "module_investment_debt_enabled": ("Investment debt", "投资债务"),
}


def module_flags_from_inputs(inputs):
    """Return flags with backwards-compatible inference for saved/imported inputs."""
    flags = dict(MODULE_DEFAULTS)
    flags.update({key: bool(inputs[key]) for key in MODULE_DEFAULTS if key in inputs})
    if "module_second_person_enabled" not in inputs:
        flags["module_second_person_enabled"] = inputs.get("household_mode", "Two People") != "One Person"
    if "module_property_enabled" not in inputs:
        flags["module_property_enabled"] = bool(inputs.get("residential_property_enabled", False))
    if "module_trust_enabled" not in inputs:
        flags["module_trust_enabled"] = bool(inputs.get("discretionary_trust_enabled", False))
    if "module_non_deductible_debt_enabled" not in inputs:
        flags["module_non_deductible_debt_enabled"] = float(inputs.get("non_deductible_debt_balance", 0.0)) > 0
    if "module_deductible_debt_enabled" not in inputs:
        flags["module_deductible_debt_enabled"] = (
            flags["module_property_enabled"]
            and float(inputs.get("residential_property_loan_balance", 0.0)) > 0
        )
    # Pension phase is a permanent part of the retirement model. A zero opening
    # pension balance means the client has no pension account today; it must not
    # disable a future accumulation-to-pension transfer at pension start age.
    flags["module_pension_enabled"] = True
    if "module_investment_debt_enabled" not in inputs:
        flags["module_investment_debt_enabled"] = bool(
            flags["module_non_deductible_debt_enabled"]
            or float(inputs.get("investment_deductible_debt_balance", 0.0)) > 0
        )
    flags["module_non_deductible_debt_enabled"] = flags["module_investment_debt_enabled"]
    flags["module_deductible_debt_enabled"] = flags["module_investment_debt_enabled"]
    return flags


def apply_module_scope(inputs):
    """Return calculation inputs with inactive modules safely neutralised."""
    scoped = copy.deepcopy(inputs)
    flags = module_flags_from_inputs(scoped)
    scoped.update(flags)

    if not flags["module_second_person_enabled"]:
        scoped["household_mode"] = "One Person"

    if not flags["module_super_enabled"]:
        for person in ("person1", "person2"):
            for suffix in ("accum_super_balance", "accum_super_cost_base"):
                scoped[f"{person}_{suffix}"] = 0.0
        scoped["contribution_events"] = []

    if not flags["module_non_super_enabled"]:
        for key in (
            "non_super_balance",
            "non_super_cost_base",
            "non_super_transition_value_2027",
            "non_super_opening_capital_losses",
            "non_super_estate_reserve",
        ):
            scoped[key] = 0.0
        scoped["cgt_reform_enabled"] = False

    if flags["module_property_enabled"]:
        scoped["residential_property_enabled"] = True
    else:
        scoped["residential_property_enabled"] = False
        for key in (
            "residential_property_value",
            "residential_property_loan_balance",
            "residential_property_annual_loan_repayment",
            "residential_property_gross_rent",
            "residential_property_operating_expenses",
            "residential_property_opening_quarantined_loss",
            "property_estate_reserve",
            "deductible_offset_balance",
        ):
            scoped[key] = 0.0

    if not flags["module_deductible_debt_enabled"]:
        scoped["investment_deductible_debt_balance"] = 0.0
        scoped["investment_deductible_offset_balance"] = 0.0
        scoped["investment_deductible_annual_repayment"] = 0.0

    if flags["module_trust_enabled"]:
        scoped["discretionary_trust_enabled"] = True
    else:
        scoped["discretionary_trust_enabled"] = False
        scoped["discretionary_trust_balance"] = 0.0
        scoped["discretionary_trust_cost_base"] = 0.0
        scoped["discretionary_trust_net_income"] = 0.0
        scoped["discretionary_trust_excluded_income"] = 0.0

    if not flags["module_non_deductible_debt_enabled"]:
        scoped["non_deductible_debt_balance"] = 0.0
        scoped["non_deductible_offset_balance"] = 0.0
        scoped["non_deductible_annual_repayment"] = 0.0

    if not flags["module_cash_surplus_enabled"]:
        scoped["surplus_allocation_order"] = ["cash"]

    withdrawal_order = list(scoped.get("withdrawal_order", []))
    allowed_withdrawals = {"cash"}
    if flags["module_non_super_enabled"]:
        allowed_withdrawals.add("non_super")
    if flags["module_super_enabled"]:
        allowed_withdrawals.add("accumulation")
    allowed_withdrawals.add("pension")
    if flags["module_property_enabled"]:
        allowed_withdrawals.add("property")
    scoped["withdrawal_order"] = [item for item in withdrawal_order if item in allowed_withdrawals]
    if "cash" not in scoped["withdrawal_order"]:
        scoped["withdrawal_order"].insert(0, "cash")

    allocation_order = list(scoped.get("surplus_allocation_order", []))
    allowed_allocations = {"cash_reserve", "cash"}
    if flags["module_non_super_enabled"]:
        allowed_allocations.add("non_super")
    if flags["module_non_deductible_debt_enabled"]:
        allowed_allocations.update({"non_deductible_offset", "non_deductible_repayment"})
    if flags["module_deductible_debt_enabled"]:
        allowed_allocations.update({"investment_deductible_offset", "investment_deductible_repayment"})
    if flags["module_property_enabled"]:
        allowed_allocations.update({"property_loan_offset", "property_loan_repayment"})
    allowed_allocations.update({"main_residence_offset", "main_residence_repayment"})
    scoped["surplus_allocation_order"] = [item for item in allocation_order if item in allowed_allocations]
    if not scoped["surplus_allocation_order"]:
        scoped["surplus_allocation_order"] = ["non_super" if flags["module_non_super_enabled"] else "cash_reserve"]

    return scoped


def active_module_names(inputs, is_chinese=False):
    flags = module_flags_from_inputs(inputs)
    index = 1 if is_chinese else 0
    return [MODULE_LABELS[key][index] for key, enabled in flags.items() if enabled and key in MODULE_LABELS]
