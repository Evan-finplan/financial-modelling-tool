import copy

import numpy as np
import pandas as pd

from debt_analysis import allocate_cash_surplus
from module_config import apply_module_scope
from policy import (
    CGT_CORE_POLICY_STATUS,
    CGT_MINIMUM_TAX_RATE,
    CGT_REFORM_START_FY,
    CGT_TRANSITION_METHOD_STATUS,
    DISCRETIONARY_TRUST_MINIMUM_TAX_RATE,
    DISCRETIONARY_TRUST_MINIMUM_TAX_START_FY,
    DISCRETIONARY_TRUST_MINIMUM_TAX_STATUS,
    DIVISION_293_TAX_RATE,
    DIVISION_293_THRESHOLD,
    LATEST_PUBLISHED_SUPER_THRESHOLD_FY,
    MEDICARE_LEVY_RATE,
    NEGATIVE_GEARING_RESTRICTION_START_FY,
    PERSONAL_TAX_SCHEDULES,
    SUPER_CONTRIBUTIONS_TAX_RATE,
    SUPER_EARNINGS_TAX_RATE,
    SUPER_GUARANTEE_RATE,
    get_concessional_contributions_cap,
    get_non_concessional_contributions_cap,
    get_policy_snapshot,
    get_super_guarantee_maximum_earnings_base,
)


# ============================================================
# SECTION: TAX CONFIGURATION
# ============================================================


# ============================================================
# SECTION: ASSUMPTION PRESETS
# ============================================================

def get_assumption_presets():
    return {
        "Conservative": {
            "super_income_return_mean": 0.020,
            "super_income_return_std": 0.020,
            "super_capital_return_mean": 0.030,
            "super_capital_return_std": 0.100,
            "non_super_income_return_mean": 0.015,
            "non_super_income_return_std": 0.015,
            "non_super_capital_return_mean": 0.025,
            "non_super_capital_return_std": 0.090,
            "inflation_rate": 0.035,
        },
        "Base Case": {
            "super_income_return_mean": 0.020,
            "super_income_return_std": 0.020,
            "super_capital_return_mean": 0.040,
            "super_capital_return_std": 0.090,
            "non_super_income_return_mean": 0.020,
            "non_super_income_return_std": 0.020,
            "non_super_capital_return_mean": 0.030,
            "non_super_capital_return_std": 0.080,
            "inflation_rate": 0.030,
        },
        "Optimistic": {
            "super_income_return_mean": 0.020,
            "super_income_return_std": 0.020,
            "super_capital_return_mean": 0.055,
            "super_capital_return_std": 0.100,
            "non_super_income_return_mean": 0.020,
            "non_super_income_return_std": 0.020,
            "non_super_capital_return_mean": 0.045,
            "non_super_capital_return_std": 0.090,
            "inflation_rate": 0.025,
        },
    }


def apply_preset_to_inputs(base_inputs, preset_name, preset_values=None):
    presets = preset_values if preset_values is not None else get_assumption_presets()
    updated_inputs = copy.deepcopy(base_inputs)

    if preset_name in presets:
        updated_inputs.update(copy.deepcopy(presets[preset_name]))

    updated_inputs["assumption_preset"] = preset_name
    return updated_inputs


# ============================================================
# SECTION: FINANCIAL YEAR HELPERS
# ============================================================

def parse_financial_year_label(financial_year_value):
    if isinstance(financial_year_value, (int, float)):
        return int(financial_year_value)

    value = str(financial_year_value).upper().replace(" ", "").replace("FY", "")
    return int(value)


def format_financial_year_label(financial_year_end_year):
    return f"{int(financial_year_end_year)}FY"


def get_financial_year_end(start_financial_year, year_index):
    return parse_financial_year_label(start_financial_year) + int(year_index)


def get_tax_schedule_key_for_financial_year(financial_year_end):
    fy_end = int(financial_year_end)

    if fy_end <= 2026:
        return 2026
    if fy_end == 2027:
        return 2027
    return "2028_PLUS"


# ============================================================
# SECTION: HOUSEHOLD MODE HELPERS
# ============================================================

def is_one_person_mode(inputs):
    return str(inputs.get("household_mode", "Two People")) == "One Person"


def normalise_household_inputs(inputs):
    normalized = apply_module_scope(copy.deepcopy(inputs))

    household_mode = str(normalized.get("household_mode", "Two People"))
    normalized["household_mode"] = household_mode

    if household_mode == "One Person":
        normalized["person2_name"] = ""
        normalized["person2_current_age"] = 0
        normalized["person2_retirement_age"] = 0
        normalized["person2_pension_start_age"] = 0

        normalized["person2_accum_super_balance"] = 0.0
        normalized["person2_pension_super_balance"] = 0.0
        normalized["person2_accum_super_cost_base"] = 0.0
        normalized["person2_pension_super_cost_base"] = 0.0
        normalized["person2_transfer_balance_cap"] = 0.0
        normalized["person2_annual_income"] = 0.0

        normalized["non_super_ownership_person1"] = 1.0
        normalized["retirement_spending_trigger"] = "Either Retired"

        events_df = normalise_contribution_events(
            normalized.get("contribution_events", []),
            household_mode=household_mode,
        )
        normalized["contribution_events"] = events_df.to_dict(orient="records")

    return normalized


# ============================================================
# SECTION: CONTRIBUTION EVENT HELPERS
# ============================================================

def normalise_contribution_events(contribution_events, household_mode="Two People"):
    if contribution_events is None:
        return pd.DataFrame(columns=["financial_year", "person", "contribution_type", "amount"])

    if isinstance(contribution_events, pd.DataFrame):
        df = contribution_events.copy()
    else:
        df = pd.DataFrame(contribution_events)

    if df.empty:
        return pd.DataFrame(columns=["financial_year", "person", "contribution_type", "amount"])

    expected_cols = ["financial_year", "person", "contribution_type", "amount"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df = df[expected_cols].copy()
    df["financial_year"] = df["financial_year"].astype(str).str.upper().str.replace(" ", "", regex=False)
    df["person"] = df["person"].astype(str)
    df["contribution_type"] = df["contribution_type"].astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    df = df[df["amount"] != 0].copy()

    if str(household_mode) == "One Person":
        df = df[df["person"] != "Person 2"].copy()

    return df.reset_index(drop=True)


def build_contribution_event_lookup(contribution_events, household_mode="Two People"):
    df = normalise_contribution_events(contribution_events, household_mode=household_mode)

    if df.empty:
        return {}

    grouped = (
        df.groupby(["financial_year", "person", "contribution_type"], as_index=False)["amount"]
        .sum()
    )

    lookup = {
        (row["financial_year"], row["person"], row["contribution_type"]): float(row["amount"])
        for _, row in grouped.iterrows()
    }
    return lookup


def get_scheduled_contribution_amount(event_lookup, financial_year_label, person, contribution_type):
    if not event_lookup:
        return 0.0

    fy_key = str(financial_year_label).upper().replace(" ", "").replace("FY", "")

    return float(
        event_lookup.get((fy_key, person, contribution_type), 0.0)
    )


# ============================================================
# SECTION: AGE AND PHASE HELPERS
# ============================================================

def get_person_age(start_age, year_index):
    return int(start_age) + int(year_index)


def get_person_phase(age, retirement_age, pension_start_age):
    is_working = age < retirement_age
    is_pension_phase = age >= pension_start_age

    if is_working and is_pension_phase:
        return "working_pension"
    if is_working:
        return "working"
    if is_pension_phase:
        return "retired_pension"
    return "retired_pre_pension"


# ============================================================
# SECTION: RETIREMENT SPENDING TRIGGER HELPER
# ============================================================

def should_use_retirement_spending(
    person1_age,
    person2_age,
    person1_retirement_age,
    person2_retirement_age,
    retirement_spending_trigger,
):
    if retirement_spending_trigger == "Either Retired":
        return (
            person1_age >= person1_retirement_age
            or person2_age >= person2_retirement_age
        )

    return (
        person1_age >= person1_retirement_age
        and person2_age >= person2_retirement_age
    )


# ============================================================
# SECTION: PROJECTION CONTEXT HELPERS
# ============================================================

def build_projection_context(inputs):
    year_rows = []
    one_person_mode = is_one_person_mode(inputs)

    for year_index in range(int(inputs["projection_years"])):
        financial_year_end = get_financial_year_end(inputs["start_financial_year"], year_index)
        financial_year_label = format_financial_year_label(financial_year_end)
        tax_schedule_key = get_tax_schedule_key_for_financial_year(financial_year_end)

        person1_age = get_person_age(inputs["person1_current_age"], year_index)
        person2_age = 0 if one_person_mode else get_person_age(inputs["person2_current_age"], year_index)

        person1_is_working = person1_age < inputs["person1_retirement_age"]
        person1_is_pension_phase = person1_age >= inputs["person1_pension_start_age"]

        if one_person_mode:
            person2_is_working = False
            person2_is_pension_phase = False
        else:
            person2_is_working = person2_age < inputs["person2_retirement_age"]
            person2_is_pension_phase = person2_age >= inputs["person2_pension_start_age"]

        def get_phase(is_working, is_pension):
            if is_working and is_pension:
                return "working_pension"
            if is_working:
                return "working"
            if is_pension:
                return "retired_pension"
            return "retired_pre_pension"

        person1_phase = get_phase(person1_is_working, person1_is_pension_phase)
        person2_phase = get_phase(person2_is_working, person2_is_pension_phase)

        person1_income_indexed = (
            inputs["person1_annual_income"] * ((1 + inputs["inflation_rate"]) ** year_index)
            if person1_is_working else 0.0
        )
        person2_income_indexed = (
            0.0 if one_person_mode else
            inputs["person2_annual_income"] * ((1 + inputs["inflation_rate"]) ** year_index)
            if person2_is_working else 0.0
        )

        use_retirement_spending = should_use_retirement_spending(
            person1_age=person1_age,
            person2_age=person2_age,
            person1_retirement_age=inputs["person1_retirement_age"],
            person2_retirement_age=(999 if one_person_mode else inputs["person2_retirement_age"]),
            retirement_spending_trigger=inputs["retirement_spending_trigger"],
        )

        indexed_retirement_spending = inputs["retirement_spending"] * ((1 + inputs["inflation_rate"]) ** year_index)

        year_rows.append(
            {
                "year_index": year_index,
                "financial_year_end": financial_year_end,
                "financial_year_label": financial_year_label,
                "tax_schedule_key": tax_schedule_key,
                "household_mode": inputs.get("household_mode", "Two People"),
                "person1_age": person1_age,
                "person2_age": person2_age,
                "person1_phase": person1_phase,
                "person2_phase": person2_phase,
                "person1_is_working": person1_is_working,
                "person2_is_working": person2_is_working,
                "person1_is_pension_phase": person1_is_pension_phase,
                "person2_is_pension_phase": person2_is_pension_phase,
                "person1_income_indexed": person1_income_indexed,
                "person2_income_indexed": person2_income_indexed,
                "use_retirement_spending": use_retirement_spending,
                "indexed_retirement_spending": indexed_retirement_spending,
            }
        )

    return {
        "year_rows": year_rows,
    }

# ============================================================
# SECTION: TAX HELPERS
# ============================================================

def calculate_progressive_income_tax(taxable_income, tax_schedule_key):
    taxable_income = max(float(taxable_income), 0.0)
    brackets = PERSONAL_TAX_SCHEDULES[tax_schedule_key]

    tax = 0.0
    for lower, upper, rate in brackets:
        if taxable_income <= lower:
            continue
        taxable_amount_in_bracket = min(taxable_income, upper) - lower
        tax += taxable_amount_in_bracket * rate

    return max(tax, 0.0)


def calculate_medicare_levy(taxable_income):
    taxable_income = max(float(taxable_income), 0.0)
    return taxable_income * MEDICARE_LEVY_RATE


def calculate_personal_income_tax(taxable_income, tax_schedule_key):
    income_tax = calculate_progressive_income_tax(
        taxable_income=taxable_income,
        tax_schedule_key=tax_schedule_key,
    )
    medicare_levy = calculate_medicare_levy(taxable_income)

    return {
        "income_tax": income_tax,
        "medicare_levy": medicare_levy,
        "personal_tax_total": income_tax + medicare_levy,
    }


def allocate_tax_proportionally(total_tax, components_dict):
    positive_components = {
        key: max(float(value), 0.0)
        for key, value in components_dict.items()
    }
    total_positive = sum(positive_components.values())

    if total_positive <= 0:
        return {key: 0.0 for key in components_dict.keys()}

    return {
        key: total_tax * value / total_positive
        for key, value in positive_components.items()
    }


def split_income_tax_and_medicare(allocated_tax, tax_result):
    total_personal_tax = tax_result["personal_tax_total"]

    if total_personal_tax <= 0:
        return 0.0, 0.0

    income_tax_component = allocated_tax * tax_result["income_tax"] / total_personal_tax
    medicare_component = allocated_tax * tax_result["medicare_levy"] / total_personal_tax
    return income_tax_component, medicare_component


def calculate_residential_property_year(
    gross_rent,
    deductible_operating_expenses,
    loan_interest,
    opening_quarantined_loss,
    financial_year_end,
    acquired_before_budget_time=False,
    is_new_build=False,
    is_exempt_housing=False,
):
    """Apply the legislated residential loss-quarantine rules for one year.

    The model aggregates one residential investment activity. It does not
    attempt property-by-property ordering or CGT treatment on disposal.
    """
    gross_rent = max(float(gross_rent), 0.0)
    deductible_operating_expenses = max(float(deductible_operating_expenses), 0.0)
    loan_interest = max(float(loan_interest), 0.0)
    opening_quarantined_loss = max(float(opening_quarantined_loss), 0.0)
    total_deductions = deductible_operating_expenses + loan_interest
    net_rental_result = gross_rent - total_deductions

    restriction_applies = (
        int(financial_year_end) >= NEGATIVE_GEARING_RESTRICTION_START_FY
        and not bool(acquired_before_budget_time)
        and not bool(is_new_build)
        and not bool(is_exempt_housing)
    )

    if not restriction_applies:
        return {
            "gross_rent": gross_rent,
            "deductible_operating_expenses": deductible_operating_expenses,
            "loan_interest": loan_interest,
            "total_deductions": total_deductions,
            "net_cashflow": net_rental_result,
            "taxable_rental_income": net_rental_result,
            "current_year_quarantined_loss": 0.0,
            "quarantined_loss_used": 0.0,
            "opening_quarantined_loss": opening_quarantined_loss,
            "closing_quarantined_loss": opening_quarantined_loss,
            "restriction_applies": False,
        }

    if net_rental_result < 0:
        current_year_quarantined_loss = -net_rental_result
        quarantined_loss_used = 0.0
        taxable_rental_income = 0.0
        closing_quarantined_loss = opening_quarantined_loss + current_year_quarantined_loss
    else:
        quarantined_loss_used = min(opening_quarantined_loss, net_rental_result)
        current_year_quarantined_loss = 0.0
        taxable_rental_income = net_rental_result - quarantined_loss_used
        closing_quarantined_loss = opening_quarantined_loss - quarantined_loss_used

    return {
        "gross_rent": gross_rent,
        "deductible_operating_expenses": deductible_operating_expenses,
        "loan_interest": loan_interest,
        "total_deductions": total_deductions,
        "net_cashflow": net_rental_result,
        "taxable_rental_income": taxable_rental_income,
        "current_year_quarantined_loss": current_year_quarantined_loss,
        "quarantined_loss_used": quarantined_loss_used,
        "opening_quarantined_loss": opening_quarantined_loss,
        "closing_quarantined_loss": closing_quarantined_loss,
        "restriction_applies": True,
    }


def calculate_discretionary_trust_minimum_tax(
    trust_net_income,
    excluded_income,
    financial_year_end,
    subject_to_minimum_tax=True,
):
    """Estimate the September 2026 exposure-draft trust minimum tax."""
    trust_net_income = max(float(trust_net_income), 0.0)
    excluded_income = min(max(float(excluded_income), 0.0), trust_net_income)
    in_scope_income = max(trust_net_income - excluded_income, 0.0)
    applies = (
        bool(subject_to_minimum_tax)
        and int(financial_year_end) >= DISCRETIONARY_TRUST_MINIMUM_TAX_START_FY
    )
    trustee_minimum_tax = (
        in_scope_income * DISCRETIONARY_TRUST_MINIMUM_TAX_RATE if applies else 0.0
    )
    return {
        "trust_net_income": trust_net_income,
        "excluded_income": excluded_income,
        "minimum_tax_income": in_scope_income if applies else 0.0,
        "trustee_minimum_tax": trustee_minimum_tax,
        "minimum_tax_applies": applies,
        "minimum_tax_rate": DISCRETIONARY_TRUST_MINIMUM_TAX_RATE,
        "policy_status": DISCRETIONARY_TRUST_MINIMUM_TAX_STATUS,
    }


def calculate_incremental_budget_tax(
    base_taxable_income,
    residential_taxable_income,
    trust_taxable_income,
    trust_tax_credit,
    tax_schedule_key,
):
    """Calculate incremental individual tax and cap the trust credit at tax payable."""
    base_taxable_income = max(float(base_taxable_income), 0.0)
    residential_taxable_income = float(residential_taxable_income)
    trust_taxable_income = max(float(trust_taxable_income), 0.0)
    trust_tax_credit = max(float(trust_tax_credit), 0.0)

    after_property_income = max(base_taxable_income + residential_taxable_income, 0.0)
    after_trust_income = after_property_income + trust_taxable_income
    base_tax = calculate_personal_income_tax(base_taxable_income, tax_schedule_key)["personal_tax_total"]
    after_property_tax = calculate_personal_income_tax(after_property_income, tax_schedule_key)["personal_tax_total"]
    after_trust_tax = calculate_personal_income_tax(after_trust_income, tax_schedule_key)["personal_tax_total"]

    property_tax_adjustment = after_property_tax - base_tax
    tax_attributable_to_trust = max(after_trust_tax - after_property_tax, 0.0)
    allowed_trust_credit = min(trust_tax_credit, tax_attributable_to_trust)
    beneficiary_trust_tax_after_credit = tax_attributable_to_trust - allowed_trust_credit

    return {
        "adjusted_taxable_income": after_trust_income,
        "property_tax_adjustment": property_tax_adjustment,
        "trust_tax_before_credit": tax_attributable_to_trust,
        "trust_tax_credit": allowed_trust_credit,
        "trust_tax_after_credit": beneficiary_trust_tax_after_credit,
        "personal_tax_adjustment": property_tax_adjustment + beneficiary_trust_tax_after_credit,
    }


# ============================================================
# SECTION: SUPER ACCOUNT HELPERS
# ============================================================

def calculate_super_contributions_tax(gross_concessional_contribution):
    gross_concessional_contribution = max(float(gross_concessional_contribution), 0.0)
    return gross_concessional_contribution * SUPER_CONTRIBUTIONS_TAX_RATE


def calculate_super_guarantee_contribution(gross_income, financial_year_end):
    """Estimate employer SG using the published maximum earnings base."""
    gross_income = max(float(gross_income), 0.0)
    maximum_earnings_base = get_super_guarantee_maximum_earnings_base(financial_year_end)
    sg_earnings_base = min(gross_income, maximum_earnings_base)
    return {
        "gross_income": gross_income,
        "maximum_earnings_base": maximum_earnings_base,
        "sg_earnings_base": sg_earnings_base,
        "sg_contribution": sg_earnings_base * SUPER_GUARANTEE_RATE,
        "income_above_sg_base": max(gross_income - maximum_earnings_base, 0.0),
    }


def calculate_division_293_tax(
    division_293_income,
    concessional_contributions,
    financial_year_end,
):
    """Estimate Division 293 tax from the income components held by the model.

    The complete statutory income definition includes items that the model does
    not currently capture, such as reportable fringe benefits and net rental
    property losses. The result is therefore an estimate based on modelled
    taxable income and concessional contributions.
    """
    division_293_income = max(float(division_293_income), 0.0)
    concessional_contributions = max(float(concessional_contributions), 0.0)
    concessional_cap = get_concessional_contributions_cap(financial_year_end)
    division_293_super_contributions = min(concessional_contributions, concessional_cap)
    combined_income_and_contributions = (
        division_293_income + division_293_super_contributions
    )
    amount_above_threshold = max(
        combined_income_and_contributions - DIVISION_293_THRESHOLD,
        0.0,
    )
    taxable_contributions = min(
        division_293_super_contributions,
        amount_above_threshold,
    )

    return {
        "division_293_income": division_293_income,
        "division_293_super_contributions": division_293_super_contributions,
        "division_293_combined_income": combined_income_and_contributions,
        "division_293_amount_above_threshold": amount_above_threshold,
        "division_293_taxable_contributions": taxable_contributions,
        "division_293_tax": taxable_contributions * DIVISION_293_TAX_RATE,
    }


def auto_transfer_to_pension(
    accum_balance,
    pension_balance,
    transfer_balance_cap,
    is_pension_phase,
    has_started_pension,
):
    accum_balance = max(float(accum_balance), 0.0)
    pension_balance = max(float(pension_balance), 0.0)
    transfer_balance_cap = max(float(transfer_balance_cap), 0.0)

    available_cap_space = max(transfer_balance_cap - pension_balance, 0.0)

    should_start_pension_this_year = (
        bool(is_pension_phase)
        and (not bool(has_started_pension))
        and pension_balance <= 0.0
    )

    if not should_start_pension_this_year:
        return {
            "transfer_to_pension": 0.0,
            "requested_transfer_amount": 0.0,
            "available_cap_space": available_cap_space,
            "excess_retained_in_accumulation": accum_balance,
            "accum_balance_after_transfer": accum_balance,
            "pension_balance_after_transfer": pension_balance,
            "started_pension_this_year": False,
            "has_started_pension_after_year": bool(has_started_pension) or (pension_balance > 0),
        }

    requested_transfer_amount = accum_balance
    transfer_to_pension = min(requested_transfer_amount, available_cap_space)
    excess_retained_in_accumulation = max(requested_transfer_amount - transfer_to_pension, 0.0)
    has_effective_transfer = transfer_to_pension > 0

    return {
        "transfer_to_pension": transfer_to_pension,
        "requested_transfer_amount": requested_transfer_amount,
        "available_cap_space": available_cap_space,
        "excess_retained_in_accumulation": excess_retained_in_accumulation,
        "accum_balance_after_transfer": accum_balance - transfer_to_pension,
        "pension_balance_after_transfer": pension_balance + transfer_to_pension,
        "started_pension_this_year": has_effective_transfer,
        "has_started_pension_after_year": bool(has_started_pension) or has_effective_transfer,
    }


def get_minimum_pension_drawdown_rate(age):
    age = int(age)

    if age < 65:
        return 0.04
    if age <= 74:
        return 0.05
    if age <= 79:
        return 0.06
    if age <= 84:
        return 0.07
    if age <= 89:
        return 0.09
    if age <= 94:
        return 0.11
    return 0.14


def calculate_minimum_pension_drawdown(opening_pension_balance, age, phase):
    if phase != "pension_phase":
        return 0.0

    opening_pension_balance = max(float(opening_pension_balance), 0.0)
    rate = get_minimum_pension_drawdown_rate(age)
    return min(opening_pension_balance, opening_pension_balance * rate)


def withdraw_from_person_super_priority(required_amount, accum_balance, pension_balance):
    required_amount = max(float(required_amount), 0.0)
    accum_balance = max(float(accum_balance), 0.0)
    pension_balance = max(float(pension_balance), 0.0)

    accum_withdrawal = min(required_amount, accum_balance)
    remaining = required_amount - accum_withdrawal
    pension_withdrawal = min(remaining, pension_balance)

    return {
        "accum_withdrawal": accum_withdrawal,
        "pension_withdrawal": pension_withdrawal,
        "total_withdrawal": accum_withdrawal + pension_withdrawal,
    }


def allocate_household_extra_super_withdrawal(
    required_amount,
    person1_accum_balance,
    person2_accum_balance,
    person1_pension_balance,
    person2_pension_balance,
):
    required_amount = max(float(required_amount), 0.0)

    total_accum_available = max(float(person1_accum_balance), 0.0) + max(float(person2_accum_balance), 0.0)
    total_pension_available = max(float(person1_pension_balance), 0.0) + max(float(person2_pension_balance), 0.0)

    p1_accum_wd = 0.0
    p2_accum_wd = 0.0
    p1_pension_wd = 0.0
    p2_pension_wd = 0.0

    remaining = required_amount

    if total_accum_available > 0 and remaining > 0:
        accum_to_take = min(remaining, total_accum_available)
        p1_share = max(float(person1_accum_balance), 0.0) / total_accum_available if total_accum_available > 0 else 0.0
        p2_share = max(float(person2_accum_balance), 0.0) / total_accum_available if total_accum_available > 0 else 0.0
        p1_accum_wd = accum_to_take * p1_share
        p2_accum_wd = accum_to_take * p2_share
        remaining -= accum_to_take

    if total_pension_available > 0 and remaining > 0:
        pension_to_take = min(remaining, total_pension_available)
        p1_share = max(float(person1_pension_balance), 0.0) / total_pension_available if total_pension_available > 0 else 0.0
        p2_share = max(float(person2_pension_balance), 0.0) / total_pension_available if total_pension_available > 0 else 0.0
        p1_pension_wd = pension_to_take * p1_share
        p2_pension_wd = pension_to_take * p2_share
        remaining -= pension_to_take

    return {
        "person1_extra_accum_withdrawal": p1_accum_wd,
        "person2_extra_accum_withdrawal": p2_accum_wd,
        "person1_extra_pension_withdrawal": p1_pension_wd,
        "person2_extra_pension_withdrawal": p2_pension_wd,
        "total_extra_super_withdrawal": p1_accum_wd + p2_accum_wd + p1_pension_wd + p2_pension_wd,
        "unfunded_after_super": remaining,
    }


def calculate_amortising_loan_year(balance, annual_interest_rate, remaining_term_years, offset_balance=0.0):
    """Calculate one annual principal-and-interest payment with offset interest savings."""
    opening_balance = max(float(balance), 0.0)
    annual_rate = max(float(annual_interest_rate), 0.0)
    years = max(int(remaining_term_years), 1)
    offset = min(max(float(offset_balance), 0.0), opening_balance)
    if opening_balance <= 0:
        return {"payment": 0.0, "interest": 0.0, "principal": 0.0, "ending_balance": 0.0}
    scheduled_payment = (
        opening_balance / years
        if annual_rate == 0
        else opening_balance * annual_rate / (1 - (1 + annual_rate) ** (-years))
    )
    interest = max(opening_balance - offset, 0.0) * annual_rate
    principal = min(max(scheduled_payment - interest, 0.0), opening_balance)
    return {
        "payment": interest + principal,
        "interest": interest,
        "principal": principal,
        "ending_balance": opening_balance - principal,
    }


def calculate_loan_year_from_annual_repayment(balance, annual_interest_rate, annual_repayment, offset_balance=0.0):
    """Apply a user-entered annual principal-and-interest repayment."""
    opening_balance = max(float(balance), 0.0)
    annual_rate = max(float(annual_interest_rate), 0.0)
    offset = min(max(float(offset_balance), 0.0), opening_balance)
    if opening_balance <= 0:
        return {"payment": 0.0, "interest": 0.0, "principal": 0.0, "ending_balance": 0.0}
    interest = max(opening_balance - offset, 0.0) * annual_rate
    requested_payment = max(float(annual_repayment), 0.0)
    payment = min(requested_payment, opening_balance + interest)
    principal = min(max(payment - interest, 0.0), opening_balance)
    ending_balance = max(opening_balance + interest - payment, 0.0)
    return {
        "payment": payment,
        "interest": interest,
        "principal": principal,
        "ending_balance": ending_balance,
    }


def allocate_shortfall_by_asset_order(
    required_amount,
    withdrawal_order,
    cash_balance,
    cash_floor,
    non_super_balance,
    non_super_floor,
    person1_accum_balance,
    person2_accum_balance,
    person1_pension_balance,
    person2_pension_balance,
    property_value=0.0,
    property_loan_balance=0.0,
    property_equity_floor=0.0,
    property_sale_cost_rate=0.0,
):
    """Allocate a cash shortfall across nominated asset sources.

    Property proceeds are an annual strategic estimate. A partial disposal
    proportionally reduces both the property value and its associated loan.
    Property CGT is deliberately excluded pending asset-level cost-base inputs.
    """
    remaining = max(float(required_amount), 0.0)
    order = list(withdrawal_order or ["cash", "non_super", "accumulation", "pension", "property"])
    valid_sources = ["cash", "non_super", "accumulation", "pension", "property"]
    order = [source for source in order if source in valid_sources]
    order.extend(source for source in valid_sources if source not in order)

    cash_available = max(float(cash_balance) - max(float(cash_floor), 0.0), 0.0)
    non_super_available = max(float(non_super_balance) - max(float(non_super_floor), 0.0), 0.0)
    accum_balances = [max(float(person1_accum_balance), 0.0), max(float(person2_accum_balance), 0.0)]
    pension_balances = [max(float(person1_pension_balance), 0.0), max(float(person2_pension_balance), 0.0)]
    property_value = max(float(property_value), 0.0)
    property_loan_balance = min(max(float(property_loan_balance), 0.0), property_value)
    sale_cost_rate = min(max(float(property_sale_cost_rate), 0.0), 0.25)
    property_net_equity = max(property_value * (1.0 - sale_cost_rate) - property_loan_balance, 0.0)
    property_available = max(property_net_equity - max(float(property_equity_floor), 0.0), 0.0)

    result = {
        "cash_withdrawal": 0.0,
        "non_super_withdrawal": 0.0,
        "person1_extra_accum_withdrawal": 0.0,
        "person2_extra_accum_withdrawal": 0.0,
        "person1_extra_pension_withdrawal": 0.0,
        "person2_extra_pension_withdrawal": 0.0,
        "residential_property_sale_proceeds": 0.0,
        "residential_property_disposal_fraction": 0.0,
    }

    for source in order:
        if remaining <= 0:
            break
        if source == "cash":
            amount = min(remaining, cash_available)
            result["cash_withdrawal"] += amount
            remaining -= amount
        elif source == "non_super":
            amount = min(remaining, non_super_available)
            result["non_super_withdrawal"] += amount
            remaining -= amount
        elif source in {"accumulation", "pension"}:
            balances = accum_balances if source == "accumulation" else pension_balances
            total_available = sum(balances)
            amount = min(remaining, total_available)
            if total_available > 0 and amount > 0:
                p1_amount = amount * balances[0] / total_available
                p2_amount = amount - p1_amount
                if source == "accumulation":
                    result["person1_extra_accum_withdrawal"] += p1_amount
                    result["person2_extra_accum_withdrawal"] += p2_amount
                else:
                    result["person1_extra_pension_withdrawal"] += p1_amount
                    result["person2_extra_pension_withdrawal"] += p2_amount
                remaining -= amount
        elif source == "property":
            amount = min(remaining, property_available)
            result["residential_property_sale_proceeds"] += amount
            if property_net_equity > 0:
                result["residential_property_disposal_fraction"] = min(amount / property_net_equity, 1.0)
            remaining -= amount

    disposal_fraction = result["residential_property_disposal_fraction"]
    result["ending_cash_reserve_balance"] = max(float(cash_balance) - result["cash_withdrawal"], 0.0)
    result["remaining_residential_property_value"] = property_value * (1.0 - disposal_fraction)
    result["remaining_residential_property_loan_balance"] = property_loan_balance * (1.0 - disposal_fraction)
    result["total_extra_super_withdrawal"] = (
        result["person1_extra_accum_withdrawal"]
        + result["person2_extra_accum_withdrawal"]
        + result["person1_extra_pension_withdrawal"]
        + result["person2_extra_pension_withdrawal"]
    )
    result["unfunded_after_assets"] = remaining
    return result


def calculate_super_account_earnings_tax(accum_balance_before_return, pension_balance_before_return, return_rate, transfer_balance_cap):
    accum_balance_before_return = float(accum_balance_before_return)
    pension_balance_before_return = float(pension_balance_before_return)

    accum_earnings = accum_balance_before_return * return_rate
    pension_earnings = pension_balance_before_return * return_rate

    accum_tax = max(accum_earnings, 0.0) * SUPER_EARNINGS_TAX_RATE
    pension_tax = 0.0

    return {
        "accum_earnings": accum_earnings,
        "pension_earnings": pension_earnings,
        "accum_earnings_tax": accum_tax,
        "pension_earnings_tax": pension_tax,
        "total_super_earnings_tax": accum_tax + pension_tax,
    }

# ============================================================
# SECTION: SUPER COST BASE AND WITHDRAWAL CGT HELPERS
# ============================================================

SUPER_CGT_DISCOUNT_RATE = 1.0 / 3.0
SUPER_CGT_TAX_RATE = 0.15


def transfer_super_cost_base_to_pension(
    accum_balance,
    accum_cost_base,
    pension_cost_base,
    transfer_to_pension,
):
    accum_balance = max(float(accum_balance), 0.0)
    accum_cost_base = max(float(accum_cost_base), 0.0)
    pension_cost_base = max(float(pension_cost_base), 0.0)
    transfer_to_pension = max(float(transfer_to_pension), 0.0)

    if accum_balance <= 0 or transfer_to_pension <= 0:
        return {
            "accum_cost_base_after_transfer": accum_cost_base,
            "pension_cost_base_after_transfer": pension_cost_base,
            "cost_base_transferred": 0.0,
        }

    transfer_ratio = min(transfer_to_pension / accum_balance, 1.0)
    cost_base_transferred = accum_cost_base * transfer_ratio

    return {
        "accum_cost_base_after_transfer": max(accum_cost_base - cost_base_transferred, 0.0),
        "pension_cost_base_after_transfer": pension_cost_base + cost_base_transferred,
        "cost_base_transferred": cost_base_transferred,
    }


def calculate_super_withdrawal_cgt(
    withdrawal_amount,
    account_balance,
    account_cost_base,
    phase,
    transfer_balance_cap=0.0,
    phase_balance_for_tax=0.0,
    cgt_discount_rate=SUPER_CGT_DISCOUNT_RATE,
):
    withdrawal_amount = max(float(withdrawal_amount), 0.0)
    account_balance = max(float(account_balance), 0.0)
    account_cost_base = max(float(account_cost_base), 0.0)

    sale_result = calculate_average_cost_cgt_on_sale(
        sale_proceeds=withdrawal_amount,
        pool_market_value=account_balance,
        pool_cost_base=account_cost_base,
        cgt_discount_rate=cgt_discount_rate,
    )

    if phase == "pension_phase":
        taxable_discounted_capital_gain = 0.0
        cgt_tax_paid = 0.0
    else:
        taxable_discounted_capital_gain = sale_result["discounted_taxable_capital_gain"]
        cgt_tax_paid = taxable_discounted_capital_gain * SUPER_CGT_TAX_RATE

    return {
        "withdrawal_amount": withdrawal_amount,
        "cost_base_reduction": sale_result["cost_base_reduction"],
        "realised_capital_gain": sale_result["realised_capital_gain"],
        "realised_capital_loss": sale_result["realised_capital_loss"],
        "discounted_taxable_capital_gain": sale_result["discounted_taxable_capital_gain"],
        "taxable_discounted_capital_gain": taxable_discounted_capital_gain,
        "cgt_tax_paid": cgt_tax_paid,
        "remaining_cost_base": sale_result["remaining_cost_base"],
    }

# ============================================================
# SECTION: VALIDATION
# ============================================================

def validate_inputs(inputs):
    inputs = normalise_household_inputs(inputs)
    errors = []

    numeric_non_negative_fields = [
        "person1_current_age",
        "person2_current_age",
        "person1_retirement_age",
        "person2_retirement_age",
        "person1_pension_start_age",
        "person2_pension_start_age",
        "projection_years",
        "person1_accum_super_balance",
        "person1_pension_super_balance",
        "person2_accum_super_balance",
        "person2_pension_super_balance",
        "person1_transfer_balance_cap",
        "person2_transfer_balance_cap",
        "non_super_balance",
        "non_super_cost_base",
        "person1_annual_income",
        "person2_annual_income",
        "annual_living_expenses",
        "retirement_spending",
        "inflation_rate",
        "super_income_return_std",
        "super_capital_return_std",
        "non_super_income_return_std",
        "non_super_capital_return_std",
        "number_of_simulations",
    ]

    for field in numeric_non_negative_fields:
        if inputs[field] < 0:
            errors.append(f"{field} cannot be negative.")

    if inputs["projection_years"] <= 0:
        errors.append("projection_years must be greater than 0.")

    if inputs["number_of_simulations"] <= 0:
        errors.append("number_of_simulations must be greater than 0.")

    active_people = ["person1"]
    if inputs.get("household_mode") != "One Person":
        active_people.append("person2")
    for person in active_people:
        current_age = int(inputs[f"{person}_current_age"])
        retirement_age = int(inputs[f"{person}_retirement_age"])
        if not 18 <= current_age <= 100:
            errors.append(f"{person}_current_age must be between 18 and 100.")
        if not 18 <= retirement_age <= 100:
            errors.append(f"{person}_retirement_age must be between 18 and 100.")
        if inputs.get("module_pension_enabled", True):
            pension_start_age = int(inputs[f"{person}_pension_start_age"])
            if not 18 <= pension_start_age <= 100:
                errors.append(f"{person}_pension_start_age must be between 18 and 100.")

        active_super_accounts = []
        if inputs.get("module_super_enabled", True):
            active_super_accounts.append("accum")
        if inputs.get("module_pension_enabled", True):
            active_super_accounts.append("pension")
        for account in active_super_accounts:
            balance = float(inputs[f"{person}_{account}_super_balance"])
            cost_base = float(inputs[f"{person}_{account}_super_cost_base"])
            if cost_base > balance + 1e-9:
                errors.append(
                    f"{person}_{account}_super_cost_base cannot exceed "
                    f"{person}_{account}_super_balance under the current pooled cost-base setup."
                )


    if inputs["super_income_return_mean"] <= -1:
        errors.append("Super Income Return Mean must be greater than -1.00.")

    if inputs["super_capital_return_mean"] <= -1:
        errors.append("Super Capital Return Mean must be greater than -1.00.")

    if inputs["non_super_income_return_mean"] <= -1:
        errors.append("Non-Super Income Return Mean must be greater than -1.00.")

    if inputs["non_super_capital_return_mean"] <= -1:
        errors.append("Non-Super Capital Return Mean must be greater than -1.00.")

    if inputs["inflation_rate"] > 0.20:
        errors.append("Inflation Rate looks unusually high. Please check your input.")

    if inputs["super_income_return_std"] > 1.00:
        errors.append("Super Income Return Std looks unusually high. Please check your input.")

    if inputs["super_capital_return_std"] > 1.00:
        errors.append("Super Capital Return Std looks unusually high. Please check your input.")

    if inputs["non_super_income_return_std"] > 1.00:
        errors.append("Non-Super Income Return Std looks unusually high. Please check your input.")

    if inputs["non_super_capital_return_std"] > 1.00:
        errors.append("Non-Super Capital Return Std looks unusually high. Please check your input.")

    if inputs["non_super_ownership_person1"] < 0 or inputs["non_super_ownership_person1"] > 1:
        errors.append("Person 1 Non-Super Ownership must be between 0 and 1.")

    if inputs.get("residential_property_enabled", False):
        for field in [
            "residential_property_value",
            "residential_property_loan_balance",
            "residential_property_gross_rent",
            "residential_property_operating_expenses",
            "residential_property_opening_quarantined_loss",
        ]:
            if float(inputs.get(field, 0.0)) < 0:
                errors.append(f"{field} cannot be negative.")
        if not 0 <= float(inputs.get("residential_property_ownership_person1", 0.5)) <= 1:
            errors.append("Residential property ownership for Person 1 must be between 0 and 1.")
        if float(inputs.get("residential_property_interest_rate", 0.0)) < 0:
            errors.append("Residential property interest rate cannot be negative.")
        if "residential_property_annual_loan_repayment" in inputs:
            property_interest = max(
                float(inputs.get("residential_property_loan_balance", 0.0))
                - float(inputs.get("deductible_offset_balance", 0.0)),
                0.0,
            ) * float(inputs.get("residential_property_interest_rate", 0.0))
            if float(inputs.get("residential_property_loan_balance", 0.0)) > 0 and float(inputs.get("residential_property_annual_loan_repayment", 0.0)) <= property_interest:
                errors.append("Residential property annual loan repayment must be greater than annual interest so principal is repaid.")

    for field in ["main_residence_value", "main_residence_loan_balance", "main_residence_offset_balance"]:
        if float(inputs.get(field, 0.0)) < 0:
            errors.append(f"{field} cannot be negative.")
    if float(inputs.get("main_residence_offset_balance", 0.0)) > float(inputs.get("main_residence_loan_balance", 0.0)):
        errors.append("main_residence_offset_balance cannot exceed main_residence_loan_balance.")
    if "main_residence_annual_loan_repayment" in inputs:
        home_interest = max(
            float(inputs.get("main_residence_loan_balance", 0.0))
            - float(inputs.get("main_residence_offset_balance", 0.0)),
            0.0,
        ) * float(inputs.get("main_residence_interest_rate", 0.0))
        if float(inputs.get("main_residence_loan_balance", 0.0)) > 0 and float(inputs.get("main_residence_annual_loan_repayment", 0.0)) <= home_interest:
            errors.append("Main residence annual loan repayment must be greater than annual interest so principal is repaid.")

    if inputs.get("discretionary_trust_enabled", False):
        trust_balance = float(inputs.get("discretionary_trust_balance", 0.0))
        trust_cost_base = float(inputs.get("discretionary_trust_cost_base", 0.0))
        if trust_balance < 0:
            errors.append("Discretionary trust balance cannot be negative.")
        if trust_cost_base < 0 or trust_cost_base > trust_balance:
            errors.append("Discretionary trust cost base must be between zero and the trust balance.")
        if not 0 <= float(inputs.get("discretionary_trust_excluded_income_pct", 0.0)) <= 1:
            errors.append("Discretionary trust excluded income percentage must be between 0 and 1.")
        for field in ["discretionary_trust_income_return_std", "discretionary_trust_capital_return_std"]:
            if float(inputs.get(field, 0.0)) < 0:
                errors.append(f"{field} cannot be negative.")
        if not 0 <= float(inputs.get("discretionary_trust_ownership_person1", 0.5)) <= 1:
            errors.append("Discretionary trust allocation for Person 1 must be between 0 and 1.")

    if inputs.get("retirement_spending_trigger") not in ["Both Retired", "Either Retired"]:
        errors.append("retirement_spending_trigger must be either 'Both Retired' or 'Either Retired'.")

    if inputs.get("cgt_discount_rate", 0.50) < 0 or inputs.get("cgt_discount_rate", 0.50) > 1:
        errors.append("cgt_discount_rate must be between 0 and 1.")

    if float(inputs.get("non_super_transition_value_2027", inputs["non_super_balance"])) < 0:
        errors.append("non_super_transition_value_2027 cannot be negative.")
    if float(inputs.get("non_super_opening_capital_losses", 0.0)) < 0:
        errors.append("non_super_opening_capital_losses cannot be negative.")
    if float(inputs.get("cgt_indexation_rate", inputs.get("inflation_rate", 0.0))) <= -1:
        errors.append("cgt_indexation_rate must be greater than -1.00.")
    if inputs.get("cgt_asset_category", "Other") not in {
        "Other", "New residential dwelling", "Affordable housing"
    }:
        errors.append("cgt_asset_category is not recognised.")
    if inputs.get("cgt_new_residential_method", "Indexation and 30% minimum tax") not in {
        "Indexation and 30% minimum tax", "50% discount"
    }:
        errors.append("cgt_new_residential_method is not recognised.")

    if inputs["non_super_cost_base"] > inputs["non_super_balance"] + 1e-9:
        errors.append("non_super_cost_base cannot exceed non_super_balance under the current average-cost setup.")

    for field in [
        "cash_reserve_balance",
        "cash_reserve_floor",
        "cash_reserve_target",
        "non_super_estate_reserve",
        "property_estate_reserve",
        "non_deductible_debt_balance",
        "non_deductible_offset_balance",
        "non_deductible_annual_repayment",
        "deductible_offset_balance",
        "investment_deductible_debt_balance",
        "investment_deductible_offset_balance",
        "investment_deductible_annual_repayment",
        "main_residence_annual_loan_repayment",
        "residential_property_annual_loan_repayment",
    ]:
        if float(inputs.get(field, 0.0)) < 0:
            errors.append(f"{field} cannot be negative.")
    if float(inputs.get("non_deductible_interest_rate", 0.0)) < 0:
        errors.append("non_deductible_interest_rate cannot be negative.")
    if float(inputs.get("non_deductible_offset_balance", 0.0)) > float(inputs.get("non_deductible_debt_balance", 0.0)):
        errors.append("non_deductible_offset_balance cannot exceed non_deductible_debt_balance.")
    if float(inputs.get("deductible_offset_balance", 0.0)) > float(inputs.get("residential_property_loan_balance", 0.0)):
        errors.append("deductible_offset_balance cannot exceed residential_property_loan_balance.")
    if float(inputs.get("investment_deductible_offset_balance", 0.0)) > float(inputs.get("investment_deductible_debt_balance", 0.0)):
        errors.append("investment_deductible_offset_balance cannot exceed investment_deductible_debt_balance.")
    if "non_deductible_annual_repayment" in inputs:
        non_deductible_interest = max(
            float(inputs.get("non_deductible_debt_balance", 0.0))
            - float(inputs.get("non_deductible_offset_balance", 0.0)),
            0.0,
        ) * float(inputs.get("non_deductible_interest_rate", 0.0))
        if float(inputs.get("non_deductible_debt_balance", 0.0)) > 0 and float(inputs.get("non_deductible_annual_repayment", 0.0)) <= non_deductible_interest:
            errors.append("Non-deductible investment debt annual repayment must be greater than annual interest so principal is repaid.")
    if "investment_deductible_annual_repayment" in inputs:
        investment_interest = max(
            float(inputs.get("investment_deductible_debt_balance", 0.0))
            - float(inputs.get("investment_deductible_offset_balance", 0.0)),
            0.0,
        ) * float(inputs.get("investment_deductible_interest_rate", 0.0))
        if float(inputs.get("investment_deductible_debt_balance", 0.0)) > 0 and float(inputs.get("investment_deductible_annual_repayment", 0.0)) <= investment_interest:
            errors.append("Deductible investment debt annual repayment must be greater than annual interest so principal is repaid.")
    valid_surplus_destinations = {
        "cash",
        "cash_reserve",
        "non_deductible_offset",
        "non_deductible_repayment",
        "deductible_offset",
        "deductible_repayment",
        "main_residence_offset",
        "main_residence_repayment",
        "property_loan_offset",
        "property_loan_repayment",
        "investment_deductible_offset",
        "investment_deductible_repayment",
        "non_super",
    }
    surplus_order = inputs.get("surplus_allocation_order", ["non_super"])
    if not isinstance(surplus_order, (list, tuple)) or set(surplus_order) - valid_surplus_destinations:
        errors.append("surplus_allocation_order contains an unrecognised destination.")
    sale_cost_rate = float(inputs.get("residential_property_sale_cost_rate", 0.0))
    if not 0.0 <= sale_cost_rate <= 0.25:
        errors.append("residential_property_sale_cost_rate must be between 0% and 25%.")
    withdrawal_order = inputs.get("withdrawal_order", [])
    valid_withdrawal_sources = {"cash", "non_super", "pension", "accumulation", "property"}
    if not isinstance(withdrawal_order, (list, tuple)) or set(withdrawal_order) - valid_withdrawal_sources:
        errors.append("withdrawal_order contains an unrecognised asset source.")

    try:
        parse_financial_year_label(inputs["start_financial_year"])
    except Exception:
        errors.append("start_financial_year must be a financial year end such as 2027.")

    events_df = normalise_contribution_events(inputs.get("contribution_events"), household_mode=inputs.get("household_mode", "Two People"))
    valid_people = {"Person 1", "Person 2"}
    valid_types = {"personal_deductible", "non_concessional"}

    if not events_df.empty:
        invalid_people = events_df.loc[~events_df["person"].isin(valid_people)]
        if not invalid_people.empty:
            errors.append("Contribution events contain invalid person values. Use 'Person 1' or 'Person 2'.")

        invalid_types = events_df.loc[~events_df["contribution_type"].isin(valid_types)]
        if not invalid_types.empty:
            errors.append("Contribution events contain invalid contribution_type values. Use 'personal_deductible' or 'non_concessional'.")

        invalid_years = []
        for fy in events_df["financial_year"].unique().tolist():
            try:
                parse_financial_year_label(fy)
            except Exception:
                invalid_years.append(fy)

        if invalid_years:
            errors.append("Contribution events contain invalid financial_year values. Use numeric year values like 2028.")

        if (events_df["amount"] < 0).any():
            errors.append("Contribution event amounts cannot be negative.")

    return errors


# ============================================================
# SECTION: WARNING GENERATION
# ============================================================

def generate_input_warnings(inputs):
    warnings = []

    start_fy = parse_financial_year_label(inputs["start_financial_year"])
    projection_end_fy = start_fy + int(inputs["projection_years"]) - 1
    if inputs.get("residential_property_enabled", False):
        if (
            projection_end_fy >= NEGATIVE_GEARING_RESTRICTION_START_FY
            and not inputs.get("residential_property_acquired_before_budget_time", False)
            and not inputs.get("residential_property_is_new_build", False)
            and not inputs.get("residential_property_is_exempt_housing", False)
        ):
            warnings.append(
                "Residential rental losses are quarantined from 2027-28 under the modelled legislated rule and carried forward against future residential income."
            )
        if "property" in inputs.get("withdrawal_order", []):
            warnings.append(
                "Residential property sale proceeds are an annual strategic estimate. Partial disposals proportionally reduce value and debt after estimated selling costs; legal feasibility, refinancing requirements, transaction-specific costs, and property CGT are not modelled."
            )
        else:
            warnings.append(
                "Residential property modelling is an aggregate projection. Property equity is included in net wealth but is not sold or refinanced to fund spending unless the property source is selected; scheduled loan amortisation, depreciation schedules, sale costs outside a disposal strategy, and property CGT are not modelled."
            )

    if float(inputs.get("non_deductible_debt_balance", 0.0)) > 0 or float(inputs.get("residential_property_loan_balance", 0.0)) > 0:
        warnings.append(
            "Debt strategies are annual cashflow estimates. Deductibility depends on the use of borrowed funds, not the security; confirm loan purpose, offset/redraw structure, refinancing terms and lender requirements before relying on the comparison."
        )

    if any(source in inputs.get("withdrawal_order", []) for source in ["accumulation", "pension"]):
        warnings.append(
            "The selected drawdown order is a strategic funding assumption. Confirm preservation age, retirement status and all conditions of release before relying on a super withdrawal result."
        )

    if inputs.get("discretionary_trust_enabled", False):
        warnings.append(
            "The discretionary trust 30% minimum tax is based on the September 2026 exposure draft and is not enacted law. Final legislation may change the result."
        )
    if inputs.get("cgt_reform_enabled", True) and projection_end_fy >= CGT_REFORM_START_FY:
        warnings.append(
            "The enacted CGT reform is modelled using one homogeneous non-super pool. The 1 July 2027 transition allocation, annual CPI indexation, loss ordering, and partial disposals are planning estimates and must be reconciled to asset-level records for tax return work."
        )
        if inputs.get("cgt_asset_category", "Other") in {
            "New residential dwelling", "Affordable housing"
        }:
            warnings.append(
                "The selected new/affordable housing CGT method is a scenario choice. Confirm statutory eligibility and compare the 50% discount with indexation using actual records at disposal."
            )
    if projection_end_fy > LATEST_PUBLISHED_SUPER_THRESHOLD_FY:
        warnings.append(
            f"Published indexed super thresholds are currently configured through "
            f"{LATEST_PUBLISHED_SUPER_THRESHOLD_FY}FY. Later projection years retain the "
            "latest known contribution caps, general transfer balance cap, and SG maximum "
            "earnings base until policy settings are refreshed."
        )

    years_to_person1_retirement = inputs["person1_retirement_age"] - inputs["person1_current_age"]
    years_to_person2_retirement = inputs["person2_retirement_age"] - inputs["person2_current_age"]

    if years_to_person1_retirement <= 5:
        warnings.append("Person 1 is scheduled to retire within 5 years. Small assumption changes may have a larger impact.")

    if years_to_person2_retirement <= 5:
        warnings.append("Person 2 is scheduled to retire within 5 years. Small assumption changes may have a larger impact.")

    if inputs["person1_retirement_age"] < 55 or inputs["person2_retirement_age"] < 55:
        warnings.append("At least one retirement age is relatively early. This may increase portfolio sustainability risk.")

    if inputs["super_capital_return_std"] >= 0.18:
        warnings.append("Super Capital Return Std is relatively high. This may produce a wide range of outcomes.")

    if inputs["non_super_capital_return_std"] >= 0.18:
        warnings.append("Non-Super Capital Return Std is relatively high. This may produce a wide range of outcomes.")

    if inputs["number_of_simulations"] < 1000:
        warnings.append("Number of Simulations is relatively low. Results may be less stable.")

    warnings.append("Non-super withdrawals use a pooled average-cost method; individual tax parcels and exact disposal ordering are not modelled.")
    warnings.append("Salary income is indexed annually using the inflation rate while the person remains in working phase.")
    warnings.append("Pension transfer is triggered from pension start age and is applied up to the person's transfer balance cap. Minimum pension drawdown is then applied from pension assets.")
    warnings.append("Personal deductible contributions reduce taxable income and also flow through concessional contribution tax inside super.")

    events_df = normalise_contribution_events(inputs.get("contribution_events"), household_mode=inputs.get("household_mode", "Two People"))
    if not events_df.empty:
        for _, row in events_df.iterrows():
            fy_end = parse_financial_year_label(row["financial_year"])
            person = row["person"]
            contribution_type = row["contribution_type"]
            amount = float(row["amount"])

            if person == "Person 1":
                person_start_age = inputs["person1_current_age"]
                person_income = inputs["person1_annual_income"]
            else:
                person_start_age = inputs["person2_current_age"]
                person_income = inputs["person2_annual_income"]

            year_offset = fy_end - start_fy
            age_in_year = person_start_age + year_offset

            if contribution_type == "personal_deductible":
                indexed_income = person_income * ((1 + inputs["inflation_rate"]) ** max(year_offset, 0))
                estimated_sg = calculate_super_guarantee_contribution(
                    indexed_income,
                    fy_end,
                )["sg_contribution"]
                estimated_total_concessional = estimated_sg + amount
                concessional_cap = get_concessional_contributions_cap(fy_end)

                if estimated_total_concessional > concessional_cap:
                    warnings.append(
                        f"{person} in {fy_end}FY: estimated concessional contributions exceed the annual cap ({concessional_cap:,.0f}). Review eligibility for carry-forward concessional contributions."
                    )

                if age_in_year >= 67:
                    warnings.append(
                        f"{person} in {fy_end}FY: personal deductible contribution entered at age 67 or above. Review eligibility and any work-test related considerations."
                    )

                if age_in_year >= 75:
                    warnings.append(
                        f"{person} in {fy_end}FY: personal deductible contribution entered at age 75 or above. Review acceptance and eligibility rules carefully."
                    )

            if contribution_type == "non_concessional":
                non_concessional_cap = get_non_concessional_contributions_cap(fy_end)

                if amount > non_concessional_cap:
                    warnings.append(
                        f"{person} in {fy_end}FY: non-concessional contributions exceed the annual cap ({non_concessional_cap:,.0f}). Review eligibility for bring-forward non-concessional contributions."
                    )

                if age_in_year >= 75:
                    warnings.append(
                        f"{person} in {fy_end}FY: non-concessional contribution entered at age 75 or above. Review acceptance and eligibility rules carefully."
                    )

    if inputs.get("non_super_cost_base", 0.0) < inputs.get("non_super_balance", 0.0):
        warnings.append("Non-super cost base is lower than market value, so future withdrawals may crystallise capital gains.")

    if inputs.get("cgt_discount_rate", 0.50) != 0.50:
        warnings.append("CGT discount rate has been changed from the default 50% assumption. Confirm this is intended.")

    return warnings


def generate_output_warnings(summary_df, failure_prob_df, det_df):
    warnings = []

    if "total_division_293_tax" in det_df.columns and det_df["total_division_293_tax"].sum() > 0:
        warnings.append(
            "Division 293 tax is an estimate based on income components available in this model. "
            "Confirm reportable fringe benefits, net investment or rental losses, defined benefit "
            "contributions, and the final ATO assessment before relying on it for advice."
        )

    success_rate = summary_df["success"].mean()
    p10_final_wealth = summary_df["final_wealth"].quantile(0.10)
    median_final_wealth = summary_df["final_wealth"].median()

    if success_rate < 0.50:
        warnings.append("Success Rate is below 50%. The plan may have a high risk of failure.")
    elif success_rate < 0.75:
        warnings.append("Success Rate is below 75%. The plan may require further review or stress testing.")

    if p10_final_wealth < 0:
        warnings.append("P10 Final Wealth is below zero. Downside outcomes may be severe.")

    if median_final_wealth < 0:
        warnings.append("Median Final Wealth is below zero. The central case may not be sustainable.")

    if det_df["unmet_shortfall"].max() > 0:
        warnings.append("The deterministic projection shows unmet shortfall in at least one year.")

    high_failure_rows = failure_prob_df[failure_prob_df["failure_probability"] >= 0.25]
    if not high_failure_rows.empty:
        first_year_25 = int(high_failure_rows["financial_year_end"].min())
        warnings.append(
            f"Cumulative failure probability reaches 25% by {first_year_25}FY."
        )

    steep_rise_rows = failure_prob_df["failure_probability"].diff().fillna(0)
    if steep_rise_rows.max() >= 0.10:
        warnings.append("Failure probability rises sharply at some point in the projection. Review sequencing and spending assumptions.")

    if "non_super_realised_capital_gain" in det_df.columns and det_df["non_super_realised_capital_gain"].sum() > 0:
        warnings.append("The deterministic projection realises capital gains on non-super withdrawals in at least one year.")

    if "ending_non_super_cost_base" in det_df.columns:
        low_cost_base_rows = det_df[
            (det_df["ending_non_super_balance"] > 0) &
            (det_df["ending_non_super_cost_base"] / det_df["ending_non_super_balance"] < 0.50)
        ]
        if not low_cost_base_rows.empty:
            warnings.append("Non-super cost base falls materially below market value in the projection, which may increase future CGT on withdrawals.")

    return warnings


# ============================================================
# SECTION: PERSONAL TAX SPLIT ENGINE
# ============================================================

def calculate_household_personal_tax_split(
    person1_salary_income,
    person2_salary_income,
    taxable_non_super_earnings_total,
    ownership_person1,
    person1_personal_deductible_contribution,
    person2_personal_deductible_contribution,
    person1_gross_concessional_contribution,
    person2_gross_concessional_contribution,
    financial_year_end,
    tax_schedule_key,
):
    ownership_person1 = float(ownership_person1)
    ownership_person2 = 1.0 - ownership_person1

    taxable_non_super_earnings_total = max(float(taxable_non_super_earnings_total), 0.0)

    person1_taxable_non_super = taxable_non_super_earnings_total * ownership_person1
    person2_taxable_non_super = taxable_non_super_earnings_total * ownership_person2

    person1_assessable_before_deduction = max(float(person1_salary_income), 0.0) + person1_taxable_non_super
    person2_assessable_before_deduction = max(float(person2_salary_income), 0.0) + person2_taxable_non_super

    person1_taxable_income = max(
        person1_assessable_before_deduction - max(float(person1_personal_deductible_contribution), 0.0),
        0.0,
    )
    person2_taxable_income = max(
        person2_assessable_before_deduction - max(float(person2_personal_deductible_contribution), 0.0),
        0.0,
    )

    person1_tax_result = calculate_personal_income_tax(
        taxable_income=person1_taxable_income,
        tax_schedule_key=tax_schedule_key,
    )
    person2_tax_result = calculate_personal_income_tax(
        taxable_income=person2_taxable_income,
        tax_schedule_key=tax_schedule_key,
    )

    person1_division_293 = calculate_division_293_tax(
        division_293_income=person1_taxable_income,
        concessional_contributions=person1_gross_concessional_contribution,
        financial_year_end=financial_year_end,
    )
    person2_division_293 = calculate_division_293_tax(
        division_293_income=person2_taxable_income,
        concessional_contributions=person2_gross_concessional_contribution,
        financial_year_end=financial_year_end,
    )

    person1_alloc = allocate_tax_proportionally(
        total_tax=person1_tax_result["personal_tax_total"],
        components_dict={
            "salary": person1_salary_income,
            "non_super": person1_taxable_non_super,
        },
    )
    person2_alloc = allocate_tax_proportionally(
        total_tax=person2_tax_result["personal_tax_total"],
        components_dict={
            "salary": person2_salary_income,
            "non_super": person2_taxable_non_super,
        },
    )

    p1_income_tax, p1_medicare = split_income_tax_and_medicare(
        person1_alloc["salary"],
        person1_tax_result,
    )
    p1_non_super_income_tax, p1_non_super_medicare = split_income_tax_and_medicare(
        person1_alloc["non_super"],
        person1_tax_result,
    )

    p2_income_tax, p2_medicare = split_income_tax_and_medicare(
        person2_alloc["salary"],
        person2_tax_result,
    )
    p2_non_super_income_tax, p2_non_super_medicare = split_income_tax_and_medicare(
        person2_alloc["non_super"],
        person2_tax_result,
    )

    return {
        "person1_taxable_non_super": person1_taxable_non_super,
        "person2_taxable_non_super": person2_taxable_non_super,
        "person1_assessable_before_deduction": person1_assessable_before_deduction,
        "person2_assessable_before_deduction": person2_assessable_before_deduction,
        "person1_taxable_income": person1_taxable_income,
        "person2_taxable_income": person2_taxable_income,
        "person1_income_tax": p1_income_tax,
        "person1_medicare_levy": p1_medicare,
        "person1_salary_tax_total": person1_alloc["salary"],
        "person1_income_tax_on_non_super_earnings": p1_non_super_income_tax,
        "person1_medicare_levy_on_non_super_earnings": p1_non_super_medicare,
        "person1_non_super_tax_total": person1_alloc["non_super"],
        "person1_personal_tax_total": person1_tax_result["personal_tax_total"],
        "person1_division_293_income": person1_division_293["division_293_income"],
        "person1_division_293_super_contributions": person1_division_293["division_293_super_contributions"],
        "person1_division_293_taxable_contributions": person1_division_293["division_293_taxable_contributions"],
        "person1_division_293_tax": person1_division_293["division_293_tax"],
        "person2_income_tax": p2_income_tax,
        "person2_medicare_levy": p2_medicare,
        "person2_salary_tax_total": person2_alloc["salary"],
        "person2_income_tax_on_non_super_earnings": p2_non_super_income_tax,
        "person2_medicare_levy_on_non_super_earnings": p2_non_super_medicare,
        "person2_non_super_tax_total": person2_alloc["non_super"],
        "person2_personal_tax_total": person2_tax_result["personal_tax_total"],
        "person2_division_293_income": person2_division_293["division_293_income"],
        "person2_division_293_super_contributions": person2_division_293["division_293_super_contributions"],
        "person2_division_293_taxable_contributions": person2_division_293["division_293_taxable_contributions"],
        "person2_division_293_tax": person2_division_293["division_293_tax"],
    }


# ============================================================
# SECTION: NON-SUPER COST BASE AND CGT HELPERS
# ============================================================

def calculate_average_cost_cgt_on_sale(
    sale_proceeds,
    pool_market_value,
    pool_cost_base,
    cgt_discount_rate,
):
    sale_proceeds = max(float(sale_proceeds), 0.0)
    pool_market_value = max(float(pool_market_value), 0.0)
    pool_cost_base = max(float(pool_cost_base), 0.0)
    cgt_discount_rate = min(max(float(cgt_discount_rate), 0.0), 1.0)

    if sale_proceeds <= 0 or pool_market_value <= 0:
        return {
            "sale_proceeds": sale_proceeds,
            "cost_base_reduction": 0.0,
            "realised_capital_gain": 0.0,
            "realised_capital_loss": 0.0,
            "net_capital_gain_before_discount": 0.0,
            "discounted_taxable_capital_gain": 0.0,
            "remaining_cost_base": pool_cost_base,
        }

    sale_proceeds = min(sale_proceeds, pool_market_value)
    average_cost_ratio = pool_cost_base / pool_market_value if pool_market_value > 0 else 0.0
    cost_base_reduction = min(pool_cost_base, sale_proceeds * average_cost_ratio)
    gain_or_loss = sale_proceeds - cost_base_reduction

    realised_capital_gain = max(gain_or_loss, 0.0)
    realised_capital_loss = max(-gain_or_loss, 0.0)
    net_capital_gain_before_discount = max(gain_or_loss, 0.0)
    discounted_taxable_capital_gain = net_capital_gain_before_discount * (1.0 - cgt_discount_rate)
    remaining_cost_base = max(pool_cost_base - cost_base_reduction, 0.0)

    return {
        "sale_proceeds": sale_proceeds,
        "cost_base_reduction": cost_base_reduction,
        "realised_capital_gain": realised_capital_gain,
        "realised_capital_loss": realised_capital_loss,
        "net_capital_gain_before_discount": net_capital_gain_before_discount,
        "discounted_taxable_capital_gain": discounted_taxable_capital_gain,
        "remaining_cost_base": remaining_cost_base,
    }


def calculate_budget_cgt_on_sale(
    sale_proceeds,
    pool_market_value,
    pool_cost_base,
    pool_indexed_cost_base,
    pool_deferred_pre_2027_gain,
    opening_capital_losses,
    financial_year_end,
    cgt_discount_rate=0.50,
    indexation_rate=0.0,
    asset_category="Other",
    new_residential_method="Indexation and 30% minimum tax",
    held_at_least_12_months=True,
    reform_enabled=True,
):
    """Estimate Budget 2026 CGT for a homogeneous non-super asset pool.

    For post-reform disposals, losses are applied first to the real gain that
    can be subject to the minimum tax, then to the deferred pre-1 July 2027
    component. This ordering is a documented modelling assumption.
    """
    sale_proceeds = max(float(sale_proceeds), 0.0)
    pool_market_value = max(float(pool_market_value), 0.0)
    pool_cost_base = max(float(pool_cost_base), 0.0)
    pool_indexed_cost_base = max(float(pool_indexed_cost_base), 0.0)
    pool_deferred_pre_2027_gain = float(pool_deferred_pre_2027_gain)
    opening_capital_losses = max(float(opening_capital_losses), 0.0)
    cgt_discount_rate = min(max(float(cgt_discount_rate), 0.0), 1.0)
    indexation_rate = max(float(indexation_rate), -0.99)
    financial_year_end = int(financial_year_end)

    reform_applies = bool(reform_enabled) and financial_year_end >= CGT_REFORM_START_FY
    indexed_cost_base_at_sale = (
        pool_indexed_cost_base * (1 + indexation_rate)
        if reform_applies else pool_indexed_cost_base
    )

    empty_result = {
        "sale_proceeds": min(sale_proceeds, pool_market_value) if pool_market_value > 0 else 0.0,
        "cost_base_reduction": 0.0,
        "indexed_cost_base_reduction": 0.0,
        "realised_capital_gain": 0.0,
        "realised_capital_loss": 0.0,
        "deferred_pre_2027_gain": 0.0,
        "post_2027_real_gain": 0.0,
        "net_capital_gain_before_discount": 0.0,
        "discounted_taxable_capital_gain": 0.0,
        "minimum_tax_capital_gain": 0.0,
        "capital_losses_applied": 0.0,
        "remaining_capital_losses": opening_capital_losses,
        "remaining_cost_base": pool_cost_base,
        "remaining_indexed_cost_base": indexed_cost_base_at_sale,
        "remaining_deferred_pre_2027_gain": pool_deferred_pre_2027_gain,
        "indexation_uplift": max(indexed_cost_base_at_sale - pool_indexed_cost_base, 0.0),
        "reform_applies": reform_applies,
        "calculation_method": "No disposal",
    }
    if sale_proceeds <= 0 or pool_market_value <= 0:
        return empty_result

    sale_proceeds = min(sale_proceeds, pool_market_value)
    disposal_fraction = min(sale_proceeds / pool_market_value, 1.0)
    nominal_cost_reduction = min(pool_cost_base, pool_cost_base * disposal_fraction)
    indexed_cost_reduction = min(
        indexed_cost_base_at_sale,
        indexed_cost_base_at_sale * disposal_fraction,
    )
    deferred_component = pool_deferred_pre_2027_gain * disposal_fraction
    remaining_cost_base = max(pool_cost_base - nominal_cost_reduction, 0.0)
    remaining_indexed_cost_base = max(indexed_cost_base_at_sale - indexed_cost_reduction, 0.0)
    remaining_deferred_gain = pool_deferred_pre_2027_gain - deferred_component

    category_uses_discount = (
        asset_category in {"New residential dwelling", "Affordable housing"}
        and new_residential_method == "50% discount"
    )
    discount_factor = (1.0 - cgt_discount_rate) if held_at_least_12_months else 1.0

    if not reform_applies or category_uses_discount:
        gain_or_loss = sale_proceeds - nominal_cost_reduction
        gross_gain = max(gain_or_loss, 0.0)
        current_loss = max(-gain_or_loss, 0.0)
        available_losses = opening_capital_losses + current_loss
        losses_applied = min(gross_gain, available_losses)
        net_gain = gross_gain - losses_applied
        remaining_losses = available_losses - losses_applied
        taxable_gain = net_gain * discount_factor
        return {
            **empty_result,
            "sale_proceeds": sale_proceeds,
            "cost_base_reduction": nominal_cost_reduction,
            "indexed_cost_base_reduction": indexed_cost_reduction,
            "realised_capital_gain": gross_gain,
            "realised_capital_loss": current_loss,
            "deferred_pre_2027_gain": net_gain if not reform_applies else 0.0,
            "net_capital_gain_before_discount": net_gain,
            "discounted_taxable_capital_gain": taxable_gain,
            "capital_losses_applied": losses_applied,
            "remaining_capital_losses": remaining_losses,
            "remaining_cost_base": remaining_cost_base,
            "remaining_indexed_cost_base": remaining_indexed_cost_base,
            "remaining_deferred_pre_2027_gain": remaining_deferred_gain,
            "calculation_method": (
                "New/affordable residential 50% discount"
                if category_uses_discount else "Pre-reform 50% discount"
            ),
        }

    post_component = sale_proceeds - indexed_cost_reduction
    pre_gain = max(deferred_component, 0.0)
    post_gain = max(post_component, 0.0)
    current_loss = max(-deferred_component, 0.0) + max(-post_component, 0.0)
    available_losses = opening_capital_losses + current_loss

    loss_to_post = min(post_gain, available_losses)
    post_gain_after_losses = post_gain - loss_to_post
    available_losses -= loss_to_post
    loss_to_pre = min(pre_gain, available_losses)
    pre_gain_after_losses = pre_gain - loss_to_pre
    available_losses -= loss_to_pre
    losses_applied = loss_to_post + loss_to_pre

    taxable_pre_gain = pre_gain_after_losses * discount_factor
    taxable_post_gain = post_gain_after_losses
    taxable_gain = taxable_pre_gain + taxable_post_gain

    return {
        **empty_result,
        "sale_proceeds": sale_proceeds,
        "cost_base_reduction": nominal_cost_reduction,
        "indexed_cost_base_reduction": indexed_cost_reduction,
        "realised_capital_gain": pre_gain + post_gain,
        "realised_capital_loss": current_loss,
        "deferred_pre_2027_gain": pre_gain_after_losses,
        "post_2027_real_gain": post_gain_after_losses,
        "net_capital_gain_before_discount": pre_gain_after_losses + post_gain_after_losses,
        "discounted_taxable_capital_gain": taxable_gain,
        "minimum_tax_capital_gain": post_gain_after_losses,
        "capital_losses_applied": losses_applied,
        "remaining_capital_losses": available_losses,
        "remaining_cost_base": remaining_cost_base,
        "remaining_indexed_cost_base": remaining_indexed_cost_base,
        "remaining_deferred_pre_2027_gain": remaining_deferred_gain,
        "calculation_method": "Transition split + indexed real gain",
    }


def calculate_cgt_minimum_tax_gap(
    taxable_income,
    minimum_tax_capital_gain,
    tax_schedule_key,
    exempt_from_minimum_tax=False,
):
    """Calculate the Division 119 top-up using basic income tax only."""
    taxable_income = max(float(taxable_income), 0.0)
    minimum_tax_capital_gain = min(
        max(float(minimum_tax_capital_gain), 0.0),
        taxable_income,
    )
    if exempt_from_minimum_tax or minimum_tax_capital_gain <= 0:
        return {
            "minimum_tax_capital_gain": minimum_tax_capital_gain,
            "basic_tax_attributable_to_gain": 0.0,
            "cgt_minimum_tax_target": 0.0,
            "cgt_minimum_tax_gap": 0.0,
        }

    current_basic_tax = calculate_progressive_income_tax(taxable_income, tax_schedule_key)
    reduced_basic_tax = calculate_progressive_income_tax(
        taxable_income - minimum_tax_capital_gain,
        tax_schedule_key,
    )
    basic_tax_attributable = max(current_basic_tax - reduced_basic_tax, 0.0)
    target = minimum_tax_capital_gain * CGT_MINIMUM_TAX_RATE
    gap = max(target - basic_tax_attributable, 0.0)
    gap = float(np.floor(gap))
    return {
        "minimum_tax_capital_gain": minimum_tax_capital_gain,
        "basic_tax_attributable_to_gain": basic_tax_attributable,
        "cgt_minimum_tax_target": target,
        "cgt_minimum_tax_gap": gap,
    }


# ============================================================
# SECTION: CASHFLOW SOLVER
# ============================================================

def solve_cashflow_before_returns(
    person1_net_income,
    person2_net_income,
    person1_min_pension_drawdown,
    person2_min_pension_drawdown,
    person1_accum_after_transfer,
    person1_pension_after_transfer,
    person2_accum_after_transfer,
    person2_pension_after_transfer,
    person1_accum_cost_base_after_transfer,
    person1_pension_cost_base_after_transfer,
    person2_accum_cost_base_after_transfer,
    person2_pension_cost_base_after_transfer,
    person1_super_phase_for_transfer,
    person2_super_phase_for_transfer,
    person1_transfer_balance_cap,
    person2_transfer_balance_cap,
    opening_non_super_balance,
    opening_non_super_cost_base,
    current_spending,
    person1_total_cash_contribution,
    person2_total_cash_contribution,
    person1_total_net_super_contribution,
    person2_total_net_super_contribution,
    cgt_discount_rate,
    financial_year_end,
    opening_non_super_indexed_cost_base,
    opening_non_super_deferred_pre_2027_gain,
    opening_non_super_capital_losses,
    cgt_indexation_rate,
    cgt_asset_category,
    cgt_new_residential_method,
    cgt_held_at_least_12_months,
    cgt_reform_enabled,
    opening_cash_reserve_balance=0.0,
    cash_reserve_floor=0.0,
    withdrawal_order=None,
    non_super_estate_reserve=0.0,
    opening_residential_property_value=0.0,
    opening_residential_property_loan_balance=0.0,
    property_estate_reserve=0.0,
    residential_property_sale_cost_rate=0.0,
    opening_non_deductible_debt_balance=0.0,
    opening_non_deductible_offset_balance=0.0,
    opening_deductible_offset_balance=0.0,
    non_deductible_interest_expense=0.0,
    opening_main_residence_loan_balance=0.0,
    opening_main_residence_offset_balance=0.0,
    opening_investment_deductible_debt_balance=0.0,
    opening_investment_deductible_offset_balance=0.0,
    additional_required_cash_outflow=0.0,
    cash_reserve_target=0.0,
    surplus_allocation_order=None,
    allow_non_super_investment=True,
):
    opening_non_super_balance = max(float(opening_non_super_balance), 0.0)
    opening_non_super_cost_base = max(float(opening_non_super_cost_base), 0.0)
    opening_non_super_indexed_cost_base = max(float(opening_non_super_indexed_cost_base), 0.0)

    # ---------- Minimum pension drawdown CGT ----------
    person1_min_pension_cgt = calculate_super_withdrawal_cgt(
        withdrawal_amount=person1_min_pension_drawdown,
        account_balance=person1_pension_after_transfer,
        account_cost_base=person1_pension_cost_base_after_transfer,
        phase=person1_super_phase_for_transfer,
        transfer_balance_cap=person1_transfer_balance_cap,
        phase_balance_for_tax=person1_pension_after_transfer,
    )
    person2_min_pension_cgt = calculate_super_withdrawal_cgt(
        withdrawal_amount=person2_min_pension_drawdown,
        account_balance=person2_pension_after_transfer,
        account_cost_base=person2_pension_cost_base_after_transfer,
        phase=person2_super_phase_for_transfer,
        transfer_balance_cap=person2_transfer_balance_cap,
        phase_balance_for_tax=person2_pension_after_transfer,
    )

    person1_pension_after_minimum = person1_pension_after_transfer - person1_min_pension_drawdown
    person2_pension_after_minimum = person2_pension_after_transfer - person2_min_pension_drawdown

    person1_pension_cost_base_after_minimum = person1_min_pension_cgt["remaining_cost_base"]
    person2_pension_cost_base_after_minimum = person2_min_pension_cgt["remaining_cost_base"]

    household_salary_net_income = person1_net_income + person2_net_income
    household_minimum_pension_drawdown = (
        person1_min_pension_drawdown + person2_min_pension_drawdown
    )

    total_cash_contributions = (
        person1_total_cash_contribution + person2_total_cash_contribution
    )

    household_cash_available_before_extra_withdrawals = (
        household_salary_net_income + household_minimum_pension_drawdown
    )

    required_cash_outflow = (
        current_spending
        + total_cash_contributions
        + max(float(non_deductible_interest_expense), 0.0)
        + max(float(additional_required_cash_outflow), 0.0)
    )

    non_super_withdrawal = 0.0
    surplus_cash_to_non_super = 0.0

    person1_extra_accum_withdrawal = 0.0
    person2_extra_accum_withdrawal = 0.0
    person1_extra_pension_withdrawal = 0.0
    person2_extra_pension_withdrawal = 0.0
    total_extra_super_withdrawal = 0.0
    unmet_shortfall = 0.0
    cash_reserve_withdrawal = 0.0
    ending_cash_reserve_balance = max(float(opening_cash_reserve_balance), 0.0)
    residential_property_sale_proceeds = 0.0
    residential_property_disposal_fraction = 0.0
    remaining_residential_property_value = max(float(opening_residential_property_value), 0.0)
    remaining_residential_property_loan_balance = max(float(opening_residential_property_loan_balance), 0.0)
    remaining_non_deductible_debt_balance = max(float(opening_non_deductible_debt_balance), 0.0)
    remaining_main_residence_loan_balance = max(float(opening_main_residence_loan_balance), 0.0)
    ending_main_residence_offset_balance = min(max(float(opening_main_residence_offset_balance), 0.0), remaining_main_residence_loan_balance)
    remaining_investment_deductible_debt_balance = max(float(opening_investment_deductible_debt_balance), 0.0)
    ending_investment_deductible_offset_balance = min(max(float(opening_investment_deductible_offset_balance), 0.0), remaining_investment_deductible_debt_balance)
    ending_non_deductible_offset_balance = min(
        max(float(opening_non_deductible_offset_balance), 0.0),
        remaining_non_deductible_debt_balance,
    )
    ending_deductible_offset_balance = min(
        max(float(opening_deductible_offset_balance), 0.0),
        remaining_residential_property_loan_balance,
    )
    cash_reserve_top_up = 0.0
    non_deductible_offset_contribution = 0.0
    non_deductible_principal_repayment = 0.0
    deductible_offset_contribution = 0.0
    deductible_principal_repayment = 0.0
    non_deductible_offset_withdrawal = 0.0
    deductible_offset_withdrawal = 0.0

    if household_cash_available_before_extra_withdrawals >= required_cash_outflow:
        annual_surplus = (
            household_cash_available_before_extra_withdrawals - required_cash_outflow
        )
        surplus_result = allocate_cash_surplus(
            surplus=annual_surplus,
            allocation_order=surplus_allocation_order or ["non_super"],
            cash_reserve_balance=opening_cash_reserve_balance,
            cash_reserve_target=cash_reserve_target,
            non_deductible_debt_balance=remaining_non_deductible_debt_balance,
            non_deductible_offset_balance=ending_non_deductible_offset_balance,
            deductible_debt_balance=remaining_residential_property_loan_balance,
            deductible_offset_balance=ending_deductible_offset_balance,
            main_residence_debt_balance=remaining_main_residence_loan_balance,
            main_residence_offset_balance=ending_main_residence_offset_balance,
            investment_deductible_debt_balance=remaining_investment_deductible_debt_balance,
            investment_deductible_offset_balance=ending_investment_deductible_offset_balance,
            allow_non_super_investment=allow_non_super_investment,
        )
        surplus_cash_to_non_super = surplus_result["surplus_cash_to_non_super"]
        ending_cash_reserve_balance = surplus_result["ending_cash_reserve_balance"]
        remaining_non_deductible_debt_balance = surplus_result["ending_non_deductible_debt_balance"]
        ending_non_deductible_offset_balance = surplus_result["ending_non_deductible_offset_balance"]
        remaining_residential_property_loan_balance = surplus_result["ending_deductible_debt_balance"]
        remaining_main_residence_loan_balance = surplus_result["ending_main_residence_debt_balance"]
        ending_main_residence_offset_balance = surplus_result["ending_main_residence_offset_balance"]
        remaining_investment_deductible_debt_balance = surplus_result["ending_investment_deductible_debt_balance"]
        ending_investment_deductible_offset_balance = surplus_result["ending_investment_deductible_offset_balance"]
        ending_deductible_offset_balance = surplus_result["ending_deductible_offset_balance"]
        cash_reserve_top_up = surplus_result["cash_reserve_top_up"]
        non_deductible_offset_contribution = surplus_result["non_deductible_offset_contribution"]
        non_deductible_principal_repayment = surplus_result["non_deductible_principal_repayment"]
        deductible_offset_contribution = surplus_result["deductible_offset_contribution"]
        deductible_principal_repayment = surplus_result["deductible_principal_repayment"]
    else:
        cash_shortfall = required_cash_outflow - household_cash_available_before_extra_withdrawals

        accessible_cash_balance = (
            max(float(opening_cash_reserve_balance), 0.0)
            + ending_non_deductible_offset_balance
            + ending_deductible_offset_balance
        )

        allocation = allocate_shortfall_by_asset_order(
            required_amount=cash_shortfall,
            withdrawal_order=withdrawal_order,
            cash_balance=accessible_cash_balance,
            cash_floor=cash_reserve_floor,
            non_super_balance=opening_non_super_balance,
            non_super_floor=non_super_estate_reserve,
            person1_accum_balance=person1_accum_after_transfer,
            person2_accum_balance=person2_accum_after_transfer,
            person1_pension_balance=person1_pension_after_minimum,
            person2_pension_balance=person2_pension_after_minimum,
            property_value=opening_residential_property_value,
            property_loan_balance=opening_residential_property_loan_balance,
            property_equity_floor=property_estate_reserve,
            property_sale_cost_rate=residential_property_sale_cost_rate,
        )

        cash_reserve_withdrawal = allocation["cash_withdrawal"]
        ordinary_cash_available = max(float(opening_cash_reserve_balance) - float(cash_reserve_floor), 0.0)
        ordinary_cash_withdrawal = min(cash_reserve_withdrawal, ordinary_cash_available)
        offset_withdrawal_remaining = max(cash_reserve_withdrawal - ordinary_cash_withdrawal, 0.0)
        non_deductible_offset_withdrawal = min(offset_withdrawal_remaining, ending_non_deductible_offset_balance)
        ending_non_deductible_offset_balance -= non_deductible_offset_withdrawal
        offset_withdrawal_remaining -= non_deductible_offset_withdrawal
        deductible_offset_withdrawal = min(offset_withdrawal_remaining, ending_deductible_offset_balance)
        ending_deductible_offset_balance -= deductible_offset_withdrawal
        ending_cash_reserve_balance = max(float(opening_cash_reserve_balance) - ordinary_cash_withdrawal, 0.0)
        non_super_withdrawal = allocation["non_super_withdrawal"]
        person1_extra_accum_withdrawal = allocation["person1_extra_accum_withdrawal"]
        person2_extra_accum_withdrawal = allocation["person2_extra_accum_withdrawal"]
        person1_extra_pension_withdrawal = allocation["person1_extra_pension_withdrawal"]
        person2_extra_pension_withdrawal = allocation["person2_extra_pension_withdrawal"]
        total_extra_super_withdrawal = allocation["total_extra_super_withdrawal"]
        residential_property_sale_proceeds = allocation["residential_property_sale_proceeds"]
        residential_property_disposal_fraction = allocation["residential_property_disposal_fraction"]
        remaining_residential_property_value = allocation["remaining_residential_property_value"]
        remaining_residential_property_loan_balance = allocation["remaining_residential_property_loan_balance"]
        ending_deductible_offset_balance = min(
            ending_deductible_offset_balance,
            remaining_residential_property_loan_balance,
        )
        unmet_shortfall = allocation["unfunded_after_assets"]

    # ---------- Non-super withdrawal CGT ----------
    non_super_sale_result = calculate_budget_cgt_on_sale(
        sale_proceeds=non_super_withdrawal,
        pool_market_value=opening_non_super_balance,
        pool_cost_base=opening_non_super_cost_base,
        pool_indexed_cost_base=opening_non_super_indexed_cost_base,
        pool_deferred_pre_2027_gain=opening_non_super_deferred_pre_2027_gain,
        opening_capital_losses=opening_non_super_capital_losses,
        financial_year_end=financial_year_end,
        cgt_discount_rate=cgt_discount_rate,
        indexation_rate=cgt_indexation_rate,
        asset_category=cgt_asset_category,
        new_residential_method=cgt_new_residential_method,
        held_at_least_12_months=cgt_held_at_least_12_months,
        reform_enabled=cgt_reform_enabled,
    )

    non_super_cost_base_after_withdrawal = non_super_sale_result["remaining_cost_base"]
    non_super_cost_base_before_return = (
        non_super_cost_base_after_withdrawal + surplus_cash_to_non_super
    )
    non_super_indexed_cost_base_before_return = (
        non_super_sale_result["remaining_indexed_cost_base"] + surplus_cash_to_non_super
    )

    # ---------- Extra super withdrawal CGT ----------
    person1_extra_accum_cgt = calculate_super_withdrawal_cgt(
        withdrawal_amount=person1_extra_accum_withdrawal,
        account_balance=person1_accum_after_transfer,
        account_cost_base=person1_accum_cost_base_after_transfer,
        phase="accumulation_phase",
    )
    person2_extra_accum_cgt = calculate_super_withdrawal_cgt(
        withdrawal_amount=person2_extra_accum_withdrawal,
        account_balance=person2_accum_after_transfer,
        account_cost_base=person2_accum_cost_base_after_transfer,
        phase="accumulation_phase",
    )
    person1_extra_pension_cgt = calculate_super_withdrawal_cgt(
        withdrawal_amount=person1_extra_pension_withdrawal,
        account_balance=person1_pension_after_minimum,
        account_cost_base=person1_pension_cost_base_after_minimum,
        phase=person1_super_phase_for_transfer,
        transfer_balance_cap=person1_transfer_balance_cap,
        phase_balance_for_tax=person1_pension_after_minimum,
    )
    person2_extra_pension_cgt = calculate_super_withdrawal_cgt(
        withdrawal_amount=person2_extra_pension_withdrawal,
        account_balance=person2_pension_after_minimum,
        account_cost_base=person2_pension_cost_base_after_minimum,
        phase=person2_super_phase_for_transfer,
        transfer_balance_cap=person2_transfer_balance_cap,
        phase_balance_for_tax=person2_pension_after_minimum,
    )

    person1_accum_cost_base_after_extra = person1_extra_accum_cgt["remaining_cost_base"]
    person2_accum_cost_base_after_extra = person2_extra_accum_cgt["remaining_cost_base"]
    person1_pension_cost_base_after_extra = person1_extra_pension_cgt["remaining_cost_base"]
    person2_pension_cost_base_after_extra = person2_extra_pension_cgt["remaining_cost_base"]

    person1_accum_before_return = (
        person1_accum_after_transfer
        - person1_extra_accum_withdrawal
        + person1_total_net_super_contribution
    )
    person2_accum_before_return = (
        person2_accum_after_transfer
        - person2_extra_accum_withdrawal
        + person2_total_net_super_contribution
    )
    person1_pension_before_return = (
        person1_pension_after_minimum - person1_extra_pension_withdrawal
    )
    person2_pension_before_return = (
        person2_pension_after_minimum - person2_extra_pension_withdrawal
    )
    non_super_before_return = max(
        opening_non_super_balance
        - non_super_withdrawal
        + surplus_cash_to_non_super,
        0.0,
    )

    person1_accum_cost_base_before_return = (
        person1_accum_cost_base_after_extra + person1_total_net_super_contribution
    )
    person2_accum_cost_base_before_return = (
        person2_accum_cost_base_after_extra + person2_total_net_super_contribution
    )
    person1_pension_cost_base_before_return = person1_pension_cost_base_after_extra
    person2_pension_cost_base_before_return = person2_pension_cost_base_after_extra

    person1_super_realised_capital_gain = (
        person1_min_pension_cgt["realised_capital_gain"]
        + person1_extra_accum_cgt["realised_capital_gain"]
        + person1_extra_pension_cgt["realised_capital_gain"]
    )
    person2_super_realised_capital_gain = (
        person2_min_pension_cgt["realised_capital_gain"]
        + person2_extra_accum_cgt["realised_capital_gain"]
        + person2_extra_pension_cgt["realised_capital_gain"]
    )

    person1_super_discounted_taxable_capital_gain = (
        person1_min_pension_cgt["taxable_discounted_capital_gain"]
        + person1_extra_accum_cgt["taxable_discounted_capital_gain"]
        + person1_extra_pension_cgt["taxable_discounted_capital_gain"]
    )
    person2_super_discounted_taxable_capital_gain = (
        person2_min_pension_cgt["taxable_discounted_capital_gain"]
        + person2_extra_accum_cgt["taxable_discounted_capital_gain"]
        + person2_extra_pension_cgt["taxable_discounted_capital_gain"]
    )

    person1_super_withdrawal_cgt_tax = (
        person1_min_pension_cgt["cgt_tax_paid"]
        + person1_extra_accum_cgt["cgt_tax_paid"]
        + person1_extra_pension_cgt["cgt_tax_paid"]
    )
    person2_super_withdrawal_cgt_tax = (
        person2_min_pension_cgt["cgt_tax_paid"]
        + person2_extra_accum_cgt["cgt_tax_paid"]
        + person2_extra_pension_cgt["cgt_tax_paid"]
    )

    return {
        "household_salary_net_income": household_salary_net_income,
        "household_minimum_pension_drawdown": household_minimum_pension_drawdown,
        "household_cash_available_before_extra_withdrawals": household_cash_available_before_extra_withdrawals,
        "required_cash_outflow": required_cash_outflow,
        "total_cash_contributions": total_cash_contributions,
        "surplus_cash_to_non_super": surplus_cash_to_non_super,
        "cash_reserve_top_up": cash_reserve_top_up,
        "cash_reserve_withdrawal": cash_reserve_withdrawal,
        "ending_cash_reserve_balance": ending_cash_reserve_balance,
        "non_deductible_interest_expense": max(float(non_deductible_interest_expense), 0.0),
        "non_deductible_offset_contribution": non_deductible_offset_contribution,
        "non_deductible_offset_withdrawal": non_deductible_offset_withdrawal,
        "non_deductible_principal_repayment": non_deductible_principal_repayment,
        "ending_non_deductible_debt_balance": remaining_non_deductible_debt_balance,
        "ending_non_deductible_offset_balance": ending_non_deductible_offset_balance,
        "deductible_offset_contribution": deductible_offset_contribution,
        "deductible_offset_withdrawal": deductible_offset_withdrawal,
        "deductible_principal_repayment": deductible_principal_repayment,
        "ending_deductible_offset_balance": ending_deductible_offset_balance,
        "ending_main_residence_loan_balance": remaining_main_residence_loan_balance,
        "ending_main_residence_offset_balance": ending_main_residence_offset_balance,
        "ending_investment_deductible_debt_balance": remaining_investment_deductible_debt_balance,
        "ending_investment_deductible_offset_balance": ending_investment_deductible_offset_balance,
        "main_residence_offset_contribution": surplus_result.get("main_residence_offset_contribution", 0.0) if 'surplus_result' in locals() else 0.0,
        "main_residence_extra_principal_repayment": surplus_result.get("main_residence_principal_repayment", 0.0) if 'surplus_result' in locals() else 0.0,
        "investment_deductible_offset_contribution": surplus_result.get("investment_deductible_offset_contribution", 0.0) if 'surplus_result' in locals() else 0.0,
        "investment_deductible_principal_repayment": surplus_result.get("investment_deductible_principal_repayment", 0.0) if 'surplus_result' in locals() else 0.0,
        "non_super_withdrawal": non_super_withdrawal,
        "non_super_cost_base_after_withdrawal": non_super_cost_base_after_withdrawal,
        "non_super_cost_base_before_return": non_super_cost_base_before_return,
        "non_super_indexed_cost_base_before_return": non_super_indexed_cost_base_before_return,
        "non_super_deferred_pre_2027_gain_after_withdrawal": non_super_sale_result["remaining_deferred_pre_2027_gain"],
        "non_super_capital_losses_after_withdrawal": non_super_sale_result["remaining_capital_losses"],
        "non_super_sale_cost_base_reduction": non_super_sale_result["cost_base_reduction"],
        "non_super_sale_indexed_cost_base_reduction": non_super_sale_result["indexed_cost_base_reduction"],
        "non_super_realised_capital_gain": non_super_sale_result["realised_capital_gain"],
        "non_super_realised_capital_loss": non_super_sale_result["realised_capital_loss"],
        "non_super_deferred_pre_2027_gain": non_super_sale_result["deferred_pre_2027_gain"],
        "non_super_post_2027_real_gain": non_super_sale_result["post_2027_real_gain"],
        "non_super_minimum_tax_capital_gain": non_super_sale_result["minimum_tax_capital_gain"],
        "non_super_capital_losses_applied": non_super_sale_result["capital_losses_applied"],
        "non_super_indexation_uplift": non_super_sale_result["indexation_uplift"],
        "non_super_cgt_reform_applies": non_super_sale_result["reform_applies"],
        "non_super_cgt_calculation_method": non_super_sale_result["calculation_method"],
        "non_super_discounted_taxable_capital_gain": non_super_sale_result["discounted_taxable_capital_gain"],
        "person1_extra_accum_withdrawal": person1_extra_accum_withdrawal,
        "person2_extra_accum_withdrawal": person2_extra_accum_withdrawal,
        "person1_extra_pension_withdrawal": person1_extra_pension_withdrawal,
        "person2_extra_pension_withdrawal": person2_extra_pension_withdrawal,
        "total_extra_super_withdrawal": total_extra_super_withdrawal,
        "residential_property_sale_proceeds": residential_property_sale_proceeds,
        "residential_property_disposal_fraction": residential_property_disposal_fraction,
        "remaining_residential_property_value": remaining_residential_property_value,
        "remaining_residential_property_loan_balance": remaining_residential_property_loan_balance,
        "unmet_shortfall": unmet_shortfall,
        "person1_accum_before_return": person1_accum_before_return,
        "person2_accum_before_return": person2_accum_before_return,
        "person1_pension_before_return": person1_pension_before_return,
        "person2_pension_before_return": person2_pension_before_return,
        "non_super_before_return": non_super_before_return,
        "person1_accum_cost_base_before_return": person1_accum_cost_base_before_return,
        "person2_accum_cost_base_before_return": person2_accum_cost_base_before_return,
        "person1_pension_cost_base_before_return": person1_pension_cost_base_before_return,
        "person2_pension_cost_base_before_return": person2_pension_cost_base_before_return,
        "person1_min_pension_cost_base_reduction": person1_min_pension_cgt["cost_base_reduction"],
        "person2_min_pension_cost_base_reduction": person2_min_pension_cgt["cost_base_reduction"],
        "person1_super_realised_capital_gain": person1_super_realised_capital_gain,
        "person2_super_realised_capital_gain": person2_super_realised_capital_gain,
        "person1_super_discounted_taxable_capital_gain": person1_super_discounted_taxable_capital_gain,
        "person2_super_discounted_taxable_capital_gain": person2_super_discounted_taxable_capital_gain,
        "person1_super_withdrawal_cgt_tax": person1_super_withdrawal_cgt_tax,
        "person2_super_withdrawal_cgt_tax": person2_super_withdrawal_cgt_tax,
        "total_super_withdrawal_cgt_tax": (
            person1_super_withdrawal_cgt_tax + person2_super_withdrawal_cgt_tax
        ),
    }


# ============================================================
# SECTION: ONE YEAR ENGINE
# ============================================================

def run_one_year(
    inputs,
    year_context,
    opening_person1_accum_super_balance,
    opening_person1_pension_super_balance,
    opening_person2_accum_super_balance,
    opening_person2_pension_super_balance,
    opening_person1_accum_super_cost_base,
    opening_person1_pension_super_cost_base,
    opening_person2_accum_super_cost_base,
    opening_person2_pension_super_cost_base,
    opening_non_super_balance,
    opening_non_super_cost_base,
    current_spending,
    super_income_return_rate,
    super_capital_return_rate,
    non_super_income_return_rate,
    non_super_capital_return_rate,
    contribution_event_lookup,
    person1_has_started_pension,
    person2_has_started_pension,
    opening_residential_property_value=0.0,
    opening_residential_quarantined_loss=0.0,
    opening_non_super_indexed_cost_base=0.0,
    opening_non_super_deferred_pre_2027_gain=0.0,
    opening_non_super_capital_losses=0.0,
    opening_cash_reserve_balance=0.0,
    opening_residential_property_loan_balance=None,
    opening_non_deductible_debt_balance=None,
    opening_non_deductible_offset_balance=None,
    opening_deductible_offset_balance=None,
    opening_main_residence_value=None,
    opening_main_residence_loan_balance=None,
    opening_main_residence_offset_balance=None,
    opening_investment_deductible_debt_balance=None,
    opening_investment_deductible_offset_balance=None,
    opening_discretionary_trust_balance=None,
    opening_discretionary_trust_cost_base=None,
    trust_income_return_rate=None,
    trust_capital_return_rate=None,
):
    year_index = year_context["year_index"]
    financial_year_end = year_context["financial_year_end"]
    tax_schedule_key = year_context["tax_schedule_key"]

    person1_age = year_context["person1_age"]
    person2_age = year_context["person2_age"]

    person1_phase = year_context["person1_phase"]
    person2_phase = year_context["person2_phase"]

    person1_is_working = year_context["person1_is_working"]
    person2_is_working = year_context["person2_is_working"]

    person1_is_pension_phase = year_context["person1_is_pension_phase"]
    person2_is_pension_phase = year_context["person2_is_pension_phase"]

    person1_income_indexed = year_context["person1_income_indexed"]
    person2_income_indexed = year_context["person2_income_indexed"]

    person1_super_phase_for_transfer = "pension_phase" if person1_is_pension_phase else "working"
    person2_super_phase_for_transfer = "pension_phase" if person2_is_pension_phase else "working"

    if person1_is_working:
        person1_gross_income = float(person1_income_indexed)
    else:
        person1_gross_income = 0.0

    super_enabled = bool(inputs.get("module_super_enabled", True))
    person1_sg_result = calculate_super_guarantee_contribution(
        gross_income=person1_gross_income if super_enabled else 0.0,
        financial_year_end=financial_year_end,
    )
    person1_sg_contribution = person1_sg_result["sg_contribution"]

    if person2_is_working:
        person2_gross_income = float(person2_income_indexed)
    else:
        person2_gross_income = 0.0

    person2_sg_result = calculate_super_guarantee_contribution(
        gross_income=person2_gross_income if super_enabled else 0.0,
        financial_year_end=financial_year_end,
    )
    person2_sg_contribution = person2_sg_result["sg_contribution"]

    financial_year_lookup_key = str(financial_year_end)

    person1_personal_deductible_contribution = get_scheduled_contribution_amount(
        contribution_event_lookup,
        financial_year_lookup_key,
        "Person 1",
        "personal_deductible",
    )
    person2_personal_deductible_contribution = get_scheduled_contribution_amount(
        contribution_event_lookup,
        financial_year_lookup_key,
        "Person 2",
        "personal_deductible",
    )
    person1_non_concessional_contribution = get_scheduled_contribution_amount(
        contribution_event_lookup,
        financial_year_lookup_key,
        "Person 1",
        "non_concessional",
    )
    person2_non_concessional_contribution = get_scheduled_contribution_amount(
        contribution_event_lookup,
        financial_year_lookup_key,
        "Person 2",
        "non_concessional",
    )

    person1_gross_concessional_contribution = (
        person1_sg_contribution + person1_personal_deductible_contribution
    )
    person2_gross_concessional_contribution = (
        person2_sg_contribution + person2_personal_deductible_contribution
    )

    person1_super_contributions_tax = calculate_super_contributions_tax(
        person1_gross_concessional_contribution
    )
    person2_super_contributions_tax = calculate_super_contributions_tax(
        person2_gross_concessional_contribution
    )

    person1_net_concessional_contribution = (
        person1_gross_concessional_contribution - person1_super_contributions_tax
    )
    person2_net_concessional_contribution = (
        person2_gross_concessional_contribution - person2_super_contributions_tax
    )

    person1_total_net_super_contribution = (
        person1_net_concessional_contribution + person1_non_concessional_contribution
    )
    person2_total_net_super_contribution = (
        person2_net_concessional_contribution + person2_non_concessional_contribution
    )

    person1_transfer_result = auto_transfer_to_pension(
        accum_balance=opening_person1_accum_super_balance,
        pension_balance=opening_person1_pension_super_balance,
        transfer_balance_cap=inputs["person1_transfer_balance_cap"],
        is_pension_phase=person1_is_pension_phase,
        has_started_pension=person1_has_started_pension,
    )
    person2_transfer_result = auto_transfer_to_pension(
        accum_balance=opening_person2_accum_super_balance,
        pension_balance=opening_person2_pension_super_balance,
        transfer_balance_cap=inputs["person2_transfer_balance_cap"],
        is_pension_phase=person2_is_pension_phase,
        has_started_pension=person2_has_started_pension,
    )

    person1_has_started_pension_after_year = person1_transfer_result["has_started_pension_after_year"]
    person2_has_started_pension_after_year = person2_transfer_result["has_started_pension_after_year"]

    person1_cost_base_transfer = transfer_super_cost_base_to_pension(
        accum_balance=opening_person1_accum_super_balance,
        accum_cost_base=opening_person1_accum_super_cost_base,
        pension_cost_base=opening_person1_pension_super_cost_base,
        transfer_to_pension=person1_transfer_result["transfer_to_pension"],
    )
    person2_cost_base_transfer = transfer_super_cost_base_to_pension(
        accum_balance=opening_person2_accum_super_balance,
        accum_cost_base=opening_person2_accum_super_cost_base,
        pension_cost_base=opening_person2_pension_super_cost_base,
        transfer_to_pension=person2_transfer_result["transfer_to_pension"],
    )

    person1_accum_after_transfer = person1_transfer_result["accum_balance_after_transfer"]
    person1_pension_after_transfer = person1_transfer_result["pension_balance_after_transfer"]
    person2_accum_after_transfer = person2_transfer_result["accum_balance_after_transfer"]
    person2_pension_after_transfer = person2_transfer_result["pension_balance_after_transfer"]

    person1_accum_cost_base_after_transfer = person1_cost_base_transfer["accum_cost_base_after_transfer"]
    person1_pension_cost_base_after_transfer = person1_cost_base_transfer["pension_cost_base_after_transfer"]
    person2_accum_cost_base_after_transfer = person2_cost_base_transfer["accum_cost_base_after_transfer"]
    person2_pension_cost_base_after_transfer = person2_cost_base_transfer["pension_cost_base_after_transfer"]

    person1_min_pension_drawdown = calculate_minimum_pension_drawdown(
        opening_pension_balance=person1_pension_after_transfer,
        age=person1_age,
        phase=person1_super_phase_for_transfer,
    )
    person2_min_pension_drawdown = calculate_minimum_pension_drawdown(
        opening_pension_balance=person2_pension_after_transfer,
        age=person2_age,
        phase=person2_super_phase_for_transfer,
    )

    main_residence_opening_value = max(float(
        inputs.get("main_residence_value", 0.0)
        if opening_main_residence_value is None else opening_main_residence_value
    ), 0.0)
    main_residence_growth_rate = float(inputs.get("main_residence_capital_growth_rate", 0.0))
    ending_main_residence_value = max(main_residence_opening_value * (1 + main_residence_growth_rate), 0.0)
    main_residence_opening_loan = max(float(
        inputs.get("main_residence_loan_balance", 0.0)
        if opening_main_residence_loan_balance is None else opening_main_residence_loan_balance
    ), 0.0)
    main_residence_opening_offset = min(max(float(
        inputs.get("main_residence_offset_balance", 0.0)
        if opening_main_residence_offset_balance is None else opening_main_residence_offset_balance
    ), 0.0), main_residence_opening_loan)
    if "main_residence_annual_loan_repayment" in inputs:
        main_residence_loan_result = calculate_loan_year_from_annual_repayment(
            main_residence_opening_loan,
            inputs.get("main_residence_interest_rate", 0.0),
            inputs.get("main_residence_annual_loan_repayment", 0.0),
            main_residence_opening_offset,
        )
    else:
        main_residence_loan_result = calculate_amortising_loan_year(
            main_residence_opening_loan,
            inputs.get("main_residence_interest_rate", 0.0),
            max(int(inputs.get("main_residence_loan_term_years", 1)) - year_index, 1),
            main_residence_opening_offset,
        )

    investment_deductible_debt_balance = max(float(
        inputs.get("investment_deductible_debt_balance", 0.0)
        if opening_investment_deductible_debt_balance is None else opening_investment_deductible_debt_balance
    ), 0.0)
    investment_deductible_offset_balance = min(max(float(
        inputs.get("investment_deductible_offset_balance", 0.0)
        if opening_investment_deductible_offset_balance is None else opening_investment_deductible_offset_balance
    ), 0.0), investment_deductible_debt_balance)
    investment_deductible_interest_only = max(
        investment_deductible_debt_balance - investment_deductible_offset_balance, 0.0
    ) * max(float(inputs.get("investment_deductible_interest_rate", 0.0)), 0.0)
    investment_deductible_loan_result = (
        calculate_loan_year_from_annual_repayment(
            investment_deductible_debt_balance,
            inputs.get("investment_deductible_interest_rate", 0.0),
            inputs.get("investment_deductible_annual_repayment", 0.0),
            investment_deductible_offset_balance,
        )
        if "investment_deductible_annual_repayment" in inputs
        else {
            "payment": investment_deductible_interest_only,
            "interest": investment_deductible_interest_only,
            "principal": 0.0,
            "ending_balance": investment_deductible_debt_balance,
        }
    )
    investment_deductible_interest_expense = investment_deductible_loan_result["interest"]

    residential_property_enabled = bool(inputs.get("residential_property_enabled", False)) and float(opening_residential_property_value) > 0
    original_property_value = max(float(inputs.get("residential_property_value", 0.0)), 0.0)
    property_scale = (
        min(max(float(opening_residential_property_value) / original_property_value, 0.0), 1.0)
        if original_property_value > 0 else 0.0
    )
    property_ownership_person1 = (
        1.0 if is_one_person_mode(inputs)
        else min(max(float(inputs.get("residential_property_ownership_person1", 0.5)), 0.0), 1.0)
    )
    property_ownership_person2 = 1.0 - property_ownership_person1
    property_gross_rent = (
        float(inputs.get("residential_property_gross_rent", 0.0)) * property_scale
        * ((1 + float(inputs.get("residential_property_rent_growth_rate", inputs.get("inflation_rate", 0.0)))) ** year_index)
        if residential_property_enabled else 0.0
    )
    property_operating_expenses = (
        float(inputs.get("residential_property_operating_expenses", 0.0)) * property_scale
        * ((1 + float(inputs.get("residential_property_expense_growth_rate", inputs.get("inflation_rate", 0.0)))) ** year_index)
        if residential_property_enabled else 0.0
    )
    property_loan_balance = (
        max(float(
            inputs.get("residential_property_loan_balance", 0.0)
            if opening_residential_property_loan_balance is None
            else opening_residential_property_loan_balance
        ), 0.0)
        if residential_property_enabled else 0.0
    )
    deductible_offset_balance = min(
        max(float(
            inputs.get("deductible_offset_balance", 0.0)
            if opening_deductible_offset_balance is None
            else opening_deductible_offset_balance
        ), 0.0),
        property_loan_balance,
    )
    if "residential_property_annual_loan_repayment" in inputs:
        property_loan_result = calculate_loan_year_from_annual_repayment(
            property_loan_balance,
            inputs.get("residential_property_interest_rate", 0.0),
            inputs.get("residential_property_annual_loan_repayment", 0.0),
            deductible_offset_balance,
        )
    elif "residential_property_loan_term_years" in inputs:
        property_loan_result = calculate_amortising_loan_year(
            property_loan_balance,
            inputs.get("residential_property_interest_rate", 0.0),
            max(int(inputs.get("residential_property_loan_term_years", 1)) - year_index, 1),
            deductible_offset_balance,
        )
    else:
        legacy_property_interest = max(property_loan_balance - deductible_offset_balance, 0.0) * max(
            float(inputs.get("residential_property_interest_rate", 0.0)), 0.0
        )
        property_loan_result = {
            "payment": legacy_property_interest,
            "interest": legacy_property_interest,
            "principal": 0.0,
            "ending_balance": property_loan_balance,
        }
    property_loan_interest = property_loan_result["interest"]
    property_loan_balance_after_scheduled_repayment = property_loan_result["ending_balance"]
    non_deductible_debt_balance = max(float(
        inputs.get("non_deductible_debt_balance", 0.0)
        if opening_non_deductible_debt_balance is None
        else opening_non_deductible_debt_balance
    ), 0.0)
    non_deductible_offset_balance = min(
        max(float(
            inputs.get("non_deductible_offset_balance", 0.0)
            if opening_non_deductible_offset_balance is None
            else opening_non_deductible_offset_balance
        ), 0.0),
        non_deductible_debt_balance,
    )
    non_deductible_interest_only = max(
        non_deductible_debt_balance - non_deductible_offset_balance,
        0.0,
    ) * max(float(inputs.get("non_deductible_interest_rate", 0.0)), 0.0)
    non_deductible_loan_result = (
        calculate_loan_year_from_annual_repayment(
            non_deductible_debt_balance,
            inputs.get("non_deductible_interest_rate", 0.0),
            inputs.get("non_deductible_annual_repayment", 0.0),
            non_deductible_offset_balance,
        )
        if "non_deductible_annual_repayment" in inputs
        else {
            "payment": non_deductible_interest_only,
            "interest": non_deductible_interest_only,
            "principal": 0.0,
            "ending_balance": non_deductible_debt_balance,
        }
    )
    non_deductible_interest_expense = non_deductible_loan_result["interest"]
    residential_result = calculate_residential_property_year(
        gross_rent=property_gross_rent,
        deductible_operating_expenses=property_operating_expenses,
        loan_interest=property_loan_interest,
        opening_quarantined_loss=opening_residential_quarantined_loss,
        financial_year_end=financial_year_end,
        acquired_before_budget_time=inputs.get("residential_property_acquired_before_budget_time", False),
        is_new_build=inputs.get("residential_property_is_new_build", False),
        is_exempt_housing=inputs.get("residential_property_is_exempt_housing", False),
    )
    property_growth_rate = float(inputs.get("residential_property_capital_growth_rate", 0.0))
    ending_residential_property_value = max(
        float(opening_residential_property_value) * (1 + property_growth_rate),
        0.0,
    ) if residential_property_enabled else 0.0

    discretionary_trust_enabled = bool(inputs.get("discretionary_trust_enabled", False))
    trust_uses_asset_pool = "discretionary_trust_balance" in inputs
    trust_opening_balance = (
        max(float(inputs.get("discretionary_trust_balance", 0.0)), 0.0)
        if opening_discretionary_trust_balance is None
        else max(float(opening_discretionary_trust_balance), 0.0)
    ) if discretionary_trust_enabled else 0.0
    trust_opening_cost_base = (
        min(max(float(inputs.get("discretionary_trust_cost_base", 0.0)), 0.0), trust_opening_balance)
        if opening_discretionary_trust_cost_base is None
        else min(max(float(opening_discretionary_trust_cost_base), 0.0), trust_opening_balance)
    ) if discretionary_trust_enabled else 0.0
    effective_trust_income_rate = float(
        inputs.get("discretionary_trust_income_return_mean", 0.0)
        if trust_income_return_rate is None else trust_income_return_rate
    )
    effective_trust_capital_rate = float(
        inputs.get("discretionary_trust_capital_return_mean", 0.0)
        if trust_capital_return_rate is None else trust_capital_return_rate
    )
    if trust_uses_asset_pool:
        trust_net_income = max(trust_opening_balance * effective_trust_income_rate, 0.0)
        trust_excluded_income_pct = min(max(float(inputs.get("discretionary_trust_excluded_income_pct", 0.0)), 0.0), 1.0)
        trust_excluded_income = trust_net_income * trust_excluded_income_pct
        trust_capital_earnings = trust_opening_balance * effective_trust_capital_rate
    else:
        trust_income_growth_rate = float(inputs.get("discretionary_trust_income_growth_rate", inputs.get("inflation_rate", 0.0)))
        trust_net_income = float(inputs.get("discretionary_trust_net_income", 0.0)) * ((1 + trust_income_growth_rate) ** year_index) if discretionary_trust_enabled else 0.0
        trust_excluded_income = float(inputs.get("discretionary_trust_excluded_income", 0.0)) * ((1 + trust_income_growth_rate) ** year_index) if discretionary_trust_enabled else 0.0
        trust_excluded_income_pct = trust_excluded_income / trust_net_income if trust_net_income > 0 else 0.0
        trust_capital_earnings = 0.0
    ending_discretionary_trust_balance = max(trust_opening_balance + trust_capital_earnings, 0.0)
    ending_discretionary_trust_cost_base = min(trust_opening_cost_base, ending_discretionary_trust_balance)
    trust_result = calculate_discretionary_trust_minimum_tax(
        trust_net_income=trust_net_income,
        excluded_income=trust_excluded_income,
        financial_year_end=financial_year_end,
        subject_to_minimum_tax=inputs.get("discretionary_trust_subject_to_minimum_tax", True),
    )
    trust_ownership_person1 = (
        1.0 if is_one_person_mode(inputs)
        else min(max(float(inputs.get("discretionary_trust_ownership_person1", 0.5)), 0.0), 1.0)
    )
    trust_ownership_person2 = 1.0 - trust_ownership_person1

    taxable_non_super_guess = max(opening_non_super_balance * non_super_income_return_rate, 0.0)
    minimum_tax_capital_gain_guess = 0.0
    cgt_minimum_tax_exempt = bool(inputs.get("cgt_minimum_tax_exempt", False))

    for _ in range(3):
        tax_split = calculate_household_personal_tax_split(
            person1_salary_income=person1_gross_income,
            person2_salary_income=person2_gross_income,
            taxable_non_super_earnings_total=taxable_non_super_guess,
            ownership_person1=inputs["non_super_ownership_person1"],
            person1_personal_deductible_contribution=(
                person1_personal_deductible_contribution
                + investment_deductible_interest_expense * float(inputs.get("non_super_ownership_person1", 1.0))
            ),
            person2_personal_deductible_contribution=(
                person2_personal_deductible_contribution
                + investment_deductible_interest_expense * (1.0 - float(inputs.get("non_super_ownership_person1", 1.0)))
            ),
            person1_gross_concessional_contribution=person1_gross_concessional_contribution,
            person2_gross_concessional_contribution=person2_gross_concessional_contribution,
            financial_year_end=financial_year_end,
            tax_schedule_key=tax_schedule_key,
        )

        person1_budget_tax = calculate_incremental_budget_tax(
            base_taxable_income=tax_split["person1_taxable_income"],
            residential_taxable_income=residential_result["taxable_rental_income"] * property_ownership_person1,
            trust_taxable_income=trust_result["trust_net_income"] * trust_ownership_person1,
            trust_tax_credit=trust_result["trustee_minimum_tax"] * trust_ownership_person1,
            tax_schedule_key=tax_schedule_key,
        )
        person2_budget_tax = calculate_incremental_budget_tax(
            base_taxable_income=tax_split["person2_taxable_income"],
            residential_taxable_income=residential_result["taxable_rental_income"] * property_ownership_person2,
            trust_taxable_income=trust_result["trust_net_income"] * trust_ownership_person2,
            trust_tax_credit=trust_result["trustee_minimum_tax"] * trust_ownership_person2,
            tax_schedule_key=tax_schedule_key,
        )
        person1_division_293 = calculate_division_293_tax(
            division_293_income=(
                person1_budget_tax["adjusted_taxable_income"]
                + max(-residential_result["taxable_rental_income"] * property_ownership_person1, 0.0)
            ),
            concessional_contributions=person1_gross_concessional_contribution,
            financial_year_end=financial_year_end,
        )
        person2_division_293 = calculate_division_293_tax(
            division_293_income=(
                person2_budget_tax["adjusted_taxable_income"]
                + max(-residential_result["taxable_rental_income"] * property_ownership_person2, 0.0)
            ),
            concessional_contributions=person2_gross_concessional_contribution,
            financial_year_end=financial_year_end,
        )
        person1_cgt_minimum_tax = calculate_cgt_minimum_tax_gap(
            taxable_income=person1_budget_tax["adjusted_taxable_income"],
            minimum_tax_capital_gain=minimum_tax_capital_gain_guess * inputs["non_super_ownership_person1"],
            tax_schedule_key=tax_schedule_key,
            exempt_from_minimum_tax=cgt_minimum_tax_exempt,
        )
        person2_cgt_minimum_tax = calculate_cgt_minimum_tax_gap(
            taxable_income=person2_budget_tax["adjusted_taxable_income"],
            minimum_tax_capital_gain=minimum_tax_capital_gain_guess * (1.0 - inputs["non_super_ownership_person1"]),
            tax_schedule_key=tax_schedule_key,
            exempt_from_minimum_tax=cgt_minimum_tax_exempt,
        )

        person1_property_and_trust_cash = (
            residential_result["net_cashflow"] * property_ownership_person1
            + trust_result["trust_net_income"] * trust_ownership_person1
            - trust_result["trustee_minimum_tax"] * trust_ownership_person1
            - person1_budget_tax["personal_tax_adjustment"]
        )
        person2_property_and_trust_cash = (
            residential_result["net_cashflow"] * property_ownership_person2
            + trust_result["trust_net_income"] * trust_ownership_person2
            - trust_result["trustee_minimum_tax"] * trust_ownership_person2
            - person2_budget_tax["personal_tax_adjustment"]
        )

        person1_net_income = (
            person1_gross_income
            - tax_split["person1_salary_tax_total"]
            - person1_division_293["division_293_tax"]
            - person1_cgt_minimum_tax["cgt_minimum_tax_gap"]
            + person1_property_and_trust_cash
        )
        person2_net_income = (
            person2_gross_income
            - tax_split["person2_salary_tax_total"]
            - person2_division_293["division_293_tax"]
            - person2_cgt_minimum_tax["cgt_minimum_tax_gap"]
            + person2_property_and_trust_cash
        )

        cashflow = solve_cashflow_before_returns(
            person1_net_income=person1_net_income,
            person2_net_income=person2_net_income,
            person1_min_pension_drawdown=person1_min_pension_drawdown,
            person2_min_pension_drawdown=person2_min_pension_drawdown,
            person1_accum_after_transfer=person1_accum_after_transfer,
            person1_pension_after_transfer=person1_pension_after_transfer,
            person2_accum_after_transfer=person2_accum_after_transfer,
            person2_pension_after_transfer=person2_pension_after_transfer,
            person1_accum_cost_base_after_transfer=person1_accum_cost_base_after_transfer,
            person1_pension_cost_base_after_transfer=person1_pension_cost_base_after_transfer,
            person2_accum_cost_base_after_transfer=person2_accum_cost_base_after_transfer,
            person2_pension_cost_base_after_transfer=person2_pension_cost_base_after_transfer,
            person1_super_phase_for_transfer=person1_super_phase_for_transfer,
            person2_super_phase_for_transfer=person2_super_phase_for_transfer,
            person1_transfer_balance_cap=inputs["person1_transfer_balance_cap"],
            person2_transfer_balance_cap=inputs["person2_transfer_balance_cap"],
            opening_non_super_balance=opening_non_super_balance,
            opening_non_super_cost_base=opening_non_super_cost_base,
            current_spending=current_spending,
            person1_total_cash_contribution=(
                person1_personal_deductible_contribution + person1_non_concessional_contribution
            ),
            person2_total_cash_contribution=(
                person2_personal_deductible_contribution + person2_non_concessional_contribution
            ),
            person1_total_net_super_contribution=person1_total_net_super_contribution,
            person2_total_net_super_contribution=person2_total_net_super_contribution,
            cgt_discount_rate=inputs.get("cgt_discount_rate", 0.50),
            financial_year_end=financial_year_end,
            opening_non_super_indexed_cost_base=opening_non_super_indexed_cost_base,
            opening_non_super_deferred_pre_2027_gain=opening_non_super_deferred_pre_2027_gain,
            opening_non_super_capital_losses=opening_non_super_capital_losses,
            cgt_indexation_rate=inputs.get("cgt_indexation_rate", inputs.get("inflation_rate", 0.0)),
            cgt_asset_category=inputs.get("cgt_asset_category", "Other"),
            cgt_new_residential_method=inputs.get("cgt_new_residential_method", "Indexation and 30% minimum tax"),
            cgt_held_at_least_12_months=inputs.get("cgt_held_at_least_12_months", True),
            cgt_reform_enabled=inputs.get("cgt_reform_enabled", True),
            opening_cash_reserve_balance=opening_cash_reserve_balance,
            cash_reserve_floor=inputs.get("cash_reserve_floor", 0.0),
            withdrawal_order=inputs.get("withdrawal_order"),
            non_super_estate_reserve=inputs.get("non_super_estate_reserve", 0.0),
            opening_residential_property_value=opening_residential_property_value,
            opening_residential_property_loan_balance=property_loan_balance_after_scheduled_repayment,
            property_estate_reserve=inputs.get("property_estate_reserve", 0.0),
            residential_property_sale_cost_rate=inputs.get("residential_property_sale_cost_rate", 0.0),
            opening_non_deductible_debt_balance=non_deductible_loan_result["ending_balance"],
            opening_non_deductible_offset_balance=non_deductible_offset_balance,
            opening_deductible_offset_balance=deductible_offset_balance,
            non_deductible_interest_expense=non_deductible_interest_expense,
            opening_main_residence_loan_balance=main_residence_loan_result["ending_balance"],
            opening_main_residence_offset_balance=main_residence_opening_offset,
            opening_investment_deductible_debt_balance=investment_deductible_loan_result["ending_balance"],
            opening_investment_deductible_offset_balance=investment_deductible_offset_balance,
            additional_required_cash_outflow=(
                property_loan_result["principal"]
                + main_residence_loan_result["payment"]
                + non_deductible_loan_result["principal"]
                + investment_deductible_loan_result["payment"]
            ),
            cash_reserve_target=inputs.get("cash_reserve_target", inputs.get("cash_reserve_floor", 0.0)),
            surplus_allocation_order=inputs.get("surplus_allocation_order", ["non_super"]),
            allow_non_super_investment=bool(inputs.get("module_non_super_enabled", True)),
        )

        taxable_non_super_guess = max(
            cashflow["non_super_before_return"] * non_super_income_return_rate,
            0.0,
        ) + max(cashflow["non_super_discounted_taxable_capital_gain"], 0.0)
        minimum_tax_capital_gain_guess = max(
            cashflow["non_super_minimum_tax_capital_gain"], 0.0
        )

    ending_residential_property_value = (
        cashflow["remaining_residential_property_value"] * (1 + property_growth_rate)
        if residential_property_enabled else 0.0
    )
    ending_residential_property_loan_balance = cashflow["remaining_residential_property_loan_balance"]
    ending_non_deductible_debt_balance = cashflow["ending_non_deductible_debt_balance"]
    ending_non_deductible_offset_balance = cashflow["ending_non_deductible_offset_balance"]
    ending_deductible_offset_balance = cashflow["ending_deductible_offset_balance"]
    ending_main_residence_loan_balance = cashflow["ending_main_residence_loan_balance"]
    ending_main_residence_offset_balance = cashflow["ending_main_residence_offset_balance"]
    ending_investment_deductible_debt_balance = cashflow["ending_investment_deductible_debt_balance"]
    ending_investment_deductible_offset_balance = cashflow["ending_investment_deductible_offset_balance"]

    total_super_return_rate = super_income_return_rate + super_capital_return_rate

    person1_super_earnings_result = calculate_super_account_earnings_tax(
        accum_balance_before_return=cashflow["person1_accum_before_return"],
        pension_balance_before_return=cashflow["person1_pension_before_return"],
        return_rate=total_super_return_rate,
        transfer_balance_cap=inputs["person1_transfer_balance_cap"],
    )
    person2_super_earnings_result = calculate_super_account_earnings_tax(
        accum_balance_before_return=cashflow["person2_accum_before_return"],
        pension_balance_before_return=cashflow["person2_pension_before_return"],
        return_rate=total_super_return_rate,
        transfer_balance_cap=inputs["person2_transfer_balance_cap"],
    )

    person1_accum_income_earnings = cashflow["person1_accum_before_return"] * super_income_return_rate
    person1_pension_income_earnings = cashflow["person1_pension_before_return"] * super_income_return_rate
    person2_accum_income_earnings = cashflow["person2_accum_before_return"] * super_income_return_rate
    person2_pension_income_earnings = cashflow["person2_pension_before_return"] * super_income_return_rate

    person1_accum_capital_earnings = cashflow["person1_accum_before_return"] * super_capital_return_rate
    person1_pension_capital_earnings = cashflow["person1_pension_before_return"] * super_capital_return_rate
    person2_accum_capital_earnings = cashflow["person2_accum_before_return"] * super_capital_return_rate
    person2_pension_capital_earnings = cashflow["person2_pension_before_return"] * super_capital_return_rate

    non_super_income_earnings = cashflow["non_super_before_return"] * non_super_income_return_rate
    non_super_capital_earnings = cashflow["non_super_before_return"] * non_super_capital_return_rate
    non_super_total_return = non_super_income_earnings + non_super_capital_earnings

    # ---------- SUPER BALANCE: cap actual CGT tax paid at available accumulation balance ----------
    person1_accum_available_before_cgt_tax = (
        cashflow["person1_accum_before_return"]
        + person1_super_earnings_result["accum_earnings"]
        - person1_super_earnings_result["accum_earnings_tax"]
    )
    person2_accum_available_before_cgt_tax = (
        cashflow["person2_accum_before_return"]
        + person2_super_earnings_result["accum_earnings"]
        - person2_super_earnings_result["accum_earnings_tax"]
    )

    # --- Person 1 CGT correction ---
    person1_theoretical_tax = cashflow["person1_super_withdrawal_cgt_tax"]

    person1_available_for_cgt = max(
        cashflow["person1_extra_accum_withdrawal"], 0.0
    )

    if person1_theoretical_tax > person1_available_for_cgt:
        person1_super_withdrawal_cgt_tax_paid = person1_available_for_cgt
    else:
        person1_super_withdrawal_cgt_tax_paid = person1_theoretical_tax

    # --- Person 2 CGT correction ---
    person2_theoretical_tax = cashflow["person2_super_withdrawal_cgt_tax"]

    person2_available_for_cgt = max(
        cashflow["person2_extra_accum_withdrawal"], 0.0
    )

    if person2_theoretical_tax > person2_available_for_cgt:
        person2_super_withdrawal_cgt_tax_paid = person2_available_for_cgt
    else:
        person2_super_withdrawal_cgt_tax_paid = person2_theoretical_tax

    ending_person1_accum_super_balance = person1_accum_available_before_cgt_tax

    if ending_person1_accum_super_balance < 1e-6:
        ending_person1_accum_super_balance = 0.0

    ending_person1_pension_super_balance = max(
        cashflow["person1_pension_before_return"]
        + person1_super_earnings_result["pension_earnings"]
        - person1_super_earnings_result["pension_earnings_tax"],
        0.0,
    )

    ending_person2_accum_super_balance = person2_accum_available_before_cgt_tax

    if ending_person2_accum_super_balance < 1e-6:
        ending_person2_accum_super_balance = 0.0

    ending_person2_pension_super_balance = max(
        cashflow["person2_pension_before_return"]
        + person2_super_earnings_result["pension_earnings"]
        - person2_super_earnings_result["pension_earnings_tax"],
        0.0,
    )

    non_super_tax_total = (
        tax_split["person1_non_super_tax_total"] + tax_split["person2_non_super_tax_total"]
    )
    available_non_super = (
        cashflow["non_super_before_return"] + non_super_total_return
    )
    non_super_tax_paid = min(non_super_tax_total, max(available_non_super, 0.0))

    ending_non_super_balance = max(
        available_non_super - non_super_tax_paid,
        0.0,
    )

    ending_person1_accum_super_cost_base = max(
        cashflow["person1_accum_cost_base_before_return"],
        0.0,
    )
    ending_person1_pension_super_cost_base = max(
        cashflow["person1_pension_cost_base_before_return"],
        0.0,
    )
    ending_person2_accum_super_cost_base = max(
        cashflow["person2_accum_cost_base_before_return"],
        0.0,
    )
    ending_person2_pension_super_cost_base = max(
        cashflow["person2_pension_cost_base_before_return"],
        0.0,
    )
    ending_non_super_cost_base = max(
        cashflow["non_super_cost_base_before_return"],
        0.0,
    )

    total_super_balance = (
        ending_person1_accum_super_balance
        + ending_person1_pension_super_balance
        + ending_person2_accum_super_balance
        + ending_person2_pension_super_balance
    )

    salary_tax_total = (
        tax_split["person1_salary_tax_total"]
        + tax_split["person2_salary_tax_total"]
    )
    total_division_293_tax = (
        person1_division_293["division_293_tax"]
        + person2_division_293["division_293_tax"]
    )
    person1_non_super_tax_paid = non_super_tax_paid * inputs["non_super_ownership_person1"]
    person2_non_super_tax_paid = non_super_tax_paid * (1.0 - inputs["non_super_ownership_person1"])
    person1_base_personal_tax = tax_split["person1_salary_tax_total"] + person1_non_super_tax_paid
    person2_base_personal_tax = tax_split["person2_salary_tax_total"] + person2_non_super_tax_paid
    person1_effective_budget_tax_adjustment = max(
        person1_budget_tax["personal_tax_adjustment"],
        -person1_base_personal_tax,
    )
    person2_effective_budget_tax_adjustment = max(
        person2_budget_tax["personal_tax_adjustment"],
        -person2_base_personal_tax,
    )
    total_budget_personal_tax_adjustment = (
        person1_effective_budget_tax_adjustment
        + person2_effective_budget_tax_adjustment
    )
    total_cgt_minimum_tax = (
        person1_cgt_minimum_tax["cgt_minimum_tax_gap"]
        + person2_cgt_minimum_tax["cgt_minimum_tax_gap"]
    )
    person1_personal_tax_total = max(
        person1_base_personal_tax
        + person1_effective_budget_tax_adjustment
        + person1_cgt_minimum_tax["cgt_minimum_tax_gap"],
        0.0,
    )
    person2_personal_tax_total = max(
        person2_base_personal_tax
        + person2_effective_budget_tax_adjustment
        + person2_cgt_minimum_tax["cgt_minimum_tax_gap"],
        0.0,
    )
    total_personal_tax = (
        person1_personal_tax_total
        + person2_personal_tax_total
        + total_division_293_tax
        + trust_result["trustee_minimum_tax"]
    )
    total_super_contributions_tax = (
        person1_super_contributions_tax + person2_super_contributions_tax
    )
    total_super_earnings_tax = (
        person1_super_earnings_result["total_super_earnings_tax"]
        + person2_super_earnings_result["total_super_earnings_tax"]
    )
    total_super_withdrawal_cgt_tax = (
        person1_super_withdrawal_cgt_tax_paid + person2_super_withdrawal_cgt_tax_paid
    )

    total_tax_paid = (
        total_personal_tax
        + total_super_contributions_tax
        + total_super_earnings_tax
        + total_super_withdrawal_cgt_tax
    )
    residential_property_net_equity = ending_residential_property_value - ending_residential_property_loan_balance
    main_residence_net_equity = ending_main_residence_value - ending_main_residence_loan_balance
    total_wealth = (
        total_super_balance
        + ending_non_super_balance
        + cashflow["ending_cash_reserve_balance"]
        + ending_non_deductible_offset_balance
        + ending_deductible_offset_balance
        + residential_property_net_equity
        + main_residence_net_equity
        + ending_main_residence_offset_balance
        + ending_investment_deductible_offset_balance
        + ending_discretionary_trust_balance
        - ending_non_deductible_debt_balance
        - ending_investment_deductible_debt_balance
    )

    person1_net_income = (
        person1_gross_income
        - tax_split["person1_salary_tax_total"]
        - person1_division_293["division_293_tax"]
        - person1_cgt_minimum_tax["cgt_minimum_tax_gap"]
        + person1_property_and_trust_cash
    )
    person2_net_income = (
        person2_gross_income
        - tax_split["person2_salary_tax_total"]
        - person2_division_293["division_293_tax"]
        - person2_cgt_minimum_tax["cgt_minimum_tax_gap"]
        + person2_property_and_trust_cash
    )

    policy_snapshot = get_policy_snapshot(financial_year_end)

    return {
        "year_index": year_index,
        "financial_year_end": financial_year_end,
        "financial_year_label": format_financial_year_label(financial_year_end),
        "tax_schedule_key": tax_schedule_key,
        "person1_age": person1_age,
        "person2_age": person2_age,
        "person1_phase": person1_phase,
        "person2_phase": person2_phase,
        "person1_super_phase_for_transfer": person1_super_phase_for_transfer,
        "person2_super_phase_for_transfer": person2_super_phase_for_transfer,
        "person1_gross_income": person1_gross_income,
        "person2_gross_income": person2_gross_income,
        "household_gross_income": person1_gross_income + person2_gross_income,
        "person1_total_taxable_income": person1_budget_tax["adjusted_taxable_income"],
        "person2_total_taxable_income": person2_budget_tax["adjusted_taxable_income"],
        "person1_assessable_before_deduction": tax_split["person1_assessable_before_deduction"],
        "person2_assessable_before_deduction": tax_split["person2_assessable_before_deduction"],
        "person1_income_tax": tax_split["person1_income_tax"],
        "person1_medicare_levy": tax_split["person1_medicare_levy"],
        "person1_salary_tax_total": tax_split["person1_salary_tax_total"],
        "person1_income_tax_on_non_super_earnings": tax_split["person1_income_tax_on_non_super_earnings"],
        "person1_medicare_levy_on_non_super_earnings": tax_split["person1_medicare_levy_on_non_super_earnings"],
        "person1_non_super_tax_total": tax_split["person1_non_super_tax_total"],
        "person1_division_293_income": person1_division_293["division_293_income"],
        "person1_division_293_super_contributions": person1_division_293["division_293_super_contributions"],
        "person1_division_293_taxable_contributions": person1_division_293["division_293_taxable_contributions"],
        "person1_division_293_tax": person1_division_293["division_293_tax"],
        "person1_personal_tax_total": person1_personal_tax_total,
        "person1_net_income": person1_net_income,
        "person2_income_tax": tax_split["person2_income_tax"],
        "person2_medicare_levy": tax_split["person2_medicare_levy"],
        "person2_salary_tax_total": tax_split["person2_salary_tax_total"],
        "person2_income_tax_on_non_super_earnings": tax_split["person2_income_tax_on_non_super_earnings"],
        "person2_medicare_levy_on_non_super_earnings": tax_split["person2_medicare_levy_on_non_super_earnings"],
        "person2_non_super_tax_total": tax_split["person2_non_super_tax_total"],
        "person2_division_293_income": person2_division_293["division_293_income"],
        "person2_division_293_super_contributions": person2_division_293["division_293_super_contributions"],
        "person2_division_293_taxable_contributions": person2_division_293["division_293_taxable_contributions"],
        "person2_division_293_tax": person2_division_293["division_293_tax"],
        "person2_personal_tax_total": person2_personal_tax_total,
        "person2_net_income": person2_net_income,
        "household_net_income": person1_net_income + person2_net_income,
        "taxable_non_super_earnings_total": taxable_non_super_guess,
        "taxable_non_super_earnings_p1": tax_split["person1_taxable_non_super"],
        "taxable_non_super_earnings_p2": tax_split["person2_taxable_non_super"],
        "residential_property_enabled": residential_property_enabled,
        "residential_property_restriction_applies": residential_result["restriction_applies"],
        "residential_property_gross_rent": residential_result["gross_rent"],
        "residential_property_operating_expenses": residential_result["deductible_operating_expenses"],
        "residential_property_loan_interest": residential_result["loan_interest"],
        "residential_property_scheduled_loan_payment": property_loan_result["payment"],
        "residential_property_scheduled_principal": property_loan_result["principal"],
        "deductible_debt_interest": residential_result["loan_interest"],
        "residential_property_total_deductions": residential_result["total_deductions"],
        "residential_property_net_cashflow": residential_result["net_cashflow"],
        "residential_property_taxable_income": residential_result["taxable_rental_income"],
        "residential_property_current_year_quarantined_loss": residential_result["current_year_quarantined_loss"],
        "residential_property_quarantined_loss_used": residential_result["quarantined_loss_used"],
        "opening_residential_property_quarantined_loss": residential_result["opening_quarantined_loss"],
        "closing_residential_property_quarantined_loss": residential_result["closing_quarantined_loss"],
        "opening_residential_property_value": opening_residential_property_value,
        "opening_residential_property_loan_balance": property_loan_balance,
        "opening_non_deductible_debt_balance": non_deductible_debt_balance,
        "opening_non_deductible_offset_balance": non_deductible_offset_balance,
        "opening_deductible_offset_balance": deductible_offset_balance,
        "ending_residential_property_value": ending_residential_property_value,
        "residential_property_loan_balance": ending_residential_property_loan_balance,
        "deductible_offset_balance": ending_deductible_offset_balance,
        "deductible_principal_repayment": cashflow["deductible_principal_repayment"],
        "residential_property_net_equity": residential_property_net_equity,
        "residential_property_sale_proceeds": cashflow["residential_property_sale_proceeds"],
        "residential_property_disposal_fraction": cashflow["residential_property_disposal_fraction"],
        "opening_main_residence_value": main_residence_opening_value,
        "ending_main_residence_value": ending_main_residence_value,
        "opening_main_residence_loan_balance": main_residence_opening_loan,
        "opening_main_residence_offset_balance": main_residence_opening_offset,
        "main_residence_loan_interest": main_residence_loan_result["interest"],
        "main_residence_scheduled_loan_payment": main_residence_loan_result["payment"],
        "main_residence_scheduled_principal": main_residence_loan_result["principal"],
        "main_residence_offset_contribution": cashflow["main_residence_offset_contribution"],
        "main_residence_extra_principal_repayment": cashflow["main_residence_extra_principal_repayment"],
        "main_residence_loan_balance": ending_main_residence_loan_balance,
        "main_residence_offset_balance": ending_main_residence_offset_balance,
        "main_residence_net_equity": main_residence_net_equity,
        "discretionary_trust_enabled": discretionary_trust_enabled,
        "opening_discretionary_trust_balance": trust_opening_balance,
        "opening_discretionary_trust_cost_base": trust_opening_cost_base,
        "discretionary_trust_income_return_rate": effective_trust_income_rate,
        "discretionary_trust_capital_return_rate": effective_trust_capital_rate,
        "discretionary_trust_capital_earnings": trust_capital_earnings,
        "ending_discretionary_trust_balance": ending_discretionary_trust_balance,
        "ending_discretionary_trust_cost_base": ending_discretionary_trust_cost_base,
        "discretionary_trust_excluded_income_pct": trust_excluded_income_pct,
        "discretionary_trust_net_income": trust_result["trust_net_income"],
        "discretionary_trust_excluded_income": trust_result["excluded_income"],
        "discretionary_trust_minimum_tax_income": trust_result["minimum_tax_income"],
        "discretionary_trust_minimum_tax_applies": trust_result["minimum_tax_applies"],
        "discretionary_trust_trustee_minimum_tax": trust_result["trustee_minimum_tax"],
        "discretionary_trust_policy_status": trust_result["policy_status"],
        "person1_property_tax_adjustment": person1_budget_tax["property_tax_adjustment"],
        "person2_property_tax_adjustment": person2_budget_tax["property_tax_adjustment"],
        "person1_trust_tax_before_credit": person1_budget_tax["trust_tax_before_credit"],
        "person2_trust_tax_before_credit": person2_budget_tax["trust_tax_before_credit"],
        "person1_trust_tax_credit": person1_budget_tax["trust_tax_credit"],
        "person2_trust_tax_credit": person2_budget_tax["trust_tax_credit"],
        "person1_trust_tax_after_credit": person1_budget_tax["trust_tax_after_credit"],
        "person2_trust_tax_after_credit": person2_budget_tax["trust_tax_after_credit"],
        "total_budget_personal_tax_adjustment": total_budget_personal_tax_adjustment,
        "spending": current_spending,
        "person1_sg_contribution": person1_sg_contribution,
        "person2_sg_contribution": person2_sg_contribution,
        "person1_sg_earnings_base": person1_sg_result["sg_earnings_base"],
        "person2_sg_earnings_base": person2_sg_result["sg_earnings_base"],
        "person1_income_above_sg_base": person1_sg_result["income_above_sg_base"],
        "person2_income_above_sg_base": person2_sg_result["income_above_sg_base"],
        "super_guarantee_maximum_earnings_base": person1_sg_result["maximum_earnings_base"],
        "person1_personal_deductible_contribution": person1_personal_deductible_contribution,
        "person2_personal_deductible_contribution": person2_personal_deductible_contribution,
        "person1_non_concessional_contribution": person1_non_concessional_contribution,
        "person2_non_concessional_contribution": person2_non_concessional_contribution,
        "person1_gross_concessional_contribution": person1_gross_concessional_contribution,
        "person2_gross_concessional_contribution": person2_gross_concessional_contribution,
        "person1_super_contributions_tax": person1_super_contributions_tax,
        "person2_super_contributions_tax": person2_super_contributions_tax,
        "person1_total_net_super_contribution": person1_total_net_super_contribution,
        "person2_total_net_super_contribution": person2_total_net_super_contribution,
        "person1_transfer_to_pension": person1_transfer_result["transfer_to_pension"],
        "person2_transfer_to_pension": person2_transfer_result["transfer_to_pension"],
        "person1_requested_transfer_amount": person1_transfer_result["requested_transfer_amount"],
        "person2_requested_transfer_amount": person2_transfer_result["requested_transfer_amount"],
        "person1_available_cap_space": person1_transfer_result["available_cap_space"],
        "person2_available_cap_space": person2_transfer_result["available_cap_space"],
        "person1_excess_retained_in_accumulation": person1_transfer_result["excess_retained_in_accumulation"],
        "person2_excess_retained_in_accumulation": person2_transfer_result["excess_retained_in_accumulation"],
        "person1_started_pension_this_year": person1_transfer_result["started_pension_this_year"],
        "person2_started_pension_this_year": person2_transfer_result["started_pension_this_year"],
        "person1_has_started_pension": person1_has_started_pension_after_year,
        "person2_has_started_pension": person2_has_started_pension_after_year,
        "person1_transfer_balance_cap": inputs["person1_transfer_balance_cap"],
        "person2_transfer_balance_cap": inputs["person2_transfer_balance_cap"],
        "non_super_ownership_person1": inputs["non_super_ownership_person1"],
        "non_super_ownership_person2": 1.0 - inputs["non_super_ownership_person1"],
        "cgt_discount_rate": inputs.get("cgt_discount_rate", 0.50),
        "super_cgt_discount_rate": SUPER_CGT_DISCOUNT_RATE,
        "person1_min_pension_drawdown": person1_min_pension_drawdown,
        "person2_min_pension_drawdown": person2_min_pension_drawdown,
        "total_minimum_pension_drawdown": person1_min_pension_drawdown + person2_min_pension_drawdown,
        "household_cash_available_before_extra_withdrawals": cashflow["household_cash_available_before_extra_withdrawals"],
        "required_cash_outflow": cashflow["required_cash_outflow"],
        "total_cash_contributions": cashflow["total_cash_contributions"],
        "surplus_cash_to_non_super": cashflow["surplus_cash_to_non_super"],
        "cash_reserve_top_up": cashflow["cash_reserve_top_up"],
        "opening_cash_reserve_balance": opening_cash_reserve_balance,
        "cash_reserve_withdrawal": cashflow["cash_reserve_withdrawal"],
        "ending_cash_reserve_balance": cashflow["ending_cash_reserve_balance"],
        "non_deductible_debt_interest": non_deductible_interest_expense,
        "non_deductible_scheduled_loan_payment": non_deductible_loan_result["payment"],
        "non_deductible_scheduled_principal": non_deductible_loan_result["principal"],
        "non_deductible_debt_balance": ending_non_deductible_debt_balance,
        "non_deductible_offset_balance": ending_non_deductible_offset_balance,
        "non_deductible_offset_contribution": cashflow["non_deductible_offset_contribution"],
        "non_deductible_offset_withdrawal": cashflow["non_deductible_offset_withdrawal"],
        "non_deductible_principal_repayment": cashflow["non_deductible_principal_repayment"],
        "investment_deductible_debt_interest": investment_deductible_interest_expense,
        "investment_deductible_scheduled_loan_payment": investment_deductible_loan_result["payment"],
        "investment_deductible_scheduled_principal": investment_deductible_loan_result["principal"],
        "opening_investment_deductible_debt_balance": investment_deductible_debt_balance,
        "opening_investment_deductible_offset_balance": investment_deductible_offset_balance,
        "investment_deductible_debt_balance": ending_investment_deductible_debt_balance,
        "investment_deductible_offset_balance": ending_investment_deductible_offset_balance,
        "investment_deductible_principal_repayment": cashflow["investment_deductible_principal_repayment"],
        "investment_deductible_offset_contribution": cashflow["investment_deductible_offset_contribution"],
        "deductible_offset_contribution": cashflow["deductible_offset_contribution"],
        "deductible_offset_withdrawal": cashflow["deductible_offset_withdrawal"],
        "total_debt_interest": (
            residential_result["loan_interest"]
            + main_residence_loan_result["interest"]
            + non_deductible_interest_expense
            + investment_deductible_interest_expense
        ),
        "debt_strategy": inputs.get("debt_strategy_name", "Custom"),
        "surplus_allocation_order": " > ".join(inputs.get("surplus_allocation_order", [])),
        "withdrawal_strategy": inputs.get("strategy_name", "Custom"),
        "withdrawal_order": " > ".join(inputs.get("withdrawal_order", [])),
        "non_super_withdrawal": cashflow["non_super_withdrawal"],
        "non_super_sale_cost_base_reduction": cashflow["non_super_sale_cost_base_reduction"],
        "non_super_realised_capital_gain": cashflow["non_super_realised_capital_gain"],
        "non_super_realised_capital_loss": cashflow["non_super_realised_capital_loss"],
        "non_super_discounted_taxable_capital_gain": cashflow["non_super_discounted_taxable_capital_gain"],
        "non_super_deferred_pre_2027_gain": cashflow["non_super_deferred_pre_2027_gain"],
        "non_super_post_2027_real_gain": cashflow["non_super_post_2027_real_gain"],
        "non_super_minimum_tax_capital_gain": cashflow["non_super_minimum_tax_capital_gain"],
        "non_super_capital_losses_applied": cashflow["non_super_capital_losses_applied"],
        "opening_non_super_capital_losses": opening_non_super_capital_losses,
        "closing_non_super_capital_losses": cashflow["non_super_capital_losses_after_withdrawal"],
        "opening_non_super_indexed_cost_base": opening_non_super_indexed_cost_base,
        "opening_non_super_deferred_pre_2027_gain": opening_non_super_deferred_pre_2027_gain,
        "non_super_indexation_uplift": cashflow["non_super_indexation_uplift"],
        "non_super_cgt_reform_applies": cashflow["non_super_cgt_reform_applies"],
        "non_super_cgt_calculation_method": cashflow["non_super_cgt_calculation_method"],
        "person1_cgt_minimum_tax_gain": person1_cgt_minimum_tax["minimum_tax_capital_gain"],
        "person2_cgt_minimum_tax_gain": person2_cgt_minimum_tax["minimum_tax_capital_gain"],
        "person1_cgt_minimum_tax_gap": person1_cgt_minimum_tax["cgt_minimum_tax_gap"],
        "person2_cgt_minimum_tax_gap": person2_cgt_minimum_tax["cgt_minimum_tax_gap"],
        "cgt_core_policy_status": CGT_CORE_POLICY_STATUS,
        "cgt_transition_method_status": CGT_TRANSITION_METHOD_STATUS,
        "person1_extra_accum_withdrawal": cashflow["person1_extra_accum_withdrawal"],
        "person2_extra_accum_withdrawal": cashflow["person2_extra_accum_withdrawal"],
        "person1_extra_pension_withdrawal": cashflow["person1_extra_pension_withdrawal"],
        "person2_extra_pension_withdrawal": cashflow["person2_extra_pension_withdrawal"],
        "total_extra_super_withdrawal": cashflow["total_extra_super_withdrawal"],
        "unmet_shortfall": cashflow["unmet_shortfall"],
        "opening_person1_accum_super_balance": opening_person1_accum_super_balance,
        "opening_person1_pension_super_balance": opening_person1_pension_super_balance,
        "opening_person2_accum_super_balance": opening_person2_accum_super_balance,
        "opening_person2_pension_super_balance": opening_person2_pension_super_balance,
        "opening_person1_accum_super_cost_base": opening_person1_accum_super_cost_base,
        "opening_person1_pension_super_cost_base": opening_person1_pension_super_cost_base,
        "opening_person2_accum_super_cost_base": opening_person2_accum_super_cost_base,
        "opening_person2_pension_super_cost_base": opening_person2_pension_super_cost_base,
        "opening_non_super_balance": opening_non_super_balance,
        "opening_non_super_cost_base": opening_non_super_cost_base,
        "person1_accum_before_return": cashflow["person1_accum_before_return"],
        "person1_pension_before_return": cashflow["person1_pension_before_return"],
        "person2_accum_before_return": cashflow["person2_accum_before_return"],
        "person2_pension_before_return": cashflow["person2_pension_before_return"],
        "non_super_before_return": cashflow["non_super_before_return"],
        "person1_accum_cost_base_before_return": cashflow["person1_accum_cost_base_before_return"],
        "person1_pension_cost_base_before_return": cashflow["person1_pension_cost_base_before_return"],
        "person2_accum_cost_base_before_return": cashflow["person2_accum_cost_base_before_return"],
        "person2_pension_cost_base_before_return": cashflow["person2_pension_cost_base_before_return"],
        "non_super_cost_base_before_return": cashflow["non_super_cost_base_before_return"],
        "super_income_return_rate": super_income_return_rate,
        "super_capital_return_rate": super_capital_return_rate,
        "super_total_return_rate": total_super_return_rate,
        "non_super_income_return_rate": non_super_income_return_rate,
        "non_super_capital_return_rate": non_super_capital_return_rate,
        "non_super_total_return_rate": non_super_income_return_rate + non_super_capital_return_rate,
        "person1_accum_earnings": person1_super_earnings_result["accum_earnings"],
        "person1_pension_earnings": person1_super_earnings_result["pension_earnings"],
        "person1_accum_income_earnings": person1_accum_income_earnings,
        "person1_accum_capital_earnings": person1_accum_capital_earnings,
        "person1_pension_income_earnings": person1_pension_income_earnings,
        "person1_pension_capital_earnings": person1_pension_capital_earnings,
        "person1_accum_earnings_tax": person1_super_earnings_result["accum_earnings_tax"],
        "person1_pension_earnings_tax": person1_super_earnings_result["pension_earnings_tax"],
        "person1_total_super_earnings_tax": person1_super_earnings_result["total_super_earnings_tax"],
        "person2_accum_earnings": person2_super_earnings_result["accum_earnings"],
        "person2_pension_earnings": person2_super_earnings_result["pension_earnings"],
        "person2_accum_income_earnings": person2_accum_income_earnings,
        "person2_accum_capital_earnings": person2_accum_capital_earnings,
        "person2_pension_income_earnings": person2_pension_income_earnings,
        "person2_pension_capital_earnings": person2_pension_capital_earnings,
        "person2_accum_earnings_tax": person2_super_earnings_result["accum_earnings_tax"],
        "person2_pension_earnings_tax": person2_super_earnings_result["pension_earnings_tax"],
        "person2_total_super_earnings_tax": person2_super_earnings_result["total_super_earnings_tax"],
        "person1_super_realised_capital_gain": cashflow["person1_super_realised_capital_gain"],
        "person2_super_realised_capital_gain": cashflow["person2_super_realised_capital_gain"],
        "person1_super_discounted_taxable_capital_gain": cashflow["person1_super_discounted_taxable_capital_gain"],
        "person2_super_discounted_taxable_capital_gain": cashflow["person2_super_discounted_taxable_capital_gain"],
        "person1_super_withdrawal_cgt_tax": person1_super_withdrawal_cgt_tax_paid,
        "person2_super_withdrawal_cgt_tax": person2_super_withdrawal_cgt_tax_paid,
        "total_super_withdrawal_cgt_tax": total_super_withdrawal_cgt_tax,
        "non_super_income_earnings": non_super_income_earnings,
        "non_super_capital_earnings": non_super_capital_earnings,
        "non_super_earnings": non_super_total_return,
        "ending_person1_accum_super_balance": ending_person1_accum_super_balance,
        "ending_person1_pension_super_balance": ending_person1_pension_super_balance,
        "ending_person2_accum_super_balance": ending_person2_accum_super_balance,
        "ending_person2_pension_super_balance": ending_person2_pension_super_balance,
        "ending_person1_accum_super_cost_base": ending_person1_accum_super_cost_base,
        "ending_person1_pension_super_cost_base": ending_person1_pension_super_cost_base,
        "ending_person2_accum_super_cost_base": ending_person2_accum_super_cost_base,
        "ending_person2_pension_super_cost_base": ending_person2_pension_super_cost_base,
        "ending_total_super_balance": total_super_balance,
        "ending_non_super_balance": ending_non_super_balance,
        "ending_non_super_cost_base": ending_non_super_cost_base,
        "ending_non_super_indexed_cost_base": cashflow["non_super_indexed_cost_base_before_return"],
        "ending_non_super_deferred_pre_2027_gain": cashflow["non_super_deferred_pre_2027_gain_after_withdrawal"],
        "ending_non_super_capital_losses": cashflow["non_super_capital_losses_after_withdrawal"],
        "total_personal_tax": total_personal_tax,
        "non_super_tax_paid": non_super_tax_paid,
        "total_super_contributions_tax": total_super_contributions_tax,
        "total_division_293_tax": total_division_293_tax,
        "total_discretionary_trust_minimum_tax": trust_result["trustee_minimum_tax"],
        "total_cgt_minimum_tax": total_cgt_minimum_tax,
        "total_super_earnings_tax": total_super_earnings_tax,
        "total_tax_paid": total_tax_paid,
        "policy_version": policy_snapshot["policy_version"],
        "policy_concessional_contributions_cap": policy_snapshot["concessional_contributions_cap"],
        "policy_non_concessional_contributions_cap": policy_snapshot["non_concessional_contributions_cap"],
        "policy_general_transfer_balance_cap": policy_snapshot["general_transfer_balance_cap"],
        "total_wealth": total_wealth,
    }


# ============================================================
# SECTION: DETERMINISTIC PROJECTION
# ============================================================

def run_deterministic_projection(inputs):
    inputs = normalise_household_inputs(inputs)
    projection_years = int(inputs["projection_years"])
    projection_context = build_projection_context(inputs)
    contribution_event_lookup = build_contribution_event_lookup(
        inputs.get("contribution_events"),
        household_mode=inputs.get("household_mode", "Two People"),
    )

    current_person1_accum_super_balance = inputs["person1_accum_super_balance"]
    current_person1_pension_super_balance = inputs["person1_pension_super_balance"]
    current_person2_accum_super_balance = inputs["person2_accum_super_balance"]
    current_person2_pension_super_balance = inputs["person2_pension_super_balance"]

    current_person1_accum_super_cost_base = inputs["person1_accum_super_cost_base"]
    current_person1_pension_super_cost_base = inputs["person1_pension_super_cost_base"]
    current_person2_accum_super_cost_base = inputs["person2_accum_super_cost_base"]
    current_person2_pension_super_cost_base = inputs["person2_pension_super_cost_base"]

    current_non_super_balance = inputs["non_super_balance"]
    current_non_super_cost_base = inputs["non_super_cost_base"]
    current_discretionary_trust_balance = max(float(inputs.get("discretionary_trust_balance", 0.0)), 0.0)
    current_discretionary_trust_cost_base = min(
        max(float(inputs.get("discretionary_trust_cost_base", 0.0)), 0.0),
        current_discretionary_trust_balance,
    )
    configured_transition_value = float(
        inputs.get("non_super_transition_value_2027", inputs["non_super_balance"])
    )
    current_non_super_indexed_cost_base = max(
        configured_transition_value
        if inputs.get("cgt_asset_acquired_before_2027", True)
        else float(inputs["non_super_cost_base"]),
        0.0,
    )
    current_non_super_deferred_pre_2027_gain = (
        configured_transition_value - float(inputs["non_super_cost_base"])
        if inputs.get("cgt_asset_acquired_before_2027", True) else 0.0
    )
    current_non_super_capital_losses = max(
        float(inputs.get("non_super_opening_capital_losses", 0.0)), 0.0
    )
    current_residential_property_value = (
        float(inputs.get("residential_property_value", 0.0))
        if inputs.get("residential_property_enabled", False) else 0.0
    )
    current_residential_quarantined_loss = max(
        float(inputs.get("residential_property_opening_quarantined_loss", 0.0)), 0.0
    )
    current_cash_reserve_balance = max(float(inputs.get("cash_reserve_balance", 0.0)), 0.0)
    current_main_residence_value = max(float(inputs.get("main_residence_value", 0.0)), 0.0)
    current_main_residence_loan_balance = max(float(inputs.get("main_residence_loan_balance", 0.0)), 0.0)
    current_main_residence_offset_balance = min(max(float(inputs.get("main_residence_offset_balance", 0.0)), 0.0), current_main_residence_loan_balance)
    current_residential_property_loan_balance = (
        max(float(inputs.get("residential_property_loan_balance", 0.0)), 0.0)
        if inputs.get("residential_property_enabled", False) else 0.0
    )
    current_non_deductible_debt_balance = max(float(inputs.get("non_deductible_debt_balance", 0.0)), 0.0)
    current_investment_deductible_debt_balance = max(float(inputs.get("investment_deductible_debt_balance", 0.0)), 0.0)
    current_investment_deductible_offset_balance = min(max(float(inputs.get("investment_deductible_offset_balance", 0.0)), 0.0), current_investment_deductible_debt_balance)
    current_non_deductible_offset_balance = min(
        max(float(inputs.get("non_deductible_offset_balance", 0.0)), 0.0),
        current_non_deductible_debt_balance,
    )
    current_deductible_offset_balance = min(
        max(float(inputs.get("deductible_offset_balance", 0.0)), 0.0),
        current_residential_property_loan_balance,
    )

    person1_has_started_pension = inputs["person1_pension_super_balance"] > 0
    person2_has_started_pension = inputs["person2_pension_super_balance"] > 0

    current_spending = inputs["annual_living_expenses"]
    results = []

    for year_context in projection_context["year_rows"]:
        indexed_retirement_spending = year_context["indexed_retirement_spending"]

        if year_context["use_retirement_spending"]:
            current_spending = max(current_spending, indexed_retirement_spending)

        result = run_one_year(
            inputs=inputs,
            year_context=year_context,
            opening_person1_accum_super_balance=current_person1_accum_super_balance,
            opening_person1_pension_super_balance=current_person1_pension_super_balance,
            opening_person2_accum_super_balance=current_person2_accum_super_balance,
            opening_person2_pension_super_balance=current_person2_pension_super_balance,
            opening_person1_accum_super_cost_base=current_person1_accum_super_cost_base,
            opening_person1_pension_super_cost_base=current_person1_pension_super_cost_base,
            opening_person2_accum_super_cost_base=current_person2_accum_super_cost_base,
            opening_person2_pension_super_cost_base=current_person2_pension_super_cost_base,
            opening_non_super_balance=current_non_super_balance,
            opening_non_super_cost_base=current_non_super_cost_base,
            current_spending=current_spending,
            super_income_return_rate=inputs["super_income_return_mean"],
            super_capital_return_rate=inputs["super_capital_return_mean"],
            non_super_income_return_rate=inputs["non_super_income_return_mean"],
            non_super_capital_return_rate=inputs["non_super_capital_return_mean"],
            contribution_event_lookup=contribution_event_lookup,
            person1_has_started_pension=person1_has_started_pension,
            person2_has_started_pension=person2_has_started_pension,
            opening_residential_property_value=current_residential_property_value,
            opening_residential_quarantined_loss=current_residential_quarantined_loss,
            opening_non_super_indexed_cost_base=current_non_super_indexed_cost_base,
            opening_non_super_deferred_pre_2027_gain=current_non_super_deferred_pre_2027_gain,
            opening_non_super_capital_losses=current_non_super_capital_losses,
            opening_cash_reserve_balance=current_cash_reserve_balance,
            opening_residential_property_loan_balance=current_residential_property_loan_balance,
            opening_non_deductible_debt_balance=current_non_deductible_debt_balance,
            opening_non_deductible_offset_balance=current_non_deductible_offset_balance,
            opening_deductible_offset_balance=current_deductible_offset_balance,
            opening_main_residence_value=current_main_residence_value,
            opening_main_residence_loan_balance=current_main_residence_loan_balance,
            opening_main_residence_offset_balance=current_main_residence_offset_balance,
            opening_investment_deductible_debt_balance=current_investment_deductible_debt_balance,
            opening_investment_deductible_offset_balance=current_investment_deductible_offset_balance,
            opening_discretionary_trust_balance=current_discretionary_trust_balance,
            opening_discretionary_trust_cost_base=current_discretionary_trust_cost_base,
            trust_income_return_rate=inputs.get("discretionary_trust_income_return_mean", 0.0),
            trust_capital_return_rate=inputs.get("discretionary_trust_capital_return_mean", 0.0),
        )

        results.append(result)

        current_person1_accum_super_balance = result["ending_person1_accum_super_balance"]
        current_person1_pension_super_balance = result["ending_person1_pension_super_balance"]
        current_person2_accum_super_balance = result["ending_person2_accum_super_balance"]
        current_person2_pension_super_balance = result["ending_person2_pension_super_balance"]

        current_person1_accum_super_cost_base = result["ending_person1_accum_super_cost_base"]
        current_person1_pension_super_cost_base = result["ending_person1_pension_super_cost_base"]
        current_person2_accum_super_cost_base = result["ending_person2_accum_super_cost_base"]
        current_person2_pension_super_cost_base = result["ending_person2_pension_super_cost_base"]

        current_non_super_balance = result["ending_non_super_balance"]
        current_non_super_cost_base = result["ending_non_super_cost_base"]
        current_non_super_indexed_cost_base = result["ending_non_super_indexed_cost_base"]
        current_non_super_deferred_pre_2027_gain = result["ending_non_super_deferred_pre_2027_gain"]
        current_non_super_capital_losses = result["ending_non_super_capital_losses"]
        current_discretionary_trust_balance = result["ending_discretionary_trust_balance"]
        current_discretionary_trust_cost_base = result["ending_discretionary_trust_cost_base"]
        current_residential_property_value = result["ending_residential_property_value"]
        current_residential_property_loan_balance = result["residential_property_loan_balance"]
        current_residential_quarantined_loss = result["closing_residential_property_quarantined_loss"]
        current_cash_reserve_balance = result["ending_cash_reserve_balance"]
        current_main_residence_value = result["ending_main_residence_value"]
        current_main_residence_loan_balance = result["main_residence_loan_balance"]
        current_main_residence_offset_balance = result["main_residence_offset_balance"]
        current_non_deductible_debt_balance = result["non_deductible_debt_balance"]
        current_investment_deductible_debt_balance = result["investment_deductible_debt_balance"]
        current_investment_deductible_offset_balance = result["investment_deductible_offset_balance"]
        current_non_deductible_offset_balance = result["non_deductible_offset_balance"]
        current_deductible_offset_balance = result["deductible_offset_balance"]

        person1_has_started_pension = result["person1_has_started_pension"]
        person2_has_started_pension = result["person2_has_started_pension"]

        if year_context["year_index"] < projection_years - 1:
            next_year_context = projection_context["year_rows"][year_context["year_index"] + 1]
            next_indexed_retirement_spending = next_year_context["indexed_retirement_spending"]

            if next_year_context["use_retirement_spending"]:
                current_spending = max(current_spending * (1 + inputs["inflation_rate"]), next_indexed_retirement_spending)
            else:
                current_spending = current_spending * (1 + inputs["inflation_rate"])

    return pd.DataFrame(results)


# ============================================================
# SECTION: MONTE CARLO
# ============================================================

def run_single_simulation(inputs, rng, contribution_event_lookup, projection_context, simulation_id):
    inputs = normalise_household_inputs(inputs)

    current_person1_accum_super_balance = inputs["person1_accum_super_balance"]
    current_person1_pension_super_balance = inputs["person1_pension_super_balance"]
    current_person2_accum_super_balance = inputs["person2_accum_super_balance"]
    current_person2_pension_super_balance = inputs["person2_pension_super_balance"]

    current_person1_accum_super_cost_base = inputs["person1_accum_super_cost_base"]
    current_person1_pension_super_cost_base = inputs["person1_pension_super_cost_base"]
    current_person2_accum_super_cost_base = inputs["person2_accum_super_cost_base"]
    current_person2_pension_super_cost_base = inputs["person2_pension_super_cost_base"]

    current_non_super_balance = inputs["non_super_balance"]
    current_non_super_cost_base = inputs["non_super_cost_base"]
    current_discretionary_trust_balance = max(float(inputs.get("discretionary_trust_balance", 0.0)), 0.0)
    current_discretionary_trust_cost_base = min(
        max(float(inputs.get("discretionary_trust_cost_base", 0.0)), 0.0),
        current_discretionary_trust_balance,
    )
    configured_transition_value = float(
        inputs.get("non_super_transition_value_2027", inputs["non_super_balance"])
    )
    current_non_super_indexed_cost_base = max(
        configured_transition_value
        if inputs.get("cgt_asset_acquired_before_2027", True)
        else float(inputs["non_super_cost_base"]),
        0.0,
    )
    current_non_super_deferred_pre_2027_gain = (
        configured_transition_value - float(inputs["non_super_cost_base"])
        if inputs.get("cgt_asset_acquired_before_2027", True) else 0.0
    )
    current_non_super_capital_losses = max(
        float(inputs.get("non_super_opening_capital_losses", 0.0)), 0.0
    )
    current_residential_property_value = (
        float(inputs.get("residential_property_value", 0.0))
        if inputs.get("residential_property_enabled", False) else 0.0
    )
    current_residential_quarantined_loss = max(
        float(inputs.get("residential_property_opening_quarantined_loss", 0.0)), 0.0
    )
    current_cash_reserve_balance = max(float(inputs.get("cash_reserve_balance", 0.0)), 0.0)
    current_main_residence_value = max(float(inputs.get("main_residence_value", 0.0)), 0.0)
    current_main_residence_loan_balance = max(float(inputs.get("main_residence_loan_balance", 0.0)), 0.0)
    current_main_residence_offset_balance = min(max(float(inputs.get("main_residence_offset_balance", 0.0)), 0.0), current_main_residence_loan_balance)
    current_residential_property_loan_balance = (
        max(float(inputs.get("residential_property_loan_balance", 0.0)), 0.0)
        if inputs.get("residential_property_enabled", False) else 0.0
    )
    current_non_deductible_debt_balance = max(float(inputs.get("non_deductible_debt_balance", 0.0)), 0.0)
    current_investment_deductible_debt_balance = max(float(inputs.get("investment_deductible_debt_balance", 0.0)), 0.0)
    current_investment_deductible_offset_balance = min(max(float(inputs.get("investment_deductible_offset_balance", 0.0)), 0.0), current_investment_deductible_debt_balance)
    current_non_deductible_offset_balance = min(
        max(float(inputs.get("non_deductible_offset_balance", 0.0)), 0.0),
        current_non_deductible_debt_balance,
    )
    current_deductible_offset_balance = min(
        max(float(inputs.get("deductible_offset_balance", 0.0)), 0.0),
        current_residential_property_loan_balance,
    )

    person1_has_started_pension = inputs["person1_pension_super_balance"] > 0
    person2_has_started_pension = inputs["person2_pension_super_balance"] > 0

    current_spending = inputs["annual_living_expenses"]
    minimal_path_rows = []
    success = True
    final_wealth = None

    for year_context in projection_context["year_rows"]:
        indexed_retirement_spending = year_context["indexed_retirement_spending"]

        if year_context["use_retirement_spending"]:
            current_spending = max(current_spending, indexed_retirement_spending)

        result = run_one_year(
            inputs=inputs,
            year_context=year_context,
            opening_person1_accum_super_balance=current_person1_accum_super_balance,
            opening_person1_pension_super_balance=current_person1_pension_super_balance,
            opening_person2_accum_super_balance=current_person2_accum_super_balance,
            opening_person2_pension_super_balance=current_person2_pension_super_balance,
            opening_person1_accum_super_cost_base=current_person1_accum_super_cost_base,
            opening_person1_pension_super_cost_base=current_person1_pension_super_cost_base,
            opening_person2_accum_super_cost_base=current_person2_accum_super_cost_base,
            opening_person2_pension_super_cost_base=current_person2_pension_super_cost_base,
            opening_non_super_balance=current_non_super_balance,
            opening_non_super_cost_base=current_non_super_cost_base,
            current_spending=current_spending,
            super_income_return_rate=rng.normal(
                loc=inputs["super_income_return_mean"],
                scale=inputs["super_income_return_std"],
            ),
            super_capital_return_rate=rng.normal(
                loc=inputs["super_capital_return_mean"],
                scale=inputs["super_capital_return_std"],
            ),
            non_super_income_return_rate=rng.normal(
                loc=inputs["non_super_income_return_mean"],
                scale=inputs["non_super_income_return_std"],
            ),
            non_super_capital_return_rate=rng.normal(
                loc=inputs["non_super_capital_return_mean"],
                scale=inputs["non_super_capital_return_std"],
            ),
            contribution_event_lookup=contribution_event_lookup,
            person1_has_started_pension=person1_has_started_pension,
            person2_has_started_pension=person2_has_started_pension,
            opening_residential_property_value=current_residential_property_value,
            opening_residential_quarantined_loss=current_residential_quarantined_loss,
            opening_non_super_indexed_cost_base=current_non_super_indexed_cost_base,
            opening_non_super_deferred_pre_2027_gain=current_non_super_deferred_pre_2027_gain,
            opening_non_super_capital_losses=current_non_super_capital_losses,
            opening_cash_reserve_balance=current_cash_reserve_balance,
            opening_residential_property_loan_balance=current_residential_property_loan_balance,
            opening_non_deductible_debt_balance=current_non_deductible_debt_balance,
            opening_non_deductible_offset_balance=current_non_deductible_offset_balance,
            opening_deductible_offset_balance=current_deductible_offset_balance,
            opening_main_residence_value=current_main_residence_value,
            opening_main_residence_loan_balance=current_main_residence_loan_balance,
            opening_main_residence_offset_balance=current_main_residence_offset_balance,
            opening_investment_deductible_debt_balance=current_investment_deductible_debt_balance,
            opening_investment_deductible_offset_balance=current_investment_deductible_offset_balance,
            opening_discretionary_trust_balance=current_discretionary_trust_balance,
            opening_discretionary_trust_cost_base=current_discretionary_trust_cost_base,
            trust_income_return_rate=rng.normal(
                loc=inputs.get("discretionary_trust_income_return_mean", 0.0),
                scale=inputs.get("discretionary_trust_income_return_std", 0.0),
            ),
            trust_capital_return_rate=rng.normal(
                loc=inputs.get("discretionary_trust_capital_return_mean", 0.0),
                scale=inputs.get("discretionary_trust_capital_return_std", 0.0),
            ),
        )

        minimal_path_rows.append(make_minimal_path_row(result, simulation_id))

        if result["unmet_shortfall"] > 0:
            success = False

        current_person1_accum_super_balance = result["ending_person1_accum_super_balance"]
        current_person1_pension_super_balance = result["ending_person1_pension_super_balance"]
        current_person2_accum_super_balance = result["ending_person2_accum_super_balance"]
        current_person2_pension_super_balance = result["ending_person2_pension_super_balance"]

        current_person1_accum_super_cost_base = result["ending_person1_accum_super_cost_base"]
        current_person1_pension_super_cost_base = result["ending_person1_pension_super_cost_base"]
        current_person2_accum_super_cost_base = result["ending_person2_accum_super_cost_base"]
        current_person2_pension_super_cost_base = result["ending_person2_pension_super_cost_base"]

        current_non_super_balance = result["ending_non_super_balance"]
        current_non_super_cost_base = result["ending_non_super_cost_base"]
        current_non_super_indexed_cost_base = result["ending_non_super_indexed_cost_base"]
        current_non_super_deferred_pre_2027_gain = result["ending_non_super_deferred_pre_2027_gain"]
        current_non_super_capital_losses = result["ending_non_super_capital_losses"]
        current_discretionary_trust_balance = result["ending_discretionary_trust_balance"]
        current_discretionary_trust_cost_base = result["ending_discretionary_trust_cost_base"]
        current_residential_property_value = result["ending_residential_property_value"]
        current_residential_property_loan_balance = result["residential_property_loan_balance"]
        current_residential_quarantined_loss = result["closing_residential_property_quarantined_loss"]
        current_cash_reserve_balance = result["ending_cash_reserve_balance"]
        current_main_residence_value = result["ending_main_residence_value"]
        current_main_residence_loan_balance = result["main_residence_loan_balance"]
        current_main_residence_offset_balance = result["main_residence_offset_balance"]
        current_non_deductible_debt_balance = result["non_deductible_debt_balance"]
        current_investment_deductible_debt_balance = result["investment_deductible_debt_balance"]
        current_investment_deductible_offset_balance = result["investment_deductible_offset_balance"]
        current_non_deductible_offset_balance = result["non_deductible_offset_balance"]
        current_deductible_offset_balance = result["deductible_offset_balance"]

        person1_has_started_pension = result["person1_has_started_pension"]
        person2_has_started_pension = result["person2_has_started_pension"]

        final_wealth = result["total_wealth"]

        if year_context["year_index"] < int(inputs["projection_years"]) - 1:
            next_year_context = projection_context["year_rows"][year_context["year_index"] + 1]
            next_indexed_retirement_spending = next_year_context["indexed_retirement_spending"]

            if next_year_context["use_retirement_spending"]:
                current_spending = max(current_spending * (1 + inputs["inflation_rate"]), next_indexed_retirement_spending)
            else:
                current_spending = current_spending * (1 + inputs["inflation_rate"])

    return {
        "path_rows": minimal_path_rows,
        "success": success,
        "final_wealth": final_wealth if final_wealth is not None else 0.0,
    }


def run_monte_carlo(inputs, random_seed=42):
    inputs = normalise_household_inputs(inputs)
    rng = np.random.default_rng(random_seed)
    n_sims = int(inputs["number_of_simulations"])
    contribution_event_lookup = build_contribution_event_lookup(
        inputs.get("contribution_events"),
        household_mode=inputs.get("household_mode", "Two People"),
    )
    projection_context = build_projection_context(inputs)

    simulation_summaries = []
    all_path_rows = []

    for sim_id in range(n_sims):
        sim_result = run_single_simulation(
            inputs=inputs,
            rng=rng,
            contribution_event_lookup=contribution_event_lookup,
            projection_context=projection_context,
            simulation_id=sim_id,
        )

        all_path_rows.extend(sim_result["path_rows"])

        simulation_summaries.append(
            {
                "simulation_id": sim_id,
                "success": sim_result["success"],
                "final_wealth": sim_result["final_wealth"],
            }
        )

    summary_df = pd.DataFrame(simulation_summaries)
    all_paths_df = pd.DataFrame(all_path_rows)

    return summary_df, all_paths_df


# ============================================================
# SECTION: MONTE CARLO PATH ROW HELPERS
# ============================================================

def make_minimal_path_row(result_row, simulation_id):
    return {
        "simulation_id": simulation_id,
        "financial_year_end": result_row["financial_year_end"],
        "financial_year_label": result_row["financial_year_label"],
        "total_wealth": result_row["total_wealth"],
        "unmet_shortfall": result_row["unmet_shortfall"],
    }


# ============================================================
# SECTION: AGGREGATION TABLES
# ============================================================

def build_percentile_table(all_paths_df):
    percentile_df = (
        all_paths_df.groupby(["financial_year_end", "financial_year_label"])["total_wealth"]
        .agg(
            p10=lambda x: x.quantile(0.10),
            p50=lambda x: x.quantile(0.50),
            p90=lambda x: x.quantile(0.90),
        )
        .reset_index()
        .sort_values("financial_year_end")
    )
    return percentile_df


def build_failure_probability_by_age(all_paths_df):
    failed_rows = all_paths_df[all_paths_df["unmet_shortfall"] > 0].copy()

    first_failures = (
        failed_rows.groupby("simulation_id")["financial_year_end"]
        .min()
        .reset_index()
        .rename(columns={"financial_year_end": "first_failure_financial_year_end"})
    )

    all_years = sorted(all_paths_df["financial_year_end"].unique())
    total_simulations = all_paths_df["simulation_id"].nunique()

    results = []
    for fy_end in all_years:
        failed_by_year = (
            first_failures["first_failure_financial_year_end"] <= fy_end
        ).sum()
        failure_probability = failed_by_year / total_simulations

        results.append(
            {
                "financial_year_end": fy_end,
                "financial_year_label": format_financial_year_label(fy_end),
                "failed_by_year_count": failed_by_year,
                "total_simulations": total_simulations,
                "failure_probability": failure_probability,
            }
        )

    return pd.DataFrame(results)
