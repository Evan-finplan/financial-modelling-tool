import copy

import numpy as np
import pandas as pd


# ============================================================
# SECTION: TAX CONFIGURATION
# ============================================================

PERSONAL_TAX_SCHEDULES = {
    2026: [
        (0.0, 18_200.0, 0.00),
        (18_200.0, 45_000.0, 0.16),
        (45_000.0, 135_000.0, 0.30),
        (135_000.0, 190_000.0, 0.37),
        (190_000.0, float("inf"), 0.45),
    ],
    2027: [
        (0.0, 18_200.0, 0.00),
        (18_200.0, 45_000.0, 0.15),
        (45_000.0, 135_000.0, 0.30),
        (135_000.0, 190_000.0, 0.37),
        (190_000.0, float("inf"), 0.45),
    ],
    "2028_PLUS": [
        (0.0, 18_200.0, 0.00),
        (18_200.0, 45_000.0, 0.14),
        (45_000.0, 135_000.0, 0.30),
        (135_000.0, 190_000.0, 0.37),
        (190_000.0, float("inf"), 0.45),
    ],
}

MEDICARE_LEVY_RATE = 0.02
SUPER_CONTRIBUTIONS_TAX_RATE = 0.15
SUPER_EARNINGS_TAX_RATE = 0.15
SUPER_GUARANTEE_RATE = 0.12


# ---------- NEW: Dynamic contribution caps ----------
def get_concessional_contributions_cap(financial_year_end):
    fy_end = int(financial_year_end)
    if fy_end >= 2027:
        return 32_500.0
    return 30_000.0


def get_non_concessional_contributions_cap(financial_year_end):
    fy_end = int(financial_year_end)
    if fy_end >= 2027:
        return 130_000.0
    return 120_000.0


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
# SECTION: CONTRIBUTION EVENT HELPERS
# ============================================================

def normalise_contribution_events(contribution_events):
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
    return df.reset_index(drop=True)


def build_contribution_event_lookup(contribution_events):
    df = normalise_contribution_events(contribution_events)

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

    for year_index in range(int(inputs["projection_years"])):
        financial_year_end = get_financial_year_end(inputs["start_financial_year"], year_index)
        financial_year_label = format_financial_year_label(financial_year_end)
        tax_schedule_key = get_tax_schedule_key_for_financial_year(financial_year_end)

        person1_age = get_person_age(inputs["person1_current_age"], year_index)
        person2_age = get_person_age(inputs["person2_current_age"], year_index)

        # ---------- NEW: decoupled logic ----------
        person1_is_working = person1_age < inputs["person1_retirement_age"]
        person2_is_working = person2_age < inputs["person2_retirement_age"]

        person1_is_pension_phase = person1_age >= inputs["person1_pension_start_age"]
        person2_is_pension_phase = person2_age >= inputs["person2_pension_start_age"]

        # ---------- phase label (for display only) ----------
        def get_phase(is_working, is_pension):
            if is_working and is_pension:
                return "working_pension"
            elif is_working:
                return "working"
            elif is_pension:
                return "retired_pension"
            else:
                return "retired_pre_pension"

        person1_phase = get_phase(person1_is_working, person1_is_pension_phase)
        person2_phase = get_phase(person2_is_working, person2_is_pension_phase)

        # ---------- income only depends on working ----------
        person1_income_indexed = (
            inputs["person1_annual_income"] * ((1 + inputs["inflation_rate"]) ** year_index)
            if person1_is_working else 0.0
        )

        person2_income_indexed = (
            inputs["person2_annual_income"] * ((1 + inputs["inflation_rate"]) ** year_index)
            if person2_is_working else 0.0
        )

        use_retirement_spending = should_use_retirement_spending(
            person1_age=person1_age,
            person2_age=person2_age,
            person1_retirement_age=inputs["person1_retirement_age"],
            person2_retirement_age=inputs["person2_retirement_age"],
            retirement_spending_trigger=inputs["retirement_spending_trigger"],
        )

        year_rows.append(
            {
                "year_index": year_index,
                "financial_year_end": financial_year_end,
                "financial_year_label": financial_year_label,
                "tax_schedule_key": tax_schedule_key,

                "person1_age": person1_age,
                "person2_age": person2_age,

                "person1_phase": person1_phase,
                "person2_phase": person2_phase,

                # ✅ 新增关键字段
                "person1_is_working": person1_is_working,
                "person2_is_working": person2_is_working,
                "person1_is_pension_phase": person1_is_pension_phase,
                "person2_is_pension_phase": person2_is_pension_phase,

                "person1_income_indexed": person1_income_indexed,
                "person2_income_indexed": person2_income_indexed,

                "use_retirement_spending": use_retirement_spending,
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


# ============================================================
# SECTION: SUPER ACCOUNT HELPERS
# ============================================================

def calculate_super_contributions_tax(gross_concessional_contribution):
    gross_concessional_contribution = max(float(gross_concessional_contribution), 0.0)
    return gross_concessional_contribution * SUPER_CONTRIBUTIONS_TAX_RATE


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


def calculate_super_account_earnings_tax(accum_balance_before_return, pension_balance_before_return, return_rate, transfer_balance_cap):
    accum_balance_before_return = float(accum_balance_before_return)
    pension_balance_before_return = float(pension_balance_before_return)
    transfer_balance_cap = max(float(transfer_balance_cap), 0.0)

    accum_earnings = accum_balance_before_return * return_rate
    pension_earnings = pension_balance_before_return * return_rate

    accum_tax = max(accum_earnings, 0.0) * SUPER_EARNINGS_TAX_RATE

    if pension_balance_before_return <= 0 or pension_earnings <= 0:
        pension_tax = 0.0
    else:
        exempt_pension_balance = min(pension_balance_before_return, transfer_balance_cap)
        excess_pension_balance = max(pension_balance_before_return - transfer_balance_cap, 0.0)
        excess_ratio = excess_pension_balance / pension_balance_before_return if pension_balance_before_return > 0 else 0.0
        taxable_pension_earnings = pension_earnings * excess_ratio
        pension_tax = max(taxable_pension_earnings, 0.0) * SUPER_EARNINGS_TAX_RATE

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
    transfer_balance_cap = max(float(transfer_balance_cap), 0.0)
    phase_balance_for_tax = max(float(phase_balance_for_tax), 0.0)

    sale_result = calculate_average_cost_cgt_on_sale(
        sale_proceeds=withdrawal_amount,
        pool_market_value=account_balance,
        pool_cost_base=account_cost_base,
        cgt_discount_rate=cgt_discount_rate,
    )

    taxable_ratio = 1.0

    if phase == "pension_phase":
        if phase_balance_for_tax <= 0:
            taxable_ratio = 0.0
        else:
            excess_balance = max(phase_balance_for_tax - transfer_balance_cap, 0.0)
            taxable_ratio = excess_balance / phase_balance_for_tax if phase_balance_for_tax > 0 else 0.0

    taxable_discounted_capital_gain = sale_result["discounted_taxable_capital_gain"] * taxable_ratio
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

    if inputs.get("retirement_spending_trigger") not in ["Both Retired", "Either Retired"]:
        errors.append("retirement_spending_trigger must be either 'Both Retired' or 'Either Retired'.")

    if inputs.get("cgt_discount_rate", 0.50) < 0 or inputs.get("cgt_discount_rate", 0.50) > 1:
        errors.append("cgt_discount_rate must be between 0 and 1.")

    if inputs["non_super_cost_base"] > inputs["non_super_balance"] + 1e-9:
        errors.append("non_super_cost_base cannot exceed non_super_balance under the current average-cost setup.")

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

    warnings.append("This version uses an average-cost CGT approximation for non-super withdrawals used to fund cash shortfall.")
    warnings.append("Salary income is indexed annually using the inflation rate while the person remains in working phase.")
    warnings.append("Pension transfer is triggered from pension start age and is applied up to the person's transfer balance cap. Minimum pension drawdown is then applied from pension assets.")
    warnings.append("Personal deductible contributions reduce taxable income and also flow through concessional contribution tax inside super.")

    events_df = normalise_contribution_events(inputs.get("contribution_events"), household_mode=inputs.get("household_mode", "Two People"))
    if not events_df.empty:
        start_fy = parse_financial_year_label(inputs["start_financial_year"])

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
                estimated_sg = indexed_income * SUPER_GUARANTEE_RATE
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
        "person2_income_tax": p2_income_tax,
        "person2_medicare_levy": p2_medicare,
        "person2_salary_tax_total": person2_alloc["salary"],
        "person2_income_tax_on_non_super_earnings": p2_non_super_income_tax,
        "person2_medicare_levy_on_non_super_earnings": p2_non_super_medicare,
        "person2_non_super_tax_total": person2_alloc["non_super"],
        "person2_personal_tax_total": person2_tax_result["personal_tax_total"],
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
):
    opening_non_super_balance = max(float(opening_non_super_balance), 0.0)
    opening_non_super_cost_base = max(float(opening_non_super_cost_base), 0.0)

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

    required_cash_outflow = current_spending + total_cash_contributions

    non_super_withdrawal = 0.0
    surplus_cash_to_non_super = 0.0

    person1_extra_accum_withdrawal = 0.0
    person2_extra_accum_withdrawal = 0.0
    person1_extra_pension_withdrawal = 0.0
    person2_extra_pension_withdrawal = 0.0
    total_extra_super_withdrawal = 0.0
    unmet_shortfall = 0.0

    if household_cash_available_before_extra_withdrawals >= required_cash_outflow:
        surplus_cash_to_non_super = (
            household_cash_available_before_extra_withdrawals - required_cash_outflow
        )
    else:
        cash_shortfall = required_cash_outflow - household_cash_available_before_extra_withdrawals

        non_super_withdrawal = min(cash_shortfall, opening_non_super_balance)
        remaining_shortfall = cash_shortfall - non_super_withdrawal

        extra_super_result = allocate_household_extra_super_withdrawal(
            required_amount=remaining_shortfall,
            person1_accum_balance=person1_accum_after_transfer,
            person2_accum_balance=person2_accum_after_transfer,
            person1_pension_balance=person1_pension_after_minimum,
            person2_pension_balance=person2_pension_after_minimum,
        )

        person1_extra_accum_withdrawal = extra_super_result["person1_extra_accum_withdrawal"]
        person2_extra_accum_withdrawal = extra_super_result["person2_extra_accum_withdrawal"]
        person1_extra_pension_withdrawal = extra_super_result["person1_extra_pension_withdrawal"]
        person2_extra_pension_withdrawal = extra_super_result["person2_extra_pension_withdrawal"]
        total_extra_super_withdrawal = extra_super_result["total_extra_super_withdrawal"]
        unmet_shortfall = extra_super_result["unfunded_after_super"]

    # ---------- Non-super withdrawal CGT ----------
    non_super_sale_result = calculate_average_cost_cgt_on_sale(
        sale_proceeds=non_super_withdrawal,
        pool_market_value=opening_non_super_balance,
        pool_cost_base=opening_non_super_cost_base,
        cgt_discount_rate=cgt_discount_rate,
    )

    non_super_cost_base_after_withdrawal = non_super_sale_result["remaining_cost_base"]
    non_super_cost_base_before_return = (
        non_super_cost_base_after_withdrawal + surplus_cash_to_non_super
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
        "non_super_withdrawal": non_super_withdrawal,
        "non_super_cost_base_after_withdrawal": non_super_cost_base_after_withdrawal,
        "non_super_cost_base_before_return": non_super_cost_base_before_return,
        "non_super_sale_cost_base_reduction": non_super_sale_result["cost_base_reduction"],
        "non_super_realised_capital_gain": non_super_sale_result["realised_capital_gain"],
        "non_super_realised_capital_loss": non_super_sale_result["realised_capital_loss"],
        "non_super_discounted_taxable_capital_gain": non_super_sale_result["discounted_taxable_capital_gain"],
        "person1_extra_accum_withdrawal": person1_extra_accum_withdrawal,
        "person2_extra_accum_withdrawal": person2_extra_accum_withdrawal,
        "person1_extra_pension_withdrawal": person1_extra_pension_withdrawal,
        "person2_extra_pension_withdrawal": person2_extra_pension_withdrawal,
        "total_extra_super_withdrawal": total_extra_super_withdrawal,
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
        person1_sg_contribution = person1_gross_income * SUPER_GUARANTEE_RATE
    else:
        person1_gross_income = 0.0
        person1_sg_contribution = 0.0

    if person2_is_working:
        person2_gross_income = float(person2_income_indexed)
        person2_sg_contribution = person2_gross_income * SUPER_GUARANTEE_RATE
    else:
        person2_gross_income = 0.0
        person2_sg_contribution = 0.0

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

    taxable_non_super_guess = max(opening_non_super_balance * non_super_income_return_rate, 0.0)

    for _ in range(3):
        tax_split = calculate_household_personal_tax_split(
            person1_salary_income=person1_gross_income,
            person2_salary_income=person2_gross_income,
            taxable_non_super_earnings_total=taxable_non_super_guess,
            ownership_person1=inputs["non_super_ownership_person1"],
            person1_personal_deductible_contribution=person1_personal_deductible_contribution,
            person2_personal_deductible_contribution=person2_personal_deductible_contribution,
            tax_schedule_key=tax_schedule_key,
        )

        person1_net_income = person1_gross_income - tax_split["person1_salary_tax_total"]
        person2_net_income = person2_gross_income - tax_split["person2_salary_tax_total"]

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
        )

        taxable_non_super_guess = max(
            cashflow["non_super_before_return"] * non_super_income_return_rate,
            0.0,
        ) + max(cashflow["non_super_discounted_taxable_capital_gain"], 0.0)

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
        cashflow["person1_accum_cost_base_before_return"]
        + max(person1_accum_capital_earnings, 0.0),
        0.0,
    )
    ending_person1_pension_super_cost_base = max(
        cashflow["person1_pension_cost_base_before_return"]
        + max(person1_pension_capital_earnings, 0.0),
        0.0,
    )
    ending_person2_accum_super_cost_base = max(
        cashflow["person2_accum_cost_base_before_return"]
        + max(person2_accum_capital_earnings, 0.0),
        0.0,
    )
    ending_person2_pension_super_cost_base = max(
        cashflow["person2_pension_cost_base_before_return"]
        + max(person2_pension_capital_earnings, 0.0),
        0.0,
    )
    ending_non_super_cost_base = (
        cashflow["non_super_cost_base_before_return"]
        + max(non_super_capital_earnings, 0.0)
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
    total_personal_tax = salary_tax_total + non_super_tax_paid
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
    total_wealth = total_super_balance + ending_non_super_balance

    person1_net_income = person1_gross_income - tax_split["person1_salary_tax_total"]
    person2_net_income = person2_gross_income - tax_split["person2_salary_tax_total"]

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
        "person1_total_taxable_income": tax_split["person1_taxable_income"],
        "person2_total_taxable_income": tax_split["person2_taxable_income"],
        "person1_assessable_before_deduction": tax_split["person1_assessable_before_deduction"],
        "person2_assessable_before_deduction": tax_split["person2_assessable_before_deduction"],
        "person1_income_tax": tax_split["person1_income_tax"],
        "person1_medicare_levy": tax_split["person1_medicare_levy"],
        "person1_salary_tax_total": tax_split["person1_salary_tax_total"],
        "person1_income_tax_on_non_super_earnings": tax_split["person1_income_tax_on_non_super_earnings"],
        "person1_medicare_levy_on_non_super_earnings": tax_split["person1_medicare_levy_on_non_super_earnings"],
        "person1_non_super_tax_total": tax_split["person1_non_super_tax_total"],
        "person1_personal_tax_total": tax_split["person1_salary_tax_total"] + (non_super_tax_paid * inputs["non_super_ownership_person1"]),
        "person1_net_income": person1_net_income,
        "person2_income_tax": tax_split["person2_income_tax"],
        "person2_medicare_levy": tax_split["person2_medicare_levy"],
        "person2_salary_tax_total": tax_split["person2_salary_tax_total"],
        "person2_income_tax_on_non_super_earnings": tax_split["person2_income_tax_on_non_super_earnings"],
        "person2_medicare_levy_on_non_super_earnings": tax_split["person2_medicare_levy_on_non_super_earnings"],
        "person2_non_super_tax_total": tax_split["person2_non_super_tax_total"],
        "person2_personal_tax_total": tax_split["person2_salary_tax_total"] + (non_super_tax_paid * (1.0 - inputs["non_super_ownership_person1"])),
        "person2_net_income": person2_net_income,
        "household_net_income": person1_net_income + person2_net_income,
        "taxable_non_super_earnings_total": taxable_non_super_guess,
        "taxable_non_super_earnings_p1": tax_split["person1_taxable_non_super"],
        "taxable_non_super_earnings_p2": tax_split["person2_taxable_non_super"],
        "spending": current_spending,
        "person1_sg_contribution": person1_sg_contribution,
        "person2_sg_contribution": person2_sg_contribution,
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
        "non_super_withdrawal": cashflow["non_super_withdrawal"],
        "non_super_sale_cost_base_reduction": cashflow["non_super_sale_cost_base_reduction"],
        "non_super_realised_capital_gain": cashflow["non_super_realised_capital_gain"],
        "non_super_realised_capital_loss": cashflow["non_super_realised_capital_loss"],
        "non_super_discounted_taxable_capital_gain": cashflow["non_super_discounted_taxable_capital_gain"],
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
        "total_personal_tax": total_personal_tax,
        "non_super_tax_paid": non_super_tax_paid,
        "total_super_contributions_tax": total_super_contributions_tax,
        "total_super_earnings_tax": total_super_earnings_tax,
        "total_tax_paid": total_tax_paid,
        "total_wealth": total_wealth,
    }


# ============================================================
# SECTION: DETERMINISTIC PROJECTION
# ============================================================

def run_deterministic_projection(inputs):
    projection_years = int(inputs["projection_years"])
    projection_context = build_projection_context(inputs)
    contribution_event_lookup = build_contribution_event_lookup(inputs.get("contribution_events"))

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

    person1_has_started_pension = inputs["person1_pension_super_balance"] > 0
    person2_has_started_pension = inputs["person2_pension_super_balance"] > 0

    current_spending = inputs["annual_living_expenses"]
    results = []

    for year_context in projection_context["year_rows"]:
        if year_context["use_retirement_spending"]:
            current_spending = max(current_spending, inputs["retirement_spending"])

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

        person1_has_started_pension = result["person1_has_started_pension"]
        person2_has_started_pension = result["person2_has_started_pension"]

        if year_context["year_index"] < projection_years - 1:
            next_year_context = projection_context["year_rows"][year_context["year_index"] + 1]

            if next_year_context["use_retirement_spending"]:
                current_spending = inputs["retirement_spending"]
            else:
                current_spending = current_spending * (1 + inputs["inflation_rate"])

    return pd.DataFrame(results)


# ============================================================
# SECTION: MONTE CARLO
# ============================================================

def run_single_simulation(inputs, rng, contribution_event_lookup, projection_context, simulation_id):
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

    person1_has_started_pension = inputs["person1_pension_super_balance"] > 0
    person2_has_started_pension = inputs["person2_pension_super_balance"] > 0

    current_spending = inputs["annual_living_expenses"]
    minimal_path_rows = []
    success = True
    final_wealth = None

    for year_context in projection_context["year_rows"]:
        if year_context["use_retirement_spending"]:
            current_spending = max(current_spending, inputs["retirement_spending"])

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

        person1_has_started_pension = result["person1_has_started_pension"]
        person2_has_started_pension = result["person2_has_started_pension"]

        final_wealth = result["total_wealth"]

        if year_context["year_index"] < int(inputs["projection_years"]) - 1:
            next_year_context = projection_context["year_rows"][year_context["year_index"] + 1]

            if next_year_context["use_retirement_spending"]:
                current_spending = inputs["retirement_spending"]
            else:
                current_spending = current_spending * (1 + inputs["inflation_rate"])

    return {
        "path_rows": minimal_path_rows,
        "success": success,
        "final_wealth": final_wealth if final_wealth is not None else 0.0,
    }


def run_monte_carlo(inputs, random_seed=42):
    rng = np.random.default_rng(random_seed)
    n_sims = int(inputs["number_of_simulations"])
    contribution_event_lookup = build_contribution_event_lookup(inputs.get("contribution_events"))
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