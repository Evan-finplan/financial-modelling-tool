"""Debt repayment and cash-surplus strategy helpers."""

from __future__ import annotations

import pandas as pd


DEBT_STRATEGY_PROFILES = {
    "Current Strategy": {
        "description_en": "Use the surplus allocation order selected in the input screen.",
        "description_zh": "使用输入页面中选择的现金盈余分配顺序。",
    },
    "Non-deductible First": {
        "surplus_allocation_order": [
            "non_deductible_repayment",
            "non_deductible_offset",
            "main_residence_offset",
            "main_residence_repayment",
            "investment_deductible_offset",
            "property_loan_offset",
            "non_super",
        ],
        "description_en": "Prioritise other non-deductible investment debt, then home-loan and deductible-debt offsets.",
        "description_zh": "优先偿还其他不可抵扣投资债务，其后使用自住房及可抵扣债务 Offset。",
    },
    "Offset First": {
        "surplus_allocation_order": [
            "main_residence_offset",
            "non_deductible_offset",
            "property_loan_offset",
            "investment_deductible_offset",
            "main_residence_repayment",
            "non_deductible_repayment",
            "non_super",
        ],
        "description_en": "Build liquid offsets across home and investment loans before direct principal repayments.",
        "description_zh": "先累积自住房及投资贷款的流动 Offset，再直接偿还本金。",
    },
    "Deductible First": {
        "surplus_allocation_order": [
            "investment_deductible_repayment",
            "property_loan_repayment",
            "investment_deductible_offset",
            "property_loan_offset",
            "non_deductible_repayment",
            "non_deductible_offset",
            "non_super",
        ],
        "description_en": "Repay other deductible investment debt and the property loan before non-deductible investment debt.",
        "description_zh": "先偿还其他可抵扣投资债务及投资物业贷款，再偿还不可抵扣投资债务。",
    },
    "Invest Surplus": {
        "surplus_allocation_order": ["non_super"],
        "description_en": "Invest all available surplus in the non-super portfolio and retain both debts.",
        "description_zh": "把全部可用盈余投入非养老金投资组合，并保留两类债务。",
    },
}


def apply_debt_strategy_profile(base_inputs, strategy_name):
    profile = DEBT_STRATEGY_PROFILES[strategy_name]
    updated = dict(base_inputs)
    updated["debt_strategy_name"] = strategy_name
    if "surplus_allocation_order" in profile:
        updated["surplus_allocation_order"] = list(profile["surplus_allocation_order"])
    return updated


def allocate_cash_surplus(
    surplus,
    allocation_order,
    cash_reserve_balance=0.0,
    cash_reserve_target=0.0,
    non_deductible_debt_balance=0.0,
    non_deductible_offset_balance=0.0,
    deductible_debt_balance=0.0,
    deductible_offset_balance=0.0,
    main_residence_debt_balance=0.0,
    main_residence_offset_balance=0.0,
    investment_deductible_debt_balance=0.0,
    investment_deductible_offset_balance=0.0,
    allow_non_super_investment=True,
):
    """Allocate an annual cash surplus while preserving debt/offset invariants."""
    remaining = max(float(surplus), 0.0)
    cash = max(float(cash_reserve_balance), 0.0)
    cash_target = max(float(cash_reserve_target), 0.0)
    non_deductible_debt = max(float(non_deductible_debt_balance), 0.0)
    non_deductible_offset = min(max(float(non_deductible_offset_balance), 0.0), non_deductible_debt)
    deductible_debt = max(float(deductible_debt_balance), 0.0)
    deductible_offset = min(max(float(deductible_offset_balance), 0.0), deductible_debt)
    main_residence_debt = max(float(main_residence_debt_balance), 0.0)
    main_residence_offset = min(max(float(main_residence_offset_balance), 0.0), main_residence_debt)
    investment_deductible_debt = max(float(investment_deductible_debt_balance), 0.0)
    investment_deductible_offset = min(max(float(investment_deductible_offset_balance), 0.0), investment_deductible_debt)
    allocations = {
        "cash_reserve_top_up": 0.0,
        "non_deductible_offset_contribution": 0.0,
        "non_deductible_principal_repayment": 0.0,
        "deductible_offset_contribution": 0.0,
        "deductible_principal_repayment": 0.0,
        "surplus_cash_to_non_super": 0.0,
        "main_residence_offset_contribution": 0.0,
        "main_residence_principal_repayment": 0.0,
        "investment_deductible_offset_contribution": 0.0,
        "investment_deductible_principal_repayment": 0.0,
    }

    for destination in allocation_order or ["non_super"]:
        if remaining <= 1e-9:
            break
        if destination == "cash_reserve":
            amount = min(remaining, max(cash_target - cash, 0.0))
            cash += amount
            allocations["cash_reserve_top_up"] += amount
        elif destination == "cash":
            amount = remaining
            cash += amount
            allocations["cash_reserve_top_up"] += amount
        elif destination == "non_deductible_offset":
            amount = min(remaining, max(non_deductible_debt - non_deductible_offset, 0.0))
            non_deductible_offset += amount
            allocations["non_deductible_offset_contribution"] += amount
        elif destination == "non_deductible_repayment":
            amount = min(remaining, max(non_deductible_debt - non_deductible_offset, 0.0))
            non_deductible_debt -= amount
            allocations["non_deductible_principal_repayment"] += amount
        elif destination in {"deductible_offset", "property_loan_offset"}:
            amount = min(remaining, max(deductible_debt - deductible_offset, 0.0))
            deductible_offset += amount
            allocations["deductible_offset_contribution"] += amount
        elif destination in {"deductible_repayment", "property_loan_repayment"}:
            amount = min(remaining, max(deductible_debt - deductible_offset, 0.0))
            deductible_debt -= amount
            allocations["deductible_principal_repayment"] += amount
        elif destination == "main_residence_offset":
            amount = min(remaining, max(main_residence_debt - main_residence_offset, 0.0))
            main_residence_offset += amount
            allocations["main_residence_offset_contribution"] += amount
        elif destination == "main_residence_repayment":
            amount = min(remaining, max(main_residence_debt - main_residence_offset, 0.0))
            main_residence_debt -= amount
            allocations["main_residence_principal_repayment"] += amount
        elif destination == "investment_deductible_offset":
            amount = min(remaining, max(investment_deductible_debt - investment_deductible_offset, 0.0))
            investment_deductible_offset += amount
            allocations["investment_deductible_offset_contribution"] += amount
        elif destination == "investment_deductible_repayment":
            amount = min(remaining, max(investment_deductible_debt - investment_deductible_offset, 0.0))
            investment_deductible_debt -= amount
            allocations["investment_deductible_principal_repayment"] += amount
        elif destination == "non_super" and allow_non_super_investment:
            amount = remaining
            allocations["surplus_cash_to_non_super"] += amount
        else:
            continue
        remaining -= amount

    # No surplus is silently lost when a profile omits a residual destination.
    if remaining > 1e-9:
        if allow_non_super_investment:
            allocations["surplus_cash_to_non_super"] += remaining
        else:
            cash += remaining
            allocations["cash_reserve_top_up"] += remaining
        remaining = 0.0

    return {
        **allocations,
        "ending_cash_reserve_balance": cash,
        "ending_non_deductible_debt_balance": non_deductible_debt,
        "ending_non_deductible_offset_balance": non_deductible_offset,
        "ending_deductible_debt_balance": deductible_debt,
        "ending_deductible_offset_balance": deductible_offset,
        "ending_main_residence_debt_balance": main_residence_debt,
        "ending_main_residence_offset_balance": main_residence_offset,
        "ending_investment_deductible_debt_balance": investment_deductible_debt,
        "ending_investment_deductible_offset_balance": investment_deductible_offset,
        "unallocated_surplus": remaining,
    }


def _first_debt_free_year(det_df, debt_column, offset_column=None):
    if det_df.empty or debt_column not in det_df.columns:
        return None
    net_debt = det_df[debt_column].astype(float)
    if offset_column and offset_column in det_df.columns:
        net_debt = net_debt - det_df[offset_column].astype(float)
    rows = det_df.loc[net_debt <= 1.0]
    return int(rows.iloc[0]["financial_year_end"]) if not rows.empty else None


def build_debt_strategy_comparison_df(comparison_results, is_chinese=False):
    if not comparison_results:
        return pd.DataFrame()
    base_name = "Current Strategy" if "Current Strategy" in comparison_results else next(iter(comparison_results))
    base_df = comparison_results[base_name]["det_df"]
    base_interest = float(base_df.get("total_debt_interest", pd.Series(dtype=float)).sum())
    base_final = float(base_df.iloc[-1].get("total_wealth", 0.0)) if not base_df.empty else 0.0
    rows = []
    for name, result in comparison_results.items():
        det_df = result["det_df"]
        inputs = result["inputs"]
        final_row = det_df.iloc[-1] if not det_df.empty else {}
        cumulative_interest = float(det_df.get("total_debt_interest", pd.Series(dtype=float)).sum())
        ending_non_deductible = float(final_row.get("non_deductible_debt_balance", 0.0))
        ending_deductible = float(final_row.get("investment_deductible_debt_balance", 0.0))
        ending_property_loan = float(final_row.get("residential_property_loan_balance", 0.0))
        ending_home_loan = float(final_row.get("main_residence_loan_balance", 0.0))
        ending_offsets = (
            float(final_row.get("main_residence_offset_balance", 0.0))
            + float(final_row.get("non_deductible_offset_balance", 0.0))
            + float(final_row.get("investment_deductible_offset_balance", 0.0))
            + float(final_row.get("deductible_offset_balance", 0.0))
        )
        final_wealth = float(final_row.get("total_wealth", 0.0))
        profile = DEBT_STRATEGY_PROFILES.get(name, {})
        rows.append({
            "scenario": name,
            "strategy_description": profile.get("description_zh" if is_chinese else "description_en", ""),
            "allocation_order": " > ".join(inputs.get("surplus_allocation_order", [])),
            "cumulative_interest": cumulative_interest,
            "interest_saved_vs_base": base_interest - cumulative_interest,
            "cumulative_tax": float(det_df.get("total_tax_paid", pd.Series(dtype=float)).sum()),
            "ending_non_deductible_debt": ending_non_deductible,
            "ending_deductible_debt": ending_deductible,
            "ending_property_loan": ending_property_loan,
            "ending_home_loan": ending_home_loan,
            "ending_offset_balance": ending_offsets,
            "non_deductible_debt_free_year": _first_debt_free_year(det_df, "non_deductible_debt_balance", "non_deductible_offset_balance"),
            "deductible_debt_free_year": _first_debt_free_year(det_df, "investment_deductible_debt_balance", "investment_deductible_offset_balance"),
            "property_loan_free_year": _first_debt_free_year(det_df, "residential_property_loan_balance", "deductible_offset_balance"),
            "home_loan_free_year": _first_debt_free_year(det_df, "main_residence_loan_balance", "main_residence_offset_balance"),
            "final_wealth": final_wealth,
            "final_wealth_delta": final_wealth - base_final,
            "failure_probability": 1.0 - float(result.get("success_rate", 0.0)),
        })
    return pd.DataFrame(rows)
