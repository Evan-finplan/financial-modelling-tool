"""Scenario comparison helpers for adviser and report outputs."""

from __future__ import annotations

import pandas as pd


STRATEGY_PROFILES = {
    "Base Case": {
        "withdrawal_order": ["cash", "non_super", "pension", "accumulation", "property"],
        "description_en": "Use cash first, then non-super investments, pension, accumulation super and property.",
        "description_zh": "依次使用现金、非养老金投资、Pension、Accumulation Super 和物业。",
    },
    "Strategy A": {
        "withdrawal_order": ["cash", "pension", "accumulation", "non_super", "property"],
        "description_en": "Use cash and super first, preserving non-super investments for longer.",
        "description_zh": "优先使用现金及 Super，尽量延后出售非养老金投资。",
    },
    "Strategy B": {
        "withdrawal_order": ["cash", "property", "non_super", "pension", "accumulation"],
        "description_en": "Use available property equity before investment and super balances.",
        "description_zh": "先使用可动用的物业净值，再使用投资及 Super 余额。",
    },
    "Strategy C": {
        "withdrawal_order": ["cash", "pension", "accumulation", "property", "non_super"],
        "description_en": "Preserve the nominated non-super estate reserve while using other assets first.",
        "description_zh": "优先使用其他资产，保留指定的非养老金遗产储备。",
        "preserve_non_super_estate": True,
    },
}


def apply_strategy_profile(base_inputs, strategy_name):
    profile = STRATEGY_PROFILES[strategy_name]
    updated = dict(base_inputs)
    updated["strategy_name"] = strategy_name
    updated["withdrawal_order"] = list(profile["withdrawal_order"])
    if not profile.get("preserve_non_super_estate"):
        updated["non_super_estate_reserve"] = 0.0
    return updated


def _first_year_at_or_after_age(det_df, retirement_age):
    if det_df.empty or "person1_age" not in det_df.columns:
        return None
    rows = det_df.loc[det_df["person1_age"].astype(float) >= float(retirement_age)]
    return rows.iloc[0] if not rows.empty else None


def _first_advantage_year(base_df, strategy_df):
    if base_df.empty or strategy_df.empty:
        return None
    merged = base_df[["financial_year_end", "total_wealth"]].merge(
        strategy_df[["financial_year_end", "total_wealth"]],
        on="financial_year_end",
        suffixes=("_base", "_strategy"),
    )
    advantage = merged["total_wealth_strategy"] - merged["total_wealth_base"]
    rows = merged.loc[advantage > 1.0]
    return int(rows.iloc[0]["financial_year_end"]) if not rows.empty else None


def _break_even_year(base_df, strategy_df):
    if base_df.empty or strategy_df.empty:
        return None
    merged = base_df[["financial_year_end", "total_wealth"]].merge(
        strategy_df[["financial_year_end", "total_wealth"]],
        on="financial_year_end",
        suffixes=("_base", "_strategy"),
    )
    delta = merged["total_wealth_strategy"] - merged["total_wealth_base"]
    for index in range(1, len(delta)):
        if delta.iloc[index] >= 0 and delta.iloc[index - 1] < 0:
            return int(merged.iloc[index]["financial_year_end"])
    if len(delta) and delta.iloc[0] >= 0:
        return int(merged.iloc[0]["financial_year_end"])
    return None


def _risk_summary(result, is_chinese=False):
    det_df = result["det_df"]
    risks = []
    success_rate = float(result.get("success_rate", 0.0))
    p10 = float(result.get("p10_final_wealth", 0.0))
    if success_rate < 0.75:
        risks.append("成功率低于 75%" if is_chinese else "Success rate is below 75%")
    if p10 <= 0:
        risks.append("P10 期末财富为零或负数" if is_chinese else "P10 final wealth is zero or negative")
    if "unmet_shortfall" in det_df.columns and float(det_df["unmet_shortfall"].max()) > 0:
        risks.append("确定性预测出现资金缺口" if is_chinese else "Deterministic cashflow has a shortfall")
    if "residential_property_sale_proceeds" in det_df.columns and float(det_df["residential_property_sale_proceeds"].sum()) > 0:
        risks.append("结果依赖出售部分或全部物业" if is_chinese else "Outcome relies on a partial or full property sale")
    if "non_super_withdrawal" in det_df.columns and float(det_df["non_super_withdrawal"].sum()) > 0:
        risks.append("非养老金资产出售可能触发 CGT" if is_chinese else "Non-super sales may crystallise CGT")
    return "; ".join(risks) if risks else ("未发现重大模型风险" if is_chinese else "No major modelled risk identified")


def build_strategy_comparison_df(comparison_results, is_chinese=False):
    if not comparison_results:
        return pd.DataFrame()
    base_name = "Base Case" if "Base Case" in comparison_results else next(iter(comparison_results))
    base = comparison_results[base_name]
    base_det = base["det_df"]
    base_final = float(base_det.iloc[-1].get("total_wealth", 0.0)) if not base_det.empty else 0.0
    base_tax = float(base_det.get("total_tax_paid", pd.Series(dtype=float)).sum())
    rows = []
    for name, result in comparison_results.items():
        det_df = result["det_df"]
        inputs = result["inputs"]
        final_wealth = float(det_df.iloc[-1].get("total_wealth", 0.0)) if not det_df.empty else 0.0
        retirement_row = _first_year_at_or_after_age(det_df, inputs.get("person1_retirement_age", 0))
        retirement_wealth = float(retirement_row.get("total_wealth", 0.0)) if retirement_row is not None else 0.0
        cumulative_tax = float(det_df.get("total_tax_paid", pd.Series(dtype=float)).sum())
        cashflow_columns = [
            "household_net_income",
            "total_minimum_pension_drawdown",
            "cash_reserve_withdrawal",
            "non_super_withdrawal",
            "total_extra_super_withdrawal",
            "residential_property_sale_proceeds",
        ]
        after_tax_cash = sum(
            float(det_df[column].sum()) for column in cashflow_columns if column in det_df.columns
        )
        failure_probability = 1.0 - float(result.get("success_rate", 0.0))
        rows.append({
            "scenario": name,
            "strategy_description": STRATEGY_PROFILES.get(name, {}).get("description_zh" if is_chinese else "description_en", ""),
            "after_tax_cashflow": after_tax_cash,
            "retirement_wealth": retirement_wealth,
            "final_wealth": final_wealth,
            "final_wealth_delta": final_wealth - base_final,
            "failure_probability": failure_probability,
            "cumulative_tax": cumulative_tax,
            "cumulative_tax_delta": cumulative_tax - base_tax,
            "first_advantage_year": _first_advantage_year(base_det, det_df),
            "break_even_year": _break_even_year(base_det, det_df),
            "key_risks": _risk_summary(result, is_chinese=is_chinese),
        })
    return pd.DataFrame(rows)


def build_assumption_change_df(comparison_results, is_chinese=False):
    if not comparison_results:
        return pd.DataFrame()
    base_name = "Base Case" if "Base Case" in comparison_results else next(iter(comparison_results))
    base_inputs = comparison_results[base_name]["inputs"]
    rows = []
    for name, result in comparison_results.items():
        inputs = result["inputs"]
        order = inputs.get("withdrawal_order", [])
        rows.append({
            "scenario": name,
            "withdrawal_order": " > ".join(order),
            "cash_reserve_floor": float(inputs.get("cash_reserve_floor", 0.0)),
            "non_super_estate_reserve": float(inputs.get("non_super_estate_reserve", 0.0)),
            "property_estate_reserve": float(inputs.get("property_estate_reserve", 0.0)),
            "differs_from_base": inputs != base_inputs,
        })
    return pd.DataFrame(rows)
