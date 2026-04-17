
import copy
import io
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from charts import (
    create_deterministic_wealth_chart_comparison,
    create_failure_probability_chart,
    create_histogram,
    create_income_vs_spending_chart,
    create_median_wealth_comparison_chart,
    create_percentile_paths_chart,
    create_success_rate_comparison_chart,
    create_tax_breakdown_chart,
    create_total_tax_paid_chart,
)
from model import (
    apply_preset_to_inputs,
    build_failure_probability_by_age,
    build_percentile_table,
    generate_input_warnings,
    generate_output_warnings,
    get_assumption_presets,
    normalise_contribution_events,
    run_deterministic_projection,
    run_monte_carlo,
    validate_inputs,
)


# ============================================================
# SECTION: UI LANGUAGE HELPERS
# ============================================================

LANGUAGE_EN = "🇬🇧 English"
LANGUAGE_CN = "🇨🇳 中文"


def is_cn():
    return st.session_state.get("ui_language", LANGUAGE_EN) == LANGUAGE_CN


def t(en, zh):
    return zh if is_cn() else en



# ============================================================
# SECTION: FILE EXPORT HELPERS
# ============================================================

def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(dataframes_dict):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return output.getvalue()


def sanitise_filename_part(value, fallback="untitled"):
    value = str(value or "").strip()
    if not value:
        return fallback

    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or fallback


def build_export_filename(report_title, base_name, scenario_name, extension):
    title_part = sanitise_filename_part(report_title, fallback="Untitled")
    base_part = sanitise_filename_part(base_name, fallback="financial_projection")
    scenario_part = sanitise_filename_part(scenario_name, fallback="Scenario")
    date_part = datetime.now().strftime("%d-%m-%Y")
    return f"{title_part}_{base_part}_{scenario_part}_{date_part}.{extension}"


def parse_formatted_number(value, is_percentage=False):
    text = str(value or "").strip()

    if not text:
        return 0.0

    cleaned = (
        text.replace("$", "")
        .replace(",", "")
        .replace("%", "")
        .replace(" ", "")
    )

    if cleaned in {"", "-", ".", "-."}:
        return 0.0

    number = float(cleaned)
    return number / 100.0 if is_percentage else number


def currency_text_input(label, value, key, help_text=None):
    raw_value = st.text_input(
        label,
        value=f"${float(value):,.0f}",
        key=key,
        help=help_text,
    )
    return parse_formatted_number(raw_value, is_percentage=False)


def percentage_text_input(label, value, key, decimals=0, help_text=None):
    raw_value = st.text_input(
        label,
        value=f"{float(value):.{decimals}%}",
        key=key,
        help=help_text,
    )
    return parse_formatted_number(raw_value, is_percentage=True)


# ============================================================
# SECTION: CONTRIBUTION EVENT HELPERS
# ============================================================

def contribution_events_to_records(events_df):
    clean_df = normalise_contribution_events(events_df)
    if clean_df.empty:
        return []
    return clean_df.to_dict(orient="records")


def get_default_contribution_events_df():
    return pd.DataFrame(
        columns=["financial_year", "person", "contribution_type", "amount"]
    )


# ============================================================
# SECTION: PRESET TABLE HELPERS
# ============================================================

def get_default_preset_table_df():
    presets = get_assumption_presets()
    rows = []

    for preset_name, values in presets.items():
        row = {"preset": preset_name}
        row.update(values)
        rows.append(row)

    return pd.DataFrame(rows)


def ensure_valid_preset_table_df(df):
    default_df = get_default_preset_table_df()

    if df is None:
        return default_df.copy()

    if not isinstance(df, pd.DataFrame):
        return default_df.copy()

    if df.empty:
        return default_df.copy()

    required_columns = list(default_df.columns)
    if any(col not in df.columns for col in required_columns):
        return default_df.copy()

    cleaned_df = df[required_columns].copy()

    if len(cleaned_df) != len(default_df):
        return default_df.copy()

    if cleaned_df["preset"].tolist() != default_df["preset"].tolist():
        return default_df.copy()

    for col in required_columns:
        if col == "preset":
            continue
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

    if cleaned_df.drop(columns=["preset"]).isna().any().any():
        return default_df.copy()

    return cleaned_df.reset_index(drop=True)


def preset_table_to_dict(preset_df):
    clean_df = preset_df.copy()
    preset_map = {}

    for _, row in clean_df.iterrows():
        preset_name = str(row["preset"])
        preset_map[preset_name] = {
            "super_income_return_mean": float(row["super_income_return_mean"]),
            "super_income_return_std": float(row["super_income_return_std"]),
            "super_capital_return_mean": float(row["super_capital_return_mean"]),
            "super_capital_return_std": float(row["super_capital_return_std"]),
            "non_super_income_return_mean": float(row["non_super_income_return_mean"]),
            "non_super_income_return_std": float(row["non_super_income_return_std"]),
            "non_super_capital_return_mean": float(row["non_super_capital_return_mean"]),
            "non_super_capital_return_std": float(row["non_super_capital_return_std"]),
            "inflation_rate": float(row["inflation_rate"]),
        }

    return preset_map


# ============================================================
# SECTION: DISPLAY TABLE HELPERS
# ============================================================

def build_assumption_details_df(inputs_by_scenario):
    rows = []

    for scenario_name, scenario_inputs in inputs_by_scenario.items():
        rows.append(
            {
                "scenario": scenario_name,
                "report_title": scenario_inputs.get("report_title", ""),
                "person1_name": scenario_inputs.get("person1_name", ""),
                "person2_name": scenario_inputs.get("person2_name", ""),
                "preset": scenario_inputs["assumption_preset"],
                "start_financial_year": scenario_inputs["start_financial_year"],
                "projection_years": scenario_inputs["projection_years"],
                "retirement_spending_trigger": scenario_inputs["retirement_spending_trigger"],
                "person1_current_age": scenario_inputs["person1_current_age"],
                "person2_current_age": scenario_inputs["person2_current_age"],
                "person1_retirement_age": scenario_inputs["person1_retirement_age"],
                "person2_retirement_age": scenario_inputs["person2_retirement_age"],
                "person1_pension_start_age": scenario_inputs["person1_pension_start_age"],
                "person2_pension_start_age": scenario_inputs["person2_pension_start_age"],
                "person1_accum_super_balance": scenario_inputs["person1_accum_super_balance"],
                "person1_pension_super_balance": scenario_inputs["person1_pension_super_balance"],
                "person2_accum_super_balance": scenario_inputs["person2_accum_super_balance"],
                "person2_pension_super_balance": scenario_inputs["person2_pension_super_balance"],
                "person1_transfer_balance_cap": scenario_inputs["person1_transfer_balance_cap"],
                "person2_transfer_balance_cap": scenario_inputs["person2_transfer_balance_cap"],
                "non_super_balance": scenario_inputs["non_super_balance"],
                "person1_annual_income": scenario_inputs["person1_annual_income"],
                "person2_annual_income": scenario_inputs["person2_annual_income"],
                "annual_living_expenses": scenario_inputs["annual_living_expenses"],
                "retirement_spending": scenario_inputs["retirement_spending"],
                "non_super_ownership_person1": scenario_inputs["non_super_ownership_person1"],
                "inflation_rate": scenario_inputs["inflation_rate"],
                "super_income_return_mean": scenario_inputs["super_income_return_mean"],
                "super_income_return_std": scenario_inputs["super_income_return_std"],
                "super_capital_return_mean": scenario_inputs["super_capital_return_mean"],
                "super_capital_return_std": scenario_inputs["super_capital_return_std"],
                "non_super_income_return_mean": scenario_inputs["non_super_income_return_mean"],
                "non_super_income_return_std": scenario_inputs["non_super_income_return_std"],
                "non_super_capital_return_mean": scenario_inputs["non_super_capital_return_mean"],
                "non_super_capital_return_std": scenario_inputs["non_super_capital_return_std"],
            }
        )

    return pd.DataFrame(rows)



def build_input_summary_df(inputs_by_scenario):
    rows = []

    for scenario_name, scenario_inputs in inputs_by_scenario.items():
        for input_name, input_value in scenario_inputs.items():
            if input_name == "contribution_events":
                continue

            if isinstance(input_value, (list, dict, pd.DataFrame)):
                continue

            rows.append(
                {
                    "scenario": scenario_name,
                    "input_name": input_name,
                    "input_value": input_value,
                }
            )

    return pd.DataFrame(rows)


def build_contribution_schedule_export_df(inputs_by_scenario):
    export_frames = []

    for scenario_name, scenario_inputs in inputs_by_scenario.items():
        events = scenario_inputs.get("contribution_events", [])
        if not events:
            continue

        event_df = pd.DataFrame(events).copy()
        event_df.insert(0, "scenario", scenario_name)
        export_frames.append(event_df)

    if not export_frames:
        return pd.DataFrame(columns=["scenario", "financial_year", "person", "contribution_type", "amount"])

    return pd.concat(export_frames, ignore_index=True)

def format_comparison_df(comparison_df):
    df = comparison_df.copy()
    df["success_rate_label"] = df["success_rate"].map(lambda x: f"{x:.1%}")
    df["median_final_wealth_label"] = df["median_final_wealth"].map(lambda x: f"${x:,.0f}")
    df["p10_final_wealth_label"] = df["p10_final_wealth"].map(lambda x: f"${x:,.0f}")
    df["p90_final_wealth_label"] = df["p90_final_wealth"].map(lambda x: f"${x:,.0f}")
    return df


def format_assumption_display_df(df):
    display_df = df.copy()

    percentage_cols = [
        "non_super_ownership_person1",
        "inflation_rate",
        "super_income_return_mean",
        "super_income_return_std",
        "super_capital_return_mean",
        "super_capital_return_std",
        "non_super_income_return_mean",
        "non_super_income_return_std",
        "non_super_capital_return_mean",
        "non_super_capital_return_std",
    ]

    for col in percentage_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col] * 100.0

    return display_df


def build_tax_summary_df(det_df):
    summary = {
        "person1_salary_tax_total": det_df["person1_salary_tax_total"].sum() if "person1_salary_tax_total" in det_df.columns else 0.0,
        "person1_non_super_tax_total": det_df["person1_non_super_tax_total"].sum() if "person1_non_super_tax_total" in det_df.columns else 0.0,
        "person1_total_personal_tax": det_df["person1_personal_tax_total"].sum() if "person1_personal_tax_total" in det_df.columns else 0.0,
        "person2_salary_tax_total": det_df["person2_salary_tax_total"].sum() if "person2_salary_tax_total" in det_df.columns else 0.0,
        "person2_non_super_tax_total": det_df["person2_non_super_tax_total"].sum() if "person2_non_super_tax_total" in det_df.columns else 0.0,
        "person2_total_personal_tax": det_df["person2_personal_tax_total"].sum() if "person2_personal_tax_total" in det_df.columns else 0.0,
        "total_super_contributions_tax": det_df["total_super_contributions_tax"].sum() if "total_super_contributions_tax" in det_df.columns else 0.0,
        "total_super_earnings_tax": det_df["total_super_earnings_tax"].sum() if "total_super_earnings_tax" in det_df.columns else 0.0,
        "total_tax_paid": det_df["total_tax_paid"].sum() if "total_tax_paid" in det_df.columns else 0.0,
    }

    return pd.DataFrame(
        {
            "tax_component": list(summary.keys()),
            "amount": list(summary.values()),
        }
    )


def build_adviser_cashflow_df(det_df):
    df = det_df.copy()

    def safe_col(name):
        return df[name] if name in df.columns else 0.0

    df["withdrawal"] = (
        safe_col("non_super_withdrawal")
        + safe_col("total_minimum_pension_drawdown")
        + safe_col("total_extra_super_withdrawal")
    )

    df["cgt"] = (
        safe_col("non_super_realised_capital_gain")
        + safe_col("person1_super_realised_capital_gain")
        + safe_col("person2_super_realised_capital_gain")
    )

    df["tax"] = safe_col("total_tax_paid")

    df["net_cash"] = (
        safe_col("household_net_income")
        + safe_col("total_minimum_pension_drawdown")
        + safe_col("total_extra_super_withdrawal")
        + safe_col("non_super_withdrawal")
        - safe_col("total_cash_contributions")
        - safe_col("non_super_tax_paid")
        - safe_col("total_super_withdrawal_cgt_tax")
    )

    return df[
        [
            "financial_year_end",
            "withdrawal",
            "cgt",
            "tax",
            "net_cash",
        ]
    ].rename(
        columns={
            "financial_year_end": "Year",
            "withdrawal": "Withdrawal",
            "cgt": "CGT",
            "tax": "Tax",
            "net_cash": "Net Cash",
        }
    )


def build_cgt_validation_df(det_df):
    df = det_df.copy()

    def safe_col(name):
        return df[name] if name in df.columns else 0.0

    validation_df = pd.DataFrame(
        {
            "Year": df["financial_year_end"] if "financial_year_end" in df.columns else range(len(df)),
            "P1 Age": safe_col("person1_age"),
            "P2 Age": safe_col("person2_age"),
            "P1 Started Pension This Year": safe_col("person1_started_pension_this_year"),
            "P2 Started Pension This Year": safe_col("person2_started_pension_this_year"),
            "P1 Has Started Pension": safe_col("person1_has_started_pension"),
            "P2 Has Started Pension": safe_col("person2_has_started_pension"),
            "P1 Opening Accum Balance": safe_col("opening_person1_accum_super_balance"),
            "P1 Opening Pension Balance": safe_col("opening_person1_pension_super_balance"),
            "P2 Opening Accum Balance": safe_col("opening_person2_accum_super_balance"),
            "P2 Opening Pension Balance": safe_col("opening_person2_pension_super_balance"),
            "P1 Ending Accum Balance": safe_col("ending_person1_accum_super_balance"),
            "P1 Ending Pension Balance": safe_col("ending_person1_pension_super_balance"),
            "P2 Ending Accum Balance": safe_col("ending_person2_accum_super_balance"),
            "P2 Ending Pension Balance": safe_col("ending_person2_pension_super_balance"),
            "P1 Transfer to Pension": safe_col("person1_transfer_to_pension"),
            "P2 Transfer to Pension": safe_col("person2_transfer_to_pension"),
            "P1 Total Net Super Contribution": safe_col("person1_total_net_super_contribution"),
            "P2 Total Net Super Contribution": safe_col("person2_total_net_super_contribution"),
            "P1 Super Realised CGT": safe_col("person1_super_realised_capital_gain"),
            "P2 Super Realised CGT": safe_col("person2_super_realised_capital_gain"),
            "Super Withdrawal CGT Tax": safe_col("total_super_withdrawal_cgt_tax"),
            "P1 Pension Earnings Tax": safe_col("person1_pension_earnings_tax"),
            "P2 Pension Earnings Tax": safe_col("person2_pension_earnings_tax"),
            "Super Earnings Tax": safe_col("total_super_earnings_tax"),
        }
    )

    return validation_df


def build_pension_tax_free_summary_df(det_df):
    df = det_df.copy()

    def safe_col(name):
        return df[name] if name in df.columns else 0.0

    def sum_all(name):
        return df[name].sum() if name in df.columns else 0.0

    pension_phase_mask = (
        (safe_col("ending_person1_pension_super_balance") > 0)
        | (safe_col("ending_person2_pension_super_balance") > 0)
        | (safe_col("person1_transfer_to_pension") > 0)
        | (safe_col("person2_transfer_to_pension") > 0)
        | (safe_col("person1_has_started_pension") > 0)
        | (safe_col("person2_has_started_pension") > 0)
    )

    pension_phase_df = df.loc[pension_phase_mask].copy()

    def sum_pension_phase(name):
        return pension_phase_df[name].sum() if name in pension_phase_df.columns else 0.0

    rows = [
        {
            "check": "Total super withdrawal CGT tax (all years)",
            "value": sum_all("total_super_withdrawal_cgt_tax"),
        },
        {
            "check": "Total super earnings tax (all years)",
            "value": sum_all("total_super_earnings_tax"),
        },
        {
            "check": "Total super earnings tax (pension phase only)",
            "value": sum_pension_phase("total_super_earnings_tax"),
        },
        {
            "check": "Total P1 pension earnings tax",
            "value": sum_all("person1_pension_earnings_tax"),
        },
        {
            "check": "Total P2 pension earnings tax",
            "value": sum_all("person2_pension_earnings_tax"),
        },
        {
            "check": "Total P1 transfer to pension",
            "value": sum_all("person1_transfer_to_pension"),
        },
        {
            "check": "Total P2 transfer to pension",
            "value": sum_all("person2_transfer_to_pension"),
        },
    ]

    return pd.DataFrame(rows)


def render_assumption_details(df):
    display_df = format_assumption_display_df(df)

    st.subheader(t("Assumption Details", "假设明细"))
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "scenario": st.column_config.TextColumn("Scenario"),
            "report_title": st.column_config.TextColumn("Title"),
            "person1_name": st.column_config.TextColumn("Person 1 Name"),
            "person2_name": st.column_config.TextColumn("Person 2 Name"),
            "preset": st.column_config.TextColumn("Preset"),
            "start_financial_year": st.column_config.NumberColumn("Start Financial Year", format="%d"),
            "projection_years": st.column_config.NumberColumn("Projection Years", format="%d"),
            "retirement_spending_trigger": st.column_config.TextColumn("Retirement Spending Trigger"),
            "person1_current_age": st.column_config.NumberColumn("P1 Current Age", format="%d"),
            "person2_current_age": st.column_config.NumberColumn("P2 Current Age", format="%d"),
            "person1_retirement_age": st.column_config.NumberColumn("P1 Retirement Age", format="%d"),
            "person2_retirement_age": st.column_config.NumberColumn("P2 Retirement Age", format="%d"),
            "person1_pension_start_age": st.column_config.NumberColumn("P1 Pension Start Age", format="%d"),
            "person2_pension_start_age": st.column_config.NumberColumn("P2 Pension Start Age", format="%d"),
            "person1_accum_super_balance": st.column_config.NumberColumn("P1 Accum Super", format="$%.0f"),
            "person1_pension_super_balance": st.column_config.NumberColumn("P1 Pension Super", format="$%.0f"),
            "person2_accum_super_balance": st.column_config.NumberColumn("P2 Accum Super", format="$%.0f"),
            "person2_pension_super_balance": st.column_config.NumberColumn("P2 Pension Super", format="$%.0f"),
            "person1_transfer_balance_cap": st.column_config.NumberColumn("P1 TBC", format="$%.0f"),
            "person2_transfer_balance_cap": st.column_config.NumberColumn("P2 TBC", format="$%.0f"),
            "non_super_balance": st.column_config.NumberColumn("Non-Super Balance", format="$%.0f"),
            "person1_annual_income": st.column_config.NumberColumn("P1 Annual Income", format="$%.0f"),
            "person2_annual_income": st.column_config.NumberColumn("P2 Annual Income", format="$%.0f"),
            "annual_living_expenses": st.column_config.NumberColumn("Annual Living Expenses", format="$%.0f"),
            "retirement_spending": st.column_config.NumberColumn("Retirement Spending", format="$%.0f"),
            "non_super_ownership_person1": st.column_config.NumberColumn("P1 Ownership", format="%.0f%%"),
            "inflation_rate": st.column_config.NumberColumn("Inflation", format="%.1f%%"),
            "super_income_return_mean": st.column_config.NumberColumn("Super Income Mean", format="%.1f%%"),
            "super_income_return_std": st.column_config.NumberColumn("Super Income Std", format="%.1f%%"),
            "super_capital_return_mean": st.column_config.NumberColumn("Super Capital Mean", format="%.1f%%"),
            "super_capital_return_std": st.column_config.NumberColumn("Super Capital Std", format="%.1f%%"),
            "non_super_income_return_mean": st.column_config.NumberColumn("Non-Super Income Mean", format="%.1f%%"),
            "non_super_income_return_std": st.column_config.NumberColumn("Non-Super Income Std", format="%.1f%%"),
            "non_super_capital_return_mean": st.column_config.NumberColumn("Non-Super Capital Mean", format="%.1f%%"),
            "non_super_capital_return_std": st.column_config.NumberColumn("Non-Super Capital Std", format="%.1f%%"),
        },
    )


def render_warning_sections(input_warnings_by_scenario, output_warnings_by_scenario, view_mode):
    if view_mode == t("Adviser View", "顾问视图"):
        for scenario_name, warnings_list in input_warnings_by_scenario.items():
            if warnings_list:
                st.subheader(f"Input Warnings - {scenario_name}")
                for warning in warnings_list:
                    st.warning(warning)

        for scenario_name, warnings_list in output_warnings_by_scenario.items():
            if warnings_list:
                st.subheader(f"Result Warnings - {scenario_name}")
                for warning in warnings_list:
                    st.warning(warning)
    else:
        client_messages = []

        for warnings_list in input_warnings_by_scenario.values():
            client_messages.extend(warnings_list)

        for warnings_list in output_warnings_by_scenario.values():
            client_messages.extend(warnings_list)

        if client_messages:
            st.subheader(t("Important Notes", "重要提示"))
            for message in client_messages[:5]:
                st.warning(message)


def chart_key(chart_type, scenario_name, view_mode, section_name="main"):
    safe_scenario = scenario_name.lower().replace(" ", "_")
    safe_view = view_mode.lower().replace(" ", "_")
    safe_section = section_name.lower().replace(" ", "_")
    return f"{chart_type}_{safe_scenario}_{safe_view}_{safe_section}"


def build_current_result_bundle_from_session_state():
    if st.session_state.comparison_results is None:
        return None

    return {
        "comparison_results": copy.deepcopy(st.session_state.comparison_results),
        "assumption_details_df": None if st.session_state.assumption_details_df is None else st.session_state.assumption_details_df.copy(),
        "input_summary_df": None if st.session_state.input_summary_df is None else st.session_state.input_summary_df.copy(),
        "contribution_schedule_export_df": None if st.session_state.contribution_schedule_export_df is None else st.session_state.contribution_schedule_export_df.copy(),
        "input_warnings_by_scenario": copy.deepcopy(st.session_state.input_warnings_by_scenario),
        "output_warnings_by_scenario": copy.deepcopy(st.session_state.output_warnings_by_scenario),
        "last_run_inputs_by_scenario": copy.deepcopy(st.session_state.last_run_inputs_by_scenario),
    }


def get_active_result_bundle():
    active_name = st.session_state.get("active_result_set_name", "Current Results")
    if active_name == "Current Results":
        return build_current_result_bundle_from_session_state()
    return st.session_state.get("saved_result_sets", {}).get(active_name)


def save_current_results_snapshot(snapshot_name):
    current_bundle = build_current_result_bundle_from_session_state()
    if current_bundle is None:
        return False, t("There are no current results to save yet.", "当前还没有可保存的结果。")

    snapshot_name = str(snapshot_name or "").strip()
    if not snapshot_name:
        return False, t("Please enter a name for the saved results.", "请先输入保存结果的名称。")

    saved_sets = copy.deepcopy(st.session_state.get("saved_result_sets", {}))
    saved_sets[snapshot_name] = current_bundle
    st.session_state.saved_result_sets = saved_sets
    st.session_state.active_result_set_name = snapshot_name
    return True, t(f"Saved results as: {snapshot_name}", f"已保存结果：{snapshot_name}")


def rename_saved_results_snapshot(old_name, new_name):
    old_name = str(old_name or "").strip()
    new_name = str(new_name or "").strip()
    if old_name in {"", "Current Results"}:
        return False, t("Select a saved result to rename.", "请选择一个已保存结果进行重命名。")
    if not new_name:
        return False, t("Please enter a new name.", "请输入新名称。")

    saved_sets = copy.deepcopy(st.session_state.get("saved_result_sets", {}))
    if old_name not in saved_sets:
        return False, t("The selected saved result no longer exists.", "所选已保存结果不存在。")
    if new_name != old_name and new_name in saved_sets:
        return False, t("That result name already exists.", "该结果名称已存在。")

    saved_sets[new_name] = saved_sets.pop(old_name)
    st.session_state.saved_result_sets = saved_sets
    if st.session_state.get("active_result_set_name") == old_name:
        st.session_state.active_result_set_name = new_name
    return True, t(f"Renamed to: {new_name}", f"已重命名为：{new_name}")


def _discount_factor_for_year(inputs, financial_year_end):
    start_fy = int(inputs["start_financial_year"])
    year_index = max(int(financial_year_end) - start_fy, 0)
    return (1 + float(inputs.get("inflation_rate", 0.0))) ** year_index


def _is_currency_like_column(col_name):
    lowered = str(col_name).lower()
    if lowered in {
        "year", "financial_year_end", "person1_age", "person2_age", "year_index",
        "simulation_id", "failed_by_year_count", "total_simulations",
    }:
        return False
    if lowered.endswith("_rate") or lowered.endswith("_probability"):
        return False
    if "age" in lowered:
        return False
    if "success_rate" in lowered:
        return False
    if "probability" in lowered:
        return False
    if lowered.startswith("p") and lowered[1:].isdigit():
        return True
    currency_tokens = [
        "wealth", "balance", "income", "spending", "expenses", "expense", "tax", "cgt",
        "withdrawal", "drawdown", "earnings", "contribution", "cost_base", "cost base",
        "cash", "amount", "cap space", "transfer", "surplus", "shortfall", "value",
        "gain", "loss", "accum", "pension",
    ]
    return any(token in lowered for token in currency_tokens)


def convert_det_df_for_value_mode(det_df, inputs, value_mode):
    df = det_df.copy()
    if value_mode == "Future Value" or df.empty:
        return df
    if "financial_year_end" not in df.columns:
        return df

    discount_factors = df["financial_year_end"].apply(lambda fy: _discount_factor_for_year(inputs, fy))
    for col in df.columns:
        if col == "financial_year_end":
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if _is_currency_like_column(col):
            df[col] = df[col] / discount_factors
    return df


def convert_percentile_df_for_value_mode(percentile_df, inputs, value_mode):
    df = percentile_df.copy()
    if value_mode == "Future Value" or df.empty or "financial_year_end" not in df.columns:
        return df
    discount_factors = df["financial_year_end"].apply(lambda fy: _discount_factor_for_year(inputs, fy))
    for col in ["p10", "p50", "p90"]:
        if col in df.columns:
            df[col] = df[col] / discount_factors
    return df


def convert_summary_df_for_value_mode(summary_df, inputs, value_mode):
    df = summary_df.copy()
    if value_mode == "Future Value" or df.empty or "final_wealth" not in df.columns:
        return df
    projection_horizon_end = int(inputs["start_financial_year"]) + int(inputs["projection_years"]) - 1
    discount_factor = _discount_factor_for_year(inputs, projection_horizon_end)
    df["final_wealth"] = df["final_wealth"] / discount_factor
    return df


def convert_comparison_df_for_value_mode(comparison_df, selected_result_inputs, value_mode):
    df = comparison_df.copy()
    if value_mode == "Future Value" or df.empty:
        return df
    projection_horizon_end = int(selected_result_inputs["start_financial_year"]) + int(selected_result_inputs["projection_years"]) - 1
    discount_factor = _discount_factor_for_year(selected_result_inputs, projection_horizon_end)
    for col in ["median_final_wealth", "p10_final_wealth", "p90_final_wealth"]:
        if col in df.columns:
            df[col] = df[col] / discount_factor
    return format_comparison_df(df)


def display_value_label(value_mode):
    return t("Present Value", "现值") if value_mode == "Present Value" else t("Future Value", "终值")


def render_live_input_feedback(base_inputs):
    validation_errors = validate_inputs(base_inputs)
    input_warnings = generate_input_warnings(base_inputs)

    if validation_errors:
        st.error(t("Live input validation found issues.", "即时输入检查发现问题。"))
        for err in validation_errors[:8]:
            st.error(err)

    high_risk_messages = []
    if base_inputs["inflation_rate"] > 0.08:
        high_risk_messages.append(t("⚠️ High inflation assumption", "⚠️ 通胀假设偏高"))
    if base_inputs["super_capital_return_std"] >= 0.18 or base_inputs["non_super_capital_return_std"] >= 0.18:
        high_risk_messages.append(t("⚠️ High volatility assumptions", "⚠️ 波动率假设偏高"))
    if base_inputs["person1_retirement_age"] < 55 or (base_inputs["household_mode"] == "Two People" and base_inputs["person2_retirement_age"] < 55):
        high_risk_messages.append(t("⚠️ Early retirement age", "⚠️ 退休年龄偏早"))
    if base_inputs["number_of_simulations"] < 1000:
        high_risk_messages.append(t("⚠️ Low simulation count", "⚠️ 模拟次数偏低"))
    if base_inputs["non_super_cost_base"] > base_inputs["non_super_balance"]:
        high_risk_messages.append(t("⚠️ Non-super cost base exceeds balance", "⚠️ 非养老金成本基础高于余额"))

    if high_risk_messages:
        st.markdown("  ".join([f"`{msg}`" for msg in high_risk_messages]))

    if input_warnings and not validation_errors:
        with st.expander(t("Live Input Warnings", "即时输入提示"), expanded=False):
            for msg in input_warnings[:8]:
                st.warning(msg)


def render_saved_result_comparison_section(saved_result_sets, value_mode):
    if len(saved_result_sets) < 2:
        return

    st.subheader(t("Compare Two Saved Results", "比较两个已保存结果"))

    saved_names = list(saved_result_sets.keys())
    compare_col1, compare_col2 = st.columns(2)
    with compare_col1:
        left_name = st.selectbox(
            t("Saved Result A", "已保存结果 A"),
            options=saved_names,
            key="saved_compare_left",
        )
    with compare_col2:
        right_default_index = 1 if len(saved_names) > 1 else 0
        right_name = st.selectbox(
            t("Saved Result B", "已保存结果 B"),
            options=saved_names,
            index=right_default_index,
            key="saved_compare_right",
        )

    if left_name == right_name:
        st.info(t("Choose two different saved results to compare.", "请选择两个不同的已保存结果进行比较。"))
        return

    left_bundle = saved_result_sets[left_name]
    right_bundle = saved_result_sets[right_name]

    left_results = left_bundle.get("comparison_results", {})
    right_results = right_bundle.get("comparison_results", {})
    if not left_results or not right_results:
        st.info(t("One of the saved results does not contain comparison data.", "其中一个已保存结果不包含比较数据。"))
        return

    left_scenario_name = list(left_results.keys())[0]
    right_scenario_name = list(right_results.keys())[0]
    left_result = left_results[left_scenario_name]
    right_result = right_results[right_scenario_name]

    left_summary_df = convert_summary_df_for_value_mode(left_result["summary_df"], left_result["inputs"], value_mode)
    right_summary_df = convert_summary_df_for_value_mode(right_result["summary_df"], right_result["inputs"], value_mode)

    comparison_rows = [
        {
            "Saved Result": left_name,
            "Scenario": left_scenario_name,
            "Success Rate": left_result["success_rate"],
            "Median Final Wealth": left_summary_df["final_wealth"].median(),
            "P10 Final Wealth": left_summary_df["final_wealth"].quantile(0.10),
            "P90 Final Wealth": left_summary_df["final_wealth"].quantile(0.90),
        },
        {
            "Saved Result": right_name,
            "Scenario": right_scenario_name,
            "Success Rate": right_result["success_rate"],
            "Median Final Wealth": right_summary_df["final_wealth"].median(),
            "P10 Final Wealth": right_summary_df["final_wealth"].quantile(0.10),
            "P90 Final Wealth": right_summary_df["final_wealth"].quantile(0.90),
        },
    ]
    comparison_table = pd.DataFrame(comparison_rows)
    st.dataframe(
        comparison_table,
        use_container_width=True,
        column_config={
            "Success Rate": st.column_config.NumberColumn("Success Rate", format="%.1f%%"),
            "Median Final Wealth": st.column_config.NumberColumn("Median Final Wealth", format="$%.0f"),
            "P10 Final Wealth": st.column_config.NumberColumn("P10 Final Wealth", format="$%.0f"),
            "P90 Final Wealth": st.column_config.NumberColumn("P90 Final Wealth", format="$%.0f"),
        },
    )

    left_det = convert_det_df_for_value_mode(left_result["det_df"], left_result["inputs"], value_mode).copy()
    right_det = convert_det_df_for_value_mode(right_result["det_df"], right_result["inputs"], value_mode).copy()
    left_det["comparison_name"] = left_name
    right_det["comparison_name"] = right_name
    combined = pd.concat([left_det, right_det], ignore_index=True)

    fig = px.line(
        combined,
        x="financial_year_end",
        y="total_wealth",
        color="comparison_name",
        title=t("Saved Results Wealth Comparison", "已保存结果财富对比"),
    )
    fig.update_layout(
        xaxis_title=t("Financial Year", "财政年度"),
        yaxis_title=display_value_label(value_mode),
        hovermode="x unified",
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    st.plotly_chart(fig, use_container_width=True, key="saved_results_compare_chart")


def get_missing_validation_columns(det_df):
    required_cols = [
        "financial_year_end",
        "person1_age",
        "person2_age",
        "person1_started_pension_this_year",
        "person2_started_pension_this_year",
        "person1_has_started_pension",
        "person2_has_started_pension",
        "opening_person1_accum_super_balance",
        "opening_person1_pension_super_balance",
        "opening_person2_accum_super_balance",
        "opening_person2_pension_super_balance",
        "ending_person1_accum_super_balance",
        "ending_person1_pension_super_balance",
        "ending_person2_accum_super_balance",
        "ending_person2_pension_super_balance",
        "person1_transfer_to_pension",
        "person2_transfer_to_pension",
        "person1_total_net_super_contribution",
        "person2_total_net_super_contribution",
        "person1_super_realised_capital_gain",
        "person2_super_realised_capital_gain",
        "total_super_withdrawal_cgt_tax",
        "person1_pension_earnings_tax",
        "person2_pension_earnings_tax",
        "total_super_earnings_tax",
    ]
    return [col for col in required_cols if col not in det_df.columns]


def build_adviser_debug_df(det_df):
    df = det_df.copy()

    def safe_col(name):
        return df[name] if name in df.columns else 0.0

    debug_df = pd.DataFrame(
        {
            "Year": safe_col("financial_year_end"),
            "P1 Started Pension This Year": safe_col("person1_started_pension_this_year"),
            "P1 Has Started Pension": safe_col("person1_has_started_pension"),
            "P2 Started Pension This Year": safe_col("person2_started_pension_this_year"),
            "P2 Has Started Pension": safe_col("person2_has_started_pension"),
            "P1 Requested Transfer": safe_col("person1_requested_transfer_amount"),
            "P2 Requested Transfer": safe_col("person2_requested_transfer_amount"),
            "P1 Available Cap Space": safe_col("person1_available_cap_space"),
            "P2 Available Cap Space": safe_col("person2_available_cap_space"),
            "P1 Transfer to Pension": safe_col("person1_transfer_to_pension"),
            "P2 Transfer to Pension": safe_col("person2_transfer_to_pension"),
            "P1 Excess Retained in Accum": safe_col("person1_excess_retained_in_accumulation"),
            "P2 Excess Retained in Accum": safe_col("person2_excess_retained_in_accumulation"),
            "P1 Opening Accum": safe_col("opening_person1_accum_super_balance"),
            "P1 Opening Pension": safe_col("opening_person1_pension_super_balance"),
            "P2 Opening Accum": safe_col("opening_person2_accum_super_balance"),
            "P2 Opening Pension": safe_col("opening_person2_pension_super_balance"),
            "P1 Net Super Contribution": safe_col("person1_total_net_super_contribution"),
            "P2 Net Super Contribution": safe_col("person2_total_net_super_contribution"),
            "P1 Ending Accum": safe_col("ending_person1_accum_super_balance"),
            "P1 Ending Pension": safe_col("ending_person1_pension_super_balance"),
            "P2 Ending Accum": safe_col("ending_person2_accum_super_balance"),
            "P2 Ending Pension": safe_col("ending_person2_pension_super_balance"),
            "P1 Min Pension Drawdown": safe_col("person1_min_pension_drawdown"),
            "P2 Min Pension Drawdown": safe_col("person2_min_pension_drawdown"),
            "P1 Extra Pension Withdrawal": safe_col("person1_extra_pension_withdrawal"),
            "P2 Extra Pension Withdrawal": safe_col("person2_extra_pension_withdrawal"),
            "P1 Pension Earnings Tax": safe_col("person1_pension_earnings_tax"),
            "P2 Pension Earnings Tax": safe_col("person2_pension_earnings_tax"),
            "P1 Super Realised CGT": safe_col("person1_super_realised_capital_gain"),
            "P2 Super Realised CGT": safe_col("person2_super_realised_capital_gain"),
            "Super Withdrawal CGT Tax": safe_col("total_super_withdrawal_cgt_tax"),
            "Total Super Earnings Tax": safe_col("total_super_earnings_tax"),
        }
    )

    return debug_df


# ============================================================
# SECTION: PAGE SETUP
# ============================================================

st.set_page_config(page_title="Retirement Modelling Suite (AU)", page_icon="📊", layout="wide")

if "ui_language" not in st.session_state:
    st.session_state.ui_language = LANGUAGE_EN


# ============================================================
# SECTION: SESSION DEFAULTS
# ============================================================

defaults = {
    "comparison_results": None,
    "saved_result_sets": {},
    "active_result_set_name": "Current Results",
    "save_result_name": "",
    "rename_result_name": "",
    "assumption_details_df": None,
    "input_summary_df": None,
    "contribution_schedule_export_df": None,
    "input_warnings_by_scenario": None,
    "output_warnings_by_scenario": None,
    "last_run_inputs_by_scenario": None,
    "assumption_preset": "Base Case",
    "value_mode": "Future Value",
    "start_financial_year": 2027,
    "projection_years": 40,
    "retirement_spending_trigger": "Both Retired",
    "household_mode": "Two People",
    "report_title": "",
    "person1_name": "",
    "person2_name": "",
    "person1_current_age": 45,
    "person2_current_age": 43,
    "person1_retirement_age": 60,
    "person2_retirement_age": 58,
    "person1_pension_start_age": 67,
    "person2_pension_start_age": 65,
    "person1_accum_super_balance": 450000.0,
    "person1_pension_super_balance": 0.0,
    "person2_accum_super_balance": 350000.0,
    "person2_pension_super_balance": 0.0,
    "person1_accum_super_cost_base": 450000.0,
    "person1_pension_super_cost_base": 0.0,
    "person2_accum_super_cost_base": 350000.0,
    "person2_pension_super_cost_base": 0.0,
    "person1_transfer_balance_cap": 2100000.0,
    "person2_transfer_balance_cap": 2100000.0,
    "person1_annual_income": 120000.0,
    "person2_annual_income": 60000.0,
    "non_super_balance": 500000.0,
    "non_super_cost_base": 300000.0,
    "annual_living_expenses": 90000.0,
    "retirement_spending": 100000.0,
    "non_super_ownership_person1_pct": 50.0,
    "cgt_discount_rate": 0.50,
    "super_income_return_mean": 0.020,
    "super_income_return_std": 0.020,
    "super_capital_return_mean": 0.040,
    "super_capital_return_std": 0.090,
    "non_super_income_return_mean": 0.020,
    "non_super_income_return_std": 0.020,
    "non_super_capital_return_mean": 0.030,
    "non_super_capital_return_std": 0.080,
    "inflation_rate": 0.030,
    "number_of_simulations": 1000,
    "random_seed": 42,
    "ui_language": LANGUAGE_EN,
    "preset_table_df": get_default_preset_table_df(),
    "contribution_events_df": get_default_contribution_events_df(),
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def apply_preset_values(preset_name, preset_map):
    preset_values = preset_map[preset_name]
    st.session_state.inflation_rate = preset_values["inflation_rate"]
    st.session_state.super_income_return_mean = preset_values["super_income_return_mean"]
    st.session_state.super_income_return_std = preset_values["super_income_return_std"]
    st.session_state.super_capital_return_mean = preset_values["super_capital_return_mean"]
    st.session_state.super_capital_return_std = preset_values["super_capital_return_std"]
    st.session_state.non_super_income_return_mean = preset_values["non_super_income_return_mean"]
    st.session_state.non_super_income_return_std = preset_values["non_super_income_return_std"]
    st.session_state.non_super_capital_return_mean = preset_values["non_super_capital_return_mean"]
    st.session_state.non_super_capital_return_std = preset_values["non_super_capital_return_std"]

    if "cgt_discount_rate" in preset_values:
        st.session_state.cgt_discount_rate = preset_values["cgt_discount_rate"]


# ============================================================
# SECTION: SESSION NORMALISATION
# ============================================================

st.session_state.preset_table_df = ensure_valid_preset_table_df(
    st.session_state.get("preset_table_df")
)

if "active_input_section" not in st.session_state:
    st.session_state.active_input_section = "projection"


# ============================================================
# SECTION: INPUT LOCAL STATE
# ============================================================

report_title = st.session_state.report_title
person1_name = st.session_state.person1_name
person2_name = st.session_state.person2_name

start_financial_year = int(st.session_state.start_financial_year)
projection_years = int(st.session_state.projection_years)
retirement_spending_trigger = st.session_state.retirement_spending_trigger
household_mode = st.session_state.household_mode
value_mode = st.session_state.value_mode
is_one_person_mode = household_mode == "One Person"

person1_current_age = int(st.session_state.person1_current_age)
person2_current_age = int(st.session_state.person2_current_age)
person1_retirement_age = int(st.session_state.person1_retirement_age)
person2_retirement_age = int(st.session_state.person2_retirement_age)
person1_pension_start_age = int(st.session_state.person1_pension_start_age)
person2_pension_start_age = int(st.session_state.person2_pension_start_age)

person1_accum_super_balance = st.session_state.person1_accum_super_balance
person1_pension_super_balance = st.session_state.person1_pension_super_balance
person2_accum_super_balance = st.session_state.person2_accum_super_balance
person2_pension_super_balance = st.session_state.person2_pension_super_balance

person1_accum_super_cost_base = st.session_state.person1_accum_super_cost_base
person1_pension_super_cost_base = st.session_state.person1_pension_super_cost_base
person2_accum_super_cost_base = st.session_state.person2_accum_super_cost_base
person2_pension_super_cost_base = st.session_state.person2_pension_super_cost_base

person1_transfer_balance_cap = st.session_state.person1_transfer_balance_cap
person2_transfer_balance_cap = st.session_state.person2_transfer_balance_cap
person1_annual_income = st.session_state.person1_annual_income
person2_annual_income = st.session_state.person2_annual_income

non_super_balance = st.session_state.non_super_balance
non_super_cost_base = st.session_state.non_super_cost_base
annual_living_expenses = st.session_state.annual_living_expenses
retirement_spending = st.session_state.retirement_spending
non_super_ownership_person1 = st.session_state.non_super_ownership_person1_pct
cgt_discount_rate = st.session_state.cgt_discount_rate

super_income_return_mean = st.session_state.super_income_return_mean
super_income_return_std = st.session_state.super_income_return_std
super_capital_return_mean = st.session_state.super_capital_return_mean
super_capital_return_std = st.session_state.super_capital_return_std
non_super_income_return_mean = st.session_state.non_super_income_return_mean
non_super_income_return_std = st.session_state.non_super_income_return_std
non_super_capital_return_mean = st.session_state.non_super_capital_return_mean
non_super_capital_return_std = st.session_state.non_super_capital_return_std
inflation_rate = st.session_state.inflation_rate

number_of_simulations = int(st.session_state.number_of_simulations)
random_seed = int(st.session_state.random_seed)
contribution_events_df = normalise_contribution_events(
    st.session_state.contribution_events_df.copy(),
    household_mode=household_mode,
)

# ============================================================
# SECTION: SIDEBAR CONTROLS
# ============================================================

with st.sidebar:
    st.markdown(f"### {t('Controls', '控制面板')}")

    selected_language = st.radio(
        t("Language", "语言"),
        options=[LANGUAGE_EN, LANGUAGE_CN],
        index=0 if st.session_state.ui_language == LANGUAGE_EN else 1,
        key="ui_language_selector",
    )
    st.session_state.ui_language = selected_language

    view_mode = st.radio(
        t("View Mode", "视图模式"),
        options=[t("Adviser View", "顾问视图"), t("Client View", "客户视图")],
    )

    household_mode = st.radio(
        t("Household Mode", "家庭模式"),
        options=["One Person", "Two People"],
        index=0 if st.session_state.household_mode == "One Person" else 1,
        help=t(
            "One Person mode excludes Person 2 from the model. Household spending is not reduced automatically.",
            "单人模式会把人物 2 排除出模型，但不会自动下调家庭支出。",
        ),
    )
    is_one_person_mode = household_mode == "One Person"

    value_mode = st.radio(
        t("Value Display", "数值显示"),
        options=["Future Value", "Present Value"],
        index=0 if st.session_state.value_mode == "Future Value" else 1,
        help=t(
            "Present Value discounts displayed monetary outputs back to the starting financial year using the inflation assumption.",
            "现值会按 inflation 假设把显示金额折算回起始财政年度。",
        ),
    )

    scenario_mode = st.radio(
        t("Scenario Mode", "情景模式"),
        options=[t("Single Scenario", "单一情景"), t("Compare Standard Presets", "比较标准预设")],
    )

    preset_choice = st.selectbox(
        t("Assumption Preset", "假设预设"),
        options=["Conservative", "Base Case", "Optimistic", "Custom"],
        key="assumption_preset",
        disabled=(scenario_mode == t("Compare Standard Presets", "比较标准预设")),
    )

    if is_one_person_mode:
        st.info(
            t(
                "One Person mode removes Person 2 from the model. Household spending remains exactly as entered, so review your spending assumptions manually.",
                "单人模式会把 Person 2 从模型中移除。家庭支出会保持原输入值不变，因此请手动检查支出假设。",
            )
        )

    st.markdown(f"### {t('Saved Results', '已保存结果')}")
    saved_result_sets = st.session_state.get("saved_result_sets", {})
    available_result_views = ["Current Results"] + list(saved_result_sets.keys())

    current_active_result_name = st.session_state.get("active_result_set_name", "Current Results")
    if current_active_result_name not in available_result_views:
        current_active_result_name = "Current Results"
        st.session_state.active_result_set_name = "Current Results"

    active_result_set_name = st.selectbox(
        t("Displayed Result Set", "当前显示结果集"),
        options=available_result_views,
        index=available_result_views.index(current_active_result_name),
        key="active_result_set_name_selector",
        help=t(
            "Switch between the latest run and any saved snapshots in this session.",
            "可在本次会话中切换查看最新结果与已保存快照。",
        ),
    )
    st.session_state.active_result_set_name = active_result_set_name

    save_result_name = st.text_input(
        t("Save Current Results As", "将当前结果另存为"),
        value=st.session_state.get("save_result_name", ""),
        key="save_result_name_input",
        placeholder=t("e.g. One person test", "例如：单人模式测试"),
    )
    st.session_state.save_result_name = save_result_name

    save_results_button = st.button(
        t("Save Current Results", "保存当前结果"),
        use_container_width=True,
        disabled=(st.session_state.comparison_results is None),
    )

    if active_result_set_name != "Current Results":
        rename_result_name = st.text_input(
            t("Rename Selected Saved Result", "重命名当前已保存结果"),
            value=st.session_state.get("rename_result_name", active_result_set_name),
            key="rename_result_name_input",
        )
        st.session_state.rename_result_name = rename_result_name

        rename_button = st.button(
            t("Rename Saved Result", "重命名已保存结果"),
            use_container_width=True,
        )

        delete_button = st.button(
            t("Delete Selected Saved Result", "删除当前已保存结果"),
            use_container_width=True,
        )
    else:
        rename_button = False
        delete_button = False

    run_button = st.button(t("Run Simulation", "运行模拟"), type="primary", use_container_width=True)


if save_results_button:
    ok, message = save_current_results_snapshot(save_result_name)
    (st.success if ok else st.error)(message)

if rename_button:
    ok, message = rename_saved_results_snapshot(
        st.session_state.get("active_result_set_name", ""),
        st.session_state.get("rename_result_name", ""),
    )
    (st.success if ok else st.error)(message)

if delete_button:
    selected_name = st.session_state.get("active_result_set_name", "")
    saved_sets = copy.deepcopy(st.session_state.get("saved_result_sets", {}))
    if selected_name in saved_sets:
        del saved_sets[selected_name]
        st.session_state.saved_result_sets = saved_sets
        st.session_state.active_result_set_name = "Current Results"
        st.success(t(f"Deleted saved result: {selected_name}", f"已删除已保存结果：{selected_name}"))
        st.rerun()


st.title(t("Retirement Modelling Suite (Australia)", "退休建模工具（澳大利亚）"))
st.subheader(t("Superannuation • Tax • CGT • Retirement Cashflow Modelling", "养老金 • 税务 • 资本利得税 • 退休现金流建模"))
st.caption(
    t(
        "A professional financial modelling tool for analysing retirement outcomes, superannuation strategies, and tax impacts under Australian rules.",
        "一个用于分析澳大利亚退休结果、养老金策略与税务影响的专业金融建模工具。"
    )
)
st.warning(
    t(
        "This tool is for modelling and educational purposes only. It does not constitute personal financial advice. Results are based on assumptions and may not reflect actual outcomes.",
        "本工具仅用于建模和学习展示，不构成个人财务建议。结果基于假设，未必反映实际结果。"
    )
)
with st.container(border=True):
    st.markdown(
        t(
            """### Overview
Use this tool to model retirement sustainability, super accumulation to pension transitions, Transfer Balance Cap constraints, and tax impacts under multiple scenarios.""",
            """### 概览
本工具可用于建模退休可持续性、养老金从积累阶段转入退休金阶段、Transfer Balance Cap 限制，以及不同情景下的税务影响。"""
        )
    )


# ============================================================
# SECTION: ASSUMPTION SETTINGS PANEL
# ============================================================

if st.button(
    t("Reset Preset Assumptions to Default", "将预设假设重置为默认值"),
    key="reset_preset_table_button_panel",
):
    st.session_state.preset_table_df = get_default_preset_table_df()
    st.rerun()

runtime_presets = preset_table_to_dict(st.session_state.preset_table_df)

with st.container(border=True):
    top_left, top_right = st.columns([2, 1])

    with top_left:
        st.subheader(t("Assumption Settings Panel", "假设设置面板"))
        st.caption(t("Manage reusable preset assumptions here. The modelling engine stores these as decimals such as 0.03 = 3.0%.", "在此管理可重复使用的预设假设。建模引擎使用小数表示，例如 0.03 = 3.0%。"))

    with top_right:
        st.metric(t("Active Preset", "当前预设"), preset_choice)
        st.caption(t("Preset selection stays in the sidebar for quick scenario control.", "预设选择保留在侧边栏，便于快速切换情景。"))

    preset_table_df = st.data_editor(
        st.session_state.preset_table_df,
        key="preset_table_editor_panel",
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={
            "preset": st.column_config.TextColumn("Preset", disabled=True),
            "super_income_return_mean": st.column_config.NumberColumn("Super Income Mean", format="%.3f"),
            "super_income_return_std": st.column_config.NumberColumn("Super Income Std", format="%.3f"),
            "super_capital_return_mean": st.column_config.NumberColumn("Super Capital Mean", format="%.3f"),
            "super_capital_return_std": st.column_config.NumberColumn("Super Capital Std", format="%.3f"),
            "non_super_income_return_mean": st.column_config.NumberColumn("Non-Super Income Mean", format="%.3f"),
            "non_super_income_return_std": st.column_config.NumberColumn("Non-Super Income Std", format="%.3f"),
            "non_super_capital_return_mean": st.column_config.NumberColumn("Non-Super Capital Mean", format="%.3f"),
            "non_super_capital_return_std": st.column_config.NumberColumn("Non-Super Capital Std", format="%.3f"),
            "inflation_rate": st.column_config.NumberColumn("Inflation", format="%.3f"),
        },
    )

    st.session_state.preset_table_df = ensure_valid_preset_table_df(preset_table_df)
    runtime_presets = preset_table_to_dict(st.session_state.preset_table_df)


# ============================================================
# SECTION: MAIN INPUT NAVIGATION
# ============================================================

section_keys = [
    "report",
    "projection",
    "person1",
    "household",
    "contributions",
    "returns",
    "simulation",
]
if not is_one_person_mode:
    section_keys.insert(3, "person2")

section_labels = {
    "report": t("Report", "报告"),
    "projection": t("Projection", "预测设置"),
    "person1": t("Person 1", "人物 1"),
    "person2": t("Person 2", "人物 2"),
    "household": t("Household", "家庭"),
    "contributions": t("Contributions", "缴款设置"),
    "returns": t("Returns", "回报假设"),
    "simulation": t("Simulation", "模拟设置"),
}

legacy_section_map = {
    "Report": "report",
    "Projection": "projection",
    "Person 1": "person1",
    "Person 2": "person2",
    "Household": "household",
    "Contributions": "contributions",
    "Returns": "returns",
    "Simulation": "simulation",
    "报告": "report",
    "预测设置": "projection",
    "人物 1": "person1",
    "人物 2": "person2",
    "家庭": "household",
    "缴款设置": "contributions",
    "回报假设": "returns",
    "模拟设置": "simulation",
}

current_section_key = st.session_state.get("active_input_section", "projection")
current_section_key = legacy_section_map.get(current_section_key, current_section_key)

if current_section_key not in section_keys:
    current_section_key = "projection"
if is_one_person_mode and current_section_key == "person2":
    current_section_key = "household"

selected_section_label = st.segmented_control(
    t("Input Section", "输入区"),
    options=[section_labels[k] for k in section_keys],
    selection_mode="single",
    default=section_labels[current_section_key],
)

active_input_section = next(
    k for k in section_keys if section_labels[k] == selected_section_label
)

st.session_state.active_input_section = active_input_section

if active_input_section == "report":
    st.subheader(t("Report", "报告"))
    report_title = st.text_input(
        t("Title (Optional)", "标题（可选）"),
        value=st.session_state.report_title,
        key="report_title_input",
        help=t("Used in exports and saved result documentation.", "用于导出文件和已保存结果说明。"),
    )

    st.subheader(t("Names", "姓名"))
    name_col1, name_col2 = st.columns(2)
    with name_col1:
        person1_name = st.text_input(
            t("Person 1 Name (Optional)", "人物 1 姓名（可选）"),
            value=st.session_state.person1_name,
            key="person1_name_input",
            help=t("Optional display name used in charts and tables.", "用于图表和表格中的可选显示名称。"),
        )
    with name_col2:
        person2_name = st.text_input(
            t("Person 2 Name (Optional)", "人物 2 姓名（可选）"),
            value=st.session_state.person2_name,
            key="person2_name_input",
            disabled=is_one_person_mode,
            help=t("Optional display name used in charts and tables.", "用于图表和表格中的可选显示名称。"),
        )
        if is_one_person_mode:
            person2_name = ""

elif active_input_section == "projection":
    st.subheader(t("Projection Timing", "预测时间设置"))
    col1, col2, col3 = st.columns(3)
    with col1:
        start_financial_year = st.number_input(
            t("Start Financial Year", "起始财政年度"),
            value=int(st.session_state.start_financial_year),
            step=1,
            min_value=2000,
            help=t("Financial year end used as year 1 of the projection, e.g. 2027 means 2026/27.", "预测起始财政年度的终点年，例如 2027 表示 2026/27 财年。"),
        )
    with col2:
        projection_years = st.number_input(
            t("Projection Years", "预测年数"),
            value=int(st.session_state.projection_years),
            step=1,
            min_value=1,
            help=t("How many financial years to project forward.", "向前预测多少个财政年度。"),
        )
    with col3:
        retirement_spending_trigger = st.selectbox(
            t("Retirement Spending Trigger", "退休支出触发条件"),
            options=[
                t("Both Retired", "双方都退休"),
                t("Either Retired", "任一方退休"),
            ],
            index=0 if st.session_state.retirement_spending_trigger in ["Both Retired", "双方都退休"] else 1,
        )

        if retirement_spending_trigger == t("Both Retired", "双方都退休"):
            retirement_spending_trigger = "Both Retired"
        else:
            retirement_spending_trigger = "Either Retired"

elif active_input_section == "person1":
    st.subheader(t("Person 1", "人物 1"))
    p1a, p1b, p1c = st.columns(3)
    with p1a:
        person1_current_age = st.number_input(
            t("Person 1 Current Age", "人物 1 当前年龄"),
            value=int(st.session_state.person1_current_age),
            step=1,
            help=t("Current age at the start of the projection.", "预测开始时的当前年龄。"),
        )
        person1_accum_super_balance = currency_text_input(
            t("Person 1 Accumulation Super Balance", "人物 1 累积型养老金余额"),
            st.session_state.person1_accum_super_balance,
            "person1_accum_super_balance_input",
            help_text=t("Opening accumulation super balance.", "期初 accumulation super 余额。"),
        )
        person1_pension_super_balance = currency_text_input(
            t("Person 1 Pension Super Balance", "人物 1 养老金阶段余额"),
            st.session_state.person1_pension_super_balance,
            "person1_pension_super_balance_input",
            help_text=t("Opening pension super balance.", "期初 pension super 余额。"),
        )
    with p1b:
        person1_retirement_age = st.number_input(
            t("Person 1 Retirement Age", "人物 1 退休年龄"),
            value=int(st.session_state.person1_retirement_age),
            step=1,
        )
        person1_accum_super_cost_base = currency_text_input(
            t("Person 1 Accumulation Super Cost Base", "人物 1 累积型养老金成本基础"),
            st.session_state.person1_accum_super_cost_base,
            "person1_accum_super_cost_base_input",
            help_text=t("Cost base used for super withdrawal CGT approximation in accumulation phase.", "用于 accumulation 阶段提取 CGT 近似计算的成本基础。"),
        )
        person1_pension_super_cost_base = currency_text_input(
            t("Person 1 Pension Super Cost Base", "人物 1 养老金阶段成本基础"),
            st.session_state.person1_pension_super_cost_base,
            "person1_pension_super_cost_base_input",
            help_text=t("Cost base carried inside the pension pool for internal tracking.", "用于 pension 池内部追踪的成本基础。"),
        )
    with p1c:
        person1_pension_start_age = st.number_input(
            t("Person 1 Pension Start Age", "人物 1 养老金开始年龄"),
            value=int(st.session_state.person1_pension_start_age),
            step=1,
        )
        person1_transfer_balance_cap = currency_text_input(
            t("Person 1 Transfer Balance Cap", "人物 1 转移余额上限"),
            st.session_state.person1_transfer_balance_cap,
            "person1_transfer_balance_cap_input",
            help_text=t("Transfer Balance Cap used when moving accumulation super to pension.", "accumulation 转 pension 时使用的 Transfer Balance Cap。"),
        )
        person1_annual_income = currency_text_input(
            t("Person 1 Annual Income", "人物 1 年收入"),
            st.session_state.person1_annual_income,
            "person1_annual_income_input",
            help_text=t("Gross annual employment income while still working.", "仍在工作时的税前年收入。"),
        )

elif active_input_section == "person2":
    st.subheader(t("Person 2", "人物 2"))
    p2a, p2b, p2c = st.columns(3)
    with p2a:
        person2_current_age = st.number_input(
            t("Person 2 Current Age", "人物 2 当前年龄"),
            value=int(st.session_state.person2_current_age),
            step=1,
            help=t("Current age at the start of the projection.", "预测开始时的当前年龄。"),
        )
        person2_accum_super_balance = currency_text_input(
            t("Person 2 Accumulation Super Balance", "人物 2 累积型养老金余额"),
            st.session_state.person2_accum_super_balance,
            "person2_accum_super_balance_input",
            help_text=t("Opening accumulation super balance.", "期初 accumulation super 余额。"),
        )
        person2_pension_super_balance = currency_text_input(
            t("Person 2 Pension Super Balance", "人物 2 养老金阶段余额"),
            st.session_state.person2_pension_super_balance,
            "person2_pension_super_balance_input",
            help_text=t("Opening pension super balance.", "期初 pension super 余额。"),
        )
    with p2b:
        person2_retirement_age = st.number_input(
            t("Person 2 Retirement Age", "人物 2 退休年龄"),
            value=int(st.session_state.person2_retirement_age),
            step=1,
            help=t("Employment income stops once current age reaches retirement age.", "达到退休年龄后，employment income 停止。"),
        )
        person2_accum_super_cost_base = currency_text_input(
            t("Person 2 Accumulation Super Cost Base", "人物 2 累积型养老金成本基础"),
            st.session_state.person2_accum_super_cost_base,
            "person2_accum_super_cost_base_input",
            help_text=t("Cost base used for super withdrawal CGT approximation in accumulation phase.", "用于 accumulation 阶段提取 CGT 近似计算的成本基础。"),
        )
        person2_pension_super_cost_base = currency_text_input(
            t("Person 2 Pension Super Cost Base", "人物 2 养老金阶段成本基础"),
            st.session_state.person2_pension_super_cost_base,
            "person2_pension_super_cost_base_input",
            help_text=t("Cost base carried inside the pension pool for internal tracking.", "用于 pension 池内部追踪的成本基础。"),
        )
    with p2c:
        person2_pension_start_age = st.number_input(
            t("Person 2 Pension Start Age", "人物 2 养老金开始年龄"),
            value=int(st.session_state.person2_pension_start_age),
            step=1,
            help=t("Age when accumulation super can start transferring into pension phase in the model.", "模型中 accumulation super 开始转入 pension 的年龄。"),
        )
        person2_transfer_balance_cap = currency_text_input(
            t("Person 2 Transfer Balance Cap", "人物 2 转移余额上限"),
            st.session_state.person2_transfer_balance_cap,
            "person2_transfer_balance_cap_input",
            help_text=t("Transfer Balance Cap used when moving accumulation super to pension.", "accumulation 转 pension 时使用的 Transfer Balance Cap。"),
        )
        person2_annual_income = currency_text_input(
            t("Person 2 Annual Income", "人物 2 年收入"),
            st.session_state.person2_annual_income,
            "person2_annual_income_input",
            help_text=t("Gross annual employment income while still working.", "仍在工作时的税前年收入。"),
        )

elif active_input_section == "household":
    st.subheader(t("Household", "家庭"))
    if is_one_person_mode:
        st.info(
            t(
                "One Person mode removes Person 2 from the model, but the household spending fields below stay exactly as entered. Reduce them manually if you want a true one-person budget.",
                "单人模式会把 Person 2 从模型中移除，但下面的家庭支出栏位不会自动变化。如果你希望按单人预算建模，请手动调低这些数值。",
            )
        )
    hh1, hh2 = st.columns(2)
    with hh1:
        non_super_balance = currency_text_input(
            t("Non-Super Balance", "非养老金资产余额"),
            st.session_state.non_super_balance,
            "non_super_balance_input",
            help_text=t("Opening non-super investment pool market value.", "期初非养老金投资池市值。"),
        )
        annual_living_expenses = currency_text_input(
            t("Annual Living Expenses", "年度生活支出"),
            st.session_state.annual_living_expenses,
            "annual_living_expenses_input",
            help_text=t("Current annual household spending before retirement trigger applies.", "退休支出触发前的当前年度家庭支出。"),
        )
        cgt_discount_rate = percentage_text_input(
            t("CGT Discount Rate", "资本利得税折扣率"),
            float(st.session_state.cgt_discount_rate),
            "cgt_discount_rate_input",
            decimals=1,
            help_text=t("Discount applied to non-super realised capital gains under the average-cost model.", "在 average-cost 模型下适用于非养老金已实现资本利得的折扣率。"),
        )
    with hh2:
        non_super_cost_base = currency_text_input(
            t("Non-Super Cost Base", "非养老金资产成本基础"),
            st.session_state.non_super_cost_base,
            "non_super_cost_base_input",
            help_text=t("Cost base of the non-super investment pool. It must not exceed market value.", "非养老金投资池的成本基础，不能高于当前市值。"),
        )
        retirement_spending = currency_text_input(
            t("Retirement Spending", "退休后支出"),
            st.session_state.retirement_spending,
            "retirement_spending_input",
            help_text=t("Target household spending after the retirement trigger is reached. This amount is internally indexed by inflation before activation.", "达到退休触发条件后的目标家庭支出。该数值在生效前也会按 inflation 内部递增。"),
        )
        if is_one_person_mode:
            st.text_input(
                t("Person 1 Ownership %", "人物 1 持有比例 %"),
                value="100.0%",
                disabled=True,
                key="non_super_ownership_person1_display",
                help=t("Single-person mode fixes non-super ownership to 100% Person 1.", "单人模式下非养老金持有比例固定为 Person 1 的 100%。"),
            )
            non_super_ownership_person1 = 100.0
        else:
            non_super_ownership_person1 = percentage_text_input(
                t("Person 1 Ownership %", "人物 1 持有比例 %"),
                st.session_state.non_super_ownership_person1_pct / 100.0,
                "non_super_ownership_person1_input",
                decimals=1,
                help_text=t("Share of non-super taxable income and tax allocated to Person 1.", "分配给 Person 1 的非养老金应税收入与税负比例。"),
            ) * 100.0

elif active_input_section == "contributions":
    st.subheader(t("Contribution Schedule", "缴款计划"))
    contribution_person_options = ["Person 1"] if is_one_person_mode else ["Person 1", "Person 2"]

    contribution_source_df = st.session_state.contribution_events_df.copy()
    if is_one_person_mode and not contribution_source_df.empty:
        contribution_source_df = contribution_source_df[contribution_source_df["person"] != "Person 2"].reset_index(drop=True)
        st.caption(t("Single-person mode automatically ignores any existing Person 2 contribution rows.", "单人模式会自动忽略现有的 Person 2 缴款行。"))

    contribution_events_df = st.data_editor(
        contribution_source_df,
        key="contribution_events_editor",
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "financial_year": st.column_config.NumberColumn(
                t("Financial Year", "财政年度"),
                min_value=2000,
                step=1,
            ),
            "person": st.column_config.SelectboxColumn(
                t("Person", "人物"),
                options=contribution_person_options,
            ),
            "contribution_type": st.column_config.SelectboxColumn(
                t("Contribution Type", "缴款类型"),
                options=["personal_deductible", "non_concessional"],
            ),
            "amount": st.column_config.NumberColumn(
                t("Amount", "金额"),
                min_value=0.0,
                step=1000.0,
                format="$%.0f",
            ),
        },
    )

elif active_input_section == "returns":
    if scenario_mode == "Single Scenario" and preset_choice == "Custom":
        st.subheader(t("Return Assumptions", "回报假设"))
        r1, r2, r3 = st.columns(3)
        with r1:
            super_income_return_mean = percentage_text_input(
                t("Super Income Return Mean", "养老金收益型回报均值"),
                st.session_state.super_income_return_mean,
                "super_income_return_mean_input",
                decimals=1,
                help_text=t("Expected annual income-style return on super assets.", "养老金资产的年度收益型回报假设。"),
            )
            super_capital_return_mean = percentage_text_input(
                t("Super Capital Return Mean", "养老金资本增值回报均值"),
                st.session_state.super_capital_return_mean,
                "super_capital_return_mean_input",
                decimals=1,
                help_text=t("Expected annual capital growth on super assets.", "养老金资产的年度资本增值回报假设。"),
            )
            inflation_rate = percentage_text_input(
                t("Inflation Rate", "通胀率"),
                st.session_state.inflation_rate,
                "inflation_rate_input",
                decimals=1,
                help_text=t("Inflation used to index salary and spending assumptions.", "用于收入与支出递增的 inflation 假设。"),
            )
        with r2:
            super_income_return_std = percentage_text_input(
                t("Super Income Return Std", "养老金收益型回报波动"),
                st.session_state.super_income_return_std,
                "super_income_return_std_input",
                decimals=1,
            )
            super_capital_return_std = percentage_text_input(
                t("Super Capital Return Std", "养老金资本增值回报波动"),
                st.session_state.super_capital_return_std,
                "super_capital_return_std_input",
                decimals=1,
            )
        with r3:
            non_super_income_return_mean = percentage_text_input(
                t("Non-Super Income Return Mean", "非养老金收益型回报均值"),
                st.session_state.non_super_income_return_mean,
                "non_super_income_return_mean_input",
                decimals=1,
            )
            non_super_capital_return_mean = percentage_text_input(
                t("Non-Super Capital Return Mean", "非养老金资本增值回报均值"),
                st.session_state.non_super_capital_return_mean,
                "non_super_capital_return_mean_input",
                decimals=1,
            )
            non_super_income_return_std = percentage_text_input(
                t("Non-Super Income Return Std", "非养老金收益型回报波动"),
                st.session_state.non_super_income_return_std,
                "non_super_income_return_std_input",
                decimals=1,
            )
            non_super_capital_return_std = percentage_text_input(
                t("Non-Super Capital Return Std", "非养老金资本增值回报波动"),
                st.session_state.non_super_capital_return_std,
                "non_super_capital_return_std_input",
                decimals=1,
            )
    else:
        selected_preset = preset_choice if preset_choice in runtime_presets else "Base Case"
        selected_values = runtime_presets[selected_preset]

        st.subheader(t("Return Assumptions", "回报假设"))
        st.info(
            t(
                f"Using values from Assumption Settings Panel: {selected_preset}",
                f"使用假设设置面板中的参数：{selected_preset}",
            )
        )

        display_df = pd.DataFrame(
            {
                t("Assumption", "假设"): [
                    t("Super Income Return Mean", "养老金收益型回报均值"),
                    t("Super Income Return Std", "养老金收益型回报波动"),
                    t("Super Capital Return Mean", "养老金资本增值回报均值"),
                    t("Super Capital Return Std", "养老金资本增值回报波动"),
                    t("Non-Super Income Return Mean", "非养老金收益型回报均值"),
                    t("Non-Super Income Return Std", "非养老金收益型回报波动"),
                    t("Non-Super Capital Return Mean", "非养老金资本增值回报均值"),
                    t("Non-Super Capital Return Std", "非养老金资本增值回报波动"),
                    t("Inflation Rate", "通胀率"),
                ],
                t("Value", "数值"): [
                    selected_values["super_income_return_mean"] * 100.0,
                    selected_values["super_income_return_std"] * 100.0,
                    selected_values["super_capital_return_mean"] * 100.0,
                    selected_values["super_capital_return_std"] * 100.0,
                    selected_values["non_super_income_return_mean"] * 100.0,
                    selected_values["non_super_income_return_std"] * 100.0,
                    selected_values["non_super_capital_return_mean"] * 100.0,
                    selected_values["non_super_capital_return_std"] * 100.0,
                    selected_values["inflation_rate"] * 100.0,
                ],
            }
        )
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                t("Assumption", "假设"): st.column_config.TextColumn(t("Assumption", "假设")),
                t("Value", "数值"): st.column_config.NumberColumn(t("Value", "数值"), format="%.1f%%"),
            },
        )

        super_income_return_mean = float(selected_values["super_income_return_mean"])
        super_income_return_std = float(selected_values["super_income_return_std"])
        super_capital_return_mean = float(selected_values["super_capital_return_mean"])
        super_capital_return_std = float(selected_values["super_capital_return_std"])
        non_super_income_return_mean = float(selected_values["non_super_income_return_mean"])
        non_super_income_return_std = float(selected_values["non_super_income_return_std"])
        non_super_capital_return_mean = float(selected_values["non_super_capital_return_mean"])
        non_super_capital_return_std = float(selected_values["non_super_capital_return_std"])
        inflation_rate = float(selected_values["inflation_rate"])

elif active_input_section == "simulation":
    st.subheader(t("Simulation", "模拟设置"))
    sim1, sim2 = st.columns(2)
    with sim1:
        number_of_simulations = st.number_input(
            t("Number of Simulations", "模拟次数"),
            value=int(st.session_state.number_of_simulations),
            step=1000,
        )
    with sim2:
        random_seed = st.number_input(
            t("Random Seed", "随机种子"),
            value=int(st.session_state.random_seed),
            step=1,
        )
        

# ============================================================
# SECTION: SESSION UPDATE
# ============================================================

st.session_state.start_financial_year = int(start_financial_year)
st.session_state.projection_years = int(projection_years)
st.session_state.retirement_spending_trigger = retirement_spending_trigger
st.session_state.household_mode = household_mode
st.session_state.value_mode = value_mode
st.session_state.report_title = report_title
st.session_state.person1_name = person1_name
st.session_state.person2_name = person2_name
st.session_state.person1_current_age = int(person1_current_age)
st.session_state.person2_current_age = int(person2_current_age)
st.session_state.person1_retirement_age = int(person1_retirement_age)
st.session_state.person2_retirement_age = int(person2_retirement_age)
st.session_state.person1_pension_start_age = int(person1_pension_start_age)
st.session_state.person2_pension_start_age = int(person2_pension_start_age)
st.session_state.person1_accum_super_balance = person1_accum_super_balance
st.session_state.person1_pension_super_balance = person1_pension_super_balance
st.session_state.person2_accum_super_balance = person2_accum_super_balance
st.session_state.person2_pension_super_balance = person2_pension_super_balance
st.session_state.person1_accum_super_cost_base = person1_accum_super_cost_base
st.session_state.person1_pension_super_cost_base = person1_pension_super_cost_base
st.session_state.person2_accum_super_cost_base = person2_accum_super_cost_base
st.session_state.person2_pension_super_cost_base = person2_pension_super_cost_base
st.session_state.person1_transfer_balance_cap = person1_transfer_balance_cap
st.session_state.person2_transfer_balance_cap = person2_transfer_balance_cap
st.session_state.person1_annual_income = person1_annual_income
st.session_state.person2_annual_income = person2_annual_income
st.session_state.non_super_balance = non_super_balance
st.session_state.non_super_cost_base = non_super_cost_base
st.session_state.annual_living_expenses = annual_living_expenses
st.session_state.retirement_spending = retirement_spending
st.session_state.non_super_ownership_person1_pct = non_super_ownership_person1
st.session_state.cgt_discount_rate = cgt_discount_rate
st.session_state.preset_table_df = preset_table_df.copy()
st.session_state.contribution_events_df = contribution_events_df.copy()
st.session_state.super_income_return_mean = super_income_return_mean
st.session_state.super_income_return_std = super_income_return_std
st.session_state.super_capital_return_mean = super_capital_return_mean
st.session_state.super_capital_return_std = super_capital_return_std
st.session_state.non_super_income_return_mean = non_super_income_return_mean
st.session_state.non_super_income_return_std = non_super_income_return_std
st.session_state.non_super_capital_return_mean = non_super_capital_return_mean
st.session_state.non_super_capital_return_std = non_super_capital_return_std
st.session_state.inflation_rate = inflation_rate
st.session_state.number_of_simulations = int(number_of_simulations)
st.session_state.random_seed = int(random_seed)

if is_one_person_mode:
    person2_name = ""
    person2_current_age = 0
    person2_retirement_age = 0
    person2_pension_start_age = 0
    person2_accum_super_balance = 0.0
    person2_pension_super_balance = 0.0
    person2_accum_super_cost_base = 0.0
    person2_pension_super_cost_base = 0.0
    person2_transfer_balance_cap = 0.0
    person2_annual_income = 0.0
    non_super_ownership_person1 = 100.0
    if not contribution_events_df.empty:
        contribution_events_df = contribution_events_df[contribution_events_df["person"] != "Person 2"].reset_index(drop=True)

# ============================================================
# SECTION: BASE INPUT MAP
# ============================================================

base_inputs = {
    "report_title": report_title,
    "person1_name": person1_name,
    "person2_name": person2_name,
    "start_financial_year": int(start_financial_year),
    "projection_years": int(projection_years),
    "retirement_spending_trigger": retirement_spending_trigger,
    "household_mode": household_mode,
    "person1_current_age": int(person1_current_age),
    "person2_current_age": int(person2_current_age),
    "person1_retirement_age": int(person1_retirement_age),
    "person2_retirement_age": int(person2_retirement_age),
    "person1_pension_start_age": int(person1_pension_start_age),
    "person2_pension_start_age": int(person2_pension_start_age),
    "person1_accum_super_balance": person1_accum_super_balance,
    "person1_pension_super_balance": person1_pension_super_balance,
    "person2_accum_super_balance": person2_accum_super_balance,
    "person2_pension_super_balance": person2_pension_super_balance,
    "person1_transfer_balance_cap": person1_transfer_balance_cap,
    "person2_transfer_balance_cap": person2_transfer_balance_cap,
    "non_super_balance": non_super_balance,
    "non_super_cost_base": non_super_cost_base,
    "person1_annual_income": person1_annual_income,
    "person2_annual_income": person2_annual_income,
    "annual_living_expenses": annual_living_expenses,
    "retirement_spending": retirement_spending,
    "non_super_ownership_person1": non_super_ownership_person1 / 100.0,
    "cgt_discount_rate": cgt_discount_rate,
    "inflation_rate": inflation_rate,
    "super_income_return_mean": super_income_return_mean,
    "super_income_return_std": super_income_return_std,
    "super_capital_return_mean": super_capital_return_mean,
    "super_capital_return_std": super_capital_return_std,
    "non_super_income_return_mean": non_super_income_return_mean,
    "non_super_income_return_std": non_super_income_return_std,
    "non_super_capital_return_mean": non_super_capital_return_mean,
    "non_super_capital_return_std": non_super_capital_return_std,
    "number_of_simulations": int(number_of_simulations),
    "assumption_preset": preset_choice,
    "contribution_events": contribution_events_to_records(contribution_events_df, household_mode=household_mode),
    "person1_accum_super_cost_base": person1_accum_super_cost_base,
    "person1_pension_super_cost_base": person1_pension_super_cost_base,
    "person2_accum_super_cost_base": person2_accum_super_cost_base,
    "person2_pension_super_cost_base": person2_pension_super_cost_base,
}

render_live_input_feedback(base_inputs)

# ============================================================
# SECTION: RUN LOGIC
# ============================================================

if run_button:
    if scenario_mode == t("Single Scenario", "单一情景"):
        if preset_choice == "Custom":
            scenario_inputs_map = {"Custom": base_inputs.copy()}
        else:
            scenario_inputs_map = {preset_choice: apply_preset_to_inputs(base_inputs, preset_choice, runtime_presets)}
    else:
        scenario_inputs_map = {
            "Conservative": apply_preset_to_inputs(base_inputs, "Conservative", runtime_presets),
            "Base Case": apply_preset_to_inputs(base_inputs, "Base Case", runtime_presets),
            "Optimistic": apply_preset_to_inputs(base_inputs, "Optimistic", runtime_presets),
        }

    all_validation_errors = []
    for scenario_name, scenario_inputs in scenario_inputs_map.items():
        scenario_errors = validate_inputs(scenario_inputs)
        for error in scenario_errors:
            all_validation_errors.append(f"[{scenario_name}] {error}")

    if all_validation_errors:
        st.session_state.comparison_results = None
        st.session_state.assumption_details_df = None
        st.session_state.input_summary_df = None
        st.session_state.contribution_schedule_export_df = None
        st.session_state.input_warnings_by_scenario = None
        st.session_state.output_warnings_by_scenario = None
        st.session_state.last_run_inputs_by_scenario = None

        for error in all_validation_errors:
            st.error(error)
    else:
        comparison_results = {}
        input_warnings_by_scenario = {}
        output_warnings_by_scenario = {}

        for scenario_name, scenario_inputs in scenario_inputs_map.items():
            det_df = run_deterministic_projection(scenario_inputs)
            summary_df, all_paths_df = run_monte_carlo(
                scenario_inputs,
                random_seed=int(random_seed),
            )
            percentile_df = build_percentile_table(all_paths_df)
            failure_prob_df = build_failure_probability_by_age(all_paths_df)

            success_rate = summary_df["success"].mean()
            median_final_wealth = summary_df["final_wealth"].median()
            p10_final_wealth = summary_df["final_wealth"].quantile(0.10)
            p90_final_wealth = summary_df["final_wealth"].quantile(0.90)

            input_warnings_by_scenario[scenario_name] = generate_input_warnings(scenario_inputs)
            output_warnings_by_scenario[scenario_name] = generate_output_warnings(
                summary_df,
                failure_prob_df,
                det_df,
            )

            comparison_results[scenario_name] = {
                "inputs": scenario_inputs,
                "det_df": det_df,
                "summary_df": summary_df,
                "all_paths_df": all_paths_df,
                "percentile_df": percentile_df,
                "failure_prob_df": failure_prob_df,
                "success_rate": success_rate,
                "median_final_wealth": median_final_wealth,
                "p10_final_wealth": p10_final_wealth,
                "p90_final_wealth": p90_final_wealth,
            }

        st.session_state.comparison_results = comparison_results
        st.session_state.assumption_details_df = build_assumption_details_df(scenario_inputs_map)
        st.session_state.input_summary_df = build_input_summary_df(scenario_inputs_map)
        st.session_state.contribution_schedule_export_df = build_contribution_schedule_export_df(scenario_inputs_map)
        st.session_state.input_warnings_by_scenario = input_warnings_by_scenario
        st.session_state.output_warnings_by_scenario = output_warnings_by_scenario
        st.session_state.last_run_inputs_by_scenario = scenario_inputs_map
        st.session_state.active_result_set_name = "Current Results"


# ============================================================
# SECTION: RESULTS RENDERING
# ============================================================

active_result_bundle = get_active_result_bundle()

if active_result_bundle is not None:
    comparison_results = active_result_bundle["comparison_results"]
    assumption_details_df = active_result_bundle["assumption_details_df"]
    input_summary_df = active_result_bundle["input_summary_df"]
    contribution_schedule_export_df = active_result_bundle["contribution_schedule_export_df"]
    input_warnings_by_scenario = active_result_bundle["input_warnings_by_scenario"]
    output_warnings_by_scenario = active_result_bundle["output_warnings_by_scenario"]

    active_name_display = st.session_state.get("active_result_set_name", "Current Results")
    if active_name_display == "Current Results":
        st.caption(t("Showing: Current Results", "当前显示：最新结果"))
    else:
        st.caption(t(f"Showing saved snapshot: {active_name_display}", f"当前显示：已保存快照：{active_name_display}"))

    render_saved_result_comparison_section(st.session_state.get("saved_result_sets", {}), value_mode)

    render_assumption_details(assumption_details_df)
    render_warning_sections(input_warnings_by_scenario, output_warnings_by_scenario, view_mode)

    comparison_rows = []
    det_scenarios_df_list = []

    for scenario_name, result in comparison_results.items():
        comparison_rows.append(
            {
                "scenario": scenario_name,
                "success_rate": result["success_rate"],
                "median_final_wealth": result["median_final_wealth"],
                "p10_final_wealth": result["p10_final_wealth"],
                "p90_final_wealth": result["p90_final_wealth"],
            }
        )

        temp_det_df = result["det_df"].copy()
        temp_det_df["scenario"] = scenario_name
        det_scenarios_df_list.append(temp_det_df)

    comparison_df = pd.DataFrame(comparison_rows)
    common_inputs = next(iter(comparison_results.values()))["inputs"]
    comparison_df = format_comparison_df(comparison_df)
    comparison_df = convert_comparison_df_for_value_mode(comparison_df, common_inputs, value_mode)
    det_scenarios_df = pd.concat(det_scenarios_df_list, ignore_index=True)
    det_scenarios_df = convert_det_df_for_value_mode(det_scenarios_df, common_inputs, value_mode)

    st.subheader(t("Scenario Comparison Summary", "情景比较摘要"))
    st.dataframe(
        comparison_df[["scenario", "success_rate_label", "median_final_wealth_label", "p10_final_wealth_label", "p90_final_wealth_label"]],
        use_container_width=True,
    )

    success_fig = create_success_rate_comparison_chart(comparison_df)
    st.plotly_chart(success_fig, use_container_width=True, key="success_rate_comparison")

    median_fig = create_median_wealth_comparison_chart(comparison_df)
    st.plotly_chart(median_fig, use_container_width=True, key="median_wealth_comparison")

    selected_scenario = st.selectbox(
        t("Select Scenario", "选择情景"),
        options=list(comparison_results.keys()),
        key="selected_scenario_results",
    )
    selected_result = comparison_results[selected_scenario]
    display_det_df = convert_det_df_for_value_mode(selected_result["det_df"], selected_result["inputs"], value_mode)
    display_percentile_df = convert_percentile_df_for_value_mode(selected_result["percentile_df"], selected_result["inputs"], value_mode)
    display_summary_df = convert_summary_df_for_value_mode(selected_result["summary_df"], selected_result["inputs"], value_mode)

    selected_success_rate = selected_result["success_rate"]
    selected_median_final_wealth = display_summary_df["final_wealth"].median()
    selected_p10_final_wealth = display_summary_df["final_wealth"].quantile(0.10)
    selected_p90_final_wealth = display_summary_df["final_wealth"].quantile(0.90)

    if view_mode == t("Adviser View", "顾问视图"):
        st.subheader(t(f"Adviser Summary - {selected_scenario}", f"顾问摘要 - {selected_scenario}"))
        st.info(t("Adviser Note: Outputs are indicative only and should be reviewed in the context of client objectives, risk profile, and current legislation before forming advice.", "顾问提示：本输出仅供指示参考，在形成建议前应结合客户目标、风险承受能力及现行法规进行审阅。"))

        top_col1, top_col2 = st.columns(2)
        top_col1.metric(t("Success Rate", "成功率"), f"{selected_result['success_rate']:.1%}")
        top_col2.metric(t("Median Final Wealth", "最终财富中位数"), f"${selected_median_final_wealth:,.0f}")

        bottom_col1, bottom_col2, bottom_col3 = st.columns(3)
        bottom_col1.metric(t("P10 Final Wealth", "P10 最终财富"), f"${selected_p10_final_wealth:,.0f}")
        bottom_col2.metric(t("P90 Final Wealth", "P90 最终财富"), f"${selected_p90_final_wealth:,.0f}")
        bottom_col3.metric(t("Spread (P90 - P10)", "区间差值（P90 - P10）"), f"${selected_p90_final_wealth - selected_p10_final_wealth:,.0f}")

        missing_validation_cols = get_missing_validation_columns(display_det_df)
        if missing_validation_cols:
            st.warning(t("Validation table is using fallback zeros for missing columns.", "验证表对缺失栏位使用了回退零值。"))
            st.caption(", ".join(missing_validation_cols))

        adviser_cashflow_df = build_adviser_cashflow_df(display_det_df)
        st.subheader(t("Adviser Cashflow Summary", "顾问现金流摘要"))
        st.dataframe(
            adviser_cashflow_df,
            use_container_width=True,
            column_config={
                "Year": st.column_config.NumberColumn("Year", format="%d"),
                "Withdrawal": st.column_config.NumberColumn("Withdrawal", format="$%.0f"),
                "CGT": st.column_config.NumberColumn("CGT", format="$%.0f"),
                "Tax": st.column_config.NumberColumn("Tax", format="$%.0f"),
                "Net Cash": st.column_config.NumberColumn("Net Cash", format="$%.0f"),
            },
        )

        pension_tax_free_summary_df = build_pension_tax_free_summary_df(display_det_df)
        st.subheader(t("Pension Tax-Free Validation Summary", "退休金免税验证摘要"))
        st.dataframe(
            pension_tax_free_summary_df,
            use_container_width=True,
            column_config={
                "check": st.column_config.TextColumn("Check"),
                "value": st.column_config.NumberColumn("Value", format="$%.0f"),
            },
        )

        debug_df = build_adviser_debug_df(display_det_df)
        st.subheader(t("Adviser Debug Table", "顾问调试表"))
        st.dataframe(
            debug_df,
            use_container_width=True,
            column_config={
                "Year": st.column_config.NumberColumn("Year", format="%d"),
                "P1 Started Pension This Year": st.column_config.CheckboxColumn("P1 Started Pension This Year"),
                "P1 Has Started Pension": st.column_config.CheckboxColumn("P1 Has Started Pension"),
                "P2 Started Pension This Year": st.column_config.CheckboxColumn("P2 Started Pension This Year"),
                "P2 Has Started Pension": st.column_config.CheckboxColumn("P2 Has Started Pension"),
                "P1 Requested Transfer": st.column_config.NumberColumn("P1 Requested Transfer", format="$%.0f"),
                "P2 Requested Transfer": st.column_config.NumberColumn("P2 Requested Transfer", format="$%.0f"),
                "P1 Available Cap Space": st.column_config.NumberColumn("P1 Available Cap Space", format="$%.0f"),
                "P2 Available Cap Space": st.column_config.NumberColumn("P2 Available Cap Space", format="$%.0f"),
                "P1 Transfer to Pension": st.column_config.NumberColumn("P1 Transfer to Pension", format="$%.0f"),
                "P2 Transfer to Pension": st.column_config.NumberColumn("P2 Transfer to Pension", format="$%.0f"),
                "P1 Excess Retained in Accum": st.column_config.NumberColumn("P1 Excess Retained in Accum", format="$%.0f"),
                "P2 Excess Retained in Accum": st.column_config.NumberColumn("P2 Excess Retained in Accum", format="$%.0f"),
                "P1 Opening Accum": st.column_config.NumberColumn("P1 Opening Accum", format="$%.0f"),
                "P1 Opening Pension": st.column_config.NumberColumn("P1 Opening Pension", format="$%.0f"),
                "P2 Opening Accum": st.column_config.NumberColumn("P2 Opening Accum", format="$%.0f"),
                "P2 Opening Pension": st.column_config.NumberColumn("P2 Opening Pension", format="$%.0f"),
                "P1 Net Super Contribution": st.column_config.NumberColumn("P1 Net Super Contribution", format="$%.0f"),
                "P2 Net Super Contribution": st.column_config.NumberColumn("P2 Net Super Contribution", format="$%.0f"),
                "P1 Ending Accum": st.column_config.NumberColumn("P1 Ending Accum", format="$%.0f"),
                "P1 Ending Pension": st.column_config.NumberColumn("P1 Ending Pension", format="$%.0f"),
                "P2 Ending Accum": st.column_config.NumberColumn("P2 Ending Accum", format="$%.0f"),
                "P2 Ending Pension": st.column_config.NumberColumn("P2 Ending Pension", format="$%.0f"),
                "P1 Min Pension Drawdown": st.column_config.NumberColumn("P1 Min Pension Drawdown", format="$%.0f"),
                "P2 Min Pension Drawdown": st.column_config.NumberColumn("P2 Min Pension Drawdown", format="$%.0f"),
                "P1 Extra Pension Withdrawal": st.column_config.NumberColumn("P1 Extra Pension Withdrawal", format="$%.0f"),
                "P2 Extra Pension Withdrawal": st.column_config.NumberColumn("P2 Extra Pension Withdrawal", format="$%.0f"),
                "P1 Pension Earnings Tax": st.column_config.NumberColumn("P1 Pension Earnings Tax", format="$%.0f"),
                "P2 Pension Earnings Tax": st.column_config.NumberColumn("P2 Pension Earnings Tax", format="$%.0f"),
                "P1 Super Realised CGT": st.column_config.NumberColumn("P1 Super Realised CGT", format="$%.0f"),
                "P2 Super Realised CGT": st.column_config.NumberColumn("P2 Super Realised CGT", format="$%.0f"),
                "Super Withdrawal CGT Tax": st.column_config.NumberColumn("Super Withdrawal CGT Tax", format="$%.0f"),
                "Total Super Earnings Tax": st.column_config.NumberColumn("Total Super Earnings Tax", format="$%.0f"),
            },
        )

        cgt_validation_df = build_cgt_validation_df(display_det_df)
        with st.expander(t("Detailed CGT / Pension Validation Table", "详细 CGT / 退休金验证表"), expanded=False):
            st.dataframe(
                cgt_validation_df,
                use_container_width=True,
                column_config={
                    "Year": st.column_config.NumberColumn("Year", format="%d"),
                    "P1 Age": st.column_config.NumberColumn("P1 Age", format="%d"),
                    "P2 Age": st.column_config.NumberColumn("P2 Age", format="%d"),
                    "P1 Started Pension This Year": st.column_config.CheckboxColumn("P1 Started Pension This Year"),
                    "P2 Started Pension This Year": st.column_config.CheckboxColumn("P2 Started Pension This Year"),
                    "P1 Has Started Pension": st.column_config.CheckboxColumn("P1 Has Started Pension"),
                    "P2 Has Started Pension": st.column_config.CheckboxColumn("P2 Has Started Pension"),
                    "P1 Opening Accum Balance": st.column_config.NumberColumn("P1 Opening Accum Balance", format="$%.0f"),
                    "P1 Opening Pension Balance": st.column_config.NumberColumn("P1 Opening Pension Balance", format="$%.0f"),
                    "P2 Opening Accum Balance": st.column_config.NumberColumn("P2 Opening Accum Balance", format="$%.0f"),
                    "P2 Opening Pension Balance": st.column_config.NumberColumn("P2 Opening Pension Balance", format="$%.0f"),
                    "P1 Ending Accum Balance": st.column_config.NumberColumn("P1 Ending Accum Balance", format="$%.0f"),
                    "P1 Ending Pension Balance": st.column_config.NumberColumn("P1 Ending Pension Balance", format="$%.0f"),
                    "P2 Ending Accum Balance": st.column_config.NumberColumn("P2 Ending Accum Balance", format="$%.0f"),
                    "P2 Ending Pension Balance": st.column_config.NumberColumn("P2 Ending Pension Balance", format="$%.0f"),
                    "P1 Transfer to Pension": st.column_config.NumberColumn("P1 Transfer to Pension", format="$%.0f"),
                    "P2 Transfer to Pension": st.column_config.NumberColumn("P2 Transfer to Pension", format="$%.0f"),
                    "P1 Total Net Super Contribution": st.column_config.NumberColumn("P1 Total Net Super Contribution", format="$%.0f"),
                    "P2 Total Net Super Contribution": st.column_config.NumberColumn("P2 Total Net Super Contribution", format="$%.0f"),
                    "P1 Super Realised CGT": st.column_config.NumberColumn("P1 Super Realised CGT", format="$%.0f"),
                    "P2 Super Realised CGT": st.column_config.NumberColumn("P2 Super Realised CGT", format="$%.0f"),
                    "Super Withdrawal CGT Tax": st.column_config.NumberColumn("Super Withdrawal CGT Tax", format="$%.0f"),
                    "P1 Pension Earnings Tax": st.column_config.NumberColumn("P1 Pension Earnings Tax", format="$%.0f"),
                    "P2 Pension Earnings Tax": st.column_config.NumberColumn("P2 Pension Earnings Tax", format="$%.0f"),
                    "Super Earnings Tax": st.column_config.NumberColumn("Super Earnings Tax", format="$%.0f"),
                },
            )

        tax_breakdown_fig = create_tax_breakdown_chart(display_det_df, selected_result["inputs"], f"Tax Breakdown - {selected_scenario}")
        st.plotly_chart(tax_breakdown_fig, use_container_width=True, key=chart_key("tax_breakdown", selected_scenario, view_mode, "adviser"))

        total_tax_fig = create_total_tax_paid_chart(display_det_df, selected_result["inputs"], f"Total Tax Paid - {selected_scenario}")
        st.plotly_chart(total_tax_fig, use_container_width=True, key=chart_key("total_tax", selected_scenario, view_mode, "adviser"))

        excel_file = dataframe_to_excel_bytes(
            {
                "input_summary": input_summary_df,
                "assumption_details": assumption_details_df,
                "contribution_schedule": contribution_schedule_export_df,
                "deterministic_projection": display_det_df,
                "simulation_summary": display_summary_df,
                "percentile_table": display_percentile_df,
                "failure_probability": selected_result["failure_prob_df"],
                "adviser_cashflow_summary": adviser_cashflow_df,
                "adviser_debug_table": debug_df,
                "pension_tax_free_summary": pension_tax_free_summary_df,
                "cgt_validation_detail": cgt_validation_df,
            }
        )

        st.download_button(
            label=t("Download Excel", "下载 Excel"),
            data=excel_file,
            file_name=build_export_filename(selected_result["inputs"].get("report_title", ""), "financial_projection", selected_scenario, "xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    else:
        st.subheader(t(f"Client Summary - {selected_scenario}", f"客户摘要 - {selected_scenario}"))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t("Success Rate", "成功率"), f"{selected_result['success_rate']:.1%}")
        col2.metric(t("Median Final Wealth", "最终财富中位数"), f"${selected_median_final_wealth:,.0f}")
        col3.metric(t("P10 Final Wealth", "P10 最终财富"), f"${selected_p10_final_wealth:,.0f}")
        col4.metric(t("P90 Final Wealth", "P90 最终财富"), f"${selected_p90_final_wealth:,.0f}")

        det_single_df = display_det_df
        det_single_compare_df = det_single_df.copy()
        det_single_compare_df["scenario"] = selected_scenario

        det_fig = create_deterministic_wealth_chart_comparison(det_single_compare_df, selected_result["inputs"])
        st.plotly_chart(det_fig, use_container_width=True, key=chart_key("deterministic", selected_scenario, view_mode, "client"))

        percentile_fig = create_percentile_paths_chart(display_percentile_df, selected_result["inputs"], f"Monte Carlo Percentile Paths - {selected_scenario}")
        st.plotly_chart(percentile_fig, use_container_width=True, key=chart_key("percentile", selected_scenario, view_mode, "client"))

        failure_fig = create_failure_probability_chart(selected_result["failure_prob_df"], selected_result["inputs"], f"Cumulative Probability of Running Out of Money - {selected_scenario}")
        st.plotly_chart(failure_fig, use_container_width=True, key=chart_key("failure", selected_scenario, view_mode, "client"))

        client_assumption_df = assumption_details_df[assumption_details_df["scenario"] == selected_scenario]
        st.subheader(t("Selected Scenario Assumptions", "所选情景假设"))
        st.dataframe(client_assumption_df, use_container_width=True)

        excel_file = dataframe_to_excel_bytes(
            {
                "input_summary": input_summary_df,
                "assumption_details": assumption_details_df,
                "contribution_schedule": contribution_schedule_export_df,
                "deterministic_projection": display_det_df,
                "simulation_summary": display_summary_df,
                "percentile_table": display_percentile_df,
                "failure_probability": selected_result["failure_prob_df"],
            }
        )

        st.download_button(
            label=t("Download Excel", "下载 Excel"),
            data=excel_file,
            file_name=build_export_filename(selected_result["inputs"].get("report_title", ""), "financial_projection", selected_scenario, "xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    st.info(t("Adjust the inputs above, then click Run Simulation in the sidebar. Saved snapshots can also be reopened from the sidebar.", "请先在上方区域调整输入，再点击侧边栏中的“运行模拟”。已保存快照也可以在侧边栏重新打开。"))
