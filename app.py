
import io
import re
from datetime import datetime

import pandas as pd
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

    st.subheader("Assumption Details")
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
    if view_mode == "Adviser View":
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
            st.subheader("Important Notes")
            for message in client_messages[:5]:
                st.warning(message)


def chart_key(chart_type, scenario_name, view_mode, section_name="main"):
    safe_scenario = scenario_name.lower().replace(" ", "_")
    safe_view = view_mode.lower().replace(" ", "_")
    safe_section = section_name.lower().replace(" ", "_")
    return f"{chart_type}_{safe_scenario}_{safe_view}_{safe_section}"


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

st.set_page_config(page_title="Financial Modelling Tool", layout="wide")
st.title("Financial Modelling Tool")
st.caption("Retirement projection, Monte Carlo simulation, scenario comparison, and client/adviser views")


# ============================================================
# SECTION: SESSION DEFAULTS
# ============================================================

defaults = {
    "comparison_results": None,
    "assumption_details_df": None,
    "input_summary_df": None,
    "contribution_schedule_export_df": None,
    "input_warnings_by_scenario": None,
    "output_warnings_by_scenario": None,
    "last_run_inputs_by_scenario": None,
    "assumption_preset": "Base Case",
    "start_financial_year": 2027,
    "projection_years": 40,
    "retirement_spending_trigger": "Both Retired",
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
    "number_of_simulations": 5000,
    "random_seed": 42,
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
    st.session_state.active_input_section = "Projection"


# ============================================================
# SECTION: INPUT LOCAL STATE
# ============================================================

report_title = st.session_state.report_title
person1_name = st.session_state.person1_name
person2_name = st.session_state.person2_name

start_financial_year = int(st.session_state.start_financial_year)
projection_years = int(st.session_state.projection_years)
retirement_spending_trigger = st.session_state.retirement_spending_trigger

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
contribution_events_df = st.session_state.contribution_events_df.copy()


# ============================================================
# SECTION: SIDEBAR CONTROLS
# ============================================================

with st.sidebar:
    st.header("Controls")

    view_mode = st.radio(
        "View Mode",
        options=["Adviser View", "Client View"],
    )

    scenario_mode = st.radio(
        "Scenario Mode",
        options=["Single Scenario", "Compare Standard Presets"],
    )

    preset_choice = st.selectbox(
        "Assumption Preset",
        options=["Conservative", "Base Case", "Optimistic", "Custom"],
        key="assumption_preset",
        disabled=(scenario_mode == "Compare Standard Presets"),
    )

    run_button = st.button("Run Simulation", type="primary", use_container_width=True)


# ============================================================
# SECTION: ASSUMPTION SETTINGS PANEL
# ============================================================

if st.button(
    "Reset Preset Assumptions to Default",
    key="reset_preset_table_button_panel",
):
    st.session_state.preset_table_df = get_default_preset_table_df()
    st.rerun()

runtime_presets = preset_table_to_dict(st.session_state.preset_table_df)

with st.container(border=True):
    top_left, top_right = st.columns([2, 1])

    with top_left:
        st.subheader("Assumption Settings Panel")
        st.caption("Manage reusable preset assumptions here. The modelling engine stores these as decimals such as 0.03 = 3.0%.")

    with top_right:
        st.metric("Active Preset", preset_choice)
        st.caption("Preset selection stays in the sidebar for quick scenario control.")

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

input_sections = [
    "Report",
    "Projection",
    "Person 1",
    "Person 2",
    "Household",
    "Contributions",
    "Returns",
    "Simulation",
]

active_input_section = st.segmented_control(
    "Input Section",
    options=input_sections,
    selection_mode="single",
    default=st.session_state.active_input_section,
    key="active_input_section",
)

if active_input_section == "Report":
    st.subheader("Report")
    report_title = st.text_input(
        "Title (Optional)",
        value=st.session_state.report_title,
        key="report_title_input",
    )

    st.subheader("Names")
    name_col1, name_col2 = st.columns(2)
    with name_col1:
        person1_name = st.text_input(
            "Person 1 Name (Optional)",
            value=st.session_state.person1_name,
            key="person1_name_input",
        )
    with name_col2:
        person2_name = st.text_input(
            "Person 2 Name (Optional)",
            value=st.session_state.person2_name,
            key="person2_name_input",
        )

elif active_input_section == "Projection":
    st.subheader("Projection Timing")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_financial_year = st.number_input(
            "Start Financial Year",
            value=int(st.session_state.start_financial_year),
            step=1,
            min_value=2000,
        )
    with col2:
        projection_years = st.number_input(
            "Projection Years",
            value=int(st.session_state.projection_years),
            step=1,
            min_value=1,
        )
    with col3:
        retirement_spending_trigger = st.selectbox(
            "Retirement Spending Trigger",
            options=["Both Retired", "Either Retired"],
            index=0 if st.session_state.retirement_spending_trigger == "Both Retired" else 1,
        )

elif active_input_section == "Person 1":
    st.subheader("Person 1")
    p1a, p1b, p1c = st.columns(3)
    with p1a:
        person1_current_age = st.number_input("Person 1 Current Age", value=int(st.session_state.person1_current_age), step=1)
        person1_accum_super_balance = currency_text_input("Person 1 Accumulation Super Balance", st.session_state.person1_accum_super_balance, "person1_accum_super_balance_input")
        person1_pension_super_balance = currency_text_input("Person 1 Pension Super Balance", st.session_state.person1_pension_super_balance, "person1_pension_super_balance_input")
    with p1b:
        person1_retirement_age = st.number_input("Person 1 Retirement Age", value=int(st.session_state.person1_retirement_age), step=1)
        person1_accum_super_cost_base = currency_text_input("Person 1 Accumulation Super Cost Base", st.session_state.person1_accum_super_cost_base, "person1_accum_super_cost_base_input")
        person1_pension_super_cost_base = currency_text_input("Person 1 Pension Super Cost Base", st.session_state.person1_pension_super_cost_base, "person1_pension_super_cost_base_input")
    with p1c:
        person1_pension_start_age = st.number_input("Person 1 Pension Start Age", value=int(st.session_state.person1_pension_start_age), step=1)
        person1_transfer_balance_cap = currency_text_input("Person 1 Transfer Balance Cap", st.session_state.person1_transfer_balance_cap, "person1_transfer_balance_cap_input")
        person1_annual_income = currency_text_input("Person 1 Annual Income", st.session_state.person1_annual_income, "person1_annual_income_input")

elif active_input_section == "Person 2":
    st.subheader("Person 2")
    p2a, p2b, p2c = st.columns(3)
    with p2a:
        person2_current_age = st.number_input("Person 2 Current Age", value=int(st.session_state.person2_current_age), step=1)
        person2_accum_super_balance = currency_text_input("Person 2 Accumulation Super Balance", st.session_state.person2_accum_super_balance, "person2_accum_super_balance_input")
        person2_pension_super_balance = currency_text_input("Person 2 Pension Super Balance", st.session_state.person2_pension_super_balance, "person2_pension_super_balance_input")
    with p2b:
        person2_retirement_age = st.number_input("Person 2 Retirement Age", value=int(st.session_state.person2_retirement_age), step=1)
        person2_accum_super_cost_base = currency_text_input("Person 2 Accumulation Super Cost Base", st.session_state.person2_accum_super_cost_base, "person2_accum_super_cost_base_input")
        person2_pension_super_cost_base = currency_text_input("Person 2 Pension Super Cost Base", st.session_state.person2_pension_super_cost_base, "person2_pension_super_cost_base_input")
    with p2c:
        person2_pension_start_age = st.number_input("Person 2 Pension Start Age", value=int(st.session_state.person2_pension_start_age), step=1)
        person2_transfer_balance_cap = currency_text_input("Person 2 Transfer Balance Cap", st.session_state.person2_transfer_balance_cap, "person2_transfer_balance_cap_input")
        person2_annual_income = currency_text_input("Person 2 Annual Income", st.session_state.person2_annual_income, "person2_annual_income_input")

elif active_input_section == "Household":
    st.subheader("Household")
    hh1, hh2 = st.columns(2)
    with hh1:
        non_super_balance = currency_text_input("Non-Super Balance", st.session_state.non_super_balance, "non_super_balance_input")
        annual_living_expenses = currency_text_input("Annual Living Expenses", st.session_state.annual_living_expenses, "annual_living_expenses_input")
        cgt_discount_rate = percentage_text_input("CGT Discount Rate", float(st.session_state.cgt_discount_rate), "cgt_discount_rate_input", decimals=1)
    with hh2:
        non_super_cost_base = currency_text_input("Non-Super Cost Base", st.session_state.non_super_cost_base, "non_super_cost_base_input")
        retirement_spending = currency_text_input("Retirement Spending", st.session_state.retirement_spending, "retirement_spending_input")
        non_super_ownership_person1 = percentage_text_input("Person 1 Ownership %", st.session_state.non_super_ownership_person1_pct / 100.0, "non_super_ownership_person1_input", decimals=1) * 100.0

elif active_input_section == "Contributions":
    st.subheader("Contribution Schedule")
    contribution_events_df = st.data_editor(
        st.session_state.contribution_events_df,
        key="contribution_events_editor",
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "financial_year": st.column_config.NumberColumn("Financial Year", min_value=2000, step=1),
            "person": st.column_config.SelectboxColumn("Person", options=["Person 1", "Person 2"]),
            "contribution_type": st.column_config.SelectboxColumn("Contribution Type", options=["personal_deductible", "non_concessional"]),
            "amount": st.column_config.NumberColumn("Amount", min_value=0.0, step=1000.0, format="$%.0f"),
        },
    )

elif active_input_section == "Returns":
    if scenario_mode == "Single Scenario" and preset_choice == "Custom":
        st.subheader("Return Assumptions")
        r1, r2, r3 = st.columns(3)
        with r1:
            super_income_return_mean = percentage_text_input("Super Income Return Mean", st.session_state.super_income_return_mean, "super_income_return_mean_input", decimals=1)
            super_capital_return_mean = percentage_text_input("Super Capital Return Mean", st.session_state.super_capital_return_mean, "super_capital_return_mean_input", decimals=1)
            inflation_rate = percentage_text_input("Inflation Rate", st.session_state.inflation_rate, "inflation_rate_input", decimals=1)
        with r2:
            super_income_return_std = percentage_text_input("Super Income Return Std", st.session_state.super_income_return_std, "super_income_return_std_input", decimals=1)
            super_capital_return_std = percentage_text_input("Super Capital Return Std", st.session_state.super_capital_return_std, "super_capital_return_std_input", decimals=1)
        with r3:
            non_super_income_return_mean = percentage_text_input("Non-Super Income Return Mean", st.session_state.non_super_income_return_mean, "non_super_income_return_mean_input", decimals=1)
            non_super_capital_return_mean = percentage_text_input("Non-Super Capital Return Mean", st.session_state.non_super_capital_return_mean, "non_super_capital_return_mean_input", decimals=1)
            non_super_income_return_std = percentage_text_input("Non-Super Income Return Std", st.session_state.non_super_income_return_std, "non_super_income_return_std_input", decimals=1)
            non_super_capital_return_std = percentage_text_input("Non-Super Capital Return Std", st.session_state.non_super_capital_return_std, "non_super_capital_return_std_input", decimals=1)
    else:
        selected_preset = preset_choice if preset_choice in runtime_presets else "Base Case"
        selected_values = runtime_presets[selected_preset]

        st.subheader("Return Assumptions")
        st.info(f"Using values from Assumption Settings Panel: {selected_preset}")

        display_df = pd.DataFrame(
            {
                "Assumption": [
                    "Super Income Return Mean",
                    "Super Income Return Std",
                    "Super Capital Return Mean",
                    "Super Capital Return Std",
                    "Non-Super Income Return Mean",
                    "Non-Super Income Return Std",
                    "Non-Super Capital Return Mean",
                    "Non-Super Capital Return Std",
                    "Inflation Rate",
                ],
                "Value": [
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
                "Assumption": st.column_config.TextColumn("Assumption"),
                "Value": st.column_config.NumberColumn("Value", format="%.1f%%"),
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

elif active_input_section == "Simulation":
    st.subheader("Simulation")
    sim1, sim2 = st.columns(2)
    with sim1:
        number_of_simulations = st.number_input("Number of Simulations", value=int(st.session_state.number_of_simulations), step=1000)
    with sim2:
        random_seed = st.number_input("Random Seed", value=int(st.session_state.random_seed), step=1)

# ============================================================
# SECTION: SESSION UPDATE
# ============================================================

st.session_state.start_financial_year = int(start_financial_year)
st.session_state.projection_years = int(projection_years)
st.session_state.retirement_spending_trigger = retirement_spending_trigger
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
    "contribution_events": contribution_events_to_records(contribution_events_df),
    "person1_accum_super_cost_base": person1_accum_super_cost_base,
    "person1_pension_super_cost_base": person1_pension_super_cost_base,
    "person2_accum_super_cost_base": person2_accum_super_cost_base,
    "person2_pension_super_cost_base": person2_pension_super_cost_base,
}


# ============================================================
# SECTION: RUN LOGIC
# ============================================================

if run_button:
    if scenario_mode == "Single Scenario":
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


# ============================================================
# SECTION: RESULTS RENDERING
# ============================================================

if st.session_state.comparison_results is not None:
    comparison_results = st.session_state.comparison_results
    assumption_details_df = st.session_state.assumption_details_df
    input_summary_df = st.session_state.input_summary_df
    contribution_schedule_export_df = st.session_state.contribution_schedule_export_df
    input_warnings_by_scenario = st.session_state.input_warnings_by_scenario
    output_warnings_by_scenario = st.session_state.output_warnings_by_scenario

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
    comparison_df = format_comparison_df(comparison_df)
    det_scenarios_df = pd.concat(det_scenarios_df_list, ignore_index=True)

    common_inputs = next(iter(comparison_results.values()))["inputs"]

    st.subheader("Scenario Comparison Summary")
    st.dataframe(
        comparison_df[["scenario", "success_rate_label", "median_final_wealth_label", "p10_final_wealth_label", "p90_final_wealth_label"]],
        use_container_width=True,
    )

    success_fig = create_success_rate_comparison_chart(comparison_df)
    st.plotly_chart(success_fig, use_container_width=True, key="success_rate_comparison")

    median_fig = create_median_wealth_comparison_chart(comparison_df)
    st.plotly_chart(median_fig, use_container_width=True, key="median_wealth_comparison")

    selected_scenario = st.selectbox(
        "Select Scenario",
        options=list(comparison_results.keys()),
        key="selected_scenario_results",
    )
    selected_result = comparison_results[selected_scenario]

    if view_mode == "Adviser View":
        st.subheader(f"Adviser Summary - {selected_scenario}")

        top_col1, top_col2 = st.columns(2)
        top_col1.metric("Success Rate", f"{selected_result['success_rate']:.1%}")
        top_col2.metric("Median Final Wealth", f"${selected_result['median_final_wealth']:,.0f}")

        bottom_col1, bottom_col2, bottom_col3 = st.columns(3)
        bottom_col1.metric("P10 Final Wealth", f"${selected_result['p10_final_wealth']:,.0f}")
        bottom_col2.metric("P90 Final Wealth", f"${selected_result['p90_final_wealth']:,.0f}")
        bottom_col3.metric("Spread (P90 - P10)", f"${selected_result['p90_final_wealth'] - selected_result['p10_final_wealth']:,.0f}")

        missing_validation_cols = get_missing_validation_columns(selected_result["det_df"])
        if missing_validation_cols:
            st.warning("Validation table is using fallback zeros for missing columns.")
            st.caption(", ".join(missing_validation_cols))

        adviser_cashflow_df = build_adviser_cashflow_df(selected_result["det_df"])
        st.subheader("Adviser Cashflow Summary")
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

        pension_tax_free_summary_df = build_pension_tax_free_summary_df(selected_result["det_df"])
        st.subheader("Pension Tax-Free Validation Summary")
        st.dataframe(
            pension_tax_free_summary_df,
            use_container_width=True,
            column_config={
                "check": st.column_config.TextColumn("Check"),
                "value": st.column_config.NumberColumn("Value", format="$%.0f"),
            },
        )

        debug_df = build_adviser_debug_df(selected_result["det_df"])
        st.subheader("Adviser Debug Table")
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

        cgt_validation_df = build_cgt_validation_df(selected_result["det_df"])
        with st.expander("Detailed CGT / Pension Validation Table", expanded=False):
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

        tax_breakdown_fig = create_tax_breakdown_chart(selected_result["det_df"], selected_result["inputs"], f"Tax Breakdown - {selected_scenario}")
        st.plotly_chart(tax_breakdown_fig, use_container_width=True, key=chart_key("tax_breakdown", selected_scenario, view_mode, "adviser"))

        total_tax_fig = create_total_tax_paid_chart(selected_result["det_df"], selected_result["inputs"], f"Total Tax Paid - {selected_scenario}")
        st.plotly_chart(total_tax_fig, use_container_width=True, key=chart_key("total_tax", selected_scenario, view_mode, "adviser"))

        excel_file = dataframe_to_excel_bytes(
            {
                "input_summary": input_summary_df,
                "assumption_details": assumption_details_df,
                "contribution_schedule": contribution_schedule_export_df,
                "deterministic_projection": selected_result["det_df"],
                "simulation_summary": selected_result["summary_df"],
                "percentile_table": selected_result["percentile_df"],
                "failure_probability": selected_result["failure_prob_df"],
                "adviser_cashflow_summary": adviser_cashflow_df,
                "adviser_debug_table": debug_df,
                "pension_tax_free_summary": pension_tax_free_summary_df,
                "cgt_validation_detail": cgt_validation_df,
            }
        )

        st.download_button(
            label="Download Excel",
            data=excel_file,
            file_name=build_export_filename(selected_result["inputs"].get("report_title", ""), "financial_projection", selected_scenario, "xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    else:
        st.subheader(f"Client Summary - {selected_scenario}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Success Rate", f"{selected_result['success_rate']:.1%}")
        col2.metric("Median Final Wealth", f"${selected_result['median_final_wealth']:,.0f}")
        col3.metric("P10 Final Wealth", f"${selected_result['p10_final_wealth']:,.0f}")
        col4.metric("P90 Final Wealth", f"${selected_result['p90_final_wealth']:,.0f}")

        det_single_df = selected_result["det_df"]
        det_single_compare_df = det_single_df.copy()
        det_single_compare_df["scenario"] = selected_scenario

        det_fig = create_deterministic_wealth_chart_comparison(det_single_compare_df, selected_result["inputs"])
        st.plotly_chart(det_fig, use_container_width=True, key=chart_key("deterministic", selected_scenario, view_mode, "client"))

        percentile_fig = create_percentile_paths_chart(selected_result["percentile_df"], selected_result["inputs"], f"Monte Carlo Percentile Paths - {selected_scenario}")
        st.plotly_chart(percentile_fig, use_container_width=True, key=chart_key("percentile", selected_scenario, view_mode, "client"))

        failure_fig = create_failure_probability_chart(selected_result["failure_prob_df"], selected_result["inputs"], f"Cumulative Probability of Running Out of Money - {selected_scenario}")
        st.plotly_chart(failure_fig, use_container_width=True, key=chart_key("failure", selected_scenario, view_mode, "client"))

        client_assumption_df = assumption_details_df[assumption_details_df["scenario"] == selected_scenario]
        st.subheader("Selected Scenario Assumptions")
        st.dataframe(client_assumption_df, use_container_width=True)

        excel_file = dataframe_to_excel_bytes(
            {
                "input_summary": input_summary_df,
                "assumption_details": assumption_details_df,
                "contribution_schedule": contribution_schedule_export_df,
                "deterministic_projection": selected_result["det_df"],
                "simulation_summary": selected_result["summary_df"],
                "percentile_table": selected_result["percentile_df"],
                "failure_probability": selected_result["failure_prob_df"],
            }
        )

        st.download_button(
            label="Download Excel",
            data=excel_file,
            file_name=build_export_filename(selected_result["inputs"].get("report_title", ""), "financial_projection", selected_scenario, "xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    st.info("Adjust the inputs in the tabs above, then click Run Simulation in the sidebar.")
