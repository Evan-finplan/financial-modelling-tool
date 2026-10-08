
import copy
import io
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from debt_analysis import (
    DEBT_STRATEGY_PROFILES,
    apply_debt_strategy_profile,
    build_debt_strategy_comparison_df,
)
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

# Streamlit Cloud can briefly reload app.py before charts.py during a deploy.
# Keep the app available during that window; the chart appears automatically
# once the updated charts module is loaded on the next rerun.
try:
    from charts import create_cashflow_chart
except ImportError:
    create_cashflow_chart = None
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
from module_config import MODULE_DEFAULTS, MODULE_LABELS, active_module_names, apply_module_scope
from pdf_report import CHART_LABELS, PDF_CHART_KEYS, build_pdf_report_bytes
from strategy_analysis import (
    STRATEGY_PROFILES,
    apply_strategy_profile,
    build_assumption_change_df,
    build_strategy_comparison_df,
)
from upload_security import WorkbookUploadError, validate_xlsx_upload


# ============================================================
# SECTION: UI LANGUAGE HELPERS
# ============================================================

LANGUAGE_EN = "🇬🇧 English"
LANGUAGE_CN = "🇨🇳 中文"


def is_cn():
    return st.session_state.get("ui_language", LANGUAGE_EN) == LANGUAGE_CN


def t(en, zh):
    return zh if is_cn() else en


def inject_jbwere_styles():
    st.markdown(
        """
        <style>
        :root {
            --jbw-navy: #00205B;
            --jbw-deep-navy: #00163F;
            --jbw-blue: #34657F;
            --jbw-sky: #DDEEF4;
            --jbw-mist: #F4F8FA;
            --jbw-ink: #182A3A;
            --jbw-muted: #53697A;
            --jbw-grid: #C7D6DE;
        }

        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"] {
            font-family: Arial, "Microsoft YaHei", "PingFang SC", sans-serif;
        }

        /* Keep Streamlit's icon ligatures on their own font. Applying Arial to
           every st-* class turns icon names into visible text and can make
           uploader controls overlap. */
        [data-testid="stIconMaterial"],
        .material-symbols-rounded,
        .material-symbols-outlined,
        .material-icons {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
            font-weight: normal !important;
            font-style: normal !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            direction: ltr !important;
            -webkit-font-feature-settings: "liga" !important;
            -webkit-font-smoothing: antialiased !important;
            font-feature-settings: "liga" !important;
        }

        p, li, label, input, textarea, button,
        [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"] {
            font-size: 12px;
        }

        [data-testid="stAppViewContainer"] {
            background: #FFFFFF;
            color: var(--jbw-ink);
            border-top: 7px solid var(--jbw-navy);
        }

        [data-testid="stSidebar"] {
            background: var(--jbw-mist);
            border-right: 1px solid var(--jbw-grid);
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--jbw-navy) !important;
            font-family: Arial, "Microsoft YaHei", "PingFang SC", sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }

        h1 {
            border-bottom: 1px solid var(--jbw-grid);
            padding-bottom: 0.35rem;
        }

        [data-testid="stMetric"] {
            background: var(--jbw-sky);
            border-top: 3px solid var(--jbw-navy);
            padding: 0.85rem 1rem;
        }

        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: var(--jbw-deep-navy);
        }

        .stButton > button,
        .stDownloadButton > button {
            background: var(--jbw-navy);
            color: #FFFFFF !important;
            border: 1px solid var(--jbw-navy);
            border-radius: 2px;
            min-height: 2.6rem;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: var(--jbw-deep-navy);
            border-color: var(--jbw-deep-navy);
            color: #FFFFFF !important;
        }

        [data-testid="stExpander"],
        [data-testid="stForm"],
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--jbw-grid) !important;
            border-radius: 2px !important;
        }

        [data-testid="stAlert"] {
            border-radius: 2px;
            border-left: 4px solid var(--jbw-blue);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea {
            border-radius: 2px !important;
        }

        a {
            color: var(--jbw-blue);
        }
        </style>
        """,
        unsafe_allow_html=True,
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


def render_pdf_export_controls(
    selected_result,
    comparison_results,
    selected_scenario,
    value_mode,
    input_warnings,
    output_warnings,
    widget_scope,
):
    st.subheader(t("PDF Report", "PDF 报告"))
    st.caption(t(
        "Tick the charts to include. The report always includes a future outlook, key milestones, headline results, review items and explanatory notes.",
        "请勾选需要导出的图表。报告始终包含未来情况概述、关键节点、核心结果、审阅事项及相关解释。",
    ))

    report_detail = st.selectbox(
        t("Report Detail", "报告详细程度"),
        options=["Client Summary", "Advice Support Report", "Technical Appendix"],
        format_func=lambda option: {
            "Client Summary": t("Client Summary", "客户摘要"),
            "Advice Support Report": t("Advice Support Report", "建议支持报告"),
            "Technical Appendix": t("Technical Appendix", "技术附录"),
        }[option],
        key=f"pdf_detail_{sanitise_filename_part(widget_scope, fallback='pdf')}",
        help=t(
            "Client Summary is concise. Advice Support adds strategy analysis and adviser notes. Technical Appendix also includes assumptions, policy status and calculation methodology.",
            "Client Summary 为简要版；Advice Support 增加策略分析与顾问备注；Technical Appendix 还包含假设、政策状态及计算方法。",
        ),
    )

    default_charts = {
        "wealth_projection",
        "percentile_paths",
        "failure_probability",
        "income_spending",
        "total_tax",
    }
    selected_chart_keys = []
    columns = st.columns(2)
    safe_scope = sanitise_filename_part(widget_scope, fallback="pdf")
    for index, chart_key in enumerate(PDF_CHART_KEYS):
        label = CHART_LABELS[chart_key][1 if is_cn() else 0]
        with columns[index % 2]:
            include_chart = st.checkbox(
                label,
                value=chart_key in default_charts,
                key=f"pdf_chart_{safe_scope}_{chart_key}",
            )
        if include_chart:
            selected_chart_keys.append(chart_key)

    if not selected_chart_keys:
        st.info(t(
            "No chart is selected. The PDF will still contain the written outlook and milestone summary.",
            "目前未选择图表。PDF 仍会包含未来情况和关键节点的文字摘要。",
        ))

    try:
        pdf_file = build_pdf_report_bytes(
            selected_result=selected_result,
            comparison_results=comparison_results,
            selected_scenario=selected_scenario,
            selected_chart_keys=selected_chart_keys,
            value_mode=value_mode,
            input_warnings=input_warnings,
            output_warnings=output_warnings,
            report_language=LANGUAGE_CN if is_cn() else LANGUAGE_EN,
            report_detail=report_detail,
            adviser_notes=st.session_state.get("adviser_notes", ""),
        )
    except Exception as exc:
        st.error(t(
            f"PDF preparation failed: {exc}",
            f"PDF 准备失败：{exc}",
        ))
        return

    st.download_button(
        label=t("Export PDF Report", "一键导出 PDF 报告"),
        data=pdf_file,
        file_name=build_export_filename(
            selected_result["inputs"].get("report_title", ""),
            "financial_projection_report",
            selected_scenario,
            "pdf",
        ),
        mime="application/pdf",
        width="stretch",
        key=f"download_pdf_{safe_scope}",
    )


# ============================================================
# SECTION: EXCEL INPUT IMPORT / EXPORT HELPERS
# ============================================================

INPUT_EXCEL_FIELDS = [
    "ui_language",
    "value_mode",
    "household_mode",
    "module_second_person_enabled",
    "module_super_enabled",
    "module_pension_enabled",
    "module_non_super_enabled",
    "module_property_enabled",
    "module_trust_enabled",
    "module_cash_surplus_enabled",
    "module_investment_debt_enabled",
    "assumption_preset",
    "report_title",
    "person1_name",
    "person2_name",
    "start_financial_year",
    "projection_years",
    "retirement_spending_trigger",
    "person1_current_age",
    "person2_current_age",
    "person1_retirement_age",
    "person2_retirement_age",
    "person1_pension_start_age",
    "person2_pension_start_age",
    "person1_accum_super_balance",
    "person1_pension_super_balance",
    "person2_accum_super_balance",
    "person2_pension_super_balance",
    "person1_accum_super_cost_base",
    "person1_pension_super_cost_base",
    "person2_accum_super_cost_base",
    "person2_pension_super_cost_base",
    "person1_transfer_balance_cap",
    "person2_transfer_balance_cap",
    "person1_annual_income",
    "person2_annual_income",
    "non_super_balance",
    "non_super_cost_base",
    "cash_reserve_balance",
    "cash_reserve_floor",
    "cash_reserve_target",
    "main_residence_value",
    "main_residence_capital_growth_rate",
    "main_residence_loan_balance",
    "main_residence_interest_rate",
    "main_residence_annual_loan_repayment",
    "main_residence_offset_balance",
    "non_deductible_debt_balance",
    "non_deductible_interest_rate",
    "non_deductible_annual_repayment",
    "non_deductible_offset_balance",
    "deductible_offset_balance",
    "investment_deductible_debt_balance",
    "investment_deductible_interest_rate",
    "investment_deductible_annual_repayment",
    "investment_deductible_offset_balance",
    "surplus_allocation_profile",
    "non_super_estate_reserve",
    "property_estate_reserve",
    "drawdown_profile",
    "cgt_reform_enabled",
    "cgt_asset_acquired_before_2027",
    "non_super_transition_value_2027",
    "non_super_opening_capital_losses",
    "cgt_indexation_rate",
    "cgt_asset_category",
    "cgt_new_residential_method",
    "cgt_held_at_least_12_months",
    "cgt_minimum_tax_exempt",
    "residential_property_enabled",
    "residential_property_value",
    "residential_property_loan_balance",
    "residential_property_annual_loan_repayment",
    "residential_property_sale_cost_rate",
    "residential_property_gross_rent",
    "residential_property_operating_expenses",
    "residential_property_interest_rate",
    "residential_property_capital_growth_rate",
    "residential_property_rent_growth_rate",
    "residential_property_expense_growth_rate",
    "residential_property_opening_quarantined_loss",
    "residential_property_acquired_before_budget_time",
    "residential_property_is_new_build",
    "residential_property_is_exempt_housing",
    "residential_property_ownership_person1_pct",
    "discretionary_trust_enabled",
    "discretionary_trust_balance",
    "discretionary_trust_cost_base",
    "discretionary_trust_income_return_mean",
    "discretionary_trust_income_return_std",
    "discretionary_trust_capital_return_mean",
    "discretionary_trust_capital_return_std",
    "discretionary_trust_excluded_income_pct",
    "discretionary_trust_subject_to_minimum_tax",
    "discretionary_trust_ownership_person1_pct",
    "annual_living_expenses",
    "retirement_spending",
    "non_super_ownership_person1_pct",
    "cgt_discount_rate",
    "super_income_return_mean",
    "super_income_return_std",
    "super_capital_return_mean",
    "super_capital_return_std",
    "non_super_income_return_mean",
    "non_super_income_return_std",
    "non_super_capital_return_mean",
    "non_super_capital_return_std",
    "inflation_rate",
    "number_of_simulations",
    "random_seed",
]


def _coerce_uploaded_input_value(field_name, value):
    if pd.isna(value):
        return "" if isinstance(defaults.get(field_name, ""), str) else defaults.get(field_name, value)

    default_value = defaults.get(field_name, "")

    if isinstance(default_value, bool):
        if isinstance(value, str):
            normalised = value.strip().lower()
            if normalised in {"true", "yes", "y", "1", "on"}:
                return True
            if normalised in {"false", "no", "n", "0", "off", ""}:
                return False
            raise ValueError(f"{field_name} must be TRUE or FALSE.")
        return bool(value)
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(float(value))
    if isinstance(default_value, float):
        return float(value)

    text_value = str(value).strip()

    if field_name == "household_mode":
        if text_value not in {"One Person", "Two People"}:
            raise ValueError("household_mode must be 'One Person' or 'Two People'.")
    if field_name == "value_mode":
        if text_value not in {"Future Value", "Present Value"}:
            raise ValueError("value_mode must be 'Future Value' or 'Present Value'.")
    if field_name == "retirement_spending_trigger":
        if text_value not in {"Both Retired", "Either Retired"}:
            raise ValueError("retirement_spending_trigger must be 'Both Retired' or 'Either Retired'.")
    if field_name == "assumption_preset":
        if text_value not in {"Conservative", "Base Case", "Optimistic", "Custom"}:
            raise ValueError("assumption_preset must be Conservative, Base Case, Optimistic, or Custom.")
    if field_name == "cgt_asset_category":
        if text_value not in {"Other", "New residential dwelling", "Affordable housing"}:
            raise ValueError("cgt_asset_category is not recognised.")
    if field_name == "cgt_new_residential_method":
        if text_value not in {"Indexation and 30% minimum tax", "50% discount"}:
            raise ValueError("cgt_new_residential_method is not recognised.")

    return text_value


def build_input_state_df_from_session_state():
    rows = []
    for field_name in INPUT_EXCEL_FIELDS:
        rows.append({
            "input_name": field_name,
            "input_value": st.session_state.get(field_name, defaults.get(field_name, "")),
            "notes": "Edit input_value only. Keep input_name unchanged.",
        })
    return pd.DataFrame(rows)


def build_excel_input_workbook_bytes(include_current_values=True):
    input_df = build_input_state_df_from_session_state() if include_current_values else pd.DataFrame([
        {
            "input_name": field_name,
            "input_value": defaults.get(field_name, ""),
            "notes": "Edit input_value only. Keep input_name unchanged.",
        }
        for field_name in INPUT_EXCEL_FIELDS
    ])

    contribution_df = st.session_state.get(
        "contribution_events_df",
        pd.DataFrame(columns=["financial_year", "person", "contribution_type", "amount"]),
    ).copy() if include_current_values else pd.DataFrame(
        columns=["financial_year", "person", "contribution_type", "amount"]
    )

    preset_df = st.session_state.get(
        "preset_table_df",
        get_default_preset_table_df(),
    ).copy() if include_current_values else get_default_preset_table_df()

    instructions_df = pd.DataFrame([
        {"item": "inputs", "instruction": "Edit the input_value column. Do not rename input_name."},
        {"item": "contribution_schedule", "instruction": "Optional. Use financial_year, person, contribution_type, amount."},
        {"item": "preset_assumptions", "instruction": "Optional. Keep preset names and columns unchanged."},
        {"item": "percentages", "instruction": "Use decimals for rate assumptions, e.g. 0.03 means 3%. non_super_ownership_person1_pct uses percent value, e.g. 100 means 100%."},
        {"item": "household_mode", "instruction": "Use One Person or Two People."},
    ])

    return dataframe_to_excel_bytes({
        "instructions": instructions_df,
        "inputs": input_df,
        "contribution_schedule": contribution_df,
        "preset_assumptions": preset_df,
    })


def apply_uploaded_input_workbook(uploaded_file):
    if uploaded_file is None:
        return False, t("Please upload an Excel file first.", "请先上传一个 Excel 文件。")

    try:
        uploaded_bytes = uploaded_file.getvalue()
        safe_workbook = validate_xlsx_upload(
            uploaded_bytes,
            filename=getattr(uploaded_file, "name", ""),
        )
        workbook = pd.read_excel(safe_workbook, sheet_name=None)
    except WorkbookUploadError as exc:
        return False, t(f"Upload rejected: {exc}", f"上传已被拒绝：{exc}")
    except Exception as exc:
        return False, t(f"Could not read Excel file: {exc}", f"无法读取 Excel 文件：{exc}")

    if len(workbook) > 12:
        return False, t(
            "The workbook contains too many worksheets (maximum 12).",
            "工作簿包含过多工作表（最多 12 个）。",
        )
    if any(frame.shape[0] > 10_000 or frame.shape[1] > 100 for frame in workbook.values()):
        return False, t(
            "A worksheet exceeds the supported size of 10,000 rows by 100 columns.",
            "工作表超出支持范围：最多 10,000 行、100 列。",
        )

    if "inputs" not in workbook:
        return False, t("The workbook must contain a sheet named 'inputs'.", "Excel 文件必须包含名为 'inputs' 的工作表。")

    input_df = workbook["inputs"].copy()
    if "input_name" not in input_df.columns or "input_value" not in input_df.columns:
        return False, t("The inputs sheet must contain input_name and input_value columns.", "inputs 工作表必须包含 input_name 和 input_value 两列。")

    updated_count = 0
    errors = []
    pending_module_values = {}
    uploaded_field_names = {
        str(value).strip()
        for value in input_df["input_name"].dropna().tolist()
    }
    for _, row in input_df.iterrows():
        field_name = str(row.get("input_name", "")).strip()
        if not field_name or field_name not in INPUT_EXCEL_FIELDS:
            continue

        try:
            uploaded_value = _coerce_uploaded_input_value(
                field_name,
                row.get("input_value")
            )

            # ==========================================
            # Streamlit widget lifecycle safe handling
            # ==========================================

            if field_name in MODULE_DEFAULTS:
                pending_module_values[field_name] = uploaded_value
            elif field_name == "assumption_preset":
                st.session_state[
                    "pending_uploaded_assumption_preset"
                ] = uploaded_value
            else:
                st.session_state[field_name] = uploaded_value

            updated_count += 1
            
        except Exception as exc:
            errors.append(f"{field_name}: {exc}")

    # Older workbooks pre-date module switches. Infer their intended scope from
    # the existing input fields so importing them does not unexpectedly hide data.
    if "module_second_person_enabled" not in uploaded_field_names:
        pending_module_values["module_second_person_enabled"] = st.session_state.get("household_mode", "Two People") != "One Person"
    if "module_property_enabled" not in uploaded_field_names:
        pending_module_values["module_property_enabled"] = bool(st.session_state.get("residential_property_enabled", False))
    if "module_trust_enabled" not in uploaded_field_names:
        pending_module_values["module_trust_enabled"] = bool(st.session_state.get("discretionary_trust_enabled", False))
    if "module_non_deductible_debt_enabled" not in uploaded_field_names:
        pending_module_values["module_non_deductible_debt_enabled"] = any(
            float(st.session_state.get(field, 0.0) or 0.0) > 0
            for field in ["non_deductible_debt_balance", "non_deductible_offset_balance"]
        )
    if "module_deductible_debt_enabled" not in uploaded_field_names:
        property_enabled = pending_module_values.get(
            "module_property_enabled",
            bool(st.session_state.get("module_property_enabled", False)),
        )
        pending_module_values["module_deductible_debt_enabled"] = property_enabled and any(
            float(st.session_state.get(field, 0.0) or 0.0) > 0
            for field in ["residential_property_loan_balance", "deductible_offset_balance"]
        )
    if "module_investment_debt_enabled" not in uploaded_field_names:
        pending_module_values["module_investment_debt_enabled"] = any(
            float(st.session_state.get(field, 0.0) or 0.0) > 0
            for field in [
                "non_deductible_debt_balance",
                "non_deductible_offset_balance",
                "investment_deductible_debt_balance",
                "investment_deductible_offset_balance",
            ]
        )
    if pending_module_values:
        st.session_state.pending_uploaded_module_values = pending_module_values

    if errors:
        return False, t(
            "Some uploaded inputs could not be applied: " + "; ".join(errors[:5]),
            "部分上传输入无法应用：" + "; ".join(errors[:5]),
        )

    if "contribution_schedule" in workbook:
        contribution_df = workbook["contribution_schedule"].copy()
        st.session_state.contribution_events_df = normalise_contribution_events(
            contribution_df,
            household_mode=st.session_state.get("household_mode", "Two People"),
        )

    if "preset_assumptions" in workbook:
        preset_df = workbook["preset_assumptions"].copy()
        st.session_state.preset_table_df = ensure_valid_preset_table_df(preset_df)

    if st.session_state.get("household_mode") == "One Person":
        st.session_state.person2_name = ""
        st.session_state.person2_current_age = 0
        st.session_state.person2_retirement_age = 0
        st.session_state.person2_pension_start_age = 0
        st.session_state.person2_accum_super_balance = 0.0
        st.session_state.person2_pension_super_balance = 0.0
        st.session_state.person2_accum_super_cost_base = 0.0
        st.session_state.person2_pension_super_cost_base = 0.0
        st.session_state.person2_transfer_balance_cap = 0.0
        st.session_state.person2_annual_income = 0.0
        st.session_state.non_super_ownership_person1_pct = 100.0
        st.session_state.retirement_spending_trigger = "Either Retired"
        st.session_state.contribution_events_df = normalise_contribution_events(
            st.session_state.contribution_events_df,
            household_mode="One Person",
        )

    st.session_state.comparison_results = None
    st.session_state.active_result_set_name = "Current Results"

    return True, t(
        f"Uploaded input applied successfully ({updated_count} fields updated). Please review inputs, then run the simulation.",
        f"上传输入已应用成功（更新 {updated_count} 个字段）。请检查输入后再运行模拟。",
    )


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

def contribution_events_to_records(events_df, household_mode="Two People"):
    clean_df = normalise_contribution_events(
        events_df,
        household_mode=household_mode,
    )
    if clean_df.empty:
        return []
    return clean_df.to_dict(orient="records")


def get_default_contribution_events_df():
    return pd.DataFrame(
        columns=["financial_year", "person", "contribution_type", "amount"]
    )


def is_one_person_inputs(inputs):
    return str(inputs.get("household_mode", "Two People")) == "One Person"


def drop_person2_columns_if_single(df, inputs):
    if df is None:
        return df
    if not is_one_person_inputs(inputs):
        return df
    keep_cols = [col for col in df.columns if "P2" not in str(col)]
    return df[keep_cols].copy()


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
                "household_mode": scenario_inputs.get("household_mode", "Two People"),
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
                "cash_reserve_balance": scenario_inputs.get("cash_reserve_balance", 0.0),
                "cash_reserve_floor": scenario_inputs.get("cash_reserve_floor", 0.0),
                "cash_reserve_target": scenario_inputs.get("cash_reserve_target", 0.0),
                "non_deductible_debt_balance": scenario_inputs.get("non_deductible_debt_balance", 0.0),
                "non_deductible_interest_rate": scenario_inputs.get("non_deductible_interest_rate", 0.0),
                "non_deductible_offset_balance": scenario_inputs.get("non_deductible_offset_balance", 0.0),
                "deductible_offset_balance": scenario_inputs.get("deductible_offset_balance", 0.0),
                "surplus_allocation_order": " > ".join(scenario_inputs.get("surplus_allocation_order", [])),
                "withdrawal_order": " > ".join(scenario_inputs.get("withdrawal_order", [])),
                "non_super_estate_reserve": scenario_inputs.get("non_super_estate_reserve", 0.0),
                "property_estate_reserve": scenario_inputs.get("property_estate_reserve", 0.0),
                "cgt_reform_enabled": scenario_inputs.get("cgt_reform_enabled", True),
                "non_super_transition_value_2027": scenario_inputs.get("non_super_transition_value_2027", scenario_inputs["non_super_balance"]),
                "non_super_opening_capital_losses": scenario_inputs.get("non_super_opening_capital_losses", 0.0),
                "cgt_indexation_rate": scenario_inputs.get("cgt_indexation_rate", scenario_inputs.get("inflation_rate", 0.0)),
                "cgt_asset_category": scenario_inputs.get("cgt_asset_category", "Other"),
                "cgt_new_residential_method": scenario_inputs.get("cgt_new_residential_method", "Indexation and 30% minimum tax"),
                "residential_property_enabled": scenario_inputs.get("residential_property_enabled", False),
                "residential_property_value": scenario_inputs.get("residential_property_value", 0.0),
                "residential_property_loan_balance": scenario_inputs.get("residential_property_loan_balance", 0.0),
                "main_residence_value": scenario_inputs.get("main_residence_value", 0.0),
                "main_residence_loan_balance": scenario_inputs.get("main_residence_loan_balance", 0.0),
                "investment_deductible_debt_balance": scenario_inputs.get("investment_deductible_debt_balance", 0.0),
                "residential_property_gross_rent": scenario_inputs.get("residential_property_gross_rent", 0.0),
                "residential_property_operating_expenses": scenario_inputs.get("residential_property_operating_expenses", 0.0),
                "residential_property_interest_rate": scenario_inputs.get("residential_property_interest_rate", 0.0),
                "residential_property_acquired_before_budget_time": scenario_inputs.get("residential_property_acquired_before_budget_time", False),
                "residential_property_is_new_build": scenario_inputs.get("residential_property_is_new_build", False),
                "discretionary_trust_enabled": scenario_inputs.get("discretionary_trust_enabled", False),
                "discretionary_trust_balance": scenario_inputs.get("discretionary_trust_balance", 0.0),
                "discretionary_trust_cost_base": scenario_inputs.get("discretionary_trust_cost_base", 0.0),
                "discretionary_trust_excluded_income_pct": scenario_inputs.get("discretionary_trust_excluded_income_pct", 0.0),
                "discretionary_trust_subject_to_minimum_tax": scenario_inputs.get("discretionary_trust_subject_to_minimum_tax", True),
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
        "person1_division_293_tax": det_df["person1_division_293_tax"].sum() if "person1_division_293_tax" in det_df.columns else 0.0,
        "person1_total_personal_tax": det_df["person1_personal_tax_total"].sum() if "person1_personal_tax_total" in det_df.columns else 0.0,
        "person2_salary_tax_total": det_df["person2_salary_tax_total"].sum() if "person2_salary_tax_total" in det_df.columns else 0.0,
        "person2_non_super_tax_total": det_df["person2_non_super_tax_total"].sum() if "person2_non_super_tax_total" in det_df.columns else 0.0,
        "person2_division_293_tax": det_df["person2_division_293_tax"].sum() if "person2_division_293_tax" in det_df.columns else 0.0,
        "person2_total_personal_tax": det_df["person2_personal_tax_total"].sum() if "person2_personal_tax_total" in det_df.columns else 0.0,
        "total_super_contributions_tax": det_df["total_super_contributions_tax"].sum() if "total_super_contributions_tax" in det_df.columns else 0.0,
        "total_division_293_tax": det_df["total_division_293_tax"].sum() if "total_division_293_tax" in det_df.columns else 0.0,
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
        + safe_col("cash_reserve_withdrawal")
        + safe_col("non_super_withdrawal")
        + safe_col("residential_property_sale_proceeds")
        - safe_col("total_cash_contributions")
        - safe_col("non_deductible_scheduled_loan_payment")
        - safe_col("investment_deductible_scheduled_loan_payment")
        - safe_col("main_residence_scheduled_loan_payment")
        - safe_col("residential_property_scheduled_principal")
        - safe_col("non_deductible_principal_repayment")
        - safe_col("deductible_principal_repayment")
        - safe_col("investment_deductible_principal_repayment")
        - safe_col("main_residence_extra_principal_repayment")
        - safe_col("non_deductible_offset_contribution")
        - safe_col("deductible_offset_contribution")
        - safe_col("investment_deductible_offset_contribution")
        - safe_col("main_residence_offset_contribution")
        - safe_col("cash_reserve_top_up")
        - safe_col("surplus_cash_to_non_super")
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


def build_adviser_cashflow_asset_movement_tax_df(det_df, inputs):
    df = det_df.copy()

    def safe_col(name):
        return df[name] if name in df.columns else 0.0

    opening_net_assets = (
        safe_col("opening_non_super_balance")
        + safe_col("opening_cash_reserve_balance")
        + safe_col("opening_person1_accum_super_balance")
        + safe_col("opening_person1_pension_super_balance")
        + safe_col("opening_person2_accum_super_balance")
        + safe_col("opening_person2_pension_super_balance")
        + safe_col("opening_residential_property_value")
        + safe_col("opening_main_residence_value")
        + safe_col("opening_discretionary_trust_balance")
        + safe_col("opening_non_deductible_offset_balance")
        + safe_col("opening_deductible_offset_balance")
        + safe_col("opening_main_residence_offset_balance")
        + safe_col("opening_investment_deductible_offset_balance")
        - safe_col("opening_residential_property_loan_balance")
        - safe_col("opening_main_residence_loan_balance")
        - safe_col("opening_non_deductible_debt_balance")
        - safe_col("opening_investment_deductible_debt_balance")
    )

    closing_net_assets = safe_col("total_wealth")
    investment_earnings = (
        safe_col("non_super_earnings")
        + safe_col("person1_accum_earnings")
        + safe_col("person1_pension_earnings")
        + safe_col("person2_accum_earnings")
        + safe_col("person2_pension_earnings")
    )
    total_withdrawals = (
        safe_col("non_super_withdrawal")
        + safe_col("total_minimum_pension_drawdown")
        + safe_col("total_extra_super_withdrawal")
    )
    total_income_tax = (
        safe_col("person1_personal_tax_total")
        + safe_col("person2_personal_tax_total")
        + safe_col("total_division_293_tax")
        + safe_col("total_discretionary_trust_minimum_tax")
    )

    movement_df = pd.DataFrame({
        "Year": safe_col("financial_year_end"),
        "Opening Net Assets": opening_net_assets,
        "Employment Income": safe_col("household_gross_income"),
        "Investment Earnings": investment_earnings,
        "Residential Property Net Cashflow": safe_col("residential_property_net_cashflow"),
        "Discretionary Trust Net Income": safe_col("discretionary_trust_net_income"),
        "Total Income": safe_col("household_gross_income") + investment_earnings,
        "Household Spending": safe_col("spending"),
        "Cash Contributions": safe_col("total_cash_contributions"),
        "Non-deductible Interest": safe_col("non_deductible_debt_interest"),
        "Other Deductible Investment Interest": safe_col("investment_deductible_debt_interest"),
        "Main Residence Loan Payment": safe_col("main_residence_scheduled_loan_payment"),
        "Other Non-deductible Investment Loan Payment": safe_col("non_deductible_scheduled_loan_payment"),
        "Other Deductible Investment Loan Payment": safe_col("investment_deductible_scheduled_loan_payment"),
        "Investment Property Scheduled Principal": safe_col("residential_property_scheduled_principal"),
        "Non-deductible Principal Repayment": safe_col("non_deductible_principal_repayment"),
        "Deductible Principal Repayment": safe_col("deductible_principal_repayment"),
        "Other Deductible Investment Principal Repayment": safe_col("investment_deductible_principal_repayment"),
        "Main Residence Extra Principal Repayment": safe_col("main_residence_extra_principal_repayment"),
        "Non-deductible Offset Contribution": safe_col("non_deductible_offset_contribution"),
        "Deductible Offset Contribution": safe_col("deductible_offset_contribution"),
        "Minimum Pension Drawdown": safe_col("total_minimum_pension_drawdown"),
        "Non-Super Withdrawal": safe_col("non_super_withdrawal"),
        "Extra Super Withdrawal": safe_col("total_extra_super_withdrawal"),
        "Total Withdrawals": total_withdrawals,
        "P1 Total Income Tax": safe_col("person1_personal_tax_total"),
        "P1 Division 293 Tax": safe_col("person1_division_293_tax"),
        "P2 Total Income Tax": safe_col("person2_personal_tax_total"),
        "P2 Division 293 Tax": safe_col("person2_division_293_tax"),
        "CGT Minimum-Tax Top-Up": safe_col("total_cgt_minimum_tax"),
        "Trustee Minimum Tax (Draft)": safe_col("total_discretionary_trust_minimum_tax"),
        "Closing Residential Property Equity": safe_col("residential_property_net_equity"),
        "Closing Main Residence Equity": safe_col("main_residence_net_equity"),
        "Closing Trust Balance": safe_col("ending_discretionary_trust_balance"),
        "Total Income Tax Per Household": total_income_tax,
        "Super Contributions Tax": safe_col("total_super_contributions_tax"),
        "Super Earnings Tax": safe_col("total_super_earnings_tax"),
        "Super Withdrawal CGT Tax": safe_col("total_super_withdrawal_cgt_tax"),
        "Total Tax Paid": safe_col("total_tax_paid"),
        "Surplus Cash to Non-Super": safe_col("surplus_cash_to_non_super"),
        "Closing Non-deductible Debt": safe_col("non_deductible_debt_balance"),
        "Closing Deductible Debt": safe_col("residential_property_loan_balance"),
        "Closing Main Residence Loan": safe_col("main_residence_loan_balance"),
        "Closing Other Deductible Investment Debt": safe_col("investment_deductible_debt_balance"),
        "Closing Non-deductible Offset": safe_col("non_deductible_offset_balance"),
        "Closing Deductible Offset": safe_col("deductible_offset_balance"),
        "Unmet Shortfall": safe_col("unmet_shortfall"),
        "Closing Net Assets": closing_net_assets,
        "Net Asset Movement": closing_net_assets - opening_net_assets,
    })

    if is_one_person_inputs(inputs):
        movement_df = movement_df.drop(
            columns=["P2 Total Income Tax", "P2 Division 293 Tax"],
            errors="ignore",
        )

    return movement_df


def build_residential_trust_tax_detail_df(det_df):
    df = det_df.copy()

    def safe_col(name):
        return df[name] if name in df.columns else 0.0

    return pd.DataFrame({
        "Year": safe_col("financial_year_end"),
        "Property Restriction Applies": safe_col("residential_property_restriction_applies"),
        "Gross Rent": safe_col("residential_property_gross_rent"),
        "Property Deductions": safe_col("residential_property_total_deductions"),
        "Property Net Cashflow": safe_col("residential_property_net_cashflow"),
        "Taxable Rental Income": safe_col("residential_property_taxable_income"),
        "Current Quarantined Loss": safe_col("residential_property_current_year_quarantined_loss"),
        "Quarantined Loss Used": safe_col("residential_property_quarantined_loss_used"),
        "Closing Quarantined Loss": safe_col("closing_residential_property_quarantined_loss"),
        "Property Net Equity": safe_col("residential_property_net_equity"),
        "Opening Trust Balance": safe_col("opening_discretionary_trust_balance"),
        "Opening Trust Cost Base": safe_col("opening_discretionary_trust_cost_base"),
        "Trust Net Income": safe_col("discretionary_trust_net_income"),
        "Trust Excluded Income": safe_col("discretionary_trust_excluded_income"),
        "Trust Excluded Income %": safe_col("discretionary_trust_excluded_income_pct"),
        "Trust Minimum-Tax Income": safe_col("discretionary_trust_minimum_tax_income"),
        "Trustee Minimum Tax (Draft)": safe_col("discretionary_trust_trustee_minimum_tax"),
        "P1 Trust Credit": safe_col("person1_trust_tax_credit"),
        "P2 Trust Credit": safe_col("person2_trust_tax_credit"),
        "Closing Trust Balance": safe_col("ending_discretionary_trust_balance"),
        "Closing Trust Cost Base": safe_col("ending_discretionary_trust_cost_base"),
    })


def build_cgt_validation_df(det_df, inputs):
    df = det_df.copy()

    def safe_col(name):
        return df[name] if name in df.columns else 0.0

    validation_df = pd.DataFrame(
        {
            "Year": df["financial_year_end"] if "financial_year_end" in df.columns else range(len(df)),
            "P1 Age": safe_col("person1_age"),
            "P1 Started Pension This Year": safe_col("person1_started_pension_this_year"),
            "P1 Has Started Pension": safe_col("person1_has_started_pension"),
            "P1 Opening Accum Balance": safe_col("opening_person1_accum_super_balance"),
            "P1 Opening Pension Balance": safe_col("opening_person1_pension_super_balance"),
            "P1 Ending Accum Balance": safe_col("ending_person1_accum_super_balance"),
            "P1 Ending Pension Balance": safe_col("ending_person1_pension_super_balance"),
            "P1 Transfer to Pension": safe_col("person1_transfer_to_pension"),
            "P1 Total Net Super Contribution": safe_col("person1_total_net_super_contribution"),
            "P1 Super Realised CGT": safe_col("person1_super_realised_capital_gain"),
            "Non-Super Realised Gain": safe_col("non_super_realised_capital_gain"),
            "Deferred Pre-2027 Gain": safe_col("non_super_deferred_pre_2027_gain"),
            "Post-2027 Real Gain": safe_col("non_super_post_2027_real_gain"),
            "Taxable Non-Super Gain": safe_col("non_super_discounted_taxable_capital_gain"),
            "Minimum-Tax Capital Gain": safe_col("non_super_minimum_tax_capital_gain"),
            "CGT Minimum-Tax Top-Up": safe_col("total_cgt_minimum_tax"),
            "Indexation Uplift": safe_col("non_super_indexation_uplift"),
            "Capital Losses Applied": safe_col("non_super_capital_losses_applied"),
            "Closing Capital Losses": safe_col("ending_non_super_capital_losses"),
            "Closing Indexed Cost Base": safe_col("ending_non_super_indexed_cost_base"),
            "CGT Calculation Method": safe_col("non_super_cgt_calculation_method"),
            "Super Withdrawal CGT Tax": safe_col("total_super_withdrawal_cgt_tax"),
            "P1 Pension Earnings Tax": safe_col("person1_pension_earnings_tax"),
            "Super Earnings Tax": safe_col("total_super_earnings_tax"),
        }
    )

    if not is_one_person_inputs(inputs):
        validation_df.insert(2, "P2 Age", safe_col("person2_age"))
        validation_df.insert(4, "P2 Started Pension This Year", safe_col("person2_started_pension_this_year"))
        validation_df.insert(6, "P2 Has Started Pension", safe_col("person2_has_started_pension"))
        validation_df["P2 Opening Accum Balance"] = safe_col("opening_person2_accum_super_balance")
        validation_df["P2 Opening Pension Balance"] = safe_col("opening_person2_pension_super_balance")
        validation_df["P2 Ending Accum Balance"] = safe_col("ending_person2_accum_super_balance")
        validation_df["P2 Ending Pension Balance"] = safe_col("ending_person2_pension_super_balance")
        validation_df["P2 Transfer to Pension"] = safe_col("person2_transfer_to_pension")
        validation_df["P2 Total Net Super Contribution"] = safe_col("person2_total_net_super_contribution")
        validation_df["P2 Super Realised CGT"] = safe_col("person2_super_realised_capital_gain")
        validation_df["P2 Pension Earnings Tax"] = safe_col("person2_pension_earnings_tax")

    return validation_df


def build_pension_tax_free_summary_df(det_df, inputs):
    df = det_df.copy()

    def safe_col(name):
        return df[name] if name in df.columns else 0.0

    def sum_all(name):
        return df[name].sum() if name in df.columns else 0.0

    pension_phase_mask = (
        (safe_col("ending_person1_pension_super_balance") > 0)
        | (safe_col("person1_transfer_to_pension") > 0)
        | (safe_col("person1_has_started_pension") > 0)
    )
    if not is_one_person_inputs(inputs):
        pension_phase_mask = pension_phase_mask | (
            (safe_col("ending_person2_pension_super_balance") > 0)
            | (safe_col("person2_transfer_to_pension") > 0)
            | (safe_col("person2_has_started_pension") > 0)
        )

    pension_phase_df = df.loc[pension_phase_mask].copy()

    def sum_pension_phase(name):
        return pension_phase_df[name].sum() if name in pension_phase_df.columns else 0.0

    rows = [
        {"check": "Total super withdrawal CGT tax (all years)", "value": sum_all("total_super_withdrawal_cgt_tax")},
        {"check": "Total super earnings tax (all years)", "value": sum_all("total_super_earnings_tax")},
        {"check": "Total super earnings tax (pension phase only)", "value": sum_pension_phase("total_super_earnings_tax")},
        {"check": "Total P1 pension earnings tax", "value": sum_all("person1_pension_earnings_tax")},
        {"check": "Total P1 transfer to pension", "value": sum_all("person1_transfer_to_pension")},
    ]
    if not is_one_person_inputs(inputs):
        rows.extend([
            {"check": "Total P2 pension earnings tax", "value": sum_all("person2_pension_earnings_tax")},
            {"check": "Total P2 transfer to pension", "value": sum_all("person2_transfer_to_pension")},
        ])

    return pd.DataFrame(rows)


def render_assumption_details(df):
    display_df = format_assumption_display_df(df)
    all_one_person = ("household_mode" in display_df.columns) and display_df["household_mode"].eq("One Person").all()
    if all_one_person:
        display_df = display_df[[col for col in display_df.columns if "person2_" not in str(col).lower()]].copy()

    st.subheader(t("Assumption Details", "假设明细"))
    st.dataframe(
        display_df,
        width="stretch",
        column_config={
            "scenario": st.column_config.TextColumn("Scenario"),
            "report_title": st.column_config.TextColumn("Title"),
            "household_mode": st.column_config.TextColumn("Household Mode"),
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
    total_warning_count = sum(
        len(warnings_list)
        for warnings_list in list(input_warnings_by_scenario.values()) + list(output_warnings_by_scenario.values())
    )
    if total_warning_count == 0:
        return

    expander_label = t(
        f"Warnings and review notes ({total_warning_count})",
        f"警告与审阅事项（{total_warning_count}）",
    )

    with st.expander(expander_label, expanded=False):
        if view_mode == "Adviser View":
            for scenario_name, warnings_list in input_warnings_by_scenario.items():
                if warnings_list:
                    st.subheader(t(f"Input Warnings - {scenario_name}", f"输入警告 - {scenario_name}"))
                    for warning in warnings_list:
                        st.warning(warning)

            for scenario_name, warnings_list in output_warnings_by_scenario.items():
                if warnings_list:
                    st.subheader(t(f"Result Warnings - {scenario_name}", f"结果警告 - {scenario_name}"))
                    for warning in warnings_list:
                        st.warning(warning)
        else:
            client_messages = []

            for warnings_list in input_warnings_by_scenario.values():
                client_messages.extend(warnings_list)

            for warnings_list in output_warnings_by_scenario.values():
                client_messages.extend(warnings_list)

            st.subheader(t("Important Notes", "重要提示"))
            for message in client_messages:
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


@st.cache_data(show_spinner=False, max_entries=20)
def run_scenario_cached(scenario_inputs, random_seed, cache_version="performance_v2"):
    # Streamlit caches this by input content, so switching view/language/PV/FV does not recalculate.
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

    return {
        "inputs": scenario_inputs,
        "det_df": det_df,
        "summary_df": summary_df,
        # Deliberately do not keep all_paths_df in session; it is large and only needed to build summaries.
        "all_paths_df": None,
        "percentile_df": percentile_df,
        "failure_prob_df": failure_prob_df,
        "success_rate": success_rate,
        "median_final_wealth": median_final_wealth,
        "p10_final_wealth": p10_final_wealth,
        "p90_final_wealth": p90_final_wealth,
    }


def render_light_input_badges(base_inputs):
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
    st.caption(t("Each saved snapshot keeps its own household mode, assumptions, and displayed value basis.", "每个已保存快照都会保留各自的家庭模式、假设以及显示数值口径。"))

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
        width="stretch",
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
    st.plotly_chart(fig, width="stretch", key="saved_results_compare_chart")


def get_missing_validation_columns(det_df, inputs):
    required_cols = [
        "financial_year_end",
        "person1_age",
        "person1_started_pension_this_year",
        "person1_has_started_pension",
        "opening_person1_accum_super_balance",
        "opening_person1_pension_super_balance",
        "ending_person1_accum_super_balance",
        "ending_person1_pension_super_balance",
        "person1_transfer_to_pension",
        "person1_total_net_super_contribution",
        "person1_super_realised_capital_gain",
        "total_super_withdrawal_cgt_tax",
        "person1_pension_earnings_tax",
        "total_super_earnings_tax",
    ]
    if not is_one_person_inputs(inputs):
        required_cols.extend([
            "person2_age",
            "person2_started_pension_this_year",
            "person2_has_started_pension",
            "opening_person2_accum_super_balance",
            "opening_person2_pension_super_balance",
            "ending_person2_accum_super_balance",
            "ending_person2_pension_super_balance",
            "person2_transfer_to_pension",
            "person2_total_net_super_contribution",
            "person2_super_realised_capital_gain",
            "person2_pension_earnings_tax",
        ])
    return [col for col in required_cols if col not in det_df.columns]


def build_adviser_debug_df(det_df, inputs):
    df = det_df.copy()

    def safe_col(name):
        return df[name] if name in df.columns else 0.0

    debug_df = pd.DataFrame(
        {
            "Year": safe_col("financial_year_end"),
            "P1 Started Pension This Year": safe_col("person1_started_pension_this_year"),
            "P1 Has Started Pension": safe_col("person1_has_started_pension"),
            "P1 Requested Transfer": safe_col("person1_requested_transfer_amount"),
            "P1 Available Cap Space": safe_col("person1_available_cap_space"),
            "P1 Transfer to Pension": safe_col("person1_transfer_to_pension"),
            "P1 Excess Retained in Accum": safe_col("person1_excess_retained_in_accumulation"),
            "P1 Opening Accum": safe_col("opening_person1_accum_super_balance"),
            "P1 Opening Pension": safe_col("opening_person1_pension_super_balance"),
            "P1 Net Super Contribution": safe_col("person1_total_net_super_contribution"),
            "P1 Ending Accum": safe_col("ending_person1_accum_super_balance"),
            "P1 Ending Pension": safe_col("ending_person1_pension_super_balance"),
            "P1 Min Pension Drawdown": safe_col("person1_min_pension_drawdown"),
            "P1 Extra Pension Withdrawal": safe_col("person1_extra_pension_withdrawal"),
            "P1 Pension Earnings Tax": safe_col("person1_pension_earnings_tax"),
            "P1 Super Realised CGT": safe_col("person1_super_realised_capital_gain"),
            "Super Withdrawal CGT Tax": safe_col("total_super_withdrawal_cgt_tax"),
            "Total Super Earnings Tax": safe_col("total_super_earnings_tax"),
        }
    )

    if not is_one_person_inputs(inputs):
        insert_after = list(debug_df.columns).index("P1 Has Started Pension") + 1
        debug_df.insert(insert_after, "P2 Started Pension This Year", safe_col("person2_started_pension_this_year"))
        debug_df.insert(insert_after + 1, "P2 Has Started Pension", safe_col("person2_has_started_pension"))
        debug_df["P2 Requested Transfer"] = safe_col("person2_requested_transfer_amount")
        debug_df["P2 Available Cap Space"] = safe_col("person2_available_cap_space")
        debug_df["P2 Transfer to Pension"] = safe_col("person2_transfer_to_pension")
        debug_df["P2 Excess Retained in Accum"] = safe_col("person2_excess_retained_in_accumulation")
        debug_df["P2 Opening Accum"] = safe_col("opening_person2_accum_super_balance")
        debug_df["P2 Opening Pension"] = safe_col("opening_person2_pension_super_balance")
        debug_df["P2 Net Super Contribution"] = safe_col("person2_total_net_super_contribution")
        debug_df["P2 Ending Accum"] = safe_col("ending_person2_accum_super_balance")
        debug_df["P2 Ending Pension"] = safe_col("ending_person2_pension_super_balance")
        debug_df["P2 Min Pension Drawdown"] = safe_col("person2_min_pension_drawdown")
        debug_df["P2 Extra Pension Withdrawal"] = safe_col("person2_extra_pension_withdrawal")
        debug_df["P2 Pension Earnings Tax"] = safe_col("person2_pension_earnings_tax")
        debug_df["P2 Super Realised CGT"] = safe_col("person2_super_realised_capital_gain")

    return debug_df


# ============================================================
# SECTION: PAGE SETUP
# ============================================================

st.set_page_config(page_title="Retirement Modelling Suite (AU)", page_icon="📊", layout="wide")
inject_jbwere_styles()

if "ui_language" not in st.session_state:
    st.session_state.ui_language = LANGUAGE_EN


# ============================================================
# SECTION: UPLOAD LIFECYCLE FIX
# ============================================================

if "pending_uploaded_assumption_preset" in st.session_state:
    pending_preset = st.session_state.pop(
        "pending_uploaded_assumption_preset"
    )

    if pending_preset in [
        "Conservative",
        "Base Case",
        "Optimistic",
        "Custom",
    ]:
        st.session_state["assumption_preset"] = pending_preset

if "pending_uploaded_module_values" in st.session_state:
    for module_key, module_value in st.session_state.pop("pending_uploaded_module_values").items():
        st.session_state[module_key] = bool(module_value)


# ============================================================
# SECTION: SESSION DEFAULTS
# ============================================================

defaults = {
    "comparison_results": None,
    "saved_result_sets": {},
    "active_result_set_name": "Current Results",
    "workspace_mode": "Edit Inputs",
    "view_mode": "Adviser View",
    "scenario_mode": "Single Scenario",
    "show_assumption_panel": False,
    "show_live_input_checks": False,
    "simulation_depth": "Standard",
    "adviser_result_section": "Overview",
    "prepare_excel_export": False,
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
    **MODULE_DEFAULTS,
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
    "cash_reserve_balance": 50000.0,
    "cash_reserve_floor": 10000.0,
    "cash_reserve_target": 50000.0,
    "main_residence_value": 1500000.0,
    "main_residence_capital_growth_rate": 0.03,
    "main_residence_loan_balance": 0.0,
    "main_residence_interest_rate": 0.06,
    "main_residence_annual_loan_repayment": 0.0,
    "main_residence_offset_balance": 0.0,
    "non_deductible_debt_balance": 0.0,
    "non_deductible_interest_rate": 0.06,
    "non_deductible_annual_repayment": 0.0,
    "non_deductible_offset_balance": 0.0,
    "deductible_offset_balance": 0.0,
    "investment_deductible_debt_balance": 0.0,
    "investment_deductible_interest_rate": 0.06,
    "investment_deductible_annual_repayment": 0.0,
    "investment_deductible_offset_balance": 0.0,
    "surplus_allocation_profile": "Non-deductible Offset > Non-deductible Debt > Deductible Offset > Deductible Debt > Invest",
    "surplus_allocation_order": ["non_deductible_offset", "non_deductible_repayment", "deductible_offset", "deductible_repayment", "non_super"],
    "non_super_estate_reserve": 0.0,
    "property_estate_reserve": 0.0,
    "residential_property_sale_cost_rate": 0.025,
    "withdrawal_order": ["cash", "non_super", "pension", "accumulation", "property"],
    "drawdown_profile": "Cash > Non-super > Pension > Accumulation > Property",
    "cgt_reform_enabled": True,
    "cgt_asset_acquired_before_2027": True,
    "non_super_transition_value_2027": 500000.0,
    "non_super_opening_capital_losses": 0.0,
    "cgt_indexation_rate": 0.03,
    "cgt_asset_category": "Other",
    "cgt_new_residential_method": "Indexation and 30% minimum tax",
    "cgt_held_at_least_12_months": True,
    "cgt_minimum_tax_exempt": False,
    "residential_property_enabled": False,
    "residential_property_value": 0.0,
    "residential_property_loan_balance": 0.0,
    "residential_property_annual_loan_repayment": 0.0,
    "residential_property_gross_rent": 0.0,
    "residential_property_operating_expenses": 0.0,
    "residential_property_interest_rate": 0.06,
    "residential_property_capital_growth_rate": 0.03,
    "residential_property_rent_growth_rate": 0.03,
    "residential_property_expense_growth_rate": 0.03,
    "residential_property_opening_quarantined_loss": 0.0,
    "residential_property_acquired_before_budget_time": False,
    "residential_property_is_new_build": False,
    "residential_property_is_exempt_housing": False,
    "residential_property_ownership_person1_pct": 50.0,
    "discretionary_trust_enabled": False,
    "discretionary_trust_net_income": 0.0,
    "discretionary_trust_excluded_income": 0.0,
    "discretionary_trust_income_growth_rate": 0.03,
    "discretionary_trust_balance": 0.0,
    "discretionary_trust_cost_base": 0.0,
    "discretionary_trust_income_return_mean": 0.02,
    "discretionary_trust_income_return_std": 0.02,
    "discretionary_trust_capital_return_mean": 0.03,
    "discretionary_trust_capital_return_std": 0.08,
    "discretionary_trust_excluded_income_pct": 0.0,
    "discretionary_trust_subject_to_minimum_tax": True,
    "discretionary_trust_ownership_person1_pct": 50.0,
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
    "adviser_notes": "",
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
workspace_mode = st.session_state.workspace_mode
show_assumption_panel = bool(st.session_state.show_assumption_panel)
show_live_input_checks = bool(st.session_state.show_live_input_checks)
is_one_person_mode = household_mode == "One Person"
module_second_person_enabled = bool(st.session_state.module_second_person_enabled)
module_super_enabled = bool(st.session_state.module_super_enabled)
module_pension_enabled = bool(st.session_state.module_pension_enabled) and module_super_enabled
module_non_super_enabled = bool(st.session_state.module_non_super_enabled)
module_property_enabled = bool(st.session_state.module_property_enabled)
module_trust_enabled = bool(st.session_state.module_trust_enabled)
module_cash_surplus_enabled = bool(st.session_state.module_cash_surplus_enabled)
module_investment_debt_enabled = bool(st.session_state.module_investment_debt_enabled)
module_non_deductible_debt_enabled = module_investment_debt_enabled
module_deductible_debt_enabled = module_investment_debt_enabled

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
cash_reserve_balance = st.session_state.cash_reserve_balance
cash_reserve_floor = st.session_state.cash_reserve_floor
cash_reserve_target = st.session_state.cash_reserve_target
main_residence_value = st.session_state.main_residence_value
main_residence_capital_growth_rate = st.session_state.main_residence_capital_growth_rate
main_residence_loan_balance = st.session_state.main_residence_loan_balance
main_residence_interest_rate = st.session_state.main_residence_interest_rate
main_residence_annual_loan_repayment = st.session_state.main_residence_annual_loan_repayment
main_residence_offset_balance = st.session_state.main_residence_offset_balance
non_deductible_debt_balance = st.session_state.non_deductible_debt_balance
non_deductible_interest_rate = st.session_state.non_deductible_interest_rate
non_deductible_annual_repayment = st.session_state.non_deductible_annual_repayment
non_deductible_offset_balance = st.session_state.non_deductible_offset_balance
deductible_offset_balance = st.session_state.deductible_offset_balance
investment_deductible_debt_balance = st.session_state.investment_deductible_debt_balance
investment_deductible_interest_rate = st.session_state.investment_deductible_interest_rate
investment_deductible_annual_repayment = st.session_state.investment_deductible_annual_repayment
investment_deductible_offset_balance = st.session_state.investment_deductible_offset_balance
surplus_allocation_profile = st.session_state.surplus_allocation_profile
surplus_allocation_order = list(st.session_state.surplus_allocation_order)
non_super_estate_reserve = st.session_state.non_super_estate_reserve
property_estate_reserve = st.session_state.property_estate_reserve
residential_property_sale_cost_rate = st.session_state.residential_property_sale_cost_rate
drawdown_profile = st.session_state.drawdown_profile
withdrawal_order = list(st.session_state.withdrawal_order)
cgt_reform_enabled = st.session_state.cgt_reform_enabled
cgt_asset_acquired_before_2027 = st.session_state.cgt_asset_acquired_before_2027
non_super_transition_value_2027 = st.session_state.non_super_transition_value_2027
non_super_opening_capital_losses = st.session_state.non_super_opening_capital_losses
cgt_indexation_rate = st.session_state.cgt_indexation_rate
cgt_asset_category = st.session_state.cgt_asset_category
cgt_new_residential_method = st.session_state.cgt_new_residential_method
cgt_held_at_least_12_months = st.session_state.cgt_held_at_least_12_months
cgt_minimum_tax_exempt = st.session_state.cgt_minimum_tax_exempt
residential_property_enabled = st.session_state.residential_property_enabled
residential_property_value = st.session_state.residential_property_value
residential_property_loan_balance = st.session_state.residential_property_loan_balance
residential_property_annual_loan_repayment = st.session_state.residential_property_annual_loan_repayment
residential_property_gross_rent = st.session_state.residential_property_gross_rent
residential_property_operating_expenses = st.session_state.residential_property_operating_expenses
residential_property_interest_rate = st.session_state.residential_property_interest_rate
residential_property_capital_growth_rate = st.session_state.residential_property_capital_growth_rate
residential_property_rent_growth_rate = st.session_state.residential_property_rent_growth_rate
residential_property_expense_growth_rate = st.session_state.residential_property_expense_growth_rate
residential_property_opening_quarantined_loss = st.session_state.residential_property_opening_quarantined_loss
residential_property_acquired_before_budget_time = st.session_state.residential_property_acquired_before_budget_time
residential_property_is_new_build = st.session_state.residential_property_is_new_build
residential_property_is_exempt_housing = st.session_state.residential_property_is_exempt_housing
residential_property_ownership_person1 = st.session_state.residential_property_ownership_person1_pct
discretionary_trust_enabled = st.session_state.discretionary_trust_enabled
discretionary_trust_balance = st.session_state.discretionary_trust_balance
discretionary_trust_cost_base = st.session_state.discretionary_trust_cost_base
discretionary_trust_income_return_mean = st.session_state.discretionary_trust_income_return_mean
discretionary_trust_income_return_std = st.session_state.discretionary_trust_income_return_std
discretionary_trust_capital_return_mean = st.session_state.discretionary_trust_capital_return_mean
discretionary_trust_capital_return_std = st.session_state.discretionary_trust_capital_return_std
discretionary_trust_excluded_income_pct = st.session_state.discretionary_trust_excluded_income_pct
discretionary_trust_net_income = st.session_state.discretionary_trust_net_income
discretionary_trust_excluded_income = st.session_state.discretionary_trust_excluded_income
discretionary_trust_income_growth_rate = st.session_state.discretionary_trust_income_growth_rate
discretionary_trust_subject_to_minimum_tax = st.session_state.discretionary_trust_subject_to_minimum_tax
discretionary_trust_ownership_person1 = st.session_state.discretionary_trust_ownership_person1_pct
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

    st.radio(
        t("Language", "语言"),
        options=[LANGUAGE_EN, LANGUAGE_CN],
        key="ui_language",
    )

    if st.button(
        t("Clear This Session", "清空本次会话"),
        width="stretch",
        help=t(
            "Remove all current inputs, uploaded data, simulations and saved result snapshots from this browser session.",
            "清除本次浏览器会话中的所有输入、上传数据、模拟结果及已保存结果快照。",
        ),
    ):
        retained_language = st.session_state.ui_language
        st.session_state.clear()
        st.session_state.ui_language = retained_language
        st.rerun()

    with st.expander(t("Privacy for this session", "本次会话隐私说明"), expanded=False):
        st.caption(t(
            "Inputs and uploaded workbooks are processed for this browser session only. The app has no customer database and does not intentionally retain your data after the session is cleared, refreshed or disconnected. Do not enter identifying information that is unnecessary for the calculation.",
            "输入内容及上传工作簿仅用于本次浏览器会话。本应用不设客户数据库；清空、刷新或断开会话后，不会有意保留你的数据。请勿输入计算并不需要的身份识别信息。",
        ))

    view_mode_options = ["Adviser View", "Client View"]
    view_mode_labels = {
        "Adviser View": t("Adviser View", "顾问视图"),
        "Client View": t("Client View", "客户视图"),
    }
    stored_view_mode = st.session_state.get("view_mode", "Adviser View")
    if stored_view_mode not in view_mode_options:
        stored_view_mode = "Adviser View"
    view_mode_label = st.radio(
        t("View Mode", "视图模式"),
        options=[view_mode_labels[option] for option in view_mode_options],
        index=view_mode_options.index(stored_view_mode),
    )
    view_mode = view_mode_options[
        [view_mode_labels[option] for option in view_mode_options].index(view_mode_label)
    ]
    st.session_state.view_mode = view_mode

    with st.expander(t("Client Modules", "客户模块"), expanded=False):
        st.caption(t(
            "Untick modules that do not apply. Their saved inputs are retained but excluded from calculations, validation, navigation and reports.",
            "取消不适用的模块。已保存输入会保留，但不会进入计算、验证、页面导航和报告。",
        ))
        module_second_person_enabled = st.checkbox(
            t("Second household member", "第二位家庭成员"),
            key="module_second_person_enabled",
        )
        module_super_enabled = st.checkbox(
            t("Super accumulation", "Super 累积账户"),
            key="module_super_enabled",
        )
        if not module_super_enabled:
            st.session_state.module_pension_enabled = False
        module_pension_enabled = st.checkbox(
            t("Pension accounts", "Pension 账户"),
            key="module_pension_enabled",
            disabled=not module_super_enabled,
        )
        module_non_super_enabled = st.checkbox(
            t("Non-super investments", "非 Super 投资"),
            key="module_non_super_enabled",
        )
        module_property_enabled = st.checkbox(
            t("Residential investment property", "住宅投资物业"),
            key="module_property_enabled",
        )
        module_trust_enabled = st.checkbox(
            t("Discretionary trust", "Discretionary Trust"),
            key="module_trust_enabled",
        )
        module_cash_surplus_enabled = st.checkbox(
            t("Cash surplus strategy", "现金盈余策略"),
            key="module_cash_surplus_enabled",
        )
        module_investment_debt_enabled = st.checkbox(
            t("Investment debt", "投资债务"),
            key="module_investment_debt_enabled",
            help=t(
                "Covers other deductible and non-deductible investment borrowing. Main-residence and residential-investment-property loans are entered on the Property page.",
                "用于其他可抵扣及不可抵扣投资借款。主住宅及住宅投资物业贷款在“物业”页面输入。",
            ),
        )
        module_non_deductible_debt_enabled = module_investment_debt_enabled
        module_deductible_debt_enabled = module_investment_debt_enabled

    household_mode = "Two People" if module_second_person_enabled else "One Person"
    is_one_person_mode = not module_second_person_enabled
    st.caption(t(
        f"Active modules: {len(active_module_names(st.session_state, is_chinese=is_cn()))}",
        f"已启用模块：{len(active_module_names(st.session_state, is_chinese=is_cn()))}",
    ))

    value_mode = st.radio(
        t("Value Display", "数值显示"),
        options=["Future Value", "Present Value"],
        index=0 if st.session_state.value_mode == "Future Value" else 1,
        format_func=lambda option: {
            "Future Value": t("Future Value", "终值"),
            "Present Value": t("Present Value", "现值"),
        }[option],
        help=t(
            "Present Value discounts displayed monetary outputs back to the starting financial year using the inflation assumption.",
            "现值会按 inflation 假设把显示金额折算回起始财政年度。",
        ),
    )

    workspace_options = ["Edit Inputs", "View Results"]
    workspace_mode_label = st.radio(
        t("Workspace", "工作区"),
        options=workspace_options,
        index=0 if st.session_state.workspace_mode == "Edit Inputs" else 1,
        format_func=lambda option: {
            "Edit Inputs": t("Edit Inputs", "编辑输入"),
            "View Results": t("View Results", "查看结果"),
        }[option],
        help=t(
            "Use Edit Inputs for faster input changes. Switch to View Results when you want to review charts and tables.",
            "编辑输入时不会反复渲染大型图表和表格，速度更快。需要查看结果时切换到查看结果。",
        ),
    )
    workspace_mode = workspace_mode_label

    show_live_input_checks = st.checkbox(
        t("Live input checks", "即时输入检查"),
        value=bool(st.session_state.show_live_input_checks),
        help=t(
            "Turn this on only when you want full live validation while editing. Keeping it off makes input navigation faster.",
            "只在需要边输入边完整检查时打开。关闭可让输入区切换更快。",
        ),
    )

    show_assumption_panel = st.checkbox(
        t("Show assumption settings panel", "显示假设设置面板"),
        value=bool(st.session_state.show_assumption_panel),
        help=t(
            "Hide the editable preset table during normal input work to reduce rerun cost.",
            "日常输入时隐藏可编辑预设表，减少页面重跑开销。",
        ),
    )

    scenario_mode_options = [
        "Single Scenario",
        "Compare Standard Presets",
    ]
    if module_super_enabled or module_non_super_enabled or module_property_enabled:
        scenario_mode_options.append("Compare Asset Drawdown Strategies")
    if module_cash_surplus_enabled and (
        module_investment_debt_enabled
        or module_property_enabled
        or float(st.session_state.get("main_residence_loan_balance", 0.0)) > 0
    ):
        scenario_mode_options.append("Compare Debt Repayment Strategies")
    stored_scenario_mode = st.session_state.get("scenario_mode", "Single Scenario")
    if stored_scenario_mode not in scenario_mode_options:
        stored_scenario_mode = "Single Scenario"
    scenario_mode_labels = {
        "Single Scenario": t("Single Scenario", "单一情景"),
        "Compare Standard Presets": t("Compare Standard Presets", "比较标准预设"),
        "Compare Asset Drawdown Strategies": t("Compare Asset Drawdown Strategies", "比较资产提取策略"),
        "Compare Debt Repayment Strategies": t("Compare Debt Repayment Strategies", "比较债务偿还策略"),
    }
    scenario_mode_label = st.radio(
        t("Scenario Mode", "情景模式"),
        options=[scenario_mode_labels[option] for option in scenario_mode_options],
        index=scenario_mode_options.index(stored_scenario_mode),
    )
    scenario_mode = scenario_mode_options[
        [scenario_mode_labels[option] for option in scenario_mode_options].index(scenario_mode_label)
    ]
    st.session_state.scenario_mode = scenario_mode

    simulation_depth_options = ["Fast", "Standard", "Deep"]
    simulation_depth = st.radio(
        t("Simulation Depth", "模拟深度"),
        options=simulation_depth_options,
        index=simulation_depth_options.index(st.session_state.get("simulation_depth", "Standard")),
        format_func=lambda option: {
            "Fast": t("Fast", "快速"),
            "Standard": t("Standard", "标准"),
            "Deep": t("Deep", "深度"),
        }[option],
        help=t(
            "Fast is best while editing. Standard is suitable for review. Deep is slower and intended for final stress testing.",
            "编辑时建议用 Fast。Standard 适合复核。Deep 更慢，适合最终压力测试。",
        ),
    )
    st.session_state.simulation_depth = simulation_depth
    simulation_depth_map = {"Fast": 300, "Standard": 1000, "Deep": 3000}
    number_of_simulations = simulation_depth_map[simulation_depth]
    st.session_state.number_of_simulations = number_of_simulations
    st.caption(t(f"Monte Carlo simulations: {number_of_simulations:,}", f"蒙特卡洛模拟次数：{number_of_simulations:,}"))

    preset_choice = st.selectbox(
        t("Assumption Preset", "假设预设"),
        options=["Conservative", "Base Case", "Optimistic", "Custom"],
        format_func=lambda option: {
            "Conservative": t("Conservative", "保守情景"),
            "Base Case": t("Base Case", "基础情景"),
            "Optimistic": t("Optimistic", "乐观情景"),
            "Custom": t("Custom", "自定义"),
        }[option],
        key="assumption_preset",
        disabled=(scenario_mode != "Single Scenario"),
    )

    if is_one_person_mode:
        st.info(
            t(
                "One Person mode removes Person 2 from the model. Household spending remains exactly as entered, so review your spending assumptions manually.",
                "单人模式会把 Person 2 从模型中移除。家庭支出会保持原输入值不变，因此请手动检查支出假设。",
            )
        )

    st.markdown(f"### {t('Input Excel', '输入 Excel')}")
    st.caption(t(
        "XLSX only, maximum 10 MB. Workbooks with macros or external links are rejected.",
        "仅支持 XLSX，最大 10 MB；包含宏或外部链接的工作簿将被拒绝。",
    ))
    st.download_button(
        label=t("Download Excel Template", "下载 Excel 模板"),
        data=build_excel_input_workbook_bytes(include_current_values=False),
        file_name="financial_modelling_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        help=t(
            "Download a clean workbook template for entering model inputs offline.",
            "下载空白输入模板，便于离线填写模型输入。",
        ),
    )

    uploaded_input_file = st.file_uploader(
        t("Upload Input", "上传输入"),
        type=["xlsx"],
        key="input_excel_uploader",
        help=t(
            "Upload a completed input workbook. Existing results will be cleared after import.",
            "上传填写完成的输入工作簿。导入后当前结果会被清空。",
        ),
    )
    apply_uploaded_input_button = st.button(
        t("Apply Uploaded Input", "应用上传输入"),
        width="stretch",
        disabled=(uploaded_input_file is None),
    )

    st.download_button(
        label=t("Download Current Input", "下载当前输入"),
        data=build_excel_input_workbook_bytes(include_current_values=True),
        file_name="financial_modelling_current_input.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        help=t(
            "Download the current on-screen inputs, contribution schedule, and preset assumptions.",
            "下载当前页面输入、缴款计划和预设假设。",
        ),
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
        format_func=lambda option: t("Current Results", "最新结果") if option == "Current Results" else option,
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
        width="stretch",
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
            width="stretch",
        )

        delete_button = st.button(
            t("Delete Selected Saved Result", "删除当前已保存结果"),
            width="stretch",
        )
    else:
        rename_button = False
        delete_button = False

    run_button = st.button(t("Run Simulation", "运行模拟"), type="primary", width="stretch")


if apply_uploaded_input_button:
    ok, message = apply_uploaded_input_workbook(uploaded_input_file)
    (st.success if ok else st.error)(message)
    if ok:
        st.rerun()

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
st.subheader(t("Superannuation • Tax • CGT • Debt • Retirement Cashflow Modelling", "养老金 • 税务 • 资本利得税 • 债务 • 退休现金流建模"))
st.caption(
    t(
        "A professional financial modelling tool for analysing retirement outcomes, superannuation, tax, asset drawdown and debt strategies under Australian rules.",
        "一个用于分析澳大利亚退休结果、养老金、税务、资产提取及债务策略的专业金融建模工具。"
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
Use this tool to model retirement sustainability, super accumulation to pension transitions, Transfer Balance Cap constraints, tax impacts, asset drawdown order, and alternative debt repayment or cash-surplus strategies.""",
            """### 概览
本工具可用于建模退休可持续性、养老金从积累阶段转入退休金阶段、Transfer Balance Cap 限制、不同情景下的税务影响、资产提取顺序，以及不同债务偿还或现金盈余分配策略。"""
        )
    )

active_modules_display = active_module_names(st.session_state, is_chinese=is_cn())
with st.container(border=True):
    st.markdown(f"### {t('Active client scope', '当前客户范围')}")
    st.write(" · ".join(active_modules_display) if active_modules_display else t("Core household cashflow only", "仅核心家庭现金流"))
    st.caption(t(
        "Use Client Modules in the sidebar to include or remove optional parts of the model.",
        "可在侧边栏的“客户模块”中加入或移除可选模型部分。",
    ))


# ============================================================
# SECTION: ASSUMPTION SETTINGS PANEL
# ============================================================

runtime_presets = preset_table_to_dict(st.session_state.preset_table_df)
preset_table_df = st.session_state.preset_table_df.copy()

if show_assumption_panel:
    if st.button(
        t("Reset Preset Assumptions to Default", "将预设假设重置为默认值"),
        key="reset_preset_table_button_panel",
    ):
        st.session_state.preset_table_df = get_default_preset_table_df()
        st.rerun()

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
            width="stretch",
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
        preset_table_df = st.session_state.preset_table_df.copy()
        runtime_presets = preset_table_to_dict(st.session_state.preset_table_df)
else:
    st.caption(t(
        "Assumption Settings Panel is hidden for faster editing. Enable it in the sidebar when you need to edit preset assumptions.",
        "为提升输入速度，假设设置面板已隐藏。需要编辑预设假设时，可在侧边栏打开。",
    ))

# ============================================================
# SECTION: MAIN INPUT NAVIGATION
# ============================================================

section_keys = [
    "report",
    "projection",
    "person1",
    "household",
    "property_trust",
    "returns",
    "simulation",
]
if not is_one_person_mode:
    section_keys.insert(3, "person2")
if module_trust_enabled:
    section_keys.insert(-2, "trust")
if module_cash_surplus_enabled:
    section_keys.insert(-2, "cash_surplus")
if module_investment_debt_enabled:
    section_keys.insert(-2, "investment_debt")
if module_super_enabled:
    section_keys.insert(-2, "contributions")

section_labels = {
    "report": t("Report", "报告"),
    "projection": t("Projection", "预测设置"),
    "person1": t("Person 1", "人物 1"),
    "person2": t("Person 2", "人物 2"),
    "household": t("Household", "家庭"),
    "property_trust": t("Property", "物业"),
    "trust": t("Trust", "信托"),
    "cash_surplus": t("Cash Surplus", "现金盈余"),
    "investment_debt": t("Investment Debt", "投资债务"),
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
    "Property & Trust": "property_trust",
    "Property": "property_trust",
    "Trust": "trust",
    "Cash Surplus": "cash_surplus",
    "Investment Debt": "investment_debt",
    "Contributions": "contributions",
    "Returns": "returns",
    "Simulation": "simulation",
    "报告": "report",
    "预测设置": "projection",
    "人物 1": "person1",
    "人物 2": "person2",
    "家庭": "household",
    "住宅与信托": "property_trust",
    "物业": "property_trust",
    "信托": "trust",
    "现金盈余": "cash_surplus",
    "投资债务": "investment_debt",
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
    key="active_input_section_segmented",
)

active_input_section = next(
    k for k in section_keys
    if section_labels[k] == selected_section_label
)

with st.form("input_editor_form", clear_on_submit=False):
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
            if is_one_person_mode:
                retirement_spending_trigger = "Either Retired"
                st.text_input(
                    t("Retirement Spending Trigger", "退休支出触发条件"),
                    value=t("Person 1 Retired", "人物 1 退休后触发"),
                    disabled=True,
                    help=t(
                        "In one-person mode the retirement spending trigger is always based on Person 1 only.",
                        "单人模式下，退休支出触发条件始终只基于 Person 1。",
                    ),
                )
            else:
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
                min_value=18,
                max_value=100,
                value=min(max(int(st.session_state.person1_current_age), 18), 100),
                step=1,
                help=t("Current age at the start of the projection.", "预测开始时的当前年龄。"),
            )
            if module_super_enabled:
                person1_accum_super_balance = currency_text_input(
                    t("Person 1 Accumulation Super Balance", "人物 1 累积型养老金余额"),
                    st.session_state.person1_accum_super_balance,
                    "person1_accum_super_balance_input",
                    help_text=t("Opening accumulation super balance.", "期初 accumulation super 余额。"),
                )
            if module_pension_enabled:
                person1_pension_super_balance = currency_text_input(
                    t("Person 1 Pension Super Balance", "人物 1 养老金阶段余额"),
                    st.session_state.person1_pension_super_balance,
                    "person1_pension_super_balance_input",
                    help_text=t("Opening pension super balance.", "期初 pension super 余额。"),
                )
        with p1b:
            person1_retirement_age = st.number_input(
                t("Person 1 Retirement Age", "人物 1 退休年龄"),
                min_value=18,
                max_value=100,
                value=min(max(int(st.session_state.person1_retirement_age), 18), 100),
                step=1,
                help=t(
                    "Enter an age below current age for a client who is already retired at the projection start.",
                    "如客户在预测开始时已经退休，可输入低于当前年龄的退休年龄。",
                ),
            )
            if module_super_enabled:
                person1_accum_super_cost_base = currency_text_input(
                t("Person 1 Accumulation Super Cost Base", "人物 1 累积型养老金成本基础"),
                st.session_state.person1_accum_super_cost_base,
                "person1_accum_super_cost_base_input",
                help_text=t("Cost base used for super withdrawal CGT approximation in accumulation phase.", "用于 accumulation 阶段提取 CGT 近似计算的成本基础。"),
            )
            if module_pension_enabled:
                person1_pension_super_cost_base = currency_text_input(
                t("Person 1 Pension Super Cost Base", "人物 1 养老金阶段成本基础"),
                st.session_state.person1_pension_super_cost_base,
                "person1_pension_super_cost_base_input",
                help_text=t("Cost base carried inside the pension pool for internal tracking.", "用于 pension 池内部追踪的成本基础。"),
            )
        with p1c:
            if module_pension_enabled:
                person1_pension_start_age = st.number_input(
                t("Person 1 Pension Start Age", "人物 1 养老金开始年龄"),
                min_value=18,
                max_value=100,
                value=min(max(int(st.session_state.person1_pension_start_age), 18), 100),
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
                min_value=18,
                max_value=100,
                value=min(max(int(st.session_state.person2_current_age), 18), 100),
                step=1,
                help=t("Current age at the start of the projection.", "预测开始时的当前年龄。"),
            )
            if module_super_enabled:
                person2_accum_super_balance = currency_text_input(
                t("Person 2 Accumulation Super Balance", "人物 2 累积型养老金余额"),
                st.session_state.person2_accum_super_balance,
                "person2_accum_super_balance_input",
                help_text=t("Opening accumulation super balance.", "期初 accumulation super 余额。"),
            )
            if module_pension_enabled:
                person2_pension_super_balance = currency_text_input(
                t("Person 2 Pension Super Balance", "人物 2 养老金阶段余额"),
                st.session_state.person2_pension_super_balance,
                "person2_pension_super_balance_input",
                help_text=t("Opening pension super balance.", "期初 pension super 余额。"),
            )
        with p2b:
            person2_retirement_age = st.number_input(
                t("Person 2 Retirement Age", "人物 2 退休年龄"),
                min_value=18,
                max_value=100,
                value=min(max(int(st.session_state.person2_retirement_age), 18), 100),
                step=1,
                help=t(
                    "Employment income stops once current age reaches retirement age. Enter an age below current age if already retired.",
                    "达到退休年龄后，工作收入停止。如客户已经退休，可输入低于当前年龄的退休年龄。",
                ),
            )
            if module_super_enabled:
                person2_accum_super_cost_base = currency_text_input(
                t("Person 2 Accumulation Super Cost Base", "人物 2 累积型养老金成本基础"),
                st.session_state.person2_accum_super_cost_base,
                "person2_accum_super_cost_base_input",
                help_text=t("Cost base used for super withdrawal CGT approximation in accumulation phase.", "用于 accumulation 阶段提取 CGT 近似计算的成本基础。"),
            )
            if module_pension_enabled:
                person2_pension_super_cost_base = currency_text_input(
                t("Person 2 Pension Super Cost Base", "人物 2 养老金阶段成本基础"),
                st.session_state.person2_pension_super_cost_base,
                "person2_pension_super_cost_base_input",
                help_text=t("Cost base carried inside the pension pool for internal tracking.", "用于 pension 池内部追踪的成本基础。"),
            )
        with p2c:
            if module_pension_enabled:
                person2_pension_start_age = st.number_input(
                t("Person 2 Pension Start Age", "人物 2 养老金开始年龄"),
                min_value=18,
                max_value=100,
                value=min(max(int(st.session_state.person2_pension_start_age), 18), 100),
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
            if module_non_super_enabled:
                non_super_balance = currency_text_input(
                    t(
                        "Household Non-Super Balance (excludes trust assets)" if module_trust_enabled else "Household Non-Super Balance",
                        "家庭非 Super 资产余额（不含信托资产）" if module_trust_enabled else "家庭非 Super 资产余额",
                    ),
                    st.session_state.non_super_balance,
                    "non_super_balance_input",
                    help_text=t(
                        "Opening personally held non-super investment pool. Trust assets and trust income are modelled separately.",
                        "个人直接持有的期初非 Super 投资池；信托资产及信托收入会单独建模。",
                    ),
                )
            annual_living_expenses = currency_text_input(
                t("Annual Living Expenses", "年度生活支出"),
                st.session_state.annual_living_expenses,
                "annual_living_expenses_input",
                help_text=t("Current annual household spending before retirement trigger applies.", "退休支出触发前的当前年度家庭支出。"),
            )
            if module_non_super_enabled:
                cgt_discount_rate = percentage_text_input(
                    t("CGT Discount Rate", "资本利得税折扣率"),
                    float(st.session_state.cgt_discount_rate),
                    "cgt_discount_rate_input",
                    decimals=1,
                    help_text=t("Discount applied to non-super realised capital gains under the average-cost model.", "在 average-cost 模型下适用于非养老金已实现资本利得的折扣率。"),
                )
        with hh2:
            if module_non_super_enabled:
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
            if module_non_super_enabled and is_one_person_mode:
                st.text_input(
                    t("Person 1 Ownership %", "人物 1 持有比例 %"),
                    value="100.0%",
                    disabled=True,
                    key="non_super_ownership_person1_display",
                    help=t("Single-person mode fixes non-super ownership to 100% Person 1.", "单人模式下非养老金持有比例固定为 Person 1 的 100%。"),
                )
                non_super_ownership_person1 = 100.0
            elif module_non_super_enabled:
                non_super_ownership_person1 = percentage_text_input(
                    t("Person 1 Ownership %", "人物 1 持有比例 %"),
                    st.session_state.non_super_ownership_person1_pct / 100.0,
                    "non_super_ownership_person1_input",
                    decimals=1,
                    help_text=t("Share of non-super taxable income and tax allocated to Person 1.", "分配给 Person 1 的非养老金应税收入与税负比例。"),
                ) * 100.0

        st.divider()
        st.subheader(t("Asset Drawdown & Estate Reserves", "资产提取与遗产保留"))
        st.caption(t(
            "The selected order determines which assets fund an annual cash shortfall first. Minimum reserve amounts are preserved where possible.",
            "所选顺序决定年度现金缺口优先由哪些资产提供。模型会尽量保留所设定的最低储备金额。",
        ))
        available_drawdown_sources = {"cash"}
        if module_non_super_enabled:
            available_drawdown_sources.add("non_super")
        if module_pension_enabled:
            available_drawdown_sources.add("pension")
        if module_super_enabled:
            available_drawdown_sources.add("accumulation")
        if module_property_enabled:
            available_drawdown_sources.add("property")
        candidate_drawdown_profiles = {
            "Cash > Non-super > Pension > Accumulation > Property": ["cash", "non_super", "pension", "accumulation", "property"],
            "Cash > Pension > Accumulation > Non-super > Property": ["cash", "pension", "accumulation", "non_super", "property"],
            "Cash > Property > Non-super > Pension > Accumulation": ["cash", "property", "non_super", "pension", "accumulation"],
            "Cash > Pension > Accumulation > Property > Non-super": ["cash", "pension", "accumulation", "property", "non_super"],
        }
        source_labels = {
            "cash": t("Cash", "现金"),
            "non_super": t("Non-super", "非 Super"),
            "pension": "Pension",
            "accumulation": "Accumulation",
            "property": t("Property", "物业"),
        }
        drawdown_profiles = {}
        for candidate_order in candidate_drawdown_profiles.values():
            filtered_order = [item for item in candidate_order if item in available_drawdown_sources]
            filtered_label = " > ".join(source_labels[item] for item in filtered_order)
            drawdown_profiles.setdefault(filtered_label, filtered_order)
        current_drawdown_profile = st.session_state.get("drawdown_profile", "")
        drawdown_index = list(drawdown_profiles).index(current_drawdown_profile) if current_drawdown_profile in drawdown_profiles else 0
        drawdown_profile = st.selectbox(
            t("Drawdown Order", "资产提取顺序"),
            options=list(drawdown_profiles),
            index=drawdown_index,
            help=t("Minimum pension payments still occur before discretionary withdrawals.", "最低 Pension 提取仍会在可选择的额外提取之前发生。"),
        )
        withdrawal_order = drawdown_profiles[drawdown_profile]
        dr1, dr2, dr3, dr4 = st.columns(4)
        with dr1:
            cash_reserve_balance = currency_text_input(
                t("Opening Cash Reserve", "期初现金储备"),
                st.session_state.cash_reserve_balance,
                "cash_reserve_balance_input",
            )
        with dr2:
            cash_reserve_floor = currency_text_input(
                t("Minimum Cash Reserve", "最低现金储备"),
                st.session_state.cash_reserve_floor,
                "cash_reserve_floor_input",
            )
        with dr3:
            if module_non_super_enabled:
                non_super_estate_reserve = currency_text_input(
                    t("Non-super Estate Reserve", "非养老金遗产保留"),
                    st.session_state.non_super_estate_reserve,
                    "non_super_estate_reserve_input",
                )
        with dr4:
            if module_property_enabled:
                property_estate_reserve = currency_text_input(
                    t("Property Equity Reserve", "物业净值保留"),
                    st.session_state.property_estate_reserve,
                    "property_estate_reserve_input",
                )

        if module_non_super_enabled:
            st.divider()
            st.subheader(t("2026 Budget CGT Reform", "2026 Budget CGT 改革"))
            st.caption(t(
                "From 2027-28, the pooled model separates deferred pre-1 July 2027 gains from indexed real gains and estimates the Division 119 30% minimum-tax top-up.",
                "从 2027–28 财年起，汇总资产池会区分 2027年7月1日前递延增值与指数化后的实际增值，并估算 Division 119 的 30% 最低税补税。",
            ))
            cgt_reform_enabled = st.checkbox(
                t("Apply legislated 2026 Budget CGT reform", "应用已立法的 2026 Budget CGT 改革"),
                value=bool(st.session_state.cgt_reform_enabled),
            )
            cg1, cg2, cg3 = st.columns(3)
            with cg1:
                cgt_asset_acquired_before_2027 = st.checkbox(
                    t("Pool held before 1 July 2027", "资产池在 2027年7月1日前已持有"),
                    value=bool(st.session_state.cgt_asset_acquired_before_2027),
                )
                non_super_transition_value_2027 = currency_text_input(
                    t("Market Value at 30 June 2027", "2027年6月30日市场价值"),
                    st.session_state.non_super_transition_value_2027,
                    "non_super_transition_value_2027_input",
                    help_text=t("Used to split protected pre-reform gains from post-reform real gains.", "用于区分改革前受保护增值与改革后的实际增值。"),
                )
            with cg2:
                non_super_opening_capital_losses = currency_text_input(
                    t("Opening Carried-Forward Capital Losses", "期初结转资本亏损"),
                    st.session_state.non_super_opening_capital_losses,
                    "non_super_opening_capital_losses_input",
                )
                cgt_indexation_rate = percentage_text_input(
                    t("Annual CPI Indexation Estimate", "年度 CPI 指数化估计"),
                    st.session_state.cgt_indexation_rate,
                    "cgt_indexation_rate_input",
                    decimals=2,
                    help_text=t("Projection estimate only; actual tax calculations use published CPI index numbers.", "仅用于预测；实际报税应使用正式公布的 CPI 指数。"),
                )
            with cg3:
                cgt_asset_category = st.selectbox(
                    t("CGT Asset Category", "CGT 资产类别"),
                    options=["Other", "New residential dwelling", "Affordable housing"],
                    index=["Other", "New residential dwelling", "Affordable housing"].index(st.session_state.cgt_asset_category),
                )
                cgt_new_residential_method = st.selectbox(
                    t("New/Affordable Housing Method", "新建／可负担住房计算方式"),
                    options=["Indexation and 30% minimum tax", "50% discount"],
                    index=["Indexation and 30% minimum tax", "50% discount"].index(st.session_state.cgt_new_residential_method),
                    help=t("Relevant only when the asset category is new residential dwelling or affordable housing.", "仅在资产类别为新建住宅或可负担住房时适用。"),
                )
            cge1, cge2 = st.columns(2)
            with cge1:
                cgt_held_at_least_12_months = st.checkbox(
                    t("Asset pool held at least 12 months", "资产池持有至少 12 个月"),
                    value=bool(st.session_state.cgt_held_at_least_12_months),
                )
            with cge2:
                cgt_minimum_tax_exempt = st.checkbox(
                    t("Minimum-tax exemption confirmed", "已确认符合最低税豁免"),
                    value=bool(st.session_state.cgt_minimum_tax_exempt),
                    help=t("Use only for a confirmed statutory payment-recipient exemption.", "仅在确认符合法定政府付款领取者豁免时使用。"),
                )
            st.warning(t(
                "This is a homogeneous pooled-asset estimate. The 1 July 2027 transition allocation and annual CPI projection must be replaced with actual asset records and published CPI for tax return work.",
                "这是同质化资产池估算。用于报税时，必须以实际资产记录和正式 CPI 替换 2027年7月1日过渡分配及年度 CPI 预测。",
            ))

    elif active_input_section == "property_trust":
        st.subheader(t("Main Residence", "主住宅"))
        st.caption(t(
            "The main residence and its principal-and-interest home loan are always modelled here. This home loan is not part of the Investment Debt module.",
            "主住宅及其本息同还贷款始终在此建模；该自住房贷款不属于“投资债务”模块。",
        ))
        mr1, mr2, mr3 = st.columns(3)
        with mr1:
            main_residence_value = currency_text_input(
                t("Main Residence Value", "主住宅价值"),
                st.session_state.main_residence_value,
                "main_residence_value_input",
            )
            main_residence_capital_growth_rate = percentage_text_input(
                t("Main Residence Capital Growth", "主住宅资本增长率"),
                st.session_state.main_residence_capital_growth_rate,
                "main_residence_capital_growth_rate_input",
                decimals=1,
            )
        with mr2:
            main_residence_loan_balance = currency_text_input(
                t("Home Loan Balance (Principal & Interest)", "自住房贷款余额（本息同还）"),
                st.session_state.main_residence_loan_balance,
                "main_residence_loan_balance_input",
            )
            main_residence_interest_rate = percentage_text_input(
                t("Home Loan Interest Rate", "自住房贷款利率"),
                st.session_state.main_residence_interest_rate,
                "main_residence_interest_rate_input",
                decimals=2,
            )
        with mr3:
            main_residence_annual_loan_repayment = currency_text_input(
                t("Annual Home Loan Repayment", "自住房贷款年度还款额"),
                st.session_state.main_residence_annual_loan_repayment,
                "main_residence_annual_loan_repayment_input",
                help_text=t("Total annual principal-and-interest repayment, excluding optional extra repayments.", "年度本息还款总额，不包括可选的额外还款。"),
            )
            main_residence_offset_balance = currency_text_input(
                t("Home Loan Offset Balance", "自住房贷款 Offset 余额"),
                st.session_state.main_residence_offset_balance,
                "main_residence_offset_balance_input",
            )

        if module_property_enabled:
            st.divider()
            st.subheader(t("Residential Investment Property", "住宅投资物业"))
            st.caption(t(
                "Models one aggregate residential investment. Loss quarantine starts in 2027-28 for affected established properties.",
                "以一个汇总住宅投资建模。受影响的存量住宅从 2027–28 财年起适用亏损隔离。",
            ))
            residential_property_enabled = True
            st.caption(t("Enabled in Client Modules.", "已在客户模块中启用。"))
            rp1, rp2, rp3 = st.columns(3)
            with rp1:
                residential_property_value = currency_text_input(
                    t("Opening Property Value", "期初物业价值"),
                    st.session_state.residential_property_value,
                    "residential_property_value_input",
                )
                residential_property_gross_rent = currency_text_input(
                    t("Annual Gross Rent", "年度租金总收入"),
                    st.session_state.residential_property_gross_rent,
                    "residential_property_gross_rent_input",
                )
                residential_property_capital_growth_rate = percentage_text_input(
                    t("Property Capital Growth", "物业资本增长率"),
                    st.session_state.residential_property_capital_growth_rate,
                    "residential_property_capital_growth_rate_input",
                    decimals=1,
                )
            with rp2:
                residential_property_loan_balance = currency_text_input(
                    t("Investment Property Loan Balance (Principal & Interest)", "投资物业贷款余额（本息同还）"),
                    st.session_state.residential_property_loan_balance,
                    "residential_property_loan_balance_input",
                )
                residential_property_operating_expenses = currency_text_input(
                    t("Annual Deductible Expenses", "年度可扣除费用"),
                    st.session_state.residential_property_operating_expenses,
                    "residential_property_operating_expenses_input",
                    help_text=t("Excludes loan interest, which is calculated separately.", "不含贷款利息；利息会单独计算。"),
                )
                residential_property_interest_rate = percentage_text_input(
                    t("Loan Interest Rate", "贷款利率"),
                    st.session_state.residential_property_interest_rate,
                    "residential_property_interest_rate_input",
                    decimals=2,
                )
                residential_property_annual_loan_repayment = currency_text_input(
                    t("Annual Property Loan Repayment", "投资物业贷款年度还款额"),
                    st.session_state.residential_property_annual_loan_repayment,
                    "residential_property_annual_loan_repayment_input",
                    help_text=t("Total annual principal-and-interest repayment, excluding optional extra repayments.", "年度本息还款总额，不包括可选的额外还款。"),
                )
                deductible_offset_balance = currency_text_input(
                    t("Investment Property Loan Offset", "投资物业贷款 Offset"),
                    st.session_state.deductible_offset_balance,
                    "residential_property_offset_balance_input",
                )
            with rp3:
                residential_property_opening_quarantined_loss = currency_text_input(
                    t("Opening Quarantined Loss", "期初隔离亏损"),
                    st.session_state.residential_property_opening_quarantined_loss,
                    "residential_property_opening_quarantined_loss_input",
                )
                residential_property_rent_growth_rate = percentage_text_input(
                    t("Rent Growth", "租金增长率"),
                    st.session_state.residential_property_rent_growth_rate,
                    "residential_property_rent_growth_rate_input",
                    decimals=1,
                )
                residential_property_expense_growth_rate = percentage_text_input(
                    t("Expense Growth", "费用增长率"),
                    st.session_state.residential_property_expense_growth_rate,
                    "residential_property_expense_growth_rate_input",
                    decimals=1,
                )
                residential_property_sale_cost_rate = percentage_text_input(
                    t("Estimated Sale Costs", "预计出售成本"),
                    st.session_state.residential_property_sale_cost_rate,
                    "residential_property_sale_cost_rate_input",
                    decimals=2,
                    help_text=t("Applied proportionally when the drawdown strategy uses part or all of the property equity. Property CGT is not yet modelled.", "当资产提取策略使用部分或全部物业净值时按比例计入。物业 CGT 尚未建模。"),
                )

            rq1, rq2, rq3 = st.columns(3)
            with rq1:
                residential_property_acquired_before_budget_time = st.checkbox(
                    t("Acquired before 7:30pm AEST 12 May 2026", "在 2026年5月12日 AEST 19:30 前取得"),
                    value=bool(st.session_state.residential_property_acquired_before_budget_time),
                )
            with rq2:
                residential_property_is_new_build = st.checkbox(
                    t("Qualifying new build", "符合条件的新建住宅"),
                    value=bool(st.session_state.residential_property_is_new_build),
                )
            with rq3:
                residential_property_is_exempt_housing = st.checkbox(
                    t("Qualifying exempt housing", "符合条件的豁免住房"),
                    value=bool(st.session_state.residential_property_is_exempt_housing),
                    help=t("Use only after confirming the statutory housing exception.", "仅在确认符合法定住房例外后使用。"),
                )

            if is_one_person_mode:
                residential_property_ownership_person1 = 100.0
            else:
                residential_property_ownership_person1 = percentage_text_input(
                    t("Person 1 Property Ownership", "人物 1 物业持有比例"),
                    st.session_state.residential_property_ownership_person1_pct / 100.0,
                    "residential_property_ownership_person1_input",
                    decimals=1,
                ) * 100.0

        if False and module_trust_enabled:
            st.divider()
            st.subheader(t("Discretionary Trust Minimum Tax", "Discretionary Trust 最低税"))
            st.warning(t(
                "Policy scenario only: the 30% minimum tax is based on the September 2026 exposure draft and is not enacted law.",
                "仅作政策情景：30% 最低税依据 2026 年 9 月 exposure draft，目前尚未立法。",
            ))
            discretionary_trust_enabled = True
            st.caption(t("Enabled in Client Modules.", "已在客户模块中启用。"))
            dt1, dt2, dt3 = st.columns(3)
            with dt1:
                discretionary_trust_net_income = currency_text_input(
                    t("Annual Trust Net Income", "年度信托净收入"),
                    st.session_state.discretionary_trust_net_income,
                    "discretionary_trust_net_income_input",
                )
            with dt2:
                discretionary_trust_excluded_income = currency_text_input(
                    t("Excluded Income", "豁免收入"),
                    st.session_state.discretionary_trust_excluded_income,
                    "discretionary_trust_excluded_income_input",
                    help_text=t("For example, confirmed primary production or other draft-law exclusions.", "例如已确认的 primary production 或草案列明的其他豁免收入。"),
                )
            with dt3:
                discretionary_trust_income_growth_rate = percentage_text_input(
                    t("Trust Income Growth", "信托收入增长率"),
                    st.session_state.discretionary_trust_income_growth_rate,
                    "discretionary_trust_income_growth_rate_input",
                    decimals=1,
                )
            discretionary_trust_subject_to_minimum_tax = st.checkbox(
                t("Trust is subject to the draft minimum tax", "该信托适用最低税草案"),
                value=bool(st.session_state.discretionary_trust_subject_to_minimum_tax),
                help=t("Turn off for a confirmed excluded trust or a valid fixed-distribution election scenario.", "若已确认属于豁免信托或有效选择固定分配情景，可关闭。"),
            )
            if is_one_person_mode:
                discretionary_trust_ownership_person1 = 100.0
            else:
                discretionary_trust_ownership_person1 = percentage_text_input(
                    t("Person 1 Trust Distribution", "人物 1 信托分配比例"),
                    st.session_state.discretionary_trust_ownership_person1_pct / 100.0,
                    "discretionary_trust_ownership_person1_input",
                    decimals=1,
                ) * 100.0

    elif active_input_section == "trust":
        st.subheader(t("Discretionary Trust Investment Pool", "Discretionary Trust 投资池"))
        st.warning(t(
            "Policy scenario only: the 30% minimum tax is based on the September 2026 exposure draft and is not enacted law.",
            "仅作政策情景：30% 最低税依据 2026 年 9 月 exposure draft，目前尚未立法。",
        ))
        st.caption(t(
            "Trust income is derived from the opening balance and return assumptions below; it is not entered as a separate gross or net income amount.",
            "信托收入由期初余额及以下回报率自动计算，不再单独输入 gross income 或 net income。",
        ))
        discretionary_trust_enabled = True
        dt1, dt2, dt3 = st.columns(3)
        with dt1:
            discretionary_trust_balance = currency_text_input(
                t("Opening Trust Balance", "期初信托资产余额"),
                st.session_state.discretionary_trust_balance,
                "discretionary_trust_balance_input",
            )
            discretionary_trust_cost_base = currency_text_input(
                t("Opening Trust Cost Base", "期初信托成本基础"),
                st.session_state.discretionary_trust_cost_base,
                "discretionary_trust_cost_base_input",
            )
        with dt2:
            discretionary_trust_income_return_mean = percentage_text_input(
                t("Trust Income Return Mean", "信托收益型回报均值"),
                st.session_state.discretionary_trust_income_return_mean,
                "discretionary_trust_income_return_mean_input",
                decimals=1,
            )
            discretionary_trust_income_return_std = percentage_text_input(
                t("Trust Income Return Std", "信托收益型回报波动"),
                st.session_state.discretionary_trust_income_return_std,
                "discretionary_trust_income_return_std_input",
                decimals=1,
            )
        with dt3:
            discretionary_trust_capital_return_mean = percentage_text_input(
                t("Trust Capital Return Mean", "信托资本增值回报均值"),
                st.session_state.discretionary_trust_capital_return_mean,
                "discretionary_trust_capital_return_mean_input",
                decimals=1,
            )
            discretionary_trust_capital_return_std = percentage_text_input(
                t("Trust Capital Return Std", "信托资本增值回报波动"),
                st.session_state.discretionary_trust_capital_return_std,
                "discretionary_trust_capital_return_std_input",
                decimals=1,
            )
        discretionary_trust_excluded_income_pct = percentage_text_input(
            t("Excluded Income % of Total Trust Income", "豁免收入占信托总收入比例"),
            st.session_state.discretionary_trust_excluded_income_pct,
            "discretionary_trust_excluded_income_pct_input",
            decimals=1,
            help_text=t("Enter the confirmed excluded share as a percentage of modelled trust income.", "按模型计算的信托总收入输入已确认的豁免比例。"),
        )
        discretionary_trust_subject_to_minimum_tax = st.checkbox(
            t("Trust is subject to the draft minimum tax", "该信托适用最低税草案"),
            value=bool(st.session_state.discretionary_trust_subject_to_minimum_tax),
        )
        if is_one_person_mode:
            discretionary_trust_ownership_person1 = 100.0
        else:
            discretionary_trust_ownership_person1 = percentage_text_input(
                t("Person 1 Trust Distribution", "人物 1 信托分配比例"),
                st.session_state.discretionary_trust_ownership_person1_pct / 100.0,
                "discretionary_trust_ownership_person1_input",
                decimals=1,
            ) * 100.0

    elif active_input_section == "cash_surplus":
        st.subheader(t("Cash Surplus Strategy", "现金盈余策略"))
        st.caption(t(
            "Choose where annual surplus cash is directed after spending, tax, contributions and scheduled principal-and-interest loan payments.",
            "选择在支出、税款、缴款及计划内本息还款之后，年度现金盈余的优先去向。",
        ))
        candidate_surplus_profiles = {
            "Home Offset > Home Loan > Cash Reserve > Invest": ["main_residence_offset", "main_residence_repayment", "cash_reserve", "non_super"],
            "Cash Reserve > Home Offset > Invest": ["cash_reserve", "main_residence_offset", "non_super"],
            "Investment Debt > Property Loan > Invest": ["non_deductible_repayment", "investment_deductible_repayment", "property_loan_repayment", "non_super"],
            "Offsets First > Invest": ["main_residence_offset", "property_loan_offset", "non_deductible_offset", "investment_deductible_offset", "non_super"],
            "Invest All Surplus": ["non_super"],
        }
        allowed_surplus_destinations = {"cash_reserve", "main_residence_offset", "main_residence_repayment"}
        if module_non_super_enabled:
            allowed_surplus_destinations.add("non_super")
        if module_property_enabled:
            allowed_surplus_destinations.update({"property_loan_offset", "property_loan_repayment"})
        if module_investment_debt_enabled:
            allowed_surplus_destinations.update({"non_deductible_offset", "non_deductible_repayment", "investment_deductible_offset", "investment_deductible_repayment"})
        surplus_destination_labels = {
            "cash_reserve": t("Cash Reserve", "现金储备"),
            "non_super": t("Non-super Investment", "非 Super 投资"),
            "main_residence_offset": t("Home Loan Offset", "自住房贷款 Offset"),
            "main_residence_repayment": t("Home Loan Repayment", "自住房贷款还款"),
            "property_loan_offset": t("Property Loan Offset", "投资物业贷款 Offset"),
            "property_loan_repayment": t("Property Loan Repayment", "投资物业贷款还款"),
            "non_deductible_offset": t("Non-deductible Investment Offset", "不可抵扣投资债务 Offset"),
            "non_deductible_repayment": t("Non-deductible Investment Debt", "不可抵扣投资债务"),
            "investment_deductible_offset": t("Deductible Investment Offset", "可抵扣投资债务 Offset"),
            "investment_deductible_repayment": t("Deductible Investment Debt", "可抵扣投资债务"),
        }
        surplus_profiles = {}
        for candidate_order in candidate_surplus_profiles.values():
            filtered_order = [item for item in candidate_order if item in allowed_surplus_destinations]
            if not filtered_order:
                filtered_order = ["cash_reserve"]
            filtered_label = " > ".join(surplus_destination_labels[item] for item in filtered_order)
            surplus_profiles.setdefault(filtered_label, filtered_order)
        current_surplus_profile = st.session_state.get("surplus_allocation_profile", "")
        surplus_profile_index = list(surplus_profiles).index(current_surplus_profile) if current_surplus_profile in surplus_profiles else 0
        surplus_allocation_profile = st.selectbox(
            t("Annual Surplus Allocation", "年度盈余分配"),
            options=list(surplus_profiles),
            index=surplus_profile_index,
        )
        surplus_allocation_order = surplus_profiles[surplus_allocation_profile]
        cash_reserve_target = currency_text_input(
            t("Cash Reserve Target", "现金储备目标"),
            st.session_state.cash_reserve_target,
            "cash_reserve_target_input",
        )

    elif active_input_section == "investment_debt":
        st.subheader(t("Investment Debt", "投资债务"))
        st.info(t(
            "This module excludes the main-residence home loan. It also excludes the deductible residential-investment-property loan when that module is active; both are entered on the Property page.",
            "本模块不包括主住宅贷款；启用住宅投资物业时，也不包括该物业的可抵扣贷款。这两类贷款均在“物业”页面输入。",
        ))
        id1, id2 = st.columns(2)
        with id1:
            st.markdown(f"#### {t('Other Non-deductible Investment Debt', '其他不可抵扣投资债务')}")
            non_deductible_debt_balance = currency_text_input(
                t("Opening Balance", "期初余额"),
                st.session_state.non_deductible_debt_balance,
                "non_deductible_debt_balance_input",
            )
            non_deductible_interest_rate = percentage_text_input(
                t("Interest Rate", "利率"),
                st.session_state.non_deductible_interest_rate,
                "non_deductible_interest_rate_input",
                decimals=2,
            )
            non_deductible_annual_repayment = currency_text_input(
                t("Annual Repayment", "年度还款额"),
                st.session_state.non_deductible_annual_repayment,
                "non_deductible_annual_repayment_input",
                help_text=t("Total annual principal-and-interest repayment, excluding optional extra repayments.", "年度本息还款总额，不包括可选的额外还款。"),
            )
            non_deductible_offset_balance = currency_text_input(
                t("Offset Balance", "Offset 余额"),
                st.session_state.non_deductible_offset_balance,
                "non_deductible_offset_balance_input",
            )
        with id2:
            st.markdown(f"#### {t('Other Deductible Investment Debt', '其他可抵扣投资债务')}")
            investment_deductible_debt_balance = currency_text_input(
                t("Opening Balance", "期初余额"),
                st.session_state.investment_deductible_debt_balance,
                "investment_deductible_debt_balance_input",
            )
            investment_deductible_interest_rate = percentage_text_input(
                t("Interest Rate", "利率"),
                st.session_state.investment_deductible_interest_rate,
                "investment_deductible_interest_rate_input",
                decimals=2,
            )
            investment_deductible_annual_repayment = currency_text_input(
                t("Annual Repayment", "年度还款额"),
                st.session_state.investment_deductible_annual_repayment,
                "investment_deductible_annual_repayment_input",
                help_text=t("Total annual principal-and-interest repayment, excluding optional extra repayments.", "年度本息还款总额，不包括可选的额外还款。"),
            )
            investment_deductible_offset_balance = currency_text_input(
                t("Offset Balance", "Offset 余额"),
                st.session_state.investment_deductible_offset_balance,
                "investment_deductible_offset_balance_input",
            )

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
            width="stretch",
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
                if module_super_enabled:
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
                if module_super_enabled:
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
                if module_non_super_enabled:
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

            assumption_rows = []
            if module_super_enabled:
                assumption_rows.extend([
                    (t("Super Income Return Mean", "养老金收益型回报均值"), selected_values["super_income_return_mean"] * 100.0),
                    (t("Super Income Return Std", "养老金收益型回报波动"), selected_values["super_income_return_std"] * 100.0),
                    (t("Super Capital Return Mean", "养老金资本增值回报均值"), selected_values["super_capital_return_mean"] * 100.0),
                    (t("Super Capital Return Std", "养老金资本增值回报波动"), selected_values["super_capital_return_std"] * 100.0),
                ])
            if module_non_super_enabled:
                assumption_rows.extend([
                    (t("Non-Super Income Return Mean", "非养老金收益型回报均值"), selected_values["non_super_income_return_mean"] * 100.0),
                    (t("Non-Super Income Return Std", "非养老金收益型回报波动"), selected_values["non_super_income_return_std"] * 100.0),
                    (t("Non-Super Capital Return Mean", "非养老金资本增值回报均值"), selected_values["non_super_capital_return_mean"] * 100.0),
                    (t("Non-Super Capital Return Std", "非养老金资本增值回报波动"), selected_values["non_super_capital_return_std"] * 100.0),
                ])
            assumption_rows.append((t("Inflation Rate", "通胀率"), selected_values["inflation_rate"] * 100.0))
            display_df = pd.DataFrame(assumption_rows, columns=[t("Assumption", "假设"), t("Value", "数值")])
            st.dataframe(
                display_df,
                width="stretch",
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
            st.metric(t("Number of Simulations", "模拟次数"), f"{int(st.session_state.number_of_simulations):,}")
            st.caption(t("Controlled by Simulation Depth in the sidebar.", "由侧边栏的模拟深度控制。"))
            number_of_simulations = int(st.session_state.number_of_simulations)
        with sim2:
            random_seed = st.number_input(
                t("Random Seed", "随机种子"),
                value=int(st.session_state.random_seed),
                step=1,
            )
        


    apply_inputs_button = st.form_submit_button(t("Apply Inputs", "应用输入"), width="stretch")
    if apply_inputs_button:
        st.toast(t("Inputs applied.", "输入已应用。"))

# ============================================================
# SECTION: SESSION UPDATE
# ============================================================

st.session_state.start_financial_year = int(start_financial_year)
st.session_state.projection_years = int(projection_years)
st.session_state.retirement_spending_trigger = retirement_spending_trigger
st.session_state.household_mode = household_mode
st.session_state.value_mode = value_mode
st.session_state.workspace_mode = workspace_mode
st.session_state.show_assumption_panel = bool(show_assumption_panel)
st.session_state.show_live_input_checks = bool(show_live_input_checks)
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
st.session_state.cash_reserve_balance = cash_reserve_balance
st.session_state.cash_reserve_floor = cash_reserve_floor
st.session_state.cash_reserve_target = cash_reserve_target
st.session_state.main_residence_value = main_residence_value
st.session_state.main_residence_capital_growth_rate = main_residence_capital_growth_rate
st.session_state.main_residence_loan_balance = main_residence_loan_balance
st.session_state.main_residence_interest_rate = main_residence_interest_rate
st.session_state.main_residence_annual_loan_repayment = main_residence_annual_loan_repayment
st.session_state.main_residence_offset_balance = main_residence_offset_balance
st.session_state.non_deductible_debt_balance = non_deductible_debt_balance
st.session_state.non_deductible_interest_rate = non_deductible_interest_rate
st.session_state.non_deductible_annual_repayment = non_deductible_annual_repayment
st.session_state.non_deductible_offset_balance = non_deductible_offset_balance
st.session_state.deductible_offset_balance = deductible_offset_balance
st.session_state.investment_deductible_debt_balance = investment_deductible_debt_balance
st.session_state.investment_deductible_interest_rate = investment_deductible_interest_rate
st.session_state.investment_deductible_annual_repayment = investment_deductible_annual_repayment
st.session_state.investment_deductible_offset_balance = investment_deductible_offset_balance
st.session_state.surplus_allocation_profile = surplus_allocation_profile
st.session_state.surplus_allocation_order = surplus_allocation_order
st.session_state.non_super_estate_reserve = non_super_estate_reserve
st.session_state.property_estate_reserve = property_estate_reserve
st.session_state.drawdown_profile = drawdown_profile
st.session_state.withdrawal_order = withdrawal_order
st.session_state.cgt_reform_enabled = bool(cgt_reform_enabled)
st.session_state.cgt_asset_acquired_before_2027 = bool(cgt_asset_acquired_before_2027)
st.session_state.non_super_transition_value_2027 = non_super_transition_value_2027
st.session_state.non_super_opening_capital_losses = non_super_opening_capital_losses
st.session_state.cgt_indexation_rate = cgt_indexation_rate
st.session_state.cgt_asset_category = cgt_asset_category
st.session_state.cgt_new_residential_method = cgt_new_residential_method
st.session_state.cgt_held_at_least_12_months = bool(cgt_held_at_least_12_months)
st.session_state.cgt_minimum_tax_exempt = bool(cgt_minimum_tax_exempt)
st.session_state.residential_property_enabled = bool(module_property_enabled)
st.session_state.residential_property_value = residential_property_value
st.session_state.residential_property_sale_cost_rate = residential_property_sale_cost_rate
st.session_state.residential_property_loan_balance = residential_property_loan_balance
st.session_state.residential_property_annual_loan_repayment = residential_property_annual_loan_repayment
st.session_state.residential_property_gross_rent = residential_property_gross_rent
st.session_state.residential_property_operating_expenses = residential_property_operating_expenses
st.session_state.residential_property_interest_rate = residential_property_interest_rate
st.session_state.residential_property_capital_growth_rate = residential_property_capital_growth_rate
st.session_state.residential_property_rent_growth_rate = residential_property_rent_growth_rate
st.session_state.residential_property_expense_growth_rate = residential_property_expense_growth_rate
st.session_state.residential_property_opening_quarantined_loss = residential_property_opening_quarantined_loss
st.session_state.residential_property_acquired_before_budget_time = bool(residential_property_acquired_before_budget_time)
st.session_state.residential_property_is_new_build = bool(residential_property_is_new_build)
st.session_state.residential_property_is_exempt_housing = bool(residential_property_is_exempt_housing)
st.session_state.residential_property_ownership_person1_pct = residential_property_ownership_person1
st.session_state.discretionary_trust_enabled = bool(module_trust_enabled)
st.session_state.discretionary_trust_balance = discretionary_trust_balance
st.session_state.discretionary_trust_cost_base = discretionary_trust_cost_base
st.session_state.discretionary_trust_income_return_mean = discretionary_trust_income_return_mean
st.session_state.discretionary_trust_income_return_std = discretionary_trust_income_return_std
st.session_state.discretionary_trust_capital_return_mean = discretionary_trust_capital_return_mean
st.session_state.discretionary_trust_capital_return_std = discretionary_trust_capital_return_std
st.session_state.discretionary_trust_excluded_income_pct = discretionary_trust_excluded_income_pct
st.session_state.discretionary_trust_net_income = discretionary_trust_net_income
st.session_state.discretionary_trust_excluded_income = discretionary_trust_excluded_income
st.session_state.discretionary_trust_income_growth_rate = discretionary_trust_income_growth_rate
st.session_state.discretionary_trust_subject_to_minimum_tax = bool(discretionary_trust_subject_to_minimum_tax)
st.session_state.discretionary_trust_ownership_person1_pct = discretionary_trust_ownership_person1
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
    retirement_spending_trigger = "Either Retired"
    if not contribution_events_df.empty:
        contribution_events_df = contribution_events_df[contribution_events_df["person"] != "Person 2"].reset_index(drop=True)

    st.session_state.person2_name = ""
    st.session_state.person2_current_age = 0
    st.session_state.person2_retirement_age = 0
    st.session_state.person2_pension_start_age = 0
    st.session_state.person2_accum_super_balance = 0.0
    st.session_state.person2_pension_super_balance = 0.0
    st.session_state.person2_accum_super_cost_base = 0.0
    st.session_state.person2_pension_super_cost_base = 0.0
    st.session_state.person2_transfer_balance_cap = 0.0
    st.session_state.person2_annual_income = 0.0
    st.session_state.non_super_ownership_person1_pct = 100.0
    st.session_state.retirement_spending_trigger = "Either Retired"
    st.session_state.contribution_events_df = contribution_events_df.copy()

# ============================================================
# SECTION: BASE INPUT MAP
# ============================================================

base_inputs = {
    "report_title": st.session_state.report_title,
    "person1_name": st.session_state.person1_name,
    "person2_name": "" if is_one_person_mode else st.session_state.person2_name,
    "start_financial_year": int(st.session_state.start_financial_year),
    "projection_years": int(st.session_state.projection_years),
    "retirement_spending_trigger": "Either Retired" if is_one_person_mode else st.session_state.retirement_spending_trigger,
    "household_mode": st.session_state.household_mode,
    "module_second_person_enabled": bool(st.session_state.module_second_person_enabled),
    "module_super_enabled": bool(st.session_state.module_super_enabled),
    "module_pension_enabled": bool(st.session_state.module_pension_enabled),
    "module_non_super_enabled": bool(st.session_state.module_non_super_enabled),
    "module_property_enabled": bool(st.session_state.module_property_enabled),
    "module_trust_enabled": bool(st.session_state.module_trust_enabled),
    "module_cash_surplus_enabled": bool(st.session_state.module_cash_surplus_enabled),
    "module_investment_debt_enabled": bool(st.session_state.module_investment_debt_enabled),
    "module_non_deductible_debt_enabled": bool(st.session_state.module_investment_debt_enabled),
    "module_deductible_debt_enabled": bool(st.session_state.module_investment_debt_enabled),
    "ui_language": st.session_state.ui_language,
    "person1_current_age": int(st.session_state.person1_current_age),
    "person2_current_age": 0 if is_one_person_mode else int(st.session_state.person2_current_age),
    "person1_retirement_age": int(st.session_state.person1_retirement_age),
    "person2_retirement_age": 0 if is_one_person_mode else int(st.session_state.person2_retirement_age),
    "person1_pension_start_age": int(st.session_state.person1_pension_start_age),
    "person2_pension_start_age": 0 if is_one_person_mode else int(st.session_state.person2_pension_start_age),
    "person1_accum_super_balance": float(st.session_state.person1_accum_super_balance),
    "person1_pension_super_balance": float(st.session_state.person1_pension_super_balance),
    "person2_accum_super_balance": 0.0 if is_one_person_mode else float(st.session_state.person2_accum_super_balance),
    "person2_pension_super_balance": 0.0 if is_one_person_mode else float(st.session_state.person2_pension_super_balance),
    "person1_transfer_balance_cap": float(st.session_state.person1_transfer_balance_cap),
    "person2_transfer_balance_cap": 0.0 if is_one_person_mode else float(st.session_state.person2_transfer_balance_cap),
    "non_super_balance": float(st.session_state.non_super_balance),
    "non_super_cost_base": float(st.session_state.non_super_cost_base),
    "cash_reserve_balance": float(st.session_state.cash_reserve_balance),
    "cash_reserve_floor": float(st.session_state.cash_reserve_floor),
    "cash_reserve_target": float(st.session_state.cash_reserve_target),
    "main_residence_value": float(st.session_state.main_residence_value),
    "main_residence_capital_growth_rate": float(st.session_state.main_residence_capital_growth_rate),
    "main_residence_loan_balance": float(st.session_state.main_residence_loan_balance),
    "main_residence_interest_rate": float(st.session_state.main_residence_interest_rate),
    "main_residence_annual_loan_repayment": float(st.session_state.main_residence_annual_loan_repayment),
    "main_residence_offset_balance": float(st.session_state.main_residence_offset_balance),
    "non_deductible_debt_balance": float(st.session_state.non_deductible_debt_balance),
    "non_deductible_interest_rate": float(st.session_state.non_deductible_interest_rate),
    "non_deductible_annual_repayment": float(st.session_state.non_deductible_annual_repayment),
    "non_deductible_offset_balance": float(st.session_state.non_deductible_offset_balance),
    "deductible_offset_balance": float(st.session_state.deductible_offset_balance),
    "investment_deductible_debt_balance": float(st.session_state.investment_deductible_debt_balance),
    "investment_deductible_interest_rate": float(st.session_state.investment_deductible_interest_rate),
    "investment_deductible_annual_repayment": float(st.session_state.investment_deductible_annual_repayment),
    "investment_deductible_offset_balance": float(st.session_state.investment_deductible_offset_balance),
    "surplus_allocation_order": list(st.session_state.surplus_allocation_order),
    "surplus_allocation_profile": st.session_state.surplus_allocation_profile,
    "non_super_estate_reserve": float(st.session_state.non_super_estate_reserve),
    "property_estate_reserve": float(st.session_state.property_estate_reserve),
    "withdrawal_order": list(st.session_state.withdrawal_order),
    "cgt_reform_enabled": bool(st.session_state.cgt_reform_enabled),
    "cgt_asset_acquired_before_2027": bool(st.session_state.cgt_asset_acquired_before_2027),
    "non_super_transition_value_2027": float(st.session_state.non_super_transition_value_2027),
    "non_super_opening_capital_losses": float(st.session_state.non_super_opening_capital_losses),
    "cgt_indexation_rate": float(st.session_state.cgt_indexation_rate),
    "cgt_asset_category": st.session_state.cgt_asset_category,
    "cgt_new_residential_method": st.session_state.cgt_new_residential_method,
    "cgt_held_at_least_12_months": bool(st.session_state.cgt_held_at_least_12_months),
    "cgt_minimum_tax_exempt": bool(st.session_state.cgt_minimum_tax_exempt),
    "residential_property_enabled": bool(st.session_state.residential_property_enabled),
    "residential_property_value": float(st.session_state.residential_property_value),
    "residential_property_loan_balance": float(st.session_state.residential_property_loan_balance),
    "residential_property_annual_loan_repayment": float(st.session_state.residential_property_annual_loan_repayment),
    "residential_property_sale_cost_rate": float(st.session_state.residential_property_sale_cost_rate),
    "residential_property_gross_rent": float(st.session_state.residential_property_gross_rent),
    "residential_property_operating_expenses": float(st.session_state.residential_property_operating_expenses),
    "residential_property_interest_rate": float(st.session_state.residential_property_interest_rate),
    "residential_property_capital_growth_rate": float(st.session_state.residential_property_capital_growth_rate),
    "residential_property_rent_growth_rate": float(st.session_state.residential_property_rent_growth_rate),
    "residential_property_expense_growth_rate": float(st.session_state.residential_property_expense_growth_rate),
    "residential_property_opening_quarantined_loss": float(st.session_state.residential_property_opening_quarantined_loss),
    "residential_property_acquired_before_budget_time": bool(st.session_state.residential_property_acquired_before_budget_time),
    "residential_property_is_new_build": bool(st.session_state.residential_property_is_new_build),
    "residential_property_is_exempt_housing": bool(st.session_state.residential_property_is_exempt_housing),
    "residential_property_ownership_person1": 1.0 if is_one_person_mode else float(st.session_state.residential_property_ownership_person1_pct) / 100.0,
    "discretionary_trust_enabled": bool(st.session_state.discretionary_trust_enabled),
    "discretionary_trust_balance": float(st.session_state.discretionary_trust_balance),
    "discretionary_trust_cost_base": float(st.session_state.discretionary_trust_cost_base),
    "discretionary_trust_income_return_mean": float(st.session_state.discretionary_trust_income_return_mean),
    "discretionary_trust_income_return_std": float(st.session_state.discretionary_trust_income_return_std),
    "discretionary_trust_capital_return_mean": float(st.session_state.discretionary_trust_capital_return_mean),
    "discretionary_trust_capital_return_std": float(st.session_state.discretionary_trust_capital_return_std),
    "discretionary_trust_excluded_income_pct": float(st.session_state.discretionary_trust_excluded_income_pct),
    "discretionary_trust_subject_to_minimum_tax": bool(st.session_state.discretionary_trust_subject_to_minimum_tax),
    "discretionary_trust_ownership_person1": 1.0 if is_one_person_mode else float(st.session_state.discretionary_trust_ownership_person1_pct) / 100.0,
    "person1_annual_income": float(st.session_state.person1_annual_income),
    "person2_annual_income": 0.0 if is_one_person_mode else float(st.session_state.person2_annual_income),
    "annual_living_expenses": float(st.session_state.annual_living_expenses),
    "retirement_spending": float(st.session_state.retirement_spending),
    "non_super_ownership_person1": 1.0 if is_one_person_mode else float(st.session_state.non_super_ownership_person1_pct) / 100.0,
    "cgt_discount_rate": float(st.session_state.cgt_discount_rate),
    "inflation_rate": float(st.session_state.inflation_rate),
    "super_income_return_mean": float(st.session_state.super_income_return_mean),
    "super_income_return_std": float(st.session_state.super_income_return_std),
    "super_capital_return_mean": float(st.session_state.super_capital_return_mean),
    "super_capital_return_std": float(st.session_state.super_capital_return_std),
    "non_super_income_return_mean": float(st.session_state.non_super_income_return_mean),
    "non_super_income_return_std": float(st.session_state.non_super_income_return_std),
    "non_super_capital_return_mean": float(st.session_state.non_super_capital_return_mean),
    "non_super_capital_return_std": float(st.session_state.non_super_capital_return_std),
    "number_of_simulations": int(st.session_state.number_of_simulations),
    "assumption_preset": preset_choice,
    "contribution_events": contribution_events_to_records(contribution_events_df, household_mode=household_mode),
    "person1_accum_super_cost_base": float(st.session_state.person1_accum_super_cost_base),
    "person1_pension_super_cost_base": float(st.session_state.person1_pension_super_cost_base),
    "person2_accum_super_cost_base": 0.0 if is_one_person_mode else float(st.session_state.person2_accum_super_cost_base),
    "person2_pension_super_cost_base": 0.0 if is_one_person_mode else float(st.session_state.person2_pension_super_cost_base),
}
base_inputs = apply_module_scope(base_inputs)

if show_live_input_checks:
    render_live_input_feedback(base_inputs)
else:
    render_light_input_badges(base_inputs)

# ============================================================
# SECTION: RUN LOGIC
# ============================================================

if run_button:
    if scenario_mode == "Single Scenario":
        if preset_choice == "Custom":
            scenario_inputs_map = {"Custom": base_inputs.copy()}
        else:
            scenario_inputs_map = {preset_choice: apply_preset_to_inputs(base_inputs, preset_choice, runtime_presets)}
    elif scenario_mode == "Compare Standard Presets":
        scenario_inputs_map = {
            "Conservative": apply_preset_to_inputs(base_inputs, "Conservative", runtime_presets),
            "Base Case": apply_preset_to_inputs(base_inputs, "Base Case", runtime_presets),
            "Optimistic": apply_preset_to_inputs(base_inputs, "Optimistic", runtime_presets),
        }
    elif scenario_mode == "Compare Asset Drawdown Strategies":
        strategy_base = (
            apply_preset_to_inputs(base_inputs, preset_choice, runtime_presets)
            if preset_choice in runtime_presets else base_inputs.copy()
        )
        scenario_inputs_map = {
            strategy_name: apply_strategy_profile(strategy_base, strategy_name)
            for strategy_name in STRATEGY_PROFILES
        }
    else:
        debt_base = (
            apply_preset_to_inputs(base_inputs, preset_choice, runtime_presets)
            if preset_choice in runtime_presets else base_inputs.copy()
        )
        scenario_inputs_map = {
            strategy_name: apply_debt_strategy_profile(debt_base, strategy_name)
            for strategy_name in DEBT_STRATEGY_PROFILES
        }

    scenario_inputs_map = {
        scenario_name: apply_module_scope(scenario_inputs)
        for scenario_name, scenario_inputs in scenario_inputs_map.items()
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
            scenario_result = run_scenario_cached(
                scenario_inputs,
                int(random_seed),
                cache_version="debt_strategy_v1",
            )

            input_warnings_by_scenario[scenario_name] = generate_input_warnings(scenario_inputs)
            output_warnings_by_scenario[scenario_name] = generate_output_warnings(
                scenario_result["summary_df"],
                scenario_result["failure_prob_df"],
                scenario_result["det_df"],
            )

            comparison_results[scenario_name] = scenario_result

        st.session_state.comparison_results = comparison_results
        st.session_state.assumption_details_df = build_assumption_details_df(scenario_inputs_map)
        st.session_state.input_summary_df = build_input_summary_df(scenario_inputs_map)
        st.session_state.contribution_schedule_export_df = build_contribution_schedule_export_df(scenario_inputs_map)
        st.session_state.input_warnings_by_scenario = input_warnings_by_scenario
        st.session_state.output_warnings_by_scenario = output_warnings_by_scenario
        st.session_state.last_run_inputs_by_scenario = scenario_inputs_map
        st.session_state.active_result_set_name = "Current Results"
        st.session_state.workspace_mode = "View Results"


# ============================================================
# SECTION: RESULTS RENDERING
# ============================================================

active_result_bundle = get_active_result_bundle()

if active_result_bundle is not None and workspace_mode == "View Results":
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

    comparison_rows = []
    det_scenarios_df_list = []
    for scenario_name, result in comparison_results.items():
        comparison_rows.append({
            "scenario": scenario_name,
            "success_rate": result["success_rate"],
            "median_final_wealth": result["median_final_wealth"],
            "p10_final_wealth": result["p10_final_wealth"],
            "p90_final_wealth": result["p90_final_wealth"],
        })
        temp_det_df = result["det_df"].copy()
        temp_det_df["scenario"] = scenario_name
        det_scenarios_df_list.append(temp_det_df)

    comparison_df = format_comparison_df(pd.DataFrame(comparison_rows))
    common_inputs = next(iter(comparison_results.values()))["inputs"]
    comparison_df = convert_comparison_df_for_value_mode(comparison_df, common_inputs, value_mode)
    det_scenarios_df = pd.concat(det_scenarios_df_list, ignore_index=True)
    det_scenarios_df = convert_det_df_for_value_mode(det_scenarios_df, common_inputs, value_mode)

    selected_scenario = st.selectbox(
        t("Select Scenario", "选择情景"),
        options=list(comparison_results.keys()),
        key="selected_scenario_results",
    )
    selected_result = comparison_results[selected_scenario]
    display_det_df = convert_det_df_for_value_mode(selected_result["det_df"], selected_result["inputs"], value_mode)
    display_percentile_df = convert_percentile_df_for_value_mode(selected_result["percentile_df"], selected_result["inputs"], value_mode)
    display_summary_df = convert_summary_df_for_value_mode(selected_result["summary_df"], selected_result["inputs"], value_mode)
    pdf_selected_result = {
        **selected_result,
        "det_df": display_det_df,
        "percentile_df": display_percentile_df,
        "summary_df": display_summary_df,
    }
    pdf_comparison_results = {
        scenario_name: {
            **result,
            "det_df": convert_det_df_for_value_mode(result["det_df"], result["inputs"], value_mode),
        }
        for scenario_name, result in comparison_results.items()
    }

    if is_one_person_inputs(selected_result["inputs"]):
        st.info(t("This result is a single-person projection. Person 2 is fully excluded from inputs, calculations, charts, tables, and exports.", "当前结果为单人预测。Person 2 已从输入、计算、图表、表格和导出中完全排除。"))

    selected_success_rate = selected_result["success_rate"]
    selected_median_final_wealth = display_summary_df["final_wealth"].median()
    selected_p10_final_wealth = display_summary_df["final_wealth"].quantile(0.10)
    selected_p90_final_wealth = display_summary_df["final_wealth"].quantile(0.90)

    det_single_compare_df = display_det_df.copy()
    det_single_compare_df["scenario"] = selected_scenario

    if view_mode == "Adviser View":
        st.subheader(t(f"Adviser Summary - {selected_scenario}", f"顾问摘要 - {selected_scenario}"))
        st.info(t("Adviser Note: Outputs are indicative only and should be reviewed in the context of client objectives, risk profile, and current legislation before forming advice.", "顾问提示：本输出仅供指示参考，在形成建议前应结合客户目标、风险承受能力及现行法规进行审阅。"))

        top_col1, top_col2 = st.columns(2)
        top_col1.metric(t("Success Rate", "成功率"), f"{selected_success_rate:.1%}")
        top_col2.metric(t("Median Final Wealth", "最终财富中位数"), f"${selected_median_final_wealth:,.0f}")
        bottom_col1, bottom_col2, bottom_col3 = st.columns(3)
        bottom_col1.metric(t("P10 Final Wealth", "P10 最终财富"), f"${selected_p10_final_wealth:,.0f}")
        bottom_col2.metric(t("P90 Final Wealth", "P90 最终财富"), f"${selected_p90_final_wealth:,.0f}")
        bottom_col3.metric(t("Spread (P90 - P10)", "区间差值（P90 - P10）"), f"${selected_p90_final_wealth - selected_p10_final_wealth:,.0f}")

        adviser_sections = [
            "Overview",
            "Strategy Comparison",
            "Wealth Charts",
            "Monte Carlo",
            "Tax",
            "Cashflow",
            "Debug Tables",
            "Export",
        ]
        selected_module_inputs = selected_result["inputs"]
        if (
            selected_module_inputs.get("module_investment_debt_enabled", False)
            or float(selected_module_inputs.get("main_residence_loan_balance", 0.0)) > 0
            or float(selected_module_inputs.get("residential_property_loan_balance", 0.0)) > 0
        ):
            adviser_sections.insert(2, "Debt Strategies")
        adviser_section_labels = {
            "Overview": t("Overview", "总览"),
            "Strategy Comparison": t("Strategy Comparison", "策略对比"),
            "Debt Strategies": t("Debt Strategies", "债务策略"),
            "Wealth Charts": t("Wealth Charts", "财富图表"),
            "Monte Carlo": t("Monte Carlo", "蒙特卡洛"),
            "Tax": t("Tax", "税务"),
            "Cashflow": t("Cashflow", "现金流"),
            "Debug Tables": t("Debug Tables", "调试表"),
            "Export": t("Export", "导出"),
        }
        stored_adviser_section = st.session_state.get("adviser_result_section", "Overview")
        if stored_adviser_section not in adviser_sections:
            stored_adviser_section = "Overview"
        adviser_result_section_label = st.radio(
            t("Adviser Display Section", "顾问显示区"),
            options=[adviser_section_labels[option] for option in adviser_sections],
            index=adviser_sections.index(stored_adviser_section),
            horizontal=True,
            help=t("Only the selected section is rendered. This keeps result navigation fast.", "只渲染当前选择的区域，从而提升结果页切换速度。"),
        )
        adviser_result_section = adviser_sections[
            [adviser_section_labels[option] for option in adviser_sections].index(adviser_result_section_label)
        ]
        st.session_state.adviser_result_section = adviser_result_section

        if adviser_result_section == "Overview":
            render_saved_result_comparison_section(st.session_state.get("saved_result_sets", {}), value_mode)
            render_assumption_details(assumption_details_df)
            render_warning_sections(input_warnings_by_scenario, output_warnings_by_scenario, view_mode)

            st.subheader(t("Scenario Comparison Summary", "情景比较摘要"))
            st.dataframe(
                comparison_df[["scenario", "success_rate_label", "median_final_wealth_label", "p10_final_wealth_label", "p90_final_wealth_label"]],
                width="stretch",
            )

            success_fig = create_success_rate_comparison_chart(comparison_df)
            success_fig.update_layout(title=t("Success Rate by Scenario", "各情景成功率"), xaxis_title=t("Scenario", "情景"), yaxis_title=t("Success Rate", "成功率"))
            st.plotly_chart(success_fig, width="stretch", key="success_rate_comparison")

            median_fig = create_median_wealth_comparison_chart(comparison_df)
            median_fig.update_layout(title=t("Median Final Wealth by Scenario", "各情景最终财富中位数"), xaxis_title=t("Scenario", "情景"), yaxis_title=t("Median Final Wealth", "最终财富中位数"))
            st.plotly_chart(median_fig, width="stretch", key="median_wealth_comparison")

        elif adviser_result_section == "Strategy Comparison":
            st.subheader(t("Strategy Comparison", "策略对比"))
            st.caption(t(
                "Compares asset drawdown order, after-tax cashflow, retirement wealth, final wealth, failure probability, cumulative tax, advantage timing and key risks. Differences are modelled outcomes, not personal advice.",
                "比较资产提取顺序、税后现金流、退休时财富、最终财富、失败概率、累计税、优势起点、break-even 和关键风险。差异属于模型结果，不构成个人建议。",
            ))
            strategy_df = build_strategy_comparison_df(comparison_results, is_chinese=is_cn())
            assumption_change_df = build_assumption_change_df(comparison_results, is_chinese=is_cn())
            if strategy_df.empty:
                st.info(t("Run at least one scenario to build the comparison.", "请至少运行一个情景以生成比较。"))
            else:
                display_strategy_df = strategy_df.copy()
                display_strategy_df["failure_probability"] = display_strategy_df["failure_probability"].map(lambda value: f"{value:.1%}")
                for column in ["after_tax_cashflow", "retirement_wealth", "final_wealth", "final_wealth_delta", "cumulative_tax", "cumulative_tax_delta"]:
                    display_strategy_df[column] = display_strategy_df[column].map(lambda value: f"${value:,.0f}")
                display_strategy_df["first_advantage_year"] = display_strategy_df["first_advantage_year"].map(lambda value: "-" if pd.isna(value) else f"FY{int(value)}")
                display_strategy_df["break_even_year"] = display_strategy_df["break_even_year"].map(lambda value: "-" if pd.isna(value) else f"FY{int(value)}")
                display_strategy_df = display_strategy_df.rename(columns={
                    "scenario": t("Scenario", "情景"),
                    "strategy_description": t("Strategy", "策略"),
                    "after_tax_cashflow": t("Cumulative After-tax Cashflow", "累计税后现金流"),
                    "retirement_wealth": t("Wealth at Retirement", "退休时财富"),
                    "final_wealth": t("Final Wealth", "最终财富"),
                    "final_wealth_delta": t("Final Wealth vs Base", "最终财富较 Base 差异"),
                    "failure_probability": t("Failure Probability", "失败概率"),
                    "cumulative_tax": t("Cumulative Tax", "累计税款"),
                    "cumulative_tax_delta": t("Tax vs Base", "税款较 Base 差异"),
                    "first_advantage_year": t("First Advantage", "首次产生优势"),
                    "break_even_year": t("Break-even", "收支平衡年"),
                    "key_risks": t("Key Risks", "关键风险"),
                })
                st.dataframe(display_strategy_df, width="stretch", hide_index=True)

                sc1, sc2 = st.columns(2)
                with sc1:
                    wealth_delta_fig = px.bar(
                        strategy_df,
                        x="scenario",
                        y="final_wealth_delta",
                        color="scenario",
                        title=t("Final Wealth Difference vs Base Case", "最终财富相对 Base Case 的差异"),
                    )
                    wealth_delta_fig.update_layout(showlegend=False, xaxis_title=t("Scenario", "情景"), yaxis_title=t("Difference", "差异"))
                    st.plotly_chart(wealth_delta_fig, width="stretch", key="strategy_final_wealth_delta")
                with sc2:
                    tax_delta_fig = px.bar(
                        strategy_df,
                        x="scenario",
                        y="cumulative_tax_delta",
                        color="scenario",
                        title=t("Cumulative Tax Difference vs Base Case", "累计税款相对 Base Case 的差异"),
                    )
                    tax_delta_fig.update_layout(showlegend=False, xaxis_title=t("Scenario", "情景"), yaxis_title=t("Difference", "差异"))
                    st.plotly_chart(tax_delta_fig, width="stretch", key="strategy_tax_delta")

                st.subheader(t("Key Assumption Changes", "关键假设变化"))
                st.dataframe(assumption_change_df, width="stretch", hide_index=True)
                st.session_state.adviser_notes = st.text_area(
                    t("Adviser Notes", "顾问备注"),
                    value=st.session_state.get("adviser_notes", ""),
                    height=130,
                    help=t("Included in Advice Support and Technical Appendix reports.", "将纳入 Advice Support 和 Technical Appendix 报告。"),
                )

        elif adviser_result_section == "Debt Strategies":
            st.subheader(t("Debt Repayment & Surplus Allocation Comparison", "债务偿还与盈余分配比较"))
            st.caption(t(
                "Compares annual interest, tax, debt-free timing, ending debt, offset liquidity, final wealth and failure probability. A lower deductible-interest bill can also reduce tax deductions, so interest saved and tax paid should be considered together.",
                "比较年度利息、税款、清债时间、期末债务、Offset 流动性、最终财富及失败概率。降低可抵扣利息也会减少税务扣除，因此应结合利息节省与税款变化一并判断。",
            ))
            debt_df = build_debt_strategy_comparison_df(comparison_results, is_chinese=is_cn())
            if debt_df.empty:
                st.info(t("Run a scenario to build the debt comparison.", "请先运行情景以生成债务比较。"))
            else:
                display_debt_df = debt_df.copy()
                for column in [
                    "cumulative_interest",
                    "interest_saved_vs_base",
                    "cumulative_tax",
                    "ending_non_deductible_debt",
                    "ending_deductible_debt",
                    "ending_property_loan",
                    "ending_home_loan",
                    "ending_offset_balance",
                    "final_wealth",
                    "final_wealth_delta",
                ]:
                    display_debt_df[column] = display_debt_df[column].map(lambda value: f"${value:,.0f}")
                display_debt_df["failure_probability"] = display_debt_df["failure_probability"].map(lambda value: f"{value:.1%}")
                for column in ["non_deductible_debt_free_year", "deductible_debt_free_year", "property_loan_free_year", "home_loan_free_year"]:
                    display_debt_df[column] = display_debt_df[column].map(lambda value: "-" if pd.isna(value) else f"FY{int(value)}")
                display_debt_df = display_debt_df.rename(columns={
                    "scenario": t("Scenario", "情景"),
                    "strategy_description": t("Strategy", "策略"),
                    "allocation_order": t("Surplus Allocation Order", "盈余分配顺序"),
                    "cumulative_interest": t("Cumulative Interest", "累计利息"),
                    "interest_saved_vs_base": t("Interest Saved vs Base", "较 Base 节省利息"),
                    "cumulative_tax": t("Cumulative Tax", "累计税款"),
                    "ending_non_deductible_debt": t("Ending Other Non-deductible Investment Debt", "期末其他不可抵扣投资债务"),
                    "ending_deductible_debt": t("Ending Other Deductible Investment Debt", "期末其他可抵扣投资债务"),
                    "ending_property_loan": t("Ending Investment Property Loan", "期末投资物业贷款"),
                    "ending_home_loan": t("Ending Home Loan", "期末自住房贷款"),
                    "ending_offset_balance": t("Ending Offset", "期末 Offset"),
                    "non_deductible_debt_free_year": t("Other Non-deductible Debt-free", "其他不可抵扣投资债务清偿年"),
                    "deductible_debt_free_year": t("Other Deductible Debt-free", "其他可抵扣投资债务清偿年"),
                    "property_loan_free_year": t("Property Loan Debt-free", "投资物业贷款清偿年"),
                    "home_loan_free_year": t("Home Loan Debt-free", "自住房贷款清偿年"),
                    "final_wealth": t("Final Wealth", "最终财富"),
                    "final_wealth_delta": t("Final Wealth vs Base", "最终财富较 Base 差异"),
                    "failure_probability": t("Failure Probability", "失败概率"),
                })
                st.dataframe(display_debt_df, width="stretch", hide_index=True)

                dc1, dc2 = st.columns(2)
                with dc1:
                    interest_fig = px.bar(
                        debt_df,
                        x="scenario",
                        y="cumulative_interest",
                        color="scenario",
                        title=t("Cumulative Debt Interest", "累计债务利息"),
                    )
                    interest_fig.update_layout(showlegend=False, xaxis_title=t("Scenario", "情景"), yaxis_title=t("Interest", "利息"))
                    st.plotly_chart(interest_fig, width="stretch", key="debt_strategy_interest")
                with dc2:
                    debt_fig = px.bar(
                        debt_df,
                        x="scenario",
                        y=["ending_home_loan", "ending_property_loan", "ending_non_deductible_debt", "ending_deductible_debt"],
                        barmode="stack",
                        title=t("Ending Debt by Type", "按类别划分的期末债务"),
                        labels={"value": t("Debt", "债务"), "variable": t("Debt Type", "债务类别")},
                    )
                    debt_fig.update_layout(xaxis_title=t("Scenario", "情景"), yaxis_title=t("Debt", "债务"))
                    st.plotly_chart(debt_fig, width="stretch", key="debt_strategy_ending_debt")

                debt_path_columns = [
                    "financial_year_end",
                    "non_deductible_debt_balance",
                    "residential_property_loan_balance",
                    "main_residence_loan_balance",
                    "investment_deductible_debt_balance",
                    "non_deductible_offset_balance",
                    "deductible_offset_balance",
                    "main_residence_offset_balance",
                    "investment_deductible_offset_balance",
                ]
                path_frames = []
                for scenario_name, result in comparison_results.items():
                    available = [column for column in debt_path_columns if column in result["det_df"].columns]
                    path = result["det_df"][available].copy()
                    path["scenario"] = scenario_name
                    path_frames.append(path)
                if path_frames:
                    debt_paths = pd.concat(path_frames, ignore_index=True)
                    debt_paths["net_debt"] = (
                        debt_paths.get("non_deductible_debt_balance", 0)
                        + debt_paths.get("residential_property_loan_balance", 0)
                        + debt_paths.get("main_residence_loan_balance", 0)
                        + debt_paths.get("investment_deductible_debt_balance", 0)
                        - debt_paths.get("non_deductible_offset_balance", 0)
                        - debt_paths.get("deductible_offset_balance", 0)
                        - debt_paths.get("main_residence_offset_balance", 0)
                        - debt_paths.get("investment_deductible_offset_balance", 0)
                    )
                    debt_path_fig = px.line(
                        debt_paths,
                        x="financial_year_end",
                        y="net_debt",
                        color="scenario",
                        title=t("Net Debt Projection", "净债务预测"),
                    )
                    debt_path_fig.update_layout(xaxis_title=t("Financial Year", "财政年度"), yaxis_title=t("Net Debt", "净债务"))
                    st.plotly_chart(debt_path_fig, width="stretch", key="debt_strategy_paths")

        elif adviser_result_section == "Wealth Charts":
            st.subheader(t("Wealth Charts", "财富图表"))
            det_all_fig = create_deterministic_wealth_chart_comparison(det_scenarios_df, common_inputs)
            det_all_fig.update_layout(title=t("Deterministic Total Wealth Projection", "确定性总财富预测"), xaxis_title=t("Financial Year", "财政年度"), yaxis_title=t("Total Wealth", "总财富"))
            st.plotly_chart(det_all_fig, width="stretch", key=chart_key("deterministic_all", selected_scenario, view_mode, "adviser_lazy"))

            income_spending_fig = create_income_vs_spending_chart(display_det_df, selected_result["inputs"], t(f"Income vs Spending - {selected_scenario}", f"收入与支出 - {selected_scenario}"))
            income_spending_fig.update_layout(xaxis_title=t("Financial Year", "财政年度"), yaxis_title=t("Annual Amount", "年度金额"))
            st.plotly_chart(income_spending_fig, width="stretch", key=chart_key("income_spending", selected_scenario, view_mode, "adviser_lazy"))

        elif adviser_result_section == "Monte Carlo":
            st.subheader(t("Monte Carlo", "蒙特卡洛"))
            percentile_fig = create_percentile_paths_chart(display_percentile_df, selected_result["inputs"], t(f"Monte Carlo Percentile Paths - {selected_scenario}", f"蒙特卡洛百分位路径 - {selected_scenario}"))
            percentile_fig.update_layout(xaxis_title=t("Financial Year", "财政年度"), yaxis_title=t("Total Wealth", "总财富"))
            st.plotly_chart(percentile_fig, width="stretch", key=chart_key("percentile", selected_scenario, view_mode, "adviser_lazy"))

            failure_fig = create_failure_probability_chart(selected_result["failure_prob_df"], selected_result["inputs"], t(f"Cumulative Probability of Running Out of Money - {selected_scenario}", f"资金耗尽累计概率 - {selected_scenario}"))
            failure_fig.update_layout(xaxis_title=t("Financial Year", "财政年度"), yaxis_title=t("Failure Probability", "资金耗尽概率"))
            st.plotly_chart(failure_fig, width="stretch", key=chart_key("failure", selected_scenario, view_mode, "adviser_lazy"))

            histogram_fig = create_histogram(display_summary_df, show_p10=True, show_p50=True, show_p90=True, title_text=t(f"Distribution of Final Wealth - {selected_scenario}", f"最终财富分布 - {selected_scenario}"))
            histogram_fig.update_layout(xaxis_title=t("Final Wealth", "最终财富"), yaxis_title=t("Frequency", "次数"))
            st.plotly_chart(histogram_fig, width="stretch", key=chart_key("histogram", selected_scenario, view_mode, "adviser_lazy"))

        elif adviser_result_section == "Tax":
            st.subheader(t("Tax", "税务"))
            tax_breakdown_fig = create_tax_breakdown_chart(display_det_df, selected_result["inputs"], t(f"Tax Breakdown - {selected_scenario}", f"税务明细 - {selected_scenario}"))
            tax_breakdown_fig.update_layout(xaxis_title=t("Financial Year", "财政年度"), yaxis_title=t("Annual Tax", "年度税款"))
            st.plotly_chart(tax_breakdown_fig, width="stretch", key=chart_key("tax_breakdown", selected_scenario, view_mode, "adviser_lazy"))

            total_tax_fig = create_total_tax_paid_chart(display_det_df, selected_result["inputs"], t(f"Total Tax Paid - {selected_scenario}", f"总税款 - {selected_scenario}"))
            total_tax_fig.update_layout(xaxis_title=t("Financial Year", "财政年度"), yaxis_title=t("Annual Tax", "年度税款"))
            st.plotly_chart(total_tax_fig, width="stretch", key=chart_key("total_tax", selected_scenario, view_mode, "adviser_lazy"))

            if selected_result["inputs"].get("module_property_enabled", False) or selected_result["inputs"].get("module_trust_enabled", False):
                st.subheader(t("Residential Property & Trust Tax Detail", "住宅物业与信托税务明细"))
                if selected_result["inputs"].get("module_trust_enabled", False):
                    st.caption(t(
                        "Trust minimum tax rows are exposure-draft estimates, not enacted-law outcomes.",
                        "信托最低税栏位为 exposure draft 估算，并非已生效法案结果。",
                    ))
                st.dataframe(build_residential_trust_tax_detail_df(display_det_df), width="stretch")

            st.subheader(t("2026 Budget CGT Reconciliation", "2026 Budget CGT 对账"))
            st.caption(t(
                "Core CGT reform is enacted. Transition allocation and CPI values shown here are pooled planning estimates and require asset-level tax-return reconciliation.",
                "CGT 核心改革已经立法；此处的过渡分配和 CPI 数值属于汇总规划估算，报税时必须按单项资产对账。",
            ))
            st.dataframe(build_cgt_validation_df(display_det_df, selected_result["inputs"]), width="stretch")

            pension_tax_free_summary_df = build_pension_tax_free_summary_df(display_det_df, selected_result["inputs"])
            st.subheader(t("Pension Tax-Free Validation Summary", "退休金免税验证摘要"))
            st.dataframe(pension_tax_free_summary_df, width="stretch")

        elif adviser_result_section == "Cashflow":
            st.subheader(t("Cashflow", "现金流"))
            if create_cashflow_chart is None:
                st.info(t(
                    "The cash flow chart is temporarily unavailable while the app update completes. The cashflow tables remain available below.",
                    "应用更新完成前，现金流图暂时不可用；下方现金流表格仍可正常查看。",
                ))
            else:
                cashflow_fig = create_cashflow_chart(
                    display_det_df,
                    selected_result["inputs"],
                    t(f"Cash Flow - {selected_scenario}", f"现金流 - {selected_scenario}"),
                )
                st.plotly_chart(
                    cashflow_fig,
                    width="stretch",
                    key=chart_key("cashflow", selected_scenario, view_mode, "adviser_lazy"),
                )

            adviser_cashflow_df = build_adviser_cashflow_df(display_det_df)
            st.subheader(t("Adviser Cashflow Summary", "顾问现金流摘要"))
            st.dataframe(adviser_cashflow_df, width="stretch")

            adviser_cashflow_asset_movement_tax_df = build_adviser_cashflow_asset_movement_tax_df(display_det_df, selected_result["inputs"])
            st.subheader(t("Cashflow, Net Asset Movement & Income Tax by Person", "现金流、净资产变动与个人所得税明细"))
            st.caption(t(
                "This table separates cashflow, net asset movement, and total income tax per person for adviser review.",
                "该表将现金流、净资产变动以及每个人的总所得税拆开，供顾问审阅。",
            ))
            st.dataframe(adviser_cashflow_asset_movement_tax_df, width="stretch")

        elif adviser_result_section == "Debug Tables":
            st.subheader(t("Debug Tables", "调试表"))
            missing_validation_cols = get_missing_validation_columns(display_det_df, selected_result["inputs"])
            if missing_validation_cols:
                st.warning(t("Validation table is using fallback zeros for missing columns.", "验证表对缺失栏位使用了回退零值。"))
                st.caption(", ".join(missing_validation_cols))

            debug_df = build_adviser_debug_df(display_det_df, selected_result["inputs"])
            st.subheader(t("Adviser Debug Table", "顾问调试表"))
            st.dataframe(debug_df, width="stretch")

            if st.checkbox(t("Show detailed CGT / pension validation table", "显示详细 CGT / 退休金验证表"), value=False):
                cgt_validation_df = build_cgt_validation_df(display_det_df, selected_result["inputs"])
                st.dataframe(cgt_validation_df, width="stretch")

            if st.checkbox(t("Show full deterministic projection table", "显示完整确定性预测表"), value=False):
                key_cols = [col for col in ["financial_year_end", "total_wealth", "ending_total_super_balance", "ending_non_super_balance", "spending", "total_tax_paid", "unmet_shortfall"] if col in display_det_df.columns]
                st.dataframe(display_det_df[key_cols], width="stretch")

            if st.checkbox(t("Show Monte Carlo summary tables", "显示蒙特卡洛摘要表"), value=False):
                st.dataframe(display_summary_df[[col for col in ["simulation_id", "success", "final_wealth"] if col in display_summary_df.columns]], width="stretch")
                st.dataframe(display_percentile_df, width="stretch")
                st.dataframe(selected_result["failure_prob_df"], width="stretch")

        elif adviser_result_section == "Export":
            st.subheader(t("Export", "导出"))
            render_pdf_export_controls(
                selected_result=pdf_selected_result,
                comparison_results=pdf_comparison_results,
                selected_scenario=selected_scenario,
                value_mode=value_mode,
                input_warnings=(input_warnings_by_scenario or {}).get(selected_scenario, []),
                output_warnings=(output_warnings_by_scenario or {}).get(selected_scenario, []),
                widget_scope=f"{active_name_display}_{selected_scenario}_{value_mode}_adviser",
            )

            st.divider()
            st.subheader(t("Excel Workbook", "Excel 工作簿"))
            st.caption(t("Excel is prepared only on demand to avoid slowing down result navigation.", "Excel 只在需要时生成，避免拖慢结果页切换。"))
            prepare_export = st.button(t("Prepare Excel Export", "准备 Excel 导出"), width="stretch")
            if prepare_export:
                adviser_cashflow_df = build_adviser_cashflow_df(display_det_df)
                adviser_cashflow_asset_movement_tax_df = build_adviser_cashflow_asset_movement_tax_df(display_det_df, selected_result["inputs"])
                pension_tax_free_summary_df = build_pension_tax_free_summary_df(display_det_df, selected_result["inputs"])
                debug_df = build_adviser_debug_df(display_det_df, selected_result["inputs"])
                cgt_validation_df = build_cgt_validation_df(display_det_df, selected_result["inputs"])
                export_tables = {
                    "input_summary": input_summary_df,
                    "assumption_details": assumption_details_df,
                    "contribution_schedule": contribution_schedule_export_df,
                    "deterministic_projection": display_det_df,
                    "simulation_summary": display_summary_df,
                    "percentile_table": display_percentile_df,
                    "failure_probability": selected_result["failure_prob_df"],
                    "adviser_cashflow_summary": adviser_cashflow_df,
                    "cashflow_asset_tax_detail": adviser_cashflow_asset_movement_tax_df,
                    "adviser_debug_table": debug_df,
                    "pension_tax_free_summary": pension_tax_free_summary_df,
                    "cgt_validation_detail": cgt_validation_df,
                }
                if selected_result["inputs"].get("module_property_enabled", False) or selected_result["inputs"].get("module_trust_enabled", False):
                    export_tables["property_trust_tax"] = build_residential_trust_tax_detail_df(display_det_df)
                excel_file = dataframe_to_excel_bytes(export_tables)
                st.download_button(
                    label=t("Download Excel", "下载 Excel"),
                    data=excel_file,
                    file_name=build_export_filename(selected_result["inputs"].get("report_title", ""), "financial_projection", selected_scenario, "xlsx"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )

    else:
        st.subheader(t(f"Client Summary - {selected_scenario}", f"客户摘要 - {selected_scenario}"))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(t("Success Rate", "成功率"), f"{selected_success_rate:.1%}")
        col2.metric(t("Median Final Wealth", "最终财富中位数"), f"${selected_median_final_wealth:,.0f}")
        col3.metric(t("P10 Final Wealth", "P10 最终财富"), f"${selected_p10_final_wealth:,.0f}")
        col4.metric(t("P90 Final Wealth", "P90 最终财富"), f"${selected_p90_final_wealth:,.0f}")

        render_warning_sections(input_warnings_by_scenario, output_warnings_by_scenario, view_mode)

        det_fig = create_deterministic_wealth_chart_comparison(det_single_compare_df, selected_result["inputs"])
        det_fig.update_layout(title=t("Deterministic Total Wealth Projection", "确定性总财富预测"), xaxis_title=t("Financial Year", "财政年度"), yaxis_title=t("Total Wealth", "总财富"))
        st.plotly_chart(det_fig, width="stretch", key=chart_key("deterministic", selected_scenario, view_mode, "client"))

        percentile_fig = create_percentile_paths_chart(display_percentile_df, selected_result["inputs"], t(f"Monte Carlo Percentile Paths - {selected_scenario}", f"蒙特卡洛百分位路径 - {selected_scenario}"))
        percentile_fig.update_layout(xaxis_title=t("Financial Year", "财政年度"), yaxis_title=t("Total Wealth", "总财富"))
        st.plotly_chart(percentile_fig, width="stretch", key=chart_key("percentile", selected_scenario, view_mode, "client"))

        failure_fig = create_failure_probability_chart(selected_result["failure_prob_df"], selected_result["inputs"], t(f"Cumulative Probability of Running Out of Money - {selected_scenario}", f"资金耗尽累计概率 - {selected_scenario}"))
        failure_fig.update_layout(xaxis_title=t("Financial Year", "财政年度"), yaxis_title=t("Failure Probability", "资金耗尽概率"))
        st.plotly_chart(failure_fig, width="stretch", key=chart_key("failure", selected_scenario, view_mode, "client"))

        with st.expander(t("Export PDF Report", "导出 PDF 报告"), expanded=False):
            render_pdf_export_controls(
                selected_result=pdf_selected_result,
                comparison_results=pdf_comparison_results,
                selected_scenario=selected_scenario,
                value_mode=value_mode,
                input_warnings=(input_warnings_by_scenario or {}).get(selected_scenario, []),
                output_warnings=(output_warnings_by_scenario or {}).get(selected_scenario, []),
                widget_scope=f"{active_name_display}_{selected_scenario}_{value_mode}_client",
            )

elif active_result_bundle is not None and workspace_mode != "View Results":
    st.info(t(
        "Results are available but hidden while you are editing inputs. Switch Workspace to 'View Results' in the sidebar to render charts and tables.",
        "结果已经存在，但在编辑输入时已隐藏以提升速度。请在侧边栏把工作区切换为“查看结果”来显示图表和表格。",
    ))
else:
    st.info(t("Adjust the inputs above, then click Run Simulation in the sidebar. Saved snapshots can also be reopened from the sidebar.", "请先在上方区域调整输入，再点击侧边栏中的“运行模拟”。已保存快照也可以在侧边栏重新打开。"))
