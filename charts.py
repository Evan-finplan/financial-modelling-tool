
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


VIVID_COLOURWAY = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#17BECF",
    "#BCBD22",
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
]

pio.templates["jbwere"] = go.layout.Template(
    layout=go.Layout(
        colorway=VIVID_COLOURWAY,
        font=dict(family="Arial, sans-serif", size=12, color="#182A3A"),
        title=dict(font=dict(family="Arial, sans-serif", size=18, color="#00205B")),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(gridcolor="#DDE6EB", linecolor="#5B91A4", zerolinecolor="#C7D6DE"),
        yaxis=dict(gridcolor="#DDE6EB", linecolor="#5B91A4", zerolinecolor="#C7D6DE"),
        legend=dict(bgcolor="rgba(255,255,255,0.82)"),
    )
)
pio.templates.default = "plotly_white+jbwere"
px.defaults.color_discrete_sequence = VIVID_COLOURWAY


# ============================================================
# SECTION: CHART HELPERS
# ============================================================

def _person_label(inputs, key, fallback):
    name = str(inputs.get(key, "") or "").strip()
    return name if name else fallback


def _is_chinese_chart(inputs):
    language_value = str(inputs.get("ui_language", ""))
    return "中文" in language_value or "CN" in language_value.upper()


def _chart_text(inputs, en, zh):
    return zh if _is_chinese_chart(inputs) else en


def _add_lifecycle_markers(fig, inputs):
    start_fy_end = int(str(inputs["start_financial_year"]).replace("FY", ""))
    one_person_mode = str(inputs.get("household_mode", "Two People")) == "One Person"

    p1_label = _person_label(inputs, "person1_name", "P1")
    p2_label = _person_label(inputs, "person2_name", "P2")

    retirement_label = _chart_text(inputs, "Retirement", "退休")
    pension_start_label = _chart_text(inputs, "Pension Start", "退休金开始")

    raw_markers = [
        (start_fy_end + (inputs["person1_retirement_age"] - inputs["person1_current_age"]), f"{p1_label} {retirement_label}"),
        (start_fy_end + (inputs["person1_pension_start_age"] - inputs["person1_current_age"]), f"{p1_label} {pension_start_label}"),
    ]

    if not one_person_mode:
        raw_markers.extend([
            (start_fy_end + (inputs["person2_retirement_age"] - inputs["person2_current_age"]), f"{p2_label} {retirement_label}"),
            (start_fy_end + (inputs["person2_pension_start_age"] - inputs["person2_current_age"]), f"{p2_label} {pension_start_label}"),
        ])

    year_groups = {}
    seen = set()
    for x_value, label in raw_markers:
        dedupe_key = (x_value, label)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        year_groups.setdefault(x_value, []).append(label)

    line_colours = ["rgba(80,80,80,0.55)", "rgba(80,80,80,0.45)", "rgba(80,80,80,0.35)", "rgba(80,80,80,0.25)"]
    y_positions = [1.11, 1.06, 1.01, 0.96]

    for x_value, labels in sorted(year_groups.items()):
        fig.add_vline(x=x_value, line_dash="dash", line_color="rgba(90,90,90,0.45)")
        for idx, label in enumerate(labels):
            fig.add_annotation(
                x=x_value,
                y=y_positions[idx % len(y_positions)],
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                xanchor="left" if idx % 2 == 0 else "right",
                align="left",
                bgcolor="rgba(255,255,255,0.82)",
                bordercolor=line_colours[idx % len(line_colours)],
                borderwidth=1,
                borderpad=2,
                font=dict(size=11),
            )

    return fig


def _format_currency_axis(fig, axis_name="y"):
    if axis_name == "y":
        fig.update_yaxes(tickprefix="$", separatethousands=True)
    elif axis_name == "x":
        fig.update_xaxes(tickprefix="$", separatethousands=True)
    return fig


# ============================================================
# SECTION: CORE COMPARISON CHARTS
# ============================================================

def create_deterministic_wealth_chart_comparison(det_scenarios_df, inputs):
    fig = px.line(
        det_scenarios_df,
        x="financial_year_end",
        y="total_wealth",
        color="scenario",
        title=_chart_text(inputs, "Deterministic Total Wealth Projection", "确定性总财富预测"),
    )

    fig = _add_lifecycle_markers(fig, inputs)

    fig.update_layout(
        xaxis_title=_chart_text(inputs, "Financial Year", "财政年度"),
        yaxis_title=_chart_text(inputs, "Total Wealth", "总财富"),
        hovermode="x unified",
    )

    for trace in fig.data:
        trace.hovertemplate = (
            "Financial Year: %{x}FY<br>"
            + "Total Wealth: $%{y:,.0f}<br>"
            + "Scenario: %{fullData.name}<extra></extra>"
        )

    return _format_currency_axis(fig, "y")


# ============================================================
# SECTION: MONTE CARLO CHARTS
# ============================================================

def create_percentile_paths_chart(percentile_df, inputs, title_text):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=percentile_df["financial_year_end"],
            y=percentile_df["p10"],
            mode="lines",
            name="P10",
            line=dict(color="#D62728", width=3),
            hovertemplate="Financial Year: %{x}FY<br>P10: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=percentile_df["financial_year_end"],
            y=percentile_df["p50"],
            mode="lines",
            name="P50",
            line=dict(color="#1F77B4", width=4),
            hovertemplate="Financial Year: %{x}FY<br>P50: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=percentile_df["financial_year_end"],
            y=percentile_df["p90"],
            mode="lines",
            name="P90",
            line=dict(color="#2CA02C", width=3),
            hovertemplate="Financial Year: %{x}FY<br>P90: $%{y:,.0f}<extra></extra>",
        )
    )

    fig = _add_lifecycle_markers(fig, inputs)

    fig.update_layout(
        title=title_text,
        xaxis_title=_chart_text(inputs, "Financial Year", "财政年度"),
        yaxis_title=_chart_text(inputs, "Total Wealth", "总财富"),
        hovermode="x unified",
    )

    return _format_currency_axis(fig, "y")


def create_failure_probability_chart(failure_prob_df, inputs, title_text):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=failure_prob_df["financial_year_end"],
            y=failure_prob_df["failure_probability"],
            mode="lines",
            name="Failure Probability",
            line=dict(color="#D62728", width=4),
            hovertemplate="Financial Year: %{x}FY<br>Failure Probability: %{y:.1%}<extra></extra>",
        )
    )

    fig = _add_lifecycle_markers(fig, inputs)

    fig.update_layout(
        title=title_text,
        xaxis_title=_chart_text(inputs, "Financial Year", "财政年度"),
        yaxis_title=_chart_text(inputs, "Failure Probability", "资金耗尽概率"),
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


# ============================================================
# SECTION: TAX AND CASHFLOW CHARTS
# ============================================================

def create_tax_breakdown_chart(det_df, inputs, title_text):
    one_person_mode = str(inputs.get("household_mode", "Two People")) == "One Person"

    tax_columns = [
        "person1_income_tax",
        "person1_medicare_levy",
        "person1_income_tax_on_non_super_earnings",
        "person1_medicare_levy_on_non_super_earnings",
        "person1_division_293_tax",
        "person1_super_contributions_tax",
        "person1_total_super_earnings_tax",
        "total_cgt_minimum_tax",
        "total_discretionary_trust_minimum_tax",
    ]

    if not one_person_mode:
        tax_columns.extend([
            "person2_income_tax",
            "person2_medicare_levy",
            "person2_income_tax_on_non_super_earnings",
            "person2_medicare_levy_on_non_super_earnings",
            "person2_division_293_tax",
            "person2_super_contributions_tax",
            "person2_total_super_earnings_tax",
        ])

    pretty_names = {
        "person1_income_tax": "P1 Salary Income Tax",
        "person1_medicare_levy": "P1 Salary Medicare Levy",
        "person1_income_tax_on_non_super_earnings": "P1 Non-Super Income Tax",
        "person1_medicare_levy_on_non_super_earnings": "P1 Non-Super Medicare Levy",
        "person2_income_tax": "P2 Salary Income Tax",
        "person2_medicare_levy": "P2 Salary Medicare Levy",
        "person2_income_tax_on_non_super_earnings": "P2 Non-Super Income Tax",
        "person2_medicare_levy_on_non_super_earnings": "P2 Non-Super Medicare Levy",
        "person1_division_293_tax": "P1 Division 293 Tax",
        "person2_division_293_tax": "P2 Division 293 Tax",
        "person1_super_contributions_tax": "P1 Super Contributions Tax",
        "person2_super_contributions_tax": "P2 Super Contributions Tax",
        "person1_total_super_earnings_tax": "P1 Super Earnings Tax",
        "person2_total_super_earnings_tax": "P2 Super Earnings Tax",
        "total_cgt_minimum_tax": "CGT Minimum-Tax Top-Up",
        "total_discretionary_trust_minimum_tax": "Trustee Minimum Tax (Draft)",
    }

    available_columns = [col for col in tax_columns if col in det_df.columns]
    if not available_columns:
        return go.Figure()

    fig = go.Figure()

    for col in available_columns:
        colour = VIVID_COLOURWAY[len(fig.data) % len(VIVID_COLOURWAY)]
        fig.add_trace(
            go.Bar(
                x=det_df["financial_year_end"],
                y=det_df[col],
                name=pretty_names.get(col, col),
                marker=dict(color=colour, line=dict(color="#FFFFFF", width=0.6)),
                hovertemplate="Financial Year: %{x}FY<br>"
                + f"{pretty_names.get(col, col)}: "
                + "$%{y:,.0f}<extra></extra>",
            )
        )

    fig = _add_lifecycle_markers(fig, inputs)

    fig.update_layout(
        title=title_text,
        barmode="stack",
        xaxis_title=_chart_text(inputs, "Financial Year", "财政年度"),
        yaxis_title=_chart_text(inputs, "Annual Tax", "年度税款"),
        hovermode="x unified",
    )

    return _format_currency_axis(fig, "y")


def create_income_vs_spending_chart(det_df, inputs, title_text):
    fig = go.Figure()
    one_person_mode = str(inputs.get("household_mode", "Two People")) == "One Person"

    series = [
        ("person1_gross_income", "P1 Gross Income"),
        ("person1_net_income", "P1 Net Income"),
        ("taxable_non_super_earnings_p1", "P1 Taxable Non-Super Income Return"),
        ("person1_min_pension_drawdown", "P1 Minimum Pension Drawdown"),
        ("spending", "Household Spending"),
        ("surplus_cash_to_non_super", "Surplus Cash to Non-Super"),
    ]

    if not one_person_mode:
        series[2:2] = [
            ("person2_gross_income", "P2 Gross Income"),
            ("person2_net_income", "P2 Net Income"),
        ]
        series.extend([
            ("taxable_non_super_earnings_p2", "P2 Taxable Non-Super Income Return"),
            ("person2_min_pension_drawdown", "P2 Minimum Pension Drawdown"),
        ])

    for column, label in series:
        if column in det_df.columns:
            colour = VIVID_COLOURWAY[len(fig.data) % len(VIVID_COLOURWAY)]
            fig.add_trace(
                go.Scatter(
                    x=det_df["financial_year_end"],
                    y=det_df[column],
                    mode="lines",
                    name=label,
                    line=dict(color=colour, width=3),
                    hovertemplate=f"Financial Year: %{{x}}FY<br>{label}: $%{{y:,.0f}}<extra></extra>",
                )
            )

    fig = _add_lifecycle_markers(fig, inputs)

    fig.update_layout(
        title=title_text,
        xaxis_title=_chart_text(inputs, "Financial Year", "财政年度"),
        yaxis_title=_chart_text(inputs, "Annual Amount", "年度金额"),
        hovermode="x unified",
    )

    return _format_currency_axis(fig, "y")


def create_cashflow_chart(det_df, inputs, title_text):
    years = det_df["financial_year_end"]

    def values(column):
        if column in det_df.columns:
            return det_df[column].fillna(0.0).astype(float)
        return years * 0.0

    inflows = [
        ("household_net_income", "Net Household Income", "家庭税后净收入", "#1F77B4"),
        ("total_minimum_pension_drawdown", "Minimum Pension Drawdown", "最低退休金提取", "#2CA02C"),
        ("cash_reserve_withdrawal", "Cash Reserve", "现金储备提取", "#00CC96"),
        ("non_super_withdrawal", "Non-Super Withdrawal", "非养老金资产提取", "#17BECF"),
        ("total_extra_super_withdrawal", "Extra Super Withdrawal", "额外养老金提取", "#9467BD"),
        ("residential_property_sale_proceeds", "Property Sale Proceeds", "物业出售所得", "#FFA15A"),
    ]
    outflows = [
        ("spending", "Household Spending", "家庭支出", "#D62728"),
        ("total_cash_contributions", "Cash Contributions", "现金缴款", "#FF7F0E"),
        ("non_deductible_debt_interest", "Non-deductible Interest", "不可抵扣债务利息", "#B22222"),
        ("non_deductible_principal_repayment", "Non-deductible Principal", "不可抵扣债务本金偿还", "#E45756"),
        ("deductible_principal_repayment", "Deductible Principal", "可抵扣债务本金偿还", "#F58518"),
        ("non_deductible_offset_contribution", "Non-deductible Offset", "不可抵扣 Offset 存入", "#7A5195"),
        ("deductible_offset_contribution", "Deductible Offset", "可抵扣债务 Offset 存入", "#BC5090"),
        ("cash_reserve_top_up", "Cash Reserve Top-up", "现金储备补充", "#4C78A8"),
        ("surplus_cash_to_non_super", "Non-Super Investment", "非养老金投资", "#59A14F"),
        ("non_super_tax_paid", "Non-Super Tax", "非养老金投资税", "#8C564B"),
        ("total_super_withdrawal_cgt_tax", "Super Withdrawal CGT", "养老金提取资本利得税", "#E377C2"),
    ]

    fig = go.Figure()
    total_inflow = values("household_net_income") * 0.0
    total_outflow = values("household_net_income") * 0.0

    for column, english_label, chinese_label, colour in inflows:
        series_values = values(column).clip(lower=0.0)
        total_inflow = total_inflow + series_values
        if series_values.abs().sum() > 0.01:
            label = _chart_text(inputs, english_label, chinese_label)
            fig.add_trace(go.Bar(
                x=years,
                y=series_values,
                name=label,
                marker=dict(color=colour, line=dict(color="#FFFFFF", width=0.7)),
                hovertemplate=f"{_chart_text(inputs, 'Financial Year', '财政年度')}: %{{x}}FY<br>{label}: $%{{y:,.0f}}<extra></extra>",
            ))

    for column, english_label, chinese_label, colour in outflows:
        series_values = values(column).clip(lower=0.0)
        total_outflow = total_outflow + series_values
        if series_values.abs().sum() > 0.01:
            label = _chart_text(inputs, english_label, chinese_label)
            fig.add_trace(go.Bar(
                x=years,
                y=-series_values,
                name=label,
                marker=dict(color=colour, line=dict(color="#FFFFFF", width=0.7)),
                hovertemplate=f"{_chart_text(inputs, 'Financial Year', '财政年度')}: %{{x}}FY<br>{label}: $%{{customdata:,.0f}}<extra></extra>",
                customdata=series_values,
            ))

    net_cashflow = total_inflow - total_outflow
    net_label = _chart_text(inputs, "Net Cash Flow", "净现金流")
    fig.add_trace(go.Scatter(
        x=years,
        y=net_cashflow,
        mode="lines+markers",
        name=net_label,
        line=dict(color="#111111", width=4),
        marker=dict(color="#FFFFFF", line=dict(color="#111111", width=2), size=7),
        hovertemplate=f"{_chart_text(inputs, 'Financial Year', '财政年度')}: %{{x}}FY<br>{net_label}: $%{{y:,.0f}}<extra></extra>",
    ))

    fig = _add_lifecycle_markers(fig, inputs)
    fig.add_hline(y=0, line_color="#182A3A", line_width=1.2)
    fig.update_layout(
        title=title_text,
        barmode="relative",
        xaxis_title=_chart_text(inputs, "Financial Year", "财政年度"),
        yaxis_title=_chart_text(inputs, "Annual Cash Flow", "年度现金流"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.30, xanchor="left", x=0),
        margin=dict(t=170),
    )
    return _format_currency_axis(fig, "y")


def create_total_tax_paid_chart(det_df, inputs, title_text):
    if "total_tax_paid" not in det_df.columns:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=det_df["financial_year_end"],
            y=det_df["total_tax_paid"],
            mode="lines",
            name="Total Tax Paid",
            hovertemplate="Financial Year: %{x}FY<br>Total Tax Paid: $%{y:,.0f}<extra></extra>",
        )
    )

    fig = _add_lifecycle_markers(fig, inputs)

    fig.update_layout(
        title=title_text,
        xaxis_title=_chart_text(inputs, "Financial Year", "财政年度"),
        yaxis_title=_chart_text(inputs, "Annual Tax", "年度税款"),
        hovermode="x unified",
    )

    return _format_currency_axis(fig, "y")


# ============================================================
# SECTION: HISTOGRAM
# ============================================================

def create_histogram(
    summary_df,
    show_p10=True,
    show_p50=True,
    show_p90=True,
    title_text="Distribution of Final Wealth",
):
    if summary_df is None or summary_df.empty:
        return px.histogram(title=title_text)

    fig = px.histogram(
        summary_df,
        x="final_wealth",
        nbins=50,
        title=title_text,
    )

    p10 = summary_df["final_wealth"].quantile(0.10)
    p50 = summary_df["final_wealth"].quantile(0.50)
    p90 = summary_df["final_wealth"].quantile(0.90)

    if show_p10:
        fig.add_vline(
            x=p10,
            line_dash="dot",
            annotation_text="P10",
            annotation_position="top",
        )

    if show_p50:
        fig.add_vline(
            x=p50,
            line_dash="dash",
            annotation_text="Median",
            annotation_position="top",
        )

    if show_p90:
        fig.add_vline(
            x=p90,
            line_dash="dot",
            annotation_text="P90",
            annotation_position="top",
        )

    fig.update_layout(
        xaxis_title="Final Wealth",
        yaxis_title="Frequency",
    )

    fig.update_traces(
        hovertemplate="Final Wealth: $%{x:,.0f}<br>Count: %{y}<extra></extra>"
    )

    fig.update_xaxes(tickprefix="$", separatethousands=True)

    return fig


# ============================================================
# SECTION: COMPARISON SUMMARY CHARTS
# ============================================================

def create_success_rate_comparison_chart(comparison_df):
    fig = px.bar(
        comparison_df,
        x="scenario",
        y="success_rate",
        title="Success Rate by Scenario",
        text="success_rate_label" if "success_rate_label" in comparison_df.columns else None,
    )

    fig.update_layout(
        xaxis_title="Scenario",
        yaxis_title="Success Rate",
    )
    fig.update_yaxes(tickformat=".0%")

    fig.update_traces(
        hovertemplate="Scenario: %{x}<br>Success Rate: %{y:.1%}<extra></extra>"
    )

    return fig


def create_median_wealth_comparison_chart(comparison_df):
    fig = px.bar(
        comparison_df,
        x="scenario",
        y="median_final_wealth",
        title="Median Final Wealth by Scenario",
        text="median_final_wealth_label" if "median_final_wealth_label" in comparison_df.columns else None,
    )

    fig.update_layout(
        xaxis_title="Scenario",
        yaxis_title="Median Final Wealth",
    )

    fig.update_traces(
        hovertemplate="Scenario: %{x}<br>Median Final Wealth: $%{y:,.0f}<extra></extra>"
    )

    fig.update_yaxes(tickprefix="$", separatethousands=True)

    return fig
