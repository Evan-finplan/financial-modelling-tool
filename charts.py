
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# SECTION: CHART HELPERS
# ============================================================

def _person_label(inputs, key, fallback):
    name = str(inputs.get(key, "") or "").strip()
    return name if name else fallback


def _add_lifecycle_markers(fig, inputs):
    start_fy_end = int(str(inputs["start_financial_year"]).replace("FY", ""))
    household_mode = str(inputs.get("household_mode", "Two People"))
    one_person_mode = household_mode == "One Person"

    p1_retirement_fy = start_fy_end + (inputs["person1_retirement_age"] - inputs["person1_current_age"])
    p1_pension_fy = start_fy_end + (inputs["person1_pension_start_age"] - inputs["person1_current_age"])

    p1_label = _person_label(inputs, "person1_name", "P1")
    p2_label = _person_label(inputs, "person2_name", "P2")

    markers = [
        {"x": p1_retirement_fy, "label": f"{p1_label} Retirement"},
        {"x": p1_pension_fy, "label": f"{p1_label} Pension Start"},
    ]

    if not one_person_mode:
        p2_retirement_fy = start_fy_end + (inputs["person2_retirement_age"] - inputs["person2_current_age"])
        p2_pension_fy = start_fy_end + (inputs["person2_pension_start_age"] - inputs["person2_current_age"])
        markers.extend(
            [
                {"x": p2_retirement_fy, "label": f"{p2_label} Retirement"},
                {"x": p2_pension_fy, "label": f"{p2_label} Pension Start"},
            ]
        )

    deduped = []
    seen = set()
    for marker in markers:
        dedupe_key = (marker["x"], marker["label"])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(marker)

    counts_by_year = {}
    for marker in deduped:
        counts_by_year[marker["x"]] = counts_by_year.get(marker["x"], 0) + 1

    used_index_by_year = {}
    annotation_positions = ["top left", "top right", "bottom left", "bottom right"]

    for marker in deduped:
        x_value = marker["x"]
        label = marker["label"]
        year_index = used_index_by_year.get(x_value, 0)
        used_index_by_year[x_value] = year_index + 1

        annotation_position = annotation_positions[year_index % len(annotation_positions)]
        ay = -40 - (year_index * 18) if "top" in annotation_position else 40 + (year_index * 18)

        fig.add_vline(
            x=x_value,
            line_dash="dash",
            annotation_text=label,
            annotation_position=annotation_position,
            annotation_y=1.0,
            annotation_yref="paper",
            annotation=dict(
                yshift=0,
                ay=ay,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
                borderpad=2,
                font=dict(size=11),
            ),
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
        title="Deterministic Total Wealth Projection",
    )

    fig = _add_lifecycle_markers(fig, inputs)

    fig.update_layout(
        xaxis_title="Financial Year",
        yaxis_title="Total Wealth",
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
            hovertemplate="Financial Year: %{x}FY<br>P10: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=percentile_df["financial_year_end"],
            y=percentile_df["p50"],
            mode="lines",
            name="P50",
            hovertemplate="Financial Year: %{x}FY<br>P50: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=percentile_df["financial_year_end"],
            y=percentile_df["p90"],
            mode="lines",
            name="P90",
            hovertemplate="Financial Year: %{x}FY<br>P90: $%{y:,.0f}<extra></extra>",
        )
    )

    fig = _add_lifecycle_markers(fig, inputs)

    fig.update_layout(
        title=title_text,
        xaxis_title="Financial Year",
        yaxis_title="Total Wealth",
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
            hovertemplate="Financial Year: %{x}FY<br>Failure Probability: %{y:.1%}<extra></extra>",
        )
    )

    fig = _add_lifecycle_markers(fig, inputs)

    fig.update_layout(
        title=title_text,
        xaxis_title="Financial Year",
        yaxis_title="Failure Probability",
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


# ============================================================
# SECTION: TAX AND CASHFLOW CHARTS
# ============================================================

def create_tax_breakdown_chart(det_df, inputs, title_text):
    tax_columns = [
        "person1_income_tax",
        "person1_medicare_levy",
        "person1_income_tax_on_non_super_earnings",
        "person1_medicare_levy_on_non_super_earnings",
        "person2_income_tax",
        "person2_medicare_levy",
        "person2_income_tax_on_non_super_earnings",
        "person2_medicare_levy_on_non_super_earnings",
        "person1_super_contributions_tax",
        "person2_super_contributions_tax",
        "person1_total_super_earnings_tax",
        "person2_total_super_earnings_tax",
    ]

    pretty_names = {
        "person1_income_tax": "P1 Salary Income Tax",
        "person1_medicare_levy": "P1 Salary Medicare Levy",
        "person1_income_tax_on_non_super_earnings": "P1 Non-Super Income Tax",
        "person1_medicare_levy_on_non_super_earnings": "P1 Non-Super Medicare Levy",
        "person2_income_tax": "P2 Salary Income Tax",
        "person2_medicare_levy": "P2 Salary Medicare Levy",
        "person2_income_tax_on_non_super_earnings": "P2 Non-Super Income Tax",
        "person2_medicare_levy_on_non_super_earnings": "P2 Non-Super Medicare Levy",
        "person1_super_contributions_tax": "P1 Super Contributions Tax",
        "person2_super_contributions_tax": "P2 Super Contributions Tax",
        "person1_total_super_earnings_tax": "P1 Super Earnings Tax",
        "person2_total_super_earnings_tax": "P2 Super Earnings Tax",
    }

    available_columns = [col for col in tax_columns if col in det_df.columns]
    if not available_columns:
        return go.Figure()

    fig = go.Figure()

    for col in available_columns:
        fig.add_trace(
            go.Bar(
                x=det_df["financial_year_end"],
                y=det_df[col],
                name=pretty_names.get(col, col),
                hovertemplate="Financial Year: %{x}FY<br>"
                + f"{pretty_names.get(col, col)}: "
                + "$%{y:,.0f}<extra></extra>",
            )
        )

    fig = _add_lifecycle_markers(fig, inputs)

    fig.update_layout(
        title=title_text,
        barmode="stack",
        xaxis_title="Financial Year",
        yaxis_title="Annual Tax",
        hovermode="x unified",
    )

    return _format_currency_axis(fig, "y")


def create_income_vs_spending_chart(det_df, inputs, title_text):
    fig = go.Figure()

    series = [
        ("person1_gross_income", "P1 Gross Income"),
        ("person1_net_income", "P1 Net Income"),
        ("person2_gross_income", "P2 Gross Income"),
        ("person2_net_income", "P2 Net Income"),
        ("taxable_non_super_earnings_p1", "P1 Taxable Non-Super Income Return"),
        ("taxable_non_super_earnings_p2", "P2 Taxable Non-Super Income Return"),
        ("person1_min_pension_drawdown", "P1 Minimum Pension Drawdown"),
        ("person2_min_pension_drawdown", "P2 Minimum Pension Drawdown"),
        ("spending", "Household Spending"),
        ("surplus_cash_to_non_super", "Surplus Cash to Non-Super"),
    ]

    for column, label in series:
        if column in det_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=det_df["financial_year_end"],
                    y=det_df[column],
                    mode="lines",
                    name=label,
                    hovertemplate=f"Financial Year: %{{x}}FY<br>{label}: $%{{y:,.0f}}<extra></extra>",
                )
            )

    fig = _add_lifecycle_markers(fig, inputs)

    fig.update_layout(
        title=title_text,
        xaxis_title="Financial Year",
        yaxis_title="Annual Amount",
        hovermode="x unified",
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
        xaxis_title="Financial Year",
        yaxis_title="Annual Tax",
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
