import io
import math
from datetime import datetime
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    CondPageBreak,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PDF_CHART_KEYS = (
    "wealth_projection",
    "income_spending",
    "percentile_paths",
    "failure_probability",
    "final_wealth_distribution",
    "tax_breakdown",
    "total_tax",
)


CHART_LABELS = {
    "wealth_projection": ("Total wealth projection", "总财富预测"),
    "income_spending": ("Income and spending", "收入与支出"),
    "percentile_paths": ("Monte Carlo percentile paths", "蒙特卡洛百分位路径"),
    "failure_probability": ("Probability of running out of money", "资金耗尽概率"),
    "final_wealth_distribution": ("Distribution of final wealth", "最终财富分布"),
    "tax_breakdown": ("Tax breakdown", "税务构成"),
    "total_tax": ("Total tax paid", "年度总税款"),
}


CHART_EXPLANATIONS = {
    "wealth_projection": (
        "Shows the deterministic path of total household wealth. Retirement and pension-start years are important points for reviewing the change from accumulation to drawdown.",
        "展示家庭总财富的确定性变化路径。退休及养老金开始年份是检视资产由累积转为提取的重要节点。",
    ),
    "income_spending": (
        "Compares projected household income sources with spending. Periods where spending is not met from income require asset drawdowns.",
        "比较预计家庭收入来源与支出。收入不足以覆盖支出的时期，需要通过提取资产补足现金流。",
    ),
    "percentile_paths": (
        "P10, P50 and P90 show downside, median and upside Monte Carlo wealth paths. They are scenarios, not guaranteed outcomes.",
        "P10、P50 和 P90 分别展示蒙特卡洛模拟中的下行情景、中位情景及上行情景，并非保证结果。",
    ),
    "failure_probability": (
        "Shows the cumulative proportion of simulations with an unmet spending shortfall by each year. A rising line indicates increasing sustainability risk.",
        "展示截至各年度出现支出缺口的累计模拟比例。曲线上升代表退休资金可持续性风险增加。",
    ),
    "final_wealth_distribution": (
        "Shows the range of simulated wealth remaining at the end of the projection. A wide range indicates greater outcome uncertainty.",
        "展示预测期末剩余财富的模拟分布。分布范围越宽，代表最终结果的不确定性越高。",
    ),
    "tax_breakdown": (
        "Groups projected tax into personal tax, Medicare levy, super tax and CGT/trust minimum-tax items. Draft policy estimates should be reviewed separately.",
        "将预计税款分为个人所得税、Medicare Levy、养老金税及 CGT/信托最低税。政策草案估算应另行审阅。",
    ),
    "total_tax": (
        "Shows total projected tax paid each year across the household and investment structures included in the model.",
        "展示模型所涵盖家庭成员及投资结构每年的预计总税款。",
    ),
}


_PALETTE = [
    colors.HexColor("#2563EB"),
    colors.HexColor("#E11D48"),
    colors.HexColor("#059669"),
    colors.HexColor("#7C3AED"),
    colors.HexColor("#D97706"),
    colors.HexColor("#0891B2"),
]


def _is_chinese(inputs):
    language_value = str(inputs.get("ui_language", ""))
    return "中文" in language_value or "CN" in language_value.upper()


def _text(is_chinese, en, zh):
    return zh if is_chinese else en


def _safe_number(value, default=0.0):
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else default
    except (TypeError, ValueError):
        return default


def _money(value):
    value = _safe_number(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.2f}m"
    if value >= 1_000:
        return f"{sign}${value / 1_000:.0f}k"
    return f"{sign}${value:,.0f}"


def _percentage(value):
    return f"{_safe_number(value):.1%}"


def _row_for_year(det_df, year):
    if det_df.empty or "financial_year_end" not in det_df.columns:
        return None
    exact = det_df.loc[det_df["financial_year_end"] == int(year)]
    if not exact.empty:
        return exact.iloc[0]
    index = (det_df["financial_year_end"].astype(float) - float(year)).abs().idxmin()
    return det_df.loc[index]


def build_key_milestones(det_df, inputs, is_chinese=False):
    if det_df is None or det_df.empty:
        return []

    start_year = int(det_df["financial_year_end"].iloc[0])
    end_year = int(det_df["financial_year_end"].iloc[-1])
    milestones = []

    def add(label_en, label_zh, year, note_en, note_zh):
        if year < start_year or year > end_year:
            return
        row = _row_for_year(det_df, year)
        wealth = _safe_number(row.get("total_wealth", 0)) if row is not None else 0
        milestones.append({
            "event": _text(is_chinese, label_en, label_zh),
            "year": int(year),
            "wealth": wealth,
            "note": _text(is_chinese, note_en, note_zh),
        })

    add("Projection starts", "预测开始", start_year, "Opening projection position", "预测期初状况")

    people = [("person1", inputs.get("person1_name"), "Person 1", "人物 1")]
    if str(inputs.get("household_mode", "Two People")) != "One Person":
        people.append(("person2", inputs.get("person2_name"), "Person 2", "人物 2"))

    for prefix, supplied_name, english_fallback, chinese_fallback in people:
        name = str(supplied_name or _text(is_chinese, english_fallback, chinese_fallback))
        current_age = int(inputs.get(f"{prefix}_current_age", 0))
        retirement_age = int(inputs.get(f"{prefix}_retirement_age", current_age))
        pension_age = int(inputs.get(f"{prefix}_pension_start_age", retirement_age))
        retirement_year = start_year + retirement_age - current_age
        pension_year = start_year + pension_age - current_age
        add(
            f"{name} retires",
            f"{name} 退休",
            retirement_year,
            f"Planned retirement at age {retirement_age}",
            f"计划于 {retirement_age} 岁退休",
        )
        add(
            f"{name} pension starts",
            f"{name} 开始领取养老金",
            pension_year,
            f"Pension phase starts at age {pension_age}",
            f"于 {pension_age} 岁进入养老金阶段",
        )

    if "total_wealth" in det_df.columns:
        minimum_index = det_df["total_wealth"].astype(float).idxmin()
        minimum_row = det_df.loc[minimum_index]
        minimum_year = int(minimum_row["financial_year_end"])
        if minimum_year not in {start_year, end_year}:
            add(
                "Lowest deterministic wealth",
                "确定性财富最低点",
                minimum_year,
                "Lowest projected total wealth during the horizon",
                "预测期内预计总财富的最低点",
            )

    if "unmet_shortfall" in det_df.columns:
        shortfall_rows = det_df.loc[det_df["unmet_shortfall"].astype(float) > 0.01]
        if not shortfall_rows.empty:
            first_shortfall_year = int(shortfall_rows.iloc[0]["financial_year_end"])
            add(
                "First deterministic shortfall",
                "首次确定性资金缺口",
                first_shortfall_year,
                "Modelled spending could not be fully funded",
                "模型显示支出未能得到全额资金支持",
            )

    add("Projection ends", "预测结束", end_year, "End of the selected modelling horizon", "所选预测期间终点")

    seen = set()
    unique = []
    for milestone in sorted(milestones, key=lambda item: (item["year"], item["event"])):
        key = (milestone["event"], milestone["year"])
        if key not in seen:
            seen.add(key)
            unique.append(milestone)
    return unique


def _normalise_series(values):
    return [_safe_number(value) for value in values]


def _line_chart(x_values, series, font_name, percent_axis=False, width=175 * mm, height=76 * mm):
    drawing = Drawing(width, height)
    left, right, bottom, top = 16 * mm, 5 * mm, 13 * mm, 7 * mm
    plot_width = width - left - right
    plot_height = height - bottom - top
    x_values = list(x_values)
    if not x_values:
        return drawing

    all_values = [value for _, values, _ in series for value in _normalise_series(values)]
    y_min = min([0.0] + all_values)
    y_max = max([1.0] + all_values)
    if y_max <= y_min:
        y_max = y_min + 1.0
    padding = (y_max - y_min) * 0.08
    y_max += padding
    if y_min < 0:
        y_min -= padding

    drawing.add(Line(left, bottom, left, bottom + plot_height, strokeColor=colors.HexColor("#64748B"), strokeWidth=0.8))
    drawing.add(Line(left, bottom, left + plot_width, bottom, strokeColor=colors.HexColor("#64748B"), strokeWidth=0.8))

    for tick in range(5):
        ratio = tick / 4
        y = bottom + plot_height * ratio
        value = y_min + (y_max - y_min) * ratio
        drawing.add(Line(left, y, left + plot_width, y, strokeColor=colors.HexColor("#E2E8F0"), strokeWidth=0.5))
        label = f"{value:.0%}" if percent_axis else _money(value)
        drawing.add(String(left - 2 * mm, y - 2, label, textAnchor="end", fontName=font_name, fontSize=7, fillColor=colors.HexColor("#475569")))

    tick_indexes = sorted(set([0, len(x_values) // 4, len(x_values) // 2, 3 * len(x_values) // 4, len(x_values) - 1]))
    for index in tick_indexes:
        x = left + plot_width * (index / max(len(x_values) - 1, 1))
        drawing.add(String(x, bottom - 5 * mm, str(x_values[index]), textAnchor="middle", fontName=font_name, fontSize=7, fillColor=colors.HexColor("#475569")))

    for series_index, (label, raw_values, colour) in enumerate(series):
        values = _normalise_series(raw_values)
        points = []
        for index, value in enumerate(values):
            x = left + plot_width * (index / max(len(values) - 1, 1))
            y = bottom + plot_height * ((value - y_min) / (y_max - y_min))
            points.extend([x, y])
        if len(points) >= 4:
            drawing.add(PolyLine(points, strokeColor=colour, strokeWidth=1.7, fillColor=None))
        legend_x = left + (series_index % 3) * (plot_width / 3)
        legend_y = height - 3 * mm - (series_index // 3) * 4 * mm
        drawing.add(Line(legend_x, legend_y, legend_x + 5 * mm, legend_y, strokeColor=colour, strokeWidth=2))
        drawing.add(String(legend_x + 6 * mm, legend_y - 2, str(label), fontName=font_name, fontSize=7, fillColor=colors.HexColor("#334155")))
    return drawing


def _histogram_chart(values, font_name, width=175 * mm, height=76 * mm):
    drawing = Drawing(width, height)
    left, right, bottom, top = 16 * mm, 5 * mm, 13 * mm, 7 * mm
    plot_width = width - left - right
    plot_height = height - bottom - top
    values = np.asarray([_safe_number(v) for v in values], dtype=float)
    if values.size == 0:
        return drawing
    counts, edges = np.histogram(values, bins=min(20, max(8, int(math.sqrt(values.size)))))
    maximum = max(int(counts.max()), 1)
    bar_width = plot_width / max(len(counts), 1)
    drawing.add(Line(left, bottom, left, bottom + plot_height, strokeColor=colors.HexColor("#64748B"), strokeWidth=0.8))
    drawing.add(Line(left, bottom, left + plot_width, bottom, strokeColor=colors.HexColor("#64748B"), strokeWidth=0.8))
    for index, count in enumerate(counts):
        height_value = plot_height * count / maximum
        drawing.add(Rect(left + index * bar_width + 0.5, bottom, max(bar_width - 1, 1), height_value, fillColor=colors.HexColor("#60A5FA"), strokeColor=None))
    for ratio in [0, 0.5, 1]:
        value = edges[0] + (edges[-1] - edges[0]) * ratio
        x = left + plot_width * ratio
        drawing.add(String(x, bottom - 5 * mm, _money(value), textAnchor="middle", fontName=font_name, fontSize=7, fillColor=colors.HexColor("#475569")))
    median = float(np.median(values))
    if edges[-1] > edges[0]:
        median_x = left + plot_width * (median - edges[0]) / (edges[-1] - edges[0])
        drawing.add(Line(median_x, bottom, median_x, bottom + plot_height, strokeColor=colors.HexColor("#E11D48"), strokeWidth=1.5))
        drawing.add(String(median_x, bottom + plot_height + 2, "P50", textAnchor="middle", fontName=font_name, fontSize=7, fillColor=colors.HexColor("#E11D48")))
    return drawing


def _tax_series(det_df, is_chinese):
    groups = [
        (
            _text(is_chinese, "Personal income tax", "个人所得税"),
            ["person1_income_tax", "person2_income_tax", "person1_income_tax_on_non_super_earnings", "person2_income_tax_on_non_super_earnings"],
        ),
        (
            "Medicare Levy" if not is_chinese else "Medicare Levy",
            ["person1_medicare_levy", "person2_medicare_levy", "person1_medicare_levy_on_non_super_earnings", "person2_medicare_levy_on_non_super_earnings"],
        ),
        (
            _text(is_chinese, "Super tax", "养老金税"),
            ["person1_division_293_tax", "person2_division_293_tax", "person1_super_contributions_tax", "person2_super_contributions_tax", "person1_total_super_earnings_tax", "person2_total_super_earnings_tax"],
        ),
        (
            _text(is_chinese, "CGT / trust minimum tax", "CGT／信托最低税"),
            ["total_cgt_minimum_tax", "total_discretionary_trust_minimum_tax"],
        ),
    ]
    result = []
    for index, (label, columns) in enumerate(groups):
        available = [column for column in columns if column in det_df.columns]
        if available:
            values = det_df[available].fillna(0).sum(axis=1).tolist()
            if any(abs(_safe_number(value)) > 0.01 for value in values):
                result.append((label, values, _PALETTE[index]))
    return result


def _chart_flowable(chart_key, selected_result, comparison_results, selected_scenario, value_mode, font_name, is_chinese):
    det_df = selected_result["det_df"].copy()
    percentile_df = selected_result["percentile_df"].copy()
    failure_df = selected_result["failure_prob_df"].copy()
    summary_df = selected_result["summary_df"].copy()
    inputs = selected_result["inputs"]
    years = det_df["financial_year_end"].astype(int).tolist()

    if chart_key == "wealth_projection":
        series = []
        for index, (scenario, result) in enumerate(comparison_results.items()):
            series.append((scenario, result["det_df"]["total_wealth"].tolist(), _PALETTE[index % len(_PALETTE)]))
        drawing = _line_chart(years, series, font_name)
    elif chart_key == "income_spending":
        series = []
        income_columns = [column for column in ["person1_net_income", "person2_net_income", "person1_min_pension_drawdown", "person2_min_pension_drawdown"] if column in det_df.columns]
        total_income = det_df[income_columns].fillna(0).sum(axis=1).tolist() if income_columns else [0] * len(det_df)
        series.append((_text(is_chinese, "Net income and pension", "净收入及养老金"), total_income, _PALETTE[0]))
        if "spending" in det_df.columns:
            series.append((_text(is_chinese, "Household spending", "家庭支出"), det_df["spending"].tolist(), _PALETTE[1]))
        drawing = _line_chart(years, series, font_name)
    elif chart_key == "percentile_paths":
        percentile_years = percentile_df["financial_year_end"].astype(int).tolist()
        series = [("P10", percentile_df["p10"].tolist(), _PALETTE[1]), ("P50", percentile_df["p50"].tolist(), _PALETTE[0]), ("P90", percentile_df["p90"].tolist(), _PALETTE[2])]
        drawing = _line_chart(percentile_years, series, font_name)
    elif chart_key == "failure_probability":
        failure_years = failure_df["financial_year_end"].astype(int).tolist()
        series = [(_text(is_chinese, "Failure probability", "资金耗尽概率"), failure_df["failure_probability"].tolist(), _PALETTE[1])]
        drawing = _line_chart(failure_years, series, font_name, percent_axis=True)
    elif chart_key == "final_wealth_distribution":
        drawing = _histogram_chart(summary_df["final_wealth"].tolist(), font_name)
    elif chart_key == "tax_breakdown":
        drawing = _line_chart(years, _tax_series(det_df, is_chinese), font_name)
    elif chart_key == "total_tax":
        values = det_df["total_tax_paid"].tolist() if "total_tax_paid" in det_df.columns else [0] * len(det_df)
        drawing = _line_chart(years, [(_text(is_chinese, "Total tax", "总税款"), values, _PALETTE[3])], font_name)
    else:
        raise ValueError(f"Unknown chart key: {chart_key}")

    label = CHART_LABELS[chart_key][1 if is_chinese else 0]
    explanation = CHART_EXPLANATIONS[chart_key][1 if is_chinese else 0]
    return label, drawing, explanation


def _styles(is_chinese):
    if is_chinese:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            font_name = "STSong-Light"
        except Exception:
            font_name = "Helvetica"
    else:
        font_name = "Helvetica"

    base = getSampleStyleSheet()
    return font_name, {
        "title": ParagraphStyle("PdfTitle", parent=base["Title"], fontName=font_name, fontSize=23, leading=29, textColor=colors.HexColor("#0F172A"), alignment=TA_LEFT, spaceAfter=8),
        "subtitle": ParagraphStyle("PdfSubtitle", parent=base["Normal"], fontName=font_name, fontSize=10, leading=15, textColor=colors.HexColor("#475569"), spaceAfter=12),
        "h1": ParagraphStyle("PdfH1", parent=base["Heading1"], fontName=font_name, fontSize=16, leading=20, textColor=colors.HexColor("#0F3D64"), spaceBefore=8, spaceAfter=7),
        "h2": ParagraphStyle("PdfH2", parent=base["Heading2"], fontName=font_name, fontSize=12, leading=16, textColor=colors.HexColor("#1E3A5F"), spaceBefore=5, spaceAfter=5),
        "body": ParagraphStyle("PdfBody", parent=base["BodyText"], fontName=font_name, fontSize=9.2, leading=14, textColor=colors.HexColor("#334155"), spaceAfter=7),
        "small": ParagraphStyle("PdfSmall", parent=base["BodyText"], fontName=font_name, fontSize=7.5, leading=10, textColor=colors.HexColor("#64748B"), spaceAfter=4),
        "metric": ParagraphStyle("PdfMetric", parent=base["BodyText"], fontName=font_name, fontSize=10, leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#0F172A")),
    }


def _footer(canvas, doc, font_name, is_chinese):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFont(font_name, 7)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 9 * mm, _text(is_chinese, "Financial modelling report - indicative only", "财务模型报告 - 仅供参考"))
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def build_pdf_report_bytes(
    selected_result,
    comparison_results,
    selected_scenario,
    selected_chart_keys,
    value_mode="Future Value",
    input_warnings=None,
    output_warnings=None,
):
    selected_chart_keys = [key for key in selected_chart_keys if key in PDF_CHART_KEYS]
    inputs = selected_result["inputs"]
    det_df = selected_result["det_df"]
    summary_df = selected_result["summary_df"]
    is_chinese = _is_chinese(inputs)
    font_name, styles = _styles(is_chinese)

    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=19 * mm,
        title=str(inputs.get("report_title") or _text(is_chinese, "Financial Projection Report", "财务预测报告")),
        author="Retirement Modelling Suite (Australia)",
    )
    story = []

    report_title = str(inputs.get("report_title") or _text(is_chinese, "Financial Projection Report", "财务预测报告"))
    story.append(Paragraph(escape(report_title), styles["title"]))
    story.append(Paragraph(
        escape(_text(
            is_chinese,
            f"Scenario: {selected_scenario} | Value basis: {value_mode} | Generated: {datetime.now().strftime('%d %B %Y')}",
            f"情景：{selected_scenario} | 价值口径：{'现值' if value_mode == 'Present Value' else '终值'} | 生成日期：{datetime.now().strftime('%Y年%m月%d日')}",
        )),
        styles["subtitle"],
    ))
    story.append(Paragraph(
        _text(
            is_chinese,
            "This concise report summarises the selected projection, key lifecycle points and the charts chosen before export.",
            "本简要报告概述所选预测情景、关键生命周期节点，以及导出前勾选的图表。",
        ),
        styles["body"],
    ))

    success_rate = _safe_number(selected_result.get("success_rate"))
    final_values = summary_df["final_wealth"] if "final_wealth" in summary_df.columns else pd.Series(dtype=float)
    median_final = _safe_number(final_values.median()) if not final_values.empty else 0
    p10_final = _safe_number(final_values.quantile(0.10)) if not final_values.empty else 0
    p90_final = _safe_number(final_values.quantile(0.90)) if not final_values.empty else 0
    end_wealth = _safe_number(det_df.iloc[-1].get("total_wealth", 0)) if not det_df.empty else 0

    metric_data = [
        [
            Paragraph(_text(is_chinese, "Success rate", "成功率"), styles["small"]),
            Paragraph(_text(is_chinese, "Median final wealth", "最终财富中位数"), styles["small"]),
            Paragraph(_text(is_chinese, "P10 / P90 range", "P10 / P90 区间"), styles["small"]),
            Paragraph(_text(is_chinese, "Deterministic final wealth", "确定性最终财富"), styles["small"]),
        ],
        [
            Paragraph(f"<b>{_percentage(success_rate)}</b>", styles["metric"]),
            Paragraph(f"<b>{_money(median_final)}</b>", styles["metric"]),
            Paragraph(f"<b>{_money(p10_final)} / {_money(p90_final)}</b>", styles["metric"]),
            Paragraph(f"<b>{_money(end_wealth)}</b>", styles["metric"]),
        ],
    ]
    metric_table = Table(metric_data, colWidths=[43 * mm] * 4, rowHeights=[9 * mm, 11 * mm])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(metric_table)
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(_text(is_chinese, "Future outlook", "未来情况概述"), styles["h1"]))
    start_year = int(det_df["financial_year_end"].iloc[0]) if not det_df.empty else 0
    end_year = int(det_df["financial_year_end"].iloc[-1]) if not det_df.empty else 0
    start_wealth = _safe_number(det_df.iloc[0].get("total_wealth", 0)) if not det_df.empty else 0
    wealth_direction = _text(is_chinese, "increases", "增加") if end_wealth >= start_wealth else _text(is_chinese, "decreases", "减少")
    shortfall_rows = det_df.loc[det_df.get("unmet_shortfall", pd.Series(0, index=det_df.index)).astype(float) > 0.01] if not det_df.empty else pd.DataFrame()
    shortfall_sentence = (
        _text(is_chinese, "No deterministic spending shortfall is projected.", "确定性预测中未出现支出资金缺口。")
        if shortfall_rows.empty
        else _text(
            is_chinese,
            f"The first deterministic spending shortfall occurs in FY{int(shortfall_rows.iloc[0]['financial_year_end'])}.",
            f"首次确定性支出资金缺口出现在 FY{int(shortfall_rows.iloc[0]['financial_year_end'])}。",
        )
    )
    outlook = _text(
        is_chinese,
        f"From FY{start_year} to FY{end_year}, deterministic total wealth {wealth_direction} from {_money(start_wealth)} to {_money(end_wealth)}. The Monte Carlo success rate is {_percentage(success_rate)}, with final wealth centred around {_money(median_final)} and a P10-P90 range of {_money(p10_final)} to {_money(p90_final)}. {shortfall_sentence}",
        f"从 FY{start_year} 至 FY{end_year}，确定性总财富预计由 {_money(start_wealth)}{wealth_direction}至 {_money(end_wealth)}。蒙特卡洛成功率为 {_percentage(success_rate)}，期末财富中位数约为 {_money(median_final)}，P10-P90 区间为 {_money(p10_final)} 至 {_money(p90_final)}。{shortfall_sentence}",
    )
    story.append(Paragraph(outlook, styles["body"]))

    story.append(Paragraph(_text(is_chinese, "Key milestones", "关键节点"), styles["h1"]))
    milestones = build_key_milestones(det_df, inputs, is_chinese=is_chinese)
    milestone_rows = [[
        Paragraph(_text(is_chinese, "Event", "事件"), styles["small"]),
        Paragraph(_text(is_chinese, "FY", "财年"), styles["small"]),
        Paragraph(_text(is_chinese, "Total wealth", "总财富"), styles["small"]),
        Paragraph(_text(is_chinese, "Interpretation", "说明"), styles["small"]),
    ]]
    for item in milestones:
        milestone_rows.append([
            Paragraph(escape(str(item["event"])), styles["small"]),
            Paragraph(str(item["year"]), styles["small"]),
            Paragraph(_money(item["wealth"]), styles["small"]),
            Paragraph(escape(str(item["note"])), styles["small"]),
        ])
    milestone_table = Table(milestone_rows, colWidths=[43 * mm, 18 * mm, 30 * mm, 81 * mm], repeatRows=1)
    milestone_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F3D64")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(milestone_table)

    combined_warnings = list(input_warnings or []) + list(output_warnings or [])
    if combined_warnings:
        story.append(Paragraph(_text(is_chinese, "Items requiring review", "需要审阅的事项"), styles["h1"]))
        for warning in combined_warnings[:8]:
            story.append(Paragraph(f"- {escape(str(warning))}", styles["body"]))

    if selected_chart_keys:
        story.append(PageBreak())
        story.append(Paragraph(_text(is_chinese, "Selected charts and interpretation", "所选图表及解释"), styles["h1"]))
        for index, chart_key in enumerate(selected_chart_keys):
            if index > 0:
                story.append(CondPageBreak(95 * mm))
            label, drawing, explanation = _chart_flowable(
                chart_key,
                selected_result,
                comparison_results,
                selected_scenario,
                value_mode,
                font_name,
                is_chinese,
            )
            story.append(Paragraph(label, styles["h2"]))
            story.append(drawing)
            story.append(Paragraph(explanation, styles["small"]))
            story.append(Spacer(1, 3 * mm))

    story.append(CondPageBreak(55 * mm))
    story.append(Paragraph(_text(is_chinese, "Important notes", "重要说明"), styles["h1"]))
    story.append(Paragraph(
        _text(
            is_chinese,
            "This report is generated from modelling assumptions and simulated outcomes. It is for planning and educational purposes only, is not personal financial or tax advice, and should be reviewed against client objectives, risk tolerance, current legislation and asset-level records. CGT transition allocations and draft trust-tax results are estimates where noted in the application.",
            "本报告基于模型假设和模拟结果生成，仅用于规划及教育目的，不构成个人财务或税务建议。使用前应结合客户目标、风险承受能力、现行法规及单项资产记录进行审阅。APP 中标注的 CGT 过渡分配及信托税草案结果均属于估算。",
        ),
        styles["small"],
    ))

    document.build(
        story,
        onFirstPage=lambda canvas, doc: _footer(canvas, doc, font_name, is_chinese),
        onLaterPages=lambda canvas, doc: _footer(canvas, doc, font_name, is_chinese),
    )
    return output.getvalue()
