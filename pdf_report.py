import io
import math
import re
from datetime import datetime
from pathlib import Path
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
from reportlab.pdfbase.ttfonts import TTFont
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


JBW_NAVY = colors.HexColor("#00205B")
JBW_DEEP_NAVY = colors.HexColor("#00163F")
JBW_BLUE = colors.HexColor("#34657F")
JBW_TEAL = colors.HexColor("#5B91A4")
JBW_SKY = colors.HexColor("#DDEEF4")
JBW_MIST = colors.HexColor("#F4F8FA")
JBW_INK = colors.HexColor("#182A3A")
JBW_MUTED = colors.HexColor("#53697A")
JBW_GRID = colors.HexColor("#C7D6DE")

_PALETTE = [
    JBW_NAVY,
    JBW_TEAL,
    colors.HexColor("#7AAFC0"),
    colors.HexColor("#334B5C"),
    colors.HexColor("#8AA0AE"),
    colors.HexColor("#4D7890"),
]


def _is_chinese(inputs):
    language_value = str(inputs.get("ui_language", ""))
    return "中文" in language_value or "CN" in language_value.upper()


def _language_is_chinese(language_value):
    language_value = str(language_value or "")
    return "中文" in language_value or "CN" in language_value.upper() or language_value.lower().startswith("zh")


def _text(is_chinese, en, zh):
    return zh if is_chinese else en


def _scenario_label(value, is_chinese):
    if not is_chinese:
        return str(value)
    return {
        "Base Case": "基础情景",
        "Conservative": "保守情景",
        "Optimistic": "乐观情景",
        "Custom": "自定义情景",
    }.get(str(value), str(value))


def _localise_warning(value, is_chinese):
    value = str(value)
    if not is_chinese or any("\u4e00" <= character <= "\u9fff" for character in value):
        return value
    translations = {
        "Residential rental losses are quarantined from 2027-28 under the modelled legislated rule and carried forward against future residential income.": "根据模型采用的已立法规则，住宅出租亏损自 2027-28 财年起被隔离，并结转以抵减未来住宅收入。",
        "Residential property modelling is an aggregate, interest-only projection. Property equity is included in net wealth but is not sold or refinanced to fund spending; principal repayments, depreciation schedules, sale costs, and property CGT are not modelled.": "住宅物业采用汇总且仅计利息的预测。物业净值计入净财富，但不会通过出售或再融资来支付支出；本金偿还、折旧明细、出售成本及物业 CGT 均未建模。",
        "The discretionary trust 30% minimum tax is based on the September 2026 exposure draft and is not enacted law. Final legislation may change the result.": "全权信托 30% 最低税基于 2026 年 9 月的征求意见稿，尚未成为正式法律；最终立法可能改变结果。",
        "The enacted CGT reform is modelled using one homogeneous non-super pool. The 1 July 2027 transition allocation, annual CPI indexation, loss ordering, and partial disposals are planning estimates and must be reconciled to asset-level records for tax return work.": "已立法的 CGT 改革以单一同质的非养老金资产池建模。2027 年 7 月 1 日的过渡分配、年度 CPI 指数化、亏损抵减顺序及部分出售均为规划估算，报税时必须与单项资产记录核对。",
        "The selected new/affordable housing CGT method is a scenario choice. Confirm statutory eligibility and compare the 50% discount with indexation using actual records at disposal.": "所选新建／可负担住房 CGT 方法属于情景选择。出售时应确认法定资格，并依据实际记录比较 50% 折扣法与指数化方法。",
        "Person 1 is scheduled to retire within 5 years. Small assumption changes may have a larger impact.": "人物 1 计划在 5 年内退休，较小的假设变化也可能产生较大影响。",
        "Person 2 is scheduled to retire within 5 years. Small assumption changes may have a larger impact.": "人物 2 计划在 5 年内退休，较小的假设变化也可能产生较大影响。",
        "At least one retirement age is relatively early. This may increase portfolio sustainability risk.": "至少一人的退休年龄相对较早，可能增加投资组合的可持续性风险。",
        "Super Capital Return Std is relatively high. This may produce a wide range of outcomes.": "养老金资本回报波动率相对较高，可能导致结果区间较宽。",
        "Non-Super Capital Return Std is relatively high. This may produce a wide range of outcomes.": "非养老金资本回报波动率相对较高，可能导致结果区间较宽。",
        "Number of Simulations is relatively low. Results may be less stable.": "模拟次数相对较少，结果可能不够稳定。",
        "Non-super withdrawals use a pooled average-cost method; individual tax parcels and exact disposal ordering are not modelled.": "非养老金资产提取采用汇总平均成本法；模型未处理单项税务批次及精确出售顺序。",
        "Salary income is indexed annually using the inflation rate while the person remains in working phase.": "人物处于工作阶段时，工资收入按通胀率逐年调整。",
        "Pension transfer is triggered from pension start age and is applied up to the person's transfer balance cap. Minimum pension drawdown is then applied from pension assets.": "达到养老金开始年龄后触发转入养老金阶段，并以个人 Transfer Balance Cap 为上限；随后从养老金资产中执行最低提取。",
        "Personal deductible contributions reduce taxable income and also flow through concessional contribution tax inside super.": "个人可扣税缴款会降低应税收入，同时在养老金账户内适用优惠缴款税。",
        "Non-super cost base is lower than market value, so future withdrawals may crystallise capital gains.": "非养老金资产成本基础低于市场价值，因此未来提取可能实现资本利得。",
        "CGT discount rate has been changed from the default 50% assumption. Confirm this is intended.": "CGT 折扣率已偏离默认的 50% 假设，请确认该设置符合预期。",
        "Success Rate is below 50%. The plan may have a high risk of failure.": "成功率低于 50%，该方案可能存在较高失败风险。",
        "Success Rate is below 75%. The plan may require further review or stress testing.": "成功率低于 75%，该方案可能需要进一步审阅或压力测试。",
        "P10 Final Wealth is below zero. Downside outcomes may be severe.": "P10 最终财富低于零，下行情景可能较为严重。",
        "Median Final Wealth is below zero. The central case may not be sustainable.": "最终财富中位数低于零，中位情景可能不可持续。",
        "The deterministic projection shows unmet shortfall in at least one year.": "确定性预测显示至少有一个年度出现未满足的资金缺口。",
        "Failure probability rises sharply at some point in the projection. Review sequencing and spending assumptions.": "预测期间某一阶段的失败概率明显上升，请审阅回报顺序风险及支出假设。",
        "The deterministic projection realises capital gains on non-super withdrawals in at least one year.": "确定性预测显示至少有一个年度的非养老金资产提取实现了资本利得。",
        "Non-super cost base falls materially below market value in the projection, which may increase future CGT on withdrawals.": "预测中的非养老金成本基础明显低于市场价值，可能提高未来提取时的 CGT。",
        "Division 293 tax is an estimate based on income components available in this model. Confirm reportable fringe benefits, net investment or rental losses, defined benefit contributions, and the final ATO assessment before relying on it for advice.": "Division 293 税额基于模型可用收入项目估算。用于建议前，应确认应申报附加福利、净投资或出租亏损、固定福利缴款及 ATO 最终评税结果。",
    }
    if value in translations:
        return translations[value]
    if value.startswith("Published indexed super thresholds are currently configured through "):
        year_match = re.search(r"through\s+(\d+)FY", value)
        year = year_match.group(1) if year_match else "最新公布年度"
        return f"目前已公布的指数化养老金门槛仅配置至 {year}FY。后续预测年度继续采用最新已知的缴款上限、一般 Transfer Balance Cap 及 SG 最高收入基础，直至政策设置更新。"
    failure_match = re.match(r"Cumulative failure probability reaches 25% by (\d+)FY\.", value)
    if failure_match:
        return f"累计失败概率在 {failure_match.group(1)}FY 前达到 25%。"
    contribution_match = re.match(r"(Person [12]) in (\d+)FY: (.+)", value)
    if contribution_match:
        person = "人物 1" if contribution_match.group(1) == "Person 1" else "人物 2"
        detail = contribution_match.group(3)
        if "estimated concessional contributions exceed the annual cap" in detail:
            return f"{person} 在 {contribution_match.group(2)}FY 的预计优惠缴款超过年度上限，请审阅结转未使用优惠缴款额度的适用资格。"
        if "personal deductible contribution entered at age 67 or above" in detail:
            return f"{person} 在 {contribution_match.group(2)}FY 于 67 岁或以上录入个人可扣税缴款，请审阅相关资格及工作测试要求。"
        if "personal deductible contribution entered at age 75 or above" in detail:
            return f"{person} 在 {contribution_match.group(2)}FY 于 75 岁或以上录入个人可扣税缴款，请仔细审阅接收及资格规则。"
        if "non-concessional contributions exceed the annual cap" in detail:
            return f"{person} 在 {contribution_match.group(2)}FY 的非优惠缴款超过年度上限，请审阅提前使用未来年度额度的适用资格。"
        if "non-concessional contribution entered at age 75 or above" in detail:
            return f"{person} 在 {contribution_match.group(2)}FY 于 75 岁或以上录入非优惠缴款，请仔细审阅接收及资格规则。"
    return "模型检测到一项需要进一步审阅的事项，请在 APP 中核对相关输入、假设及输出。"


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

    drawing.add(Line(left, bottom, left, bottom + plot_height, strokeColor=JBW_BLUE, strokeWidth=0.8))
    drawing.add(Line(left, bottom, left + plot_width, bottom, strokeColor=JBW_BLUE, strokeWidth=0.8))

    for tick in range(5):
        ratio = tick / 4
        y = bottom + plot_height * ratio
        value = y_min + (y_max - y_min) * ratio
        drawing.add(Line(left, y, left + plot_width, y, strokeColor=JBW_GRID, strokeWidth=0.5))
        label = f"{value:.0%}" if percent_axis else _money(value)
        drawing.add(String(left - 2 * mm, y - 2, label, textAnchor="end", fontName=font_name, fontSize=8, fillColor=JBW_MUTED))

    tick_indexes = sorted(set([0, len(x_values) // 4, len(x_values) // 2, 3 * len(x_values) // 4, len(x_values) - 1]))
    for index in tick_indexes:
        x = left + plot_width * (index / max(len(x_values) - 1, 1))
        drawing.add(String(x, bottom - 5 * mm, str(x_values[index]), textAnchor="middle", fontName=font_name, fontSize=8, fillColor=JBW_MUTED))

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
        drawing.add(String(legend_x + 6 * mm, legend_y - 2, str(label), fontName=font_name, fontSize=8, fillColor=JBW_INK))
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
    drawing.add(Line(left, bottom, left, bottom + plot_height, strokeColor=JBW_BLUE, strokeWidth=0.8))
    drawing.add(Line(left, bottom, left + plot_width, bottom, strokeColor=JBW_BLUE, strokeWidth=0.8))
    for index, count in enumerate(counts):
        height_value = plot_height * count / maximum
        drawing.add(Rect(left + index * bar_width + 0.5, bottom, max(bar_width - 1, 1), height_value, fillColor=JBW_TEAL, strokeColor=None))
    for ratio in [0, 0.5, 1]:
        value = edges[0] + (edges[-1] - edges[0]) * ratio
        x = left + plot_width * ratio
        drawing.add(String(x, bottom - 5 * mm, _money(value), textAnchor="middle", fontName=font_name, fontSize=8, fillColor=JBW_MUTED))
    median = float(np.median(values))
    if edges[-1] > edges[0]:
        median_x = left + plot_width * (median - edges[0]) / (edges[-1] - edges[0])
        drawing.add(Line(median_x, bottom, median_x, bottom + plot_height, strokeColor=JBW_NAVY, strokeWidth=1.5))
        drawing.add(String(median_x, bottom + plot_height + 2, "P50", textAnchor="middle", fontName=font_name, fontSize=8, fillColor=JBW_NAVY))
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
            series.append((_scenario_label(scenario, is_chinese), result["det_df"]["total_wealth"].tolist(), _PALETTE[index % len(_PALETTE)]))
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


def _register_arial_compatible_font():
    font_candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"), Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf")),
        (Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"), Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf")),
        (Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"), Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")),
    ]
    for regular_path, bold_path in font_candidates:
        if not regular_path.exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont("ArialReport", str(regular_path)))
            if bold_path.exists():
                pdfmetrics.registerFont(TTFont("ArialReport-Bold", str(bold_path)))
                pdfmetrics.registerFontFamily(
                    "ArialReport",
                    normal="ArialReport",
                    bold="ArialReport-Bold",
                    italic="ArialReport",
                    boldItalic="ArialReport-Bold",
                )
            return "ArialReport"
        except Exception:
            continue
    return "Helvetica"


def _styles(is_chinese):
    if is_chinese:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            font_name = "STSong-Light"
        except Exception:
            font_name = "Helvetica"
        body_size = 11.0
        small_size = 9.5
    else:
        font_name = _register_arial_compatible_font()
        body_size = 12.0
        small_size = 10.0

    base = getSampleStyleSheet()
    return font_name, {
        "title": ParagraphStyle("PdfTitle", parent=base["Title"], fontName=font_name, fontSize=24, leading=29, textColor=JBW_NAVY, alignment=TA_LEFT, spaceAfter=9),
        "subtitle": ParagraphStyle("PdfSubtitle", parent=base["Normal"], fontName=font_name, fontSize=body_size, leading=17, textColor=JBW_MUTED, spaceAfter=13),
        "h1": ParagraphStyle("PdfH1", parent=base["Heading1"], fontName=font_name, fontSize=18, leading=22, textColor=JBW_NAVY, spaceBefore=9, spaceAfter=7),
        "h2": ParagraphStyle("PdfH2", parent=base["Heading2"], fontName=font_name, fontSize=14, leading=18, textColor=JBW_DEEP_NAVY, spaceBefore=6, spaceAfter=5),
        "body": ParagraphStyle("PdfBody", parent=base["BodyText"], fontName=font_name, fontSize=body_size, leading=17, textColor=JBW_INK, spaceAfter=8),
        "small": ParagraphStyle("PdfSmall", parent=base["BodyText"], fontName=font_name, fontSize=small_size, leading=13, textColor=JBW_MUTED, spaceAfter=4),
        "table_header": ParagraphStyle("PdfTableHeader", parent=base["BodyText"], fontName=font_name, fontSize=small_size, leading=13, textColor=colors.white, spaceAfter=0),
        "metric": ParagraphStyle("PdfMetric", parent=base["BodyText"], fontName=font_name, fontSize=12, leading=15, alignment=TA_CENTER, textColor=JBW_DEEP_NAVY),
    }


def _footer(canvas, doc, font_name, is_chinese):
    canvas.saveState()
    canvas.setFillColor(JBW_NAVY)
    canvas.rect(0, A4[1] - 5 * mm, A4[0], 5 * mm, fill=1, stroke=0)
    canvas.setStrokeColor(JBW_GRID)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFont(font_name, 8)
    canvas.setFillColor(JBW_MUTED)
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
    report_language=None,
):
    selected_chart_keys = [key for key in selected_chart_keys if key in PDF_CHART_KEYS]
    inputs = selected_result["inputs"]
    det_df = selected_result["det_df"]
    summary_df = selected_result["summary_df"]
    is_chinese = _language_is_chinese(report_language) if report_language is not None else _is_chinese(inputs)
    font_name, styles = _styles(is_chinese)

    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=19 * mm,
        title=str(inputs.get("report_title") or _text(is_chinese, "Financial Projection Report", "财务预测报告")),
        author="Retirement Modelling Suite (Australia)",
    )
    story = []

    report_title = str(inputs.get("report_title") or _text(is_chinese, "Financial Projection Report", "财务预测报告"))
    title_style = styles["title"]
    if is_chinese and not any("\u4e00" <= character <= "\u9fff" for character in report_title):
        _, latin_styles = _styles(False)
        title_style = latin_styles["title"]
    story.append(Paragraph(escape(report_title), title_style))
    story.append(Paragraph(
        escape(_text(
            is_chinese,
            f"Scenario: {selected_scenario} | Value basis: {value_mode} | Generated: {datetime.now().strftime('%d %B %Y')}",
            f"情景：{_scenario_label(selected_scenario, True)} | 价值口径：{'现值' if value_mode == 'Present Value' else '终值'} | 生成日期：{datetime.now().strftime('%Y年%m月%d日')}",
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
    metric_table = Table(metric_data, colWidths=[43 * mm] * 4, rowHeights=[11 * mm, 13 * mm])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), JBW_SKY),
        ("BOX", (0, 0), (-1, -1), 0.5, JBW_GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, JBW_GRID),
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
        Paragraph(_text(is_chinese, "Event", "事件"), styles["table_header"]),
        Paragraph(_text(is_chinese, "FY", "财年"), styles["table_header"]),
        Paragraph(_text(is_chinese, "Total wealth", "总财富"), styles["table_header"]),
        Paragraph(_text(is_chinese, "Interpretation", "说明"), styles["table_header"]),
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
        ("BACKGROUND", (0, 0), (-1, 0), JBW_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, JBW_GRID),
        ("BACKGROUND", (0, 1), (-1, -1), JBW_MIST),
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
            story.append(Paragraph(f"- {escape(_localise_warning(warning, is_chinese))}", styles["body"]))

    if selected_chart_keys:
        story.append(PageBreak())
        story.append(Paragraph(_text(is_chinese, "Selected charts and interpretation", "所选图表及解释"), styles["h1"]))
        for index, chart_key in enumerate(selected_chart_keys):
            if index > 0:
                story.append(PageBreak())
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
