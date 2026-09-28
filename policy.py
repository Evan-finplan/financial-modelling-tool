"""Australian tax and super policy settings used by the projection engine.

Financial years are represented by their ending calendar year. For example,
``2027`` means the 2026-27 income year.

Where a future indexed value has not been published, the most recently known
value is carried forward. This is deliberate and keeps policy assumptions in
one reviewable location instead of scattering hard-coded values through the
calculation engine.
"""


POLICY_VERSION = "2026-27 Budget phase 2 - residential property and trust minimum tax"
POLICY_LAST_REVIEWED = "2026-09-28"
LATEST_PUBLISHED_SUPER_THRESHOLD_FY = 2027


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
DIVISION_293_THRESHOLD = 250_000.0
DIVISION_293_TAX_RATE = 0.15

# Residential negative-gearing changes are legislated. The restriction applies
# from the 2027-28 income year to established residential property acquired
# after 7:30pm AEST on 12 May 2026, subject to statutory exceptions.
NEGATIVE_GEARING_RESTRICTION_START_FY = 2028

# The discretionary-trust measure is still exposure draft policy at the date
# above. It is modelled as a scenario assumption, not represented as enacted law.
DISCRETIONARY_TRUST_MINIMUM_TAX_START_FY = 2029
DISCRETIONARY_TRUST_MINIMUM_TAX_RATE = 0.30
DISCRETIONARY_TRUST_MINIMUM_TAX_STATUS = "Exposure draft - not enacted"


CONCESSIONAL_CONTRIBUTIONS_CAPS = {
    2026: 30_000.0,
    2027: 32_500.0,
}

NON_CONCESSIONAL_CONTRIBUTIONS_CAPS = {
    2026: 120_000.0,
    2027: 130_000.0,
}

GENERAL_TRANSFER_BALANCE_CAPS = {
    2026: 2_000_000.0,
    2027: 2_100_000.0,
}

# The annual framework starts in 2026-27. The 2025-26 value is the
# annualised equivalent of the published $62,500 quarterly base.
SUPER_GUARANTEE_MAXIMUM_EARNINGS_BASES = {
    2026: 250_000.0,
    2027: 270_830.0,
}


POLICY_SOURCES = {
    "personal_tax_rates": "https://budget.gov.au/content/02-cost-of-living.htm",
    "super_rates_and_thresholds": "https://www.ato.gov.au/tax-rates-and-codes/key-superannuation-rates-and-thresholds",
    "division_293": "https://www.ato.gov.au/individuals-and-families/super-for-individuals-and-families/super/growing-and-keeping-track-of-your-super/caps-limits-and-tax-on-super-contributions/division-293-tax-on-concessional-contributions-by-high-income-earners",
    "negative_gearing": "https://www.legislation.gov.au/C2026A00049/asmade",
    "negative_gearing_amendments": "https://www.legislation.gov.au/C2026A00071/asmade",
    "discretionary_trust_exposure_draft": "https://consult.treasury.gov.au/c2026-799771",
}


def _latest_value_for_financial_year(values_by_year, financial_year_end):
    """Return the latest published value that applies to a financial year."""
    financial_year_end = int(financial_year_end)
    eligible_years = [year for year in values_by_year if year <= financial_year_end]
    if not eligible_years:
        return values_by_year[min(values_by_year)]
    return values_by_year[max(eligible_years)]


def get_concessional_contributions_cap(financial_year_end):
    return _latest_value_for_financial_year(
        CONCESSIONAL_CONTRIBUTIONS_CAPS,
        financial_year_end,
    )


def get_non_concessional_contributions_cap(financial_year_end):
    return _latest_value_for_financial_year(
        NON_CONCESSIONAL_CONTRIBUTIONS_CAPS,
        financial_year_end,
    )


def get_general_transfer_balance_cap(financial_year_end):
    return _latest_value_for_financial_year(
        GENERAL_TRANSFER_BALANCE_CAPS,
        financial_year_end,
    )


def get_super_guarantee_maximum_earnings_base(financial_year_end):
    return _latest_value_for_financial_year(
        SUPER_GUARANTEE_MAXIMUM_EARNINGS_BASES,
        financial_year_end,
    )


def get_policy_snapshot(financial_year_end):
    """Return the policy values used for an annual projection row."""
    return {
        "financial_year_end": int(financial_year_end),
        "concessional_contributions_cap": get_concessional_contributions_cap(financial_year_end),
        "non_concessional_contributions_cap": get_non_concessional_contributions_cap(financial_year_end),
        "general_transfer_balance_cap": get_general_transfer_balance_cap(financial_year_end),
        "super_guarantee_rate": SUPER_GUARANTEE_RATE,
        "super_guarantee_maximum_earnings_base": get_super_guarantee_maximum_earnings_base(financial_year_end),
        "division_293_threshold": DIVISION_293_THRESHOLD,
        "division_293_tax_rate": DIVISION_293_TAX_RATE,
        "negative_gearing_restriction_start_fy": NEGATIVE_GEARING_RESTRICTION_START_FY,
        "discretionary_trust_minimum_tax_start_fy": DISCRETIONARY_TRUST_MINIMUM_TAX_START_FY,
        "discretionary_trust_minimum_tax_rate": DISCRETIONARY_TRUST_MINIMUM_TAX_RATE,
        "discretionary_trust_minimum_tax_status": DISCRETIONARY_TRUST_MINIMUM_TAX_STATUS,
        "policy_version": POLICY_VERSION,
    }
