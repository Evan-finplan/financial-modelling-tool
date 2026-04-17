import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from model import (
    build_percentile_table,
    run_deterministic_projection,
    run_monte_carlo,
)


# ============================================================
# SECTION: EXAMPLE INPUTS
# ============================================================

inputs = {
    "start_financial_year": 2027,
    "projection_years": 40,
    "retirement_spending_trigger": "Both Retired",
    "household_mode": "Two People",

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

    "non_super_balance": 500000.0,
    "non_super_cost_base": 300000.0,
    "cgt_discount_rate": 0.50,

    "person1_annual_income": 120000.0,
    "person2_annual_income": 60000.0,

    "annual_living_expenses": 90000.0,
    "retirement_spending": 100000.0,

    "non_super_ownership_person1": 0.50,

    "inflation_rate": 0.03,

    "super_income_return_mean": 0.02,
    "super_income_return_std": 0.02,
    "super_capital_return_mean": 0.04,
    "super_capital_return_std": 0.09,

    "non_super_income_return_mean": 0.02,
    "non_super_income_return_std": 0.02,
    "non_super_capital_return_mean": 0.03,
    "non_super_capital_return_std": 0.08,

    "number_of_simulations": 10000,
    "assumption_preset": "Custom",

    "contribution_events": [
        {
            "financial_year": 2028,
            "person": "Person 1",
            "contribution_type": "personal_deductible",
            "amount": 20000.0,
        },
        {
            "financial_year": 2029,
            "person": "Person 2",
            "contribution_type": "non_concessional",
            "amount": 30000.0,
        },
    ],
}


# ============================================================
# SECTION: HELPER FUNCTIONS
# ============================================================

def format_currency(x, pos):
    return f"${x:,.0f}"


# ============================================================
# SECTION: MAIN
# ============================================================

if __name__ == "__main__":
    det_df = run_deterministic_projection(inputs)
    summary_df, all_paths_df = run_monte_carlo(inputs, random_seed=42)

    success_rate = summary_df["success"].mean()
    median_final_wealth = summary_df["final_wealth"].median()
    p10_final_wealth = summary_df["final_wealth"].quantile(0.10)
    p90_final_wealth = summary_df["final_wealth"].quantile(0.90)

    print("\nDeterministic projection preview:\n")
    print(
        det_df[
            [
                "financial_year_label",
                "person1_age",
                "person2_age",
                "person1_phase",
                "person2_phase",
                "person1_net_income",
                "person2_net_income",
                "person1_sg_contribution",
                "person2_sg_contribution",
                "person1_personal_deductible_contribution",
                "person2_non_concessional_contribution",
                "non_super_income_earnings",
                "non_super_capital_earnings",
                "total_minimum_pension_drawdown",
                "total_tax_paid",
                "ending_total_super_balance",
                "ending_non_super_balance",
                "total_wealth",
            ]
        ].head(10)
    )

    print("\nMonte Carlo summary:\n")
    print(f"Number of simulations: {len(summary_df):,}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Median final wealth: ${median_final_wealth:,.0f}")
    print(f"P10 final wealth: ${p10_final_wealth:,.0f}")
    print(f"P90 final wealth: ${p90_final_wealth:,.0f}")

    percentile_df = build_percentile_table(all_paths_df)

    # ========================================================
    # SECTION: CHART 1 - HISTOGRAM OF FINAL WEALTH
    # ========================================================
    plt.figure(figsize=(10, 6))
    plt.hist(summary_df["final_wealth"], bins=50)
    plt.axvline(median_final_wealth, linestyle="--", label="Median")
    plt.axvline(p10_final_wealth, linestyle="--", label="P10")
    plt.axvline(p90_final_wealth, linestyle="--", label="P90")
    plt.xlabel("Final Wealth")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo Final Wealth Distribution")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_currency))
    plt.legend()
    plt.grid(True)
    plt.show()

    # ========================================================
    # SECTION: CHART 2 - DETERMINISTIC TOTAL WEALTH
    # ========================================================
    plt.figure(figsize=(10, 6))
    plt.plot(det_df["financial_year_end"], det_df["total_wealth"], label="Deterministic Total Wealth")
    plt.xlabel("Financial Year")
    plt.ylabel("Total Wealth")
    plt.title("Deterministic Total Wealth Projection")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
    plt.grid(True)
    plt.legend()
    plt.show()

    # ========================================================
    # SECTION: CHART 3 - P10 / P50 / P90 PATHS
    # ========================================================
    plt.figure(figsize=(10, 6))
    plt.plot(percentile_df["financial_year_end"], percentile_df["p10"], label="P10")
    plt.plot(percentile_df["financial_year_end"], percentile_df["p50"], label="P50")
    plt.plot(percentile_df["financial_year_end"], percentile_df["p90"], label="P90")
    plt.xlabel("Financial Year")
    plt.ylabel("Total Wealth")
    plt.title("Monte Carlo Percentile Paths")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
    plt.grid(True)
    plt.legend()
    plt.show()