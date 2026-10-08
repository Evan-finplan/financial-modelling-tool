# Retirement Modelling Suite (Australia)

A Streamlit-based financial modelling tool for analysing retirement, superannuation, tax, asset drawdown and debt-strategy outcomes under Australian rules.

---

## 🌐 Live App

👉 https://financial-modelling-tool-4ea8y7mxom9xjgbpfzzbyd.streamlit.app/

---

## 🚀 Key Features

- Deterministic projection modelling
- Monte Carlo simulation
- Dual-person modelling (Person 1 / Person 2)
- Super accumulation → pension phase
- Transfer Balance Cap (TBC) enforcement
- CGT modelling (super and non-super assets)
- Adviser View / Client View
- Excel export (inputs, assumptions, debug tables)
- Client Module controls to hide non-applicable people, super accumulation, non-super investments, residential property, discretionary trust, cash-surplus strategy and investment-debt sections while retaining saved inputs and excluding inactive modules from calculations and reports. Pension phase is always active: a zero opening pension balance represents a client who has not commenced pension yet, while future accumulation-to-pension transfer still occurs at the selected pension start age.
- One-click PDF report export with selectable charts, future outlook, milestone commentary, and chart explanations
- Strategy comparison for Base Case and Strategy A/B/C, including after-tax cashflow, retirement wealth, final wealth, failure probability, cumulative tax, advantage timing, break-even year and key risks
- Configurable asset drawdown order across cash, non-super investments, pension, accumulation super and residential property equity, with estate reserve floors
- A permanent Property page for the main residence and its principal-and-interest home loan, plus an optional residential-investment-property pool with its own principal-and-interest loan
- A separate Investment Debt module for other deductible and non-deductible investment debts, explicitly excluding the main-residence and residential-investment-property loans
- A separate Cash Surplus Strategy module for directing surplus to cash reserves, offsets, debt repayment or non-super investment, with side-by-side strategy comparison
- A separate discretionary-trust investment pool with balance, cost base, income/capital return assumptions and excluded income entered as a percentage of modelled trust income
- Three PDF detail levels: Client Summary, Advice Support Report and Technical Appendix
- Editable assumption presets
- Bilingual interface (English / 中文)

---

## 🧠 Modelling Scope

- Australian personal income tax
- Medicare levy
- Superannuation contributions & earnings tax
- Pension phase tax treatment
- Capital gains tax (average cost method)
- 2026 Budget CGT transition split, CPI-indexed cost base, capital-loss ledger, and 30% minimum-tax estimate
- Division 293 tax estimate for high-income clients
- 2026-27 SG maximum earnings base and indexed super caps
- Residential investment property cashflow and legislated negative-gearing loss quarantine
- Principal-and-interest home and residential-investment-property loans, plus other deductible and non-deductible investment debts, linked offsets, cash-surplus allocation and repayment-strategy comparison
- Discretionary trust 30% minimum-tax policy scenario (exposure draft; not enacted)
- Retirement income sustainability

---

## 🖥️ Run Locally

pip install -r requirements.txt  
streamlit run app.py

## 🔒 Public-session privacy and upload safeguards

- The app does not use a customer database or persist model inputs between browser sessions.
- Users can select **Clear This Session** to remove inputs, uploaded data, simulation results and saved snapshots from the current session.
- Excel uploads are limited to `.xlsx` files of 10 MB or less. Macros, external links, oversized worksheets and unusually large compressed workbooks are rejected before parsing.
- Production error details are hidden from public users, while cross-site request protections remain enabled.
- Do not add secrets to the repository. Store future credentials in `.streamlit/secrets.toml` locally or in Streamlit Community Cloud secrets settings.

## Automated tests

Run the policy and projection regression suite with:

```text
python -m unittest discover -s tests -v
```

The 2026-27 policy scope, exclusions, and modelling assumptions are documented in `BUDGET_2026_NOTES.md`.

---

## 🌏 Language Support

Use the sidebar to switch between:

- 🇬🇧 English  
- 🇨🇳 中文  

---

## ⚠️ Disclaimer

This tool is for modelling and educational purposes only.  
It does not constitute financial advice.

---

## 👤 Author

- Margaret (Yunfei) Chen
- Jinglei Zhang  
