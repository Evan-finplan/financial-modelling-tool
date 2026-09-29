# Retirement Modelling Suite (Australia)

A Streamlit-based financial modelling tool for analysing retirement outcomes under Australian superannuation and tax rules.

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
- One-click PDF report export with selectable charts, future outlook, milestone commentary, and chart explanations
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
- Discretionary trust 30% minimum-tax policy scenario (exposure draft; not enacted)
- Retirement income sustainability

---

## 🖥️ Run Locally

pip install -r requirements.txt  
streamlit run app.py

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
