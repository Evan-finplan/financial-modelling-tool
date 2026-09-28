# 2026–27 Budget Modelling Notes

## Current scope decision

The following legislated 2026–27 Budget measures have been identified but are intentionally excluded from the current model implementation:

1. The standard deduction for work-related expenses of up to $1,000 from the 2026–27 income year, including its interaction with actual eligible work-related expenses.
2. The Working Australians Tax Offset of up to $250 from the 2027–28 income year, including its eligibility rules and non-refundable cap.
3. Separate reporting of the standard deduction, WATO, and income tax before and after these measures.

## Reason for exclusion

The model is currently designed for a client base with comparatively high incomes. These measures are not expected to be material to the primary modelling use cases, so adding the related inputs, calculations, disclosures, and tests would increase complexity without sufficient current benefit.

This is a modelling scope decision, not a conclusion that the measures are unavailable to high-income clients. Eligibility and outcomes depend on the legislation and each client's circumstances.

## Reassessment triggers

Reconsider these measures if any of the following occurs:

- the model is extended to lower- or middle-income client segments;
- detailed personal tax return estimation becomes part of the model's purpose;
- users need a reconciliation between gross income, deductions, offsets, and final tax;
- the financial impact becomes material to a documented client scenario; or
- the relevant legislation or ATO guidance changes.

Until one of these triggers applies, these measures should remain outside the calculation engine, user interface, exports, and automated tests.

## Phase 1 implementation

Phase 1 instead focuses on measures and controls that are material to higher-income clients:

- a central policy configuration for the 2026-27 personal tax rates, super contribution caps, general transfer balance cap, SG rate, and SG maximum earnings base;
- an annual SG maximum earnings base of $270,830 for 2026-27;
- an estimated Division 293 tax calculation using the income components available in the model;
- separate Division 293 reporting in annual results, adviser tax summaries, detailed cashflow tables, charts, and exports; and
- automated boundary and projection regression tests.

The Division 293 result is an estimate. The current model does not capture every component of the statutory Division 293 income definition, including reportable fringe benefits, net rental property losses, defined benefit contributions, and other adjustments used in an ATO assessment. Phase 1 assumes the estimated Division 293 liability is paid from household cash rather than released from super.

Published indexed super thresholds are configured through 2027FY. Later projection years retain the latest known contribution caps, general transfer balance cap, and SG maximum earnings base until the policy configuration is refreshed with newly published values.

## Phase 2 implementation: residential property and discretionary trusts

Phase 2 is intentionally limited to the two items requested for this release.

### Residential investment and negative gearing

The model now includes one aggregate residential investment property with opening market value, an interest-only loan, gross rent, deductible operating expenses, ownership, and growth assumptions. It applies the legislated loss-quarantine rule from the 2027–28 income year where the property is an affected established dwelling acquired after 7:30pm AEST on 12 May 2026. A current-year restricted loss is carried forward and may offset future modelled residential property income.

The interface includes scenario flags for a property acquired before Budget time, a qualifying new build, and qualifying exempt housing. These flags require adviser confirmation against the legislation and current guidance. The August 2026 amendments preserving treatment in specified death and relationship-breakdown transfers are recognised in the policy notes, but the model does not independently identify those legal events; the adviser must select the resulting applicable status.

This is an aggregate, interest-only projection. Property equity is included in net wealth but is not treated as liquid funding for household spending. The model does not sell or refinance the property and does not model principal repayments, depreciation schedules, borrowing-cost amortisation, individual property disposal, transaction costs, or residential CGT.

### Discretionary trust 30% minimum tax

The model includes a policy-scenario estimate from the 2028–29 income year. It calculates 30% trustee minimum tax on entered in-scope trust net income, allocates non-refundable credits to the modelled individual beneficiaries, and caps each credit at tax attributable to that beneficiary's trust income. Entered excluded income is removed from the minimum-tax base.

As at 28 September 2026, this measure is based on the Treasury Laws Amendment (Tax Reform No. 4) Bill 2026 exposure draft released on 3 September 2026. It is not enacted law. The calculation is therefore labelled **Exposure draft - not enacted** in annual output and may need to change when a Bill is introduced, amended, passed, or supported by final ATO guidance.

The model does not independently test trust legal form or eligibility for every exclusion. It does not model corporate beneficiaries, non-resident withholding, UPE/Division 7A interactions, franking-credit pools, charity caps, the proposed fixed-distribution election, restructuring rollover relief, integrity rules, or trustee administrative and collection rules.

## Explicitly not implemented in this release

The following matters remain outside the calculation engine and should not be inferred from any output:

- full asset-level CGT records, pre/post 1 July 2027 gain segmentation, CPI cost-base indexation, the 30% minimum tax on real capital gains, capital-loss ledgers, and partial disposal ordering;
- property CGT, depreciation schedules, loan principal amortisation, refinancing, purchase and sale costs, multiple properties, and entity-by-entity residential loss pools;
- complete Division 293 statutory income inputs, including reportable fringe benefits, net investment loss adjustments, defined benefit contributions, and ATO assessment reconciliation;
- concessional contribution carry-forward eligibility and non-concessional bring-forward rules;
- personal transfer balance account history and proportional indexation of an individual's transfer balance cap;
- the $1,000 standard deduction, Working Australians Tax Offset, and their requested separate pre/post-tax reporting, as documented above; and
- final-law discretionary trust rules and administrative mechanisms that remain subject to the legislative process.

## Phase 2 policy sources

- Treasury Laws Amendment (Tax Reform No. 1) Act 2026, Schedule 2: https://www.legislation.gov.au/C2026A00049/asmade
- Treasury Laws Amendment (Tax Reform No. 2) Act 2026, Schedule 4: https://www.legislation.gov.au/C2026A00071/asmade
- Treasury Budget tax changes overview: https://treasury.gov.au/policy-topics/taxation/budget2026-27
- Minimum tax on discretionary trusts exposure draft: https://consult.treasury.gov.au/c2026-799771
