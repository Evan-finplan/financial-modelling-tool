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

This is an aggregate annual projection. Property equity is included in net wealth and can be selected as a strategic funding source. The debt strategy module can allocate annual surplus to extra principal repayments or an offset, but it does not model contractual amortisation schedules. The model does not include depreciation schedules, borrowing-cost amortisation, refinancing, exact transaction costs, or residential CGT.

### Discretionary trust 30% minimum tax

The model includes a policy-scenario estimate from the 2028–29 income year. It calculates 30% trustee minimum tax on entered in-scope trust net income, allocates non-refundable credits to the modelled individual beneficiaries, and caps each credit at tax attributable to that beneficiary's trust income. Entered excluded income is removed from the minimum-tax base.

As at 28 September 2026, this measure is based on the Treasury Laws Amendment (Tax Reform No. 4) Bill 2026 exposure draft released on 3 September 2026. It is not enacted law. The calculation is therefore labelled **Exposure draft - not enacted** in annual output and may need to change when a Bill is introduced, amended, passed, or supported by final ATO guidance.

The model does not independently test trust legal form or eligibility for every exclusion. It does not model corporate beneficiaries, non-resident withholding, UPE/Division 7A interactions, franking-credit pools, charity caps, the proposed fixed-distribution election, restructuring rollover relief, integrity rules, or trustee administrative and collection rules.

## Explicitly not implemented in this release

The following matters remain outside the calculation engine and should not be inferred from any output:

- full asset-by-asset CGT parcel records, exact statutory transition apportionment, published quarterly CPI index numbers, and tax-lot disposal ordering beyond the pooled Phase 3 estimate described below;
- property CGT, depreciation schedules, contractual loan amortisation, refinancing, exact purchase and sale costs, multiple properties, and entity-by-entity residential loss pools;
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

## Phase 3 implementation: 2026 Budget CGT reform

The non-super investment pool now models the core enacted CGT changes for CGT events from the 2027–28 income year:

- a 30 June 2027 transition market value separates deferred pre-1 July 2027 gains from gains accruing after that date;
- the deferred pre-reform component retains the modelled 50% discount where the 12-month condition is selected;
- the post-reform cost base is indexed annually using a user-entered CPI estimate, producing an estimated real capital gain;
- carried-forward capital losses are tracked and applied first to post-reform real gains, then to the deferred pre-reform component;
- the Division 119 minimum-tax gap is calculated as 30% of the modelled minimum-tax capital gain less basic income tax attributable to that gain, rounded down to whole dollars;
- the minimum-tax calculation excludes Medicare levy and is calculated separately for each owner according to the non-super ownership percentage;
- confirmed statutory payment-recipient exemptions can be selected; and
- qualifying new residential dwellings and affordable housing can be modelled using either the 50% discount or indexation plus the 30% minimum-tax regime.

The core indexation, transition, discount and minimum-tax provisions are enacted in the Treasury Laws Amendment (Tax Reform No. 1) Act 2026 and apply to relevant CGT events from 1 July 2027. The precise transition apportionment and several special-case rules were the subject of Tranche 2 exposure draft consultation in August 2026. The model therefore labels the core policy as enacted while labelling its annual pooled transition method as an estimate.

### Phase 3 modelling limits

The current retirement engine holds one homogeneous non-super asset pool rather than an asset register. It uses annual periods and applies one CPI assumption to the remaining indexed pool. New surplus cash is added to both nominal and indexed cost base at the end of the annual cashflow calculation. Partial sales use the pool's average disposal fraction.

This is suitable for strategic scenario comparison, but not for preparing an income tax return. Tax-return work still requires acquisition dates, individual cost-base elements, actual 30 June 2027 market values or the final permitted apportionment method, published CPI index numbers, residency history, trust statements, loss choices, exemptions, small-business concessions, gifts/conservation deductions, and event-specific legal analysis.

The following CGT matters remain outside Phase 3:

- a multi-asset register and parcel-level sale selection;
- foreign or temporary resident adjustments;
- pre-CGT asset K6 calculations;
- small-business CGT concessions and active-asset reductions;
- trust-level CGT attribution and beneficiary statement mechanics;
- deceased-estate, relationship-breakdown and rollover events;
- gifts and conservation-covenant deductions that may reduce a minimum-tax capital gain;
- automatic identification of government-payment exemptions; and
- residential investment property disposal and CGT within the separate property module.

Phase 3 sources:

- Treasury Laws Amendment (Tax Reform No. 1) Act 2026: https://www.legislation.gov.au/C2026A00049/asmade
- Treasury CGT and negative gearing Tranche 2 consultation: https://consult.treasury.gov.au/c2026-792170
- Treasury Budget tax changes overview: https://treasury.gov.au/policy-topics/taxation/budget2026-27

## Strategy comparison and asset drawdown release

The application can compare a common set of economic assumptions across Base Case and Strategy A/B/C asset drawdown profiles. The comparison reports cumulative after-tax cashflow, wealth at Person 1's retirement, deterministic and simulated final wealth, failure probability, cumulative tax, first advantage year, break-even year and modelled risk observations.

Annual cash shortfalls can be funded in a selected order from cash reserves, the pooled non-super investment account, pension balances, accumulation super and residential property equity. Cash, non-super and property estate reserve floors are respected where sufficient alternative assets are available. Partial property disposals proportionally reduce modelled property value and associated debt after estimated selling costs.

This remains strategic modelling rather than transaction or tax-return calculation. The property disposal estimate does not include property CGT, exact conveyancing costs, lender requirements, legal feasibility of partial disposal, ownership restructuring, stamp duty or transaction-specific tax advice. Super withdrawals must also be reviewed against preservation and conditions-of-release requirements before an adviser relies on a strategy result.

PDF reports are available as Client Summary, Advice Support Report and Technical Appendix. The longer formats add strategy outcomes, key risks, assumption changes, adviser notes, policy status, calculation methodology and reconciliation disclosures.

## Debt strategy comparison

The model separately tracks private-purpose non-deductible debt and the deductible investment-property loan. It calculates interest on each net of the linked offset balance. Non-deductible interest is treated as a household cash outflow; investment-property interest continues through the residential property tax calculation.

Annual cash surplus can be directed in a selected order to the cash reserve, non-deductible offset, non-deductible principal, deductible offset, deductible principal or the pooled non-super investment account. Debt comparison mode evaluates the current selection against non-deductible-first, offset-first, deductible-first and invest-surplus strategies, including cumulative interest, tax, ending debt, offset liquidity, debt-free timing, final wealth and failure probability.

Deductibility depends on the use of borrowed funds rather than the security. The comparison does not determine legal deductibility or model daily interest, minimum contractual repayments, loan fees, fixed-rate break costs, refinance costs, redraw contamination, lender serviceability or whether extra repayments remain redrawable. Actual loan statements and tax records must be reviewed before advice is implemented.
