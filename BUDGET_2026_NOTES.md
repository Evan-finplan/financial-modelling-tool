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
