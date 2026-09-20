# International Payment Eligibility

## Enabling international payments

International card acceptance is not enabled by default on a new account;
it must be requested through the dashboard and requires KYC to already be
fully verified. The request is reviewed against the merchant's declared
business category — certain categories (financial services, pharmaceuticals,
adult content) are not eligible for international acceptance regardless of
KYC status.

## Approval timeline

Approval for international payments typically takes 3-5 business days once
requested, separate from and in addition to any KYC verification time. A
merchant who requests international payments the same day their KYC clears
should expect the two timelines to run one after the other, not in
parallel.

## Currency handling

International payments are captured in the customer's local currency where
supported, then converted to the merchant's settlement currency using the
platform's exchange rate at the time of capture, not at the time of
settlement. This means the settled amount can differ slightly from a
naive conversion at today's rate, especially if there is a delay between
capture and settlement.

## Additional fees

International transactions carry a currency conversion fee in addition to
the standard processing fee, and this fee is deducted before the
transaction amount enters the settlement pipeline — it does not appear as
a separate line item on the merchant's settlement report, only as a lower
net amount per transaction.

## Higher scrutiny for international transactions

International transactions are subject to additional automated risk
screening compared to domestic ones, because cross-border card fraud rates
are structurally higher. This means a merchant who is newly approved for
international payments may see a higher proportion of `RISK_BLOCK` failures
in their first few weeks of international volume as the risk engine
calibrates to their specific transaction patterns.

## Settlement cycle for international payments

International payments follow the merchant's existing settlement cycle
(T+3 or T+1) but are settled in a separate batch from domestic
transactions, since the currency conversion step adds a processing stage
that domestic payments do not go through.

## Approval review depth

The approval review for international acceptance is the same process and
timeline for every merchant regardless of typical transaction size; there
is no separate or deeper review track for merchants expecting high-ticket
international volume. Category eligibility and KYC status are the only
factors that affect approval.
