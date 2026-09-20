# Rolling Reserves

## What a rolling reserve is

A rolling reserve is a percentage of each settlement withheld for a fixed
period as a buffer against future refunds and chargebacks, rather than paid
out immediately. The withheld amount is released automatically once its
hold period expires, on a rolling basis — it is not a one-time deduction.

## When a reserve is applied

Reserves are applied automatically when a merchant's account crosses a risk
threshold: a chargeback rate above 1% of transaction volume in a trailing
30-day window, a dispute rate above 0.65%, or a sudden volume spike (more
than 3x the merchant's trailing 90-day average) combined with a business
category flagged as higher-risk. A reserve can also be applied manually by
the risk team following a KYC re-review, independent of transaction
behavior.

## Standard reserve terms

The default reserve, once triggered, withholds 10% of each day's settlement
for a rolling 90-day period. For merchants in higher-risk categories
(travel, subscription services, high-ticket electronics), the default is
15% for a rolling 120-day period. These percentages and durations are set
per-merchant and recorded on the merchant's account; they are not published
as a single universal number because they vary by risk category and
history.

## How reserves interact with settlement

The reserve percentage is deducted from the settlement batch before payout,
so the "next payout" amount shown on the dashboard is already net of any
active reserve. Merchants sometimes expect the reserve to be a separate
withdrawal after the fact — it is not; it never leaves the reserve pool to
begin with.

## Reserve removal

A reserve is removed early only if the risk team completes a manual review
and finds the underlying trigger has resolved (e.g., chargeback rate back
under 0.5% for 60 consecutive days). There is no self-serve option to
remove a reserve from the merchant dashboard. Merchants can request a
review, but a request alone does not pause or reduce the reserve while
pending.

## Reserve balance visibility

The total amount currently held in reserve, and the schedule on which it
will be released, is visible on the merchant dashboard under "Reserve
Balance." Amounts scheduled for release appear there up to 90 days in
advance of their release date.

## What counts toward the chargeback rate

The chargeback rate used to evaluate the reserve trigger counts only
chargebacks that were actually upheld against the merchant (i.e., disputes
the merchant lost) within the trailing 30-day window. A dispute that was
opened and then resolved in the merchant's favor does not count toward this
rate, since no funds were ultimately reversed.

## Reserve scope

A reserve withholds a percentage of each day's total settlement; it is not
scoped to specific transactions, payment methods, or order types. There is
no mechanism to apply a reserve to only part of a merchant's transaction
mix — once triggered, it applies uniformly to the whole account's daily
settlement until it is removed.
