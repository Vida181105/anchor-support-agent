# Account Review and Holds

## What triggers an account review

An account is placed under review when one or more automated risk signals
fire: a sudden spike in refund volume, multiple failed KYC document
verifications in a short window, a pattern of transactions matching known
fraud signatures, or a manual flag raised by the compliance team after an
external report. A review can also be triggered by a change in business
category or website content that no longer matches what was declared at
onboarding.

## What a hold means

A hold freezes outbound settlement for the account while the review is in
progress. Incoming payments continue to be collected and recorded normally;
only the payout leg is paused. A hold does not cancel or reverse
transactions that have already settled before the hold began.

## Review duration

Standard reviews are resolved within 7 business days of being opened. Reviews
that involve external verification (for example, confirming a business
registration with a government registry) can take up to 21 business days.
There is no expedited path for a review that requires external
verification, regardless of the reason for the delay.

## Communication during a review

The platform sends an email notification when a review opens and again when
it closes, to the email address on file for the account's primary contact.
If a merchant has changed their primary contact email without updating it
on the account, review notifications go to the old address and the merchant
may not see them.

## Partial holds

In some cases only a subset of an account's settlement is held — for
example, when a review is scoped to a specific product line or a specific
batch of transactions flagged as suspicious, rather than the whole account.
A partial hold displays on the dashboard as a reduced "next payout" amount
rather than a payout of zero, which is easy to mistake for a reserve
deduction rather than a hold.

## What ends a review

A review closes when the assigned compliance analyst either clears the
flagged signal or escalates the account to permanent restriction. There is
no automatic timeout that closes a review in the merchant's favor if the
review window is exceeded — an overdue review still requires manual
closure.
