# Settlement Cycles and Timing

## Standard settlement cycle

By default, a merchant's settlement cycle is T+3: funds collected on a given
business day (T) are settled to the merchant's linked bank account three
business days later. "Business day" excludes weekends and bank holidays in
the merchant's settlement region, so a payment collected on a Thursday
typically settles the following Tuesday, not Sunday.

## Faster settlement tiers

Merchants who have completed at least 90 days on the platform with no open
disputes and a clean KYC record are eligible to request T+1 settlement. This
is not automatic — it must be requested through the merchant dashboard and
is approved within 5 business days. Merchants on new or unverified accounts
are placed on T+3 by default and are not eligible to request a faster cycle
until the 90-day mark.

## How the settlement date is calculated

The settlement date for a transaction is calculated from the date the
payment was **captured**, not the date it was authorized or the date the
order was placed. For card payments in particular, authorization and
capture can be several hours or, in rare cases, a full calendar day apart,
which shifts the T+0 anchor date later than merchants sometimes expect.

## Settlement batching

Transactions are settled in a single daily batch per merchant, not
individually. If a merchant's settlement window falls on a day their
account has an active hold (see the account review and holds document),
that day's entire batch is deferred, not just the flagged transactions
within it.

## What "next payout" means on the dashboard

The "next payout" date shown on the merchant dashboard reflects the
settlement date for the oldest unsettled batch, not a running daily payout.
A merchant checking this field on day two of a T+3 cycle will correctly see
a date that is still one business day away — this is expected behavior, not
a delay.

## Delayed settlement due to bank-side issues

In rare cases, a settlement batch is released on schedule but delayed by
the receiving bank before it lands in the merchant's account. This shows as
"settled" on the merchant dashboard even though funds have not yet cleared
the bank. Bank-side delays typically resolve within 1-2 additional business
days and are outside the platform's settlement SLA.
