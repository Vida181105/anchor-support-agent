# Chargeback and Dispute Windows

## What counts as a dispute vs. a chargeback

A dispute is opened when a customer contacts their card issuer or bank to
contest a transaction, before any funds move. A chargeback is the outcome
where the disputed amount is forcibly reversed from the merchant and
returned to the customer. Not every dispute becomes a chargeback — a
merchant that responds with sufficient evidence within the response window
can win the dispute and keep the funds.

## Response window

Merchants have 7 calendar days from the date a dispute notification is
received to submit evidence through the dashboard. This window is set by
the card network's rules, not by the platform, and cannot be extended
except in specific network-approved circumstances (e.g., a declared
regional payment-system outage).

## What counts as sufficient evidence

Acceptable evidence varies by dispute reason code but generally includes:
proof of delivery with a signature or tracking confirmation, the customer's
original order confirmation, and any customer communication acknowledging
receipt of the goods or service. A dispute reason of "product not as
described" requires additional evidence such as product photos or a
detailed description matching what was advertised — proof of delivery alone
is not sufficient for this reason code.

## Timeline after evidence is submitted

Once evidence is submitted, the issuing bank typically takes 30-45 days to
issue a final decision. This window is well outside the platform's control
and does not follow the same SLA as refunds or settlement. The dispute
status on the dashboard remains "under review" for the full duration.

## Chargeback fees

A chargeback fee is charged to the merchant's account regardless of the
final outcome of the dispute, to cover network processing costs. If the
merchant later wins the dispute, the disputed transaction amount is
returned, but the chargeback fee itself is non-refundable.

## Chargebacks and reserve triggers

A cluster of chargebacks within a short window is one of the automated
triggers for a rolling reserve (see the rolling reserves document). The
chargeback rate used for this calculation is measured over a trailing
30-day window and is recalculated daily, so an account can cross the
reserve threshold days after the chargebacks themselves occurred, once
enough of them land within the same rolling window.

## Chargeback fee amount

The chargeback fee is a flat amount per case, set by the card network and
passed through unchanged by the platform. It does not vary by dispute
reason code, transaction amount, or outcome.
