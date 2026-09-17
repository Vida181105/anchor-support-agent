# Refund Timelines

## Refund initiation

A merchant can initiate a refund from the dashboard or via API for any
captured payment within 180 days of the original transaction. Refunds
cannot be initiated against payments that were only authorized but never
captured — those should be voided instead, which follows a separate,
same-day process.

## Time to reach the customer

Once a refund is initiated on the platform side, the time for funds to
reach the customer depends on the original payment method, not on
anything the merchant controls after initiation. Card refunds typically
take 5-7 business days to reflect on the customer's statement, because this
timeline is set by the card network and issuing bank, not the platform.
UPI and wallet refunds are typically same-day to 2 business days. Net
banking refunds take 3-5 business days.

## Refund deduction from settlement

A refund is deducted from the merchant's next available settlement batch at
the time the refund is processed, not from the original transaction's
batch. If the original transaction has already settled, the refund appears
as a negative line item in a later settlement rather than reversing the
original payout.

## Partial refunds

Partial refunds are supported for any captured payment and can be issued
multiple times against the same transaction as long as the cumulative
refunded amount does not exceed the original captured amount. Each partial
refund follows its own independent timeline to the customer.

## Failed refund attempts

A refund can fail after initiation if the customer's original payment
instrument has since expired or been closed (common with cards) or if the
receiving account has since been frozen. A failed refund is flagged on the
dashboard and does not silently retry — the merchant must choose an
alternate payout method to the customer or contact support.

## Refunds and dispute status

Issuing a refund on a transaction that already has an open dispute or
chargeback filed against it does not automatically withdraw the dispute.
The dispute must be separately marked as resolved once the issuing bank
confirms receipt of the refund, which can take longer than the refund
itself.
