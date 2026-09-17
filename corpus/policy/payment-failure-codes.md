# Payment Failure Error Codes

## How to read a failure code

Every failed payment attempt is recorded with a short error code on the
transaction record. These codes originate from several different layers —
the card network, the issuing bank, the platform's own risk engine, or the
customer's UPI app — and the same underlying problem can surface under
different codes depending on which layer rejected it.

## Common card failure codes

`INSUFFICIENT_FUNDS` is returned by the issuing bank and means exactly what
it says; the platform cannot distinguish "insufficient funds" from "card
frozen for suspected fraud" because banks report both under similar
generic codes in many cases. `DO_NOT_HONOR` is a catch-all decline from the
issuing bank that can mean anything from a temporary hold to a permanent
block, and the platform has no visibility into the bank's specific reason.
`CVV_MISMATCH` and `INVALID_EXPIRY` originate from the network's own
validation before the issuing bank is even contacted.

## Common UPI failure codes

`UPI_TIMEOUT` means the customer's UPI app did not respond within the
network's timeout window (typically 90 seconds) — this is frequently a
customer-side connectivity issue, not a platform issue. `UPI_LIMIT_EXCEEDED`
means the customer has hit their bank's daily UPI transaction cap, which is
set by the customer's bank and is unrelated to the merchant's own limits.

## Risk-engine declines

`RISK_BLOCK` means the platform's own risk engine declined the transaction
before it reached the network at all, typically due to a velocity check
(too many attempts from the same instrument or IP in a short window) or a
match against a known fraud pattern. Unlike bank-side declines, a
`RISK_BLOCK` can sometimes be appealed by the merchant if they believe a
specific customer was incorrectly flagged, by raising a ticket with the
transaction ID.

## Retry behavior

The platform does not automatically retry a failed payment. Each retry
shown in a merchant's transaction log is a separate attempt initiated by
the customer (or the merchant's checkout flow, if it prompts the customer
to retry), each with its own independent error code.

## Failure codes and settlement

A failed payment never enters the settlement pipeline — it has no
associated payout, no refund is applicable, and it does not appear in any
settlement batch. Merchants sometimes confuse a failed payment with a
delayed settlement; the two are unrelated, since a failed payment was never
collected in the first place.
