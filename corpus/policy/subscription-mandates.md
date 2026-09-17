# Subscription Mandate Rules

## What a mandate is

A mandate is a customer's standing authorization for a merchant to charge a
recurring amount on a defined schedule, used for subscriptions and
installment plans. A mandate is set up once, at the time of the first
payment, and subsequent charges reference the mandate rather than requiring
the customer to re-enter payment details each cycle.

## Mandate registration and the first charge

The first charge under a new mandate typically requires full customer
authentication (e.g., an OTP or bank app approval), even if later charges
under the same mandate do not. This first-charge friction is a network
requirement, not a platform choice, and cannot be waived by the merchant.

## Recurring charge authentication

Recurring charges below a fixed per-transaction threshold are generally
exempt from repeat authentication under the network's e-mandate rules. A
charge above that threshold, or a charge amount that varies from what was
registered on the mandate (common with usage-based billing), will not be
exempt and can be silently declined by the issuing bank if the merchant's
integration does not route it through the additional authentication flow.

## Mandate failure and retry rules

If a recurring charge fails, the platform allows up to 3 automatic retry
attempts over the following 7 days, spaced at least 48 hours apart. Beyond
the third failed attempt, the mandate is marked `paused` and requires the
merchant to prompt the customer to re-authorize before charges resume — the
platform does not keep retrying indefinitely.

## Customer-initiated mandate cancellation

A customer can revoke a mandate directly through their bank or UPI app
without notifying the merchant first. The platform receives a
`mandate.revoked` webhook when this happens, but there is often a delay of
up to 24 hours between the customer's revocation and the platform
receiving confirmation from the bank, during which a scheduled charge
attempt may still fail with a generic decline rather than a clear
"mandate revoked" reason.

## Mandate amount changes

Changing the recurring amount on an existing mandate (for example, a plan
upgrade) requires creating a new mandate and cancelling the old one; the
platform does not support modifying an active mandate's amount in place.
This means an amount change always re-triggers the first-charge
authentication requirement described above.
