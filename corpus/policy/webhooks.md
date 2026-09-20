# Webhook Delivery and Retries

## What triggers a webhook

A webhook event is fired for state changes on a transaction, refund,
dispute, or settlement — for example `payment.captured`, `refund.processed`,
or `settlement.processed`. Webhooks are fired once per state change, not
once per API call, so a single dashboard action that changes several
related records can fire multiple webhook events close together.

## Delivery attempts and retry schedule

If a merchant's configured webhook endpoint does not return a 2xx response
within 10 seconds, the platform treats the attempt as failed and retries on
an exponential backoff schedule: 1 minute, 5 minutes, 30 minutes, 2 hours,
6 hours, then once every 12 hours up to a total of 24 hours after the
original event. After 24 hours of failed attempts, the event is marked
`undelivered` and no further retries occur.

## Idempotency

Because of retries, a merchant's endpoint may receive the same webhook
event more than once. Every webhook payload includes an `event_id` field
that is stable across retries of the same event, and merchant integrations
are expected to deduplicate on this field rather than assuming exactly-once
delivery.

## Checking delivery status

The dashboard's webhook log shows the status and timestamp of every
delivery attempt for each event, including the HTTP response code returned
by the merchant's endpoint on each attempt. This is the authoritative
source for whether an event was delivered — the underlying transaction,
refund, or settlement record itself does not store webhook delivery status.

## Manually replaying a webhook

A merchant can manually trigger a re-delivery of any event from the
dashboard's webhook log, regardless of whether the automatic retry window
has expired. A manual replay uses the same `event_id` as the original
event, so a correctly deduplicating integration will not double-process it.

## Webhook failures do not affect the underlying record

A failed or undelivered webhook has no effect on the underlying payment,
refund, settlement, or dispute — those records update in the platform's
systems regardless of whether the merchant's endpoint ever received
notice. A merchant who never received a `refund.processed` webhook may
still find the refund fully processed when checking the dashboard directly.

## Checking delivery status outside the dashboard

Webhook delivery status is only available through the dashboard's webhook
log; there is no separate API endpoint for querying past delivery attempts,
and delivery status is not emailed or otherwise pushed to the merchant.
