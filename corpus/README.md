# Corpus

Synthetic material only. Nothing here is copied or scraped from Razorpay's
actual documentation, support content, or systems — it is written from
scratch to be realistic in shape and plausible in substance, for the purpose
of building and evaluating the Anchor prototype.

## Layout

- `policy/` — the policy knowledge base. Ten short markdown documents covering
  settlement, reserves, KYC, disputes, and related topics.
- `merchants/` — structured account-state fixtures for 15 synthetic merchants.
- `tickets/` — synthetic support tickets, written to look like a real queue.

## Evidence-id scheme

Every piece of evidence in the corpus — whether it comes from a policy
document or from a merchant's account state — has a stable, deterministic id.
Downstream components (retrieval, the agent, the evaluation harness) cite
these ids directly, so the ids must never be renumbered by regenerating the
corpus.

### Policy chunks: `doc:<doc-slug>#c<n>`

- `<doc-slug>` is the filename of the policy doc without its `.md` extension
  (e.g. `settlement-cycles`, `rolling-reserves`).
- `<n>` is a 1-indexed chunk number, assigned in document order by splitting
  the doc on its `##` headings — one chunk per section, in the order the
  sections appear in the file.
- Example: the third section of `corpus/policy/settlement-cycles.md` is
  `doc:settlement-cycles#c3`.
- Chunk numbers are stable because they follow section order, not content
  hashes: reordering sections would renumber chunks (avoid this), but editing
  the text *within* a section, or adding a new trailing section, does not
  change any existing chunk's id. Never delete or reorder an existing `##`
  section once tickets/fixtures reference its chunk id — append new sections
  at the end instead.

### Merchant state fields: `state:<merchant-id>[.<dotted.path>]`

- `<merchant-id>` is the merchant's id as used in `corpus/merchants/*.json`
  (e.g. `merchant_7`).
- `<dotted.path>` is the JSON path to the specific field being cited, using
  `.` for object keys and `[i]` for array indices.
- Example: `state:merchant_7.settlement_schedule.next_payout`, or
  `state:merchant_3.transactions[12].status` for a single transaction's
  status field.
- Paths are stable because they name fields, not values: a field's value can
  change (e.g. a payout date moving forward) without changing its id.
- The path is optional: a bare `state:merchant_7` cites the merchant's whole
  account-state record as one unit. This is what a bulk lookup
  (`get_merchant_state`) returns evidence for; a tool that fetches one
  specific field (e.g. `get_settlement_schedule`) should cite the narrower
  path instead, since that is what the Phase 3 verifier checks a claim
  against — prefer the most specific id a tool actually has.

## Regenerating the policy KB

Chunk ids are derived purely from each doc's existing `##` section order, so
adding a new policy doc, or appending a new trailing section to an existing
one, is always safe. Do not reorder or delete existing sections.
