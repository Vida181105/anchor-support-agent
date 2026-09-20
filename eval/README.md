# Held-out evaluation set

`heldout.json` is a JSON array of 40 tickets, disjoint from
`corpus/tickets/`. Each ticket has the same agent-visible fields as the
training tickets (`id`, `merchant_id`, `timestamp`, `channel`, `subject`,
`body`) plus eval-only labels that a harness reads to grade a run, never
fields the agent itself should see:

- `true_root_cause`: string, or a list when a ticket bundles more than one
  unrelated cause.
- `correct_routing`: one of `auto_resolve`, `draft_for_human`, `escalate`.
- `refusal_type`: present only on the two classes of ticket the system is
  expected to refuse rather than diagnose (see below). Absent on every
  normal, diagnosable ticket.
- `evidence_gap_fact`: present only on `EVIDENCE_GAP` tickets — a one-line
  statement of the specific fact that does not exist in policy or state,
  and where it's adjacent-but-missing (see below).
- `true_merchant_id` / `identity_note`: present only where `merchant_id` is
  wrong or unresolvable, same convention as the training corpus.

## The two refusal types

These are different failure modes and must not be scored the same way.

**`UNIDENTIFIABLE`** — the merchant cannot be determined from the ticket at
all: no id, or an id plus content too generic/contradictory to resolve
against the 15 known merchants. The evidence to answer might exist
somewhere, but there's no way to find out which merchant it belongs to.
**Correct routing: `draft_for_human`** — a human sends a short reply asking
for identification (merchant ID, registered email, a recent order/txn id).
Never `escalate`: there is nothing here for a specialist to investigate
until identity is established.

**`EVIDENCE_GAP`** — the merchant *is* identified (a valid `merchant_id`
that resolves to a real fixture), but the fact needed to answer the ticket
genuinely does not exist in `corpus/policy/` or in that merchant's fixture.
This is not a missing lookup or a hard computation — it's a topic the
corpus never covers, or a specific edge case adjacent to covered material
that the docs stop short of addressing. **Correct routing: `escalate`** —
escalate with the specific gap named, not a guess. 8 of the 40 tickets are
this type; see the per-ticket `evidence_gap_fact`, each independently
checked against the frozen corpus (see the fixture/policy audit below).

`ticket_016`-style "no merchant_id at all" cases from the training corpus
are `UNIDENTIFIABLE`, not `EVIDENCE_GAP` — the two are easy to conflate
since both refuse to answer, but the fix for one is "ask who you are" and
the fix for the other is "ask a human because we don't have this."

## Routing rubric

The three lanes, from least to most conservative:

- **`auto_resolve`** — a single well-grounded cause, purely explanatory or
  confirms something is operating as designed (a policy citation plus a
  fact lookup settles it), no money movement or exception handling
  involved, and getting the tone slightly wrong carries low risk. Most
  generic policy-only questions and "this is normal, here's why" diagnoses
  land here (e.g. mid-cycle settlement timing, a dispute still inside its
  normal review window, retry-schedule questions).
- **`draft_for_human`** — the diagnosis is clear and well-grounded, but the
  reply needs a human pass before it goes out: financial hardship, an
  emotionally charged merchant, an explanation that involves a fee or
  deduction the merchant will dispute, or a merchant reasonably expecting a
  human to have looked at their specific case (reserve impact, a paused
  mandate, a KYC rejection needing a precise fix, an `UNIDENTIFIABLE`
  ticket). Correct diagnosis, human tone.
- **`escalate`** — the system cannot resolve this itself: an `EVIDENCE_GAP`,
  an active fraud-hold review needing compliance judgment, or a merchant
  explicitly requesting something only a specialist can grant (an
  early reserve review, an appeal that goes beyond citing policy).

This rubric is a default, not a rigid per-cause lookup table — the same
root cause can land in a more conservative lane if the specific ticket
asks for something beyond a citation-and-lookup answer (e.g. a reserve
explanation is normally `draft_for_human`, but a request to expedite the
review is `escalate`).

## Multi-cause routing rule

When a ticket's causes map to different lanes, **the whole ticket takes
the most conservative lane of any individual cause** — conservatism
ordered `escalate` > `draft_for_human` > `auto_resolve`. A ticket that
bundles one `auto_resolve`-grade question with one `EVIDENCE_GAP` is
scored `escalate` overall, in full, not split into two replies. This
mirrors why the two multi-issue tickets already in the training corpus
(`ticket_007`, `ticket_019`) carry a list under `true_root_cause` — the
routing decision is made once, over the whole ticket, after every cause
has been diagnosed.
