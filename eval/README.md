# Evaluation

## Frozen gate thresholds

**Frozen 2026-09-23 10:58:05 IST, before any held-out ticket had been run.**

| Threshold | money_movement | informational |
|---|---|---|
| `AUTO_RESOLVE_MIN_SUPPORTED_RATIO` | 1.0 | 1.0 |
| `AUTO_RESOLVE_MIN_CLAIMS` | 2 | 1 |

Set in `src/config.py` from a 21-ticket stratified sample of **training**
tickets only (`eval/run_training_sample.py` →
`eval/training_sample_results.json`), model `gemini-3.5-flash-lite`, zero
fail-closed artifacts. `eval/heldout.json` was not read or run at any point
before this freeze.

**How they were set, including what the data did not support.** The
training sample did not discriminate on the supported-ratio dimension: 20
of 21 tickets scored exactly 1.0, leaving one sub-threshold case
(`ticket_002`, 11/12 = 0.92). With a single relevant data point there is
nothing to fit, so 1.0 is chosen on the cost asymmetry rather than a
measured optimum — `PARTIALLY_SUPPORTED` means, by the verifier's own
rubric, that a claim overstates or goes beyond its evidence, and blocking
costs a human review while a wrong auto-resolve costs a merchant. Stating
this plainly because "tuned from data" would overclaim what 21 tickets
showed.

The claim-count bar is where "money_movement is stricter" is actually
implemented — with both ratios pinned at 1.0 there is nowhere else for the
asymmetry to live. Both money_movement tickets that auto-resolved in the
sample carried ≥2 claims (`ticket_014`: 4, `ticket_051`: 2), so requiring 2
costs nothing observed while ruling out a settlement or refund answer
resting on one fact. The only single-claim auto-resolve
(`ticket_059`) was informational, where one fact genuinely is the whole
answer.

These values are not to be changed after any held-out result is seen.

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

## Responsiveness check — held-out measurement

`src/responsiveness.py`, measured on the same 40 held-out tickets
(`eval/run_responsiveness.py` → `eval/heldout_responsiveness.json`, model
`gemini-3.5-flash-lite`). Replays the composer's own root cause
(`pre_verifier_root_cause`) against the ticket body — the agent was not
re-run, so these are the diagnoses the held-out evaluation already scored.
One ticket (`heldout_013`) failed closed on a connection reset and was
re-run; no other fail-closed artifacts.

**No gate rule or threshold was changed.** `_responsiveness` is recorded
and read by nothing.

| | flagged (NON_ or PARTIALLY_RESPONSIVE) |
|---|---|
| of 13 WRONG root causes | 8 (62%) |
| of 24 CORRECT root causes | 7 (29%) |
| of 6 wrong **auto-resolved** | 3 (50%) |
| of 9 correct **auto-resolved** | 0 (0%) |

The two checks are almost disjoint on the 13 wrong root causes — verifier
only `heldout_005`, responsiveness only seven tickets, **both** only
`heldout_024`. Union 9/13 against the verifier's 2/13 alone. That is the
design working: one check sees the evidence and not the question, the
other the question and not the evidence, so they fail in different places.

The 29% false-flag rate needs splitting before it is read as noise. Two of
the seven (`heldout_009`, `heldout_012`) are UNIDENTIFIABLE refusals — the
check is being handed a refusal notice and correctly saying it does not
answer the question, which is an artifact of the input, not a
misjudgement. The other five (`015`, `021`, `022`, `023`, `030`) are a real
disagreement rather than an error: each root cause is diagnostically
correct but leaves the merchant's "so what do I do" unanswered, which is
what the check is built to notice. All five were already routed
`DRAFT_FOR_HUMAN`. Zero correct auto-resolves were flagged.

## POST-HOC gate rule: `non_responsive_root_cause`

**The gate rules were frozen 2026-09-23 10:58:05 IST, before any held-out
ticket had been run. This rule was added on 2026-09-24, after the
held-out results above had been produced and read.** It is therefore not
an out-of-sample result and is never merged into the frozen numbers. The
frozen figures in this file stand unchanged; the two are reported side by
side below. `src.gate.POST_HOC_RULES` names it in code so an audit trail
carries the same distinction.

The rule: responsiveness verdict `NON_RESPONSIVE` → `DRAFT_FOR_HUMAN`.
`PARTIALLY_RESPONSIVE` does **not** block — on the held-out set those were
diagnoses that were correct but left the merchant's "so what do I do now"
unanswered, and all of them already routed to draft through other rules.
Absent responsiveness is not treated as non-responsive, the same
ablation-integrity principle as the verifier: otherwise the delta would
measure the gate rather than the check.

`eval/run_heldout_posthoc.py` → `eval/heldout_results_posthoc.json`,
scored by `eval/score_posthoc.py`. Same 40 tickets, same model
(`gemini-3.5-flash-lite`), both checks on, zero fail-closed artifacts.

| metric | FROZEN gate | POST-HOC (+1 rule) |
|---|---|---|
| routing accuracy | 24/40 = 60% | 24/40 = 60% |
| auto-resolved | 15/40 = 38% | 14/40 = 35% |
| **false auto-resolve (all tickets)** | **6/40 = 15%** | **5/40 = 12%** |
| false auto-resolve (of auto-resolved) | 6/15 = 40% | 5/14 = 36% |
| recall: auto_resolve | 9/13 = 69% | 9/13 = 69% |
| recall: draft_for_human | 14/17 = 82% | 14/17 = 82% |
| recall: escalate | 1/10 = 10% | 1/10 = 10% |

**Exactly one ticket changed lane**: `heldout_002` (asked how to change a
settlement bank account, answered with the T+3 cycle), `AUTO_RESOLVE` →
`DRAFT_FOR_HUMAN`. No correct auto-resolve was lost, which is why routing
accuracy is flat while the false auto-resolve rate falls — the frozen
labels route `heldout_002` to `auto_resolve`, so blocking it costs a
routing point and buys a wrong answer not sent. That trade is the reason
the false auto-resolve rate, not routing accuracy, is the headline.

Eight further tickets came back `NON_RESPONSIVE` and were already caught
by a **frozen** rule firing earlier (seven via
`uncorroborated_identity_discloses_account_data`, one via
`root_cause_rejected_by_verifier`). The audit trail attributes them to the
frozen rule, correctly — the new rule did not move them. The honest
reading is that the responsiveness signal is largely redundant with
existing identity-disclosure conservatism on this set, and its independent
contribution is one ticket in forty.
