# Anchor

A merchant support root-cause agent for a payments platform.

**Live console:** https://anchor-support-agent.onrender.com

---

## The problem

When a merchant writes to a payments company — *"my settlement hasn't arrived"*, *"why is my account under review"*, *"my refund isn't showing"* — the answer is usually not in any help document. It's in their account.

"Why hasn't my money arrived?" has no documentation answer. The answer is: *your cycle is T+3, your batch was captured Tuesday, it's due Friday, today is Wednesday, nothing is wrong.* That requires reading this merchant's state and comparing it to now.

This is why support here is templated and why merchants find it frustrating. A retrieval chatbot over the docs cannot solve it, because the documents don't contain the answer.

## The idea

**Read the account first, then find the rule that explains it.**

Not retrieve-then-look-up. State first, then a retrieval query built from what the state revealed.

### The case that proves it

A merchant wrote:

> refund show nahi ho raha status pending customer ne dobara email kiya hai

Every word points at refund policy. One-shot retrieval on that text returns refund timelines — correctly, because that's what the words mean. It is still the wrong answer.

The real cause was a webhook that failed to deliver with HTTP 522. The refund processed fine; the merchant was never notified. They had no way to know that, so they couldn't mention it.

The agent read the webhook log first, then searched for:

> refund processed webhook undelivered error 522 merchant notification retries

The word *webhook* appears in the query and nowhere in the ticket. That ticket went from a complete retrieval miss to a top-1 hit.

**Measured across 25 hand-labelled tickets:**

| | recall@1 | recall@3 | recall@5 |
|---|---|---|---|
| One-shot on raw ticket text | 72% | 84% | 92% |
| State-informed query | 76% | 96% | 100% |
| — terse / Hinglish subset, one-shot | 55% | 73% | 82% |
| — terse / Hinglish subset, state-informed | 67% | 93% | 100% |

The gain is largest on the hardest tickets, which is what you'd expect if the mechanism is real.

## Architecture

```
TICKET
  │
  ├─ Identity resolution ─ deterministic, outside the LLM loop
  │    reference lookup → business name → submitted id
  │    CONFIRMED / UNCORROBORATED / MISMATCH / UNIDENTIFIABLE
  │
  ├─ Evidence gathering ─ bounded tool loop
  │    merchant state · settlement · transactions · disputes
  │    derived facts (dates computed in Python, never by the model)
  │    then policy retrieval, queried using what the state revealed
  │
  ├─ Diagnosis ─ atomic claims, each citing one evidence id
  │
  ├─ Two independent checks
  │    verifier        sees evidence, never the ticket  → is it true?
  │    responsiveness  sees the ticket, never evidence  → does it answer?
  │
  └─ Gate ─ deterministic rules, no LLM
       AUTO_RESOLVE · DRAFT_FOR_HUMAN · ESCALATE · REQUEST_IDENTIFICATION
```

### What stops it doing something dangerous

**Identity is a claim, not a fact.** The merchant id on a ticket is what someone typed. It's cross-checked against references and business names in the body. A mismatch escalates immediately, with zero model calls.

**The model cannot reach another merchant's data.** State tools are bound to the resolved merchant by closure and carry no merchant id parameter. There is no argument for the model to fabricate. This is structural, not a prompt instruction.

**The gate blocks disclosure to unverified claimants.** If identity is uncorroborated and any surviving claim cites account state, it cannot auto-send — a policy explanation is safe to send to anyone, account facts are not.

**Dates are computed in Python.** Business-day arithmetic is deterministic code exposed as citable evidence. The model cites it; it never counts days.

**Everything fails closed.** Exhausted tool budget, verifier error, malformed output, unknown decline — all route to a human, never to a guess.

## Results

Held-out set of 40 tickets, disjoint from the 120 used during development. Gate thresholds were frozen and committed **before** the held-out set was run.

| | |
|---|---|
| Root-cause accuracy | 24/40 (60%) |
| Routing accuracy | 24/40 (60%) |
| False auto-resolve rate | 6/40 (15%) frozen · 5/40 (12%) post-hoc |
| EVIDENCE_GAP refusal | 1/8 |
| UNIDENTIFIABLE refusal | 4/4 |

### The verifier finding

The grounding verifier moved the false auto-resolve rate by **exactly zero**. 15% with it, 15% without.

The reason is structural, and it's the most interesting thing I learned building this.

All six false auto-resolves are claims that are faithfully grounded in the evidence they cite and answer the wrong question. One merchant asked how to change their settlement bank account; the system replied that settlement is T+3. True, correctly cited, cleanly verified, and not an answer.

The verifier never sees the ticket — deliberately, so it cannot infer the intended answer and rubber-stamp. The cost of that isolation is now measured: it can check whether a claim is **true** but never whether it is **responsive**.

So I built a second check with the mirror-image blindness: it sees the question and never the evidence. Isolation enforced by the function signature — two strings in, one verdict out.

| | wrong root causes caught |
|---|---|
| Verifier alone | 2/13 |
| Responsiveness alone | 8/13 |
| Both | 9/13 |
| Overlap | 1 ticket |

Near-zero overlap. The two checks are measuring genuinely different things, which is the empirical support for building them as separate passes rather than one combined check.

Correct diagnoses wrongly rejected by the verifier: 0/24.

## Honesty notes

- **The corpus is synthetic and I wrote it.** 10 policy documents, 15 merchant fixtures, 120 tickets, 40 held-out. No real merchant data was used. Policy content is my own, not copied from any company's documentation.
- **Adjudication was manual, by me** — the person who built the system. Every call is recorded with reasoning in `eval/heldout_adjudication.json`, and the two genuinely borderline ones are flagged rather than resolved in my favour. An LLM judge was rejected as circular; manual adjudication has its own bias I can't eliminate.
- **One gate rule is post-hoc.** `non_responsive_root_cause` was added after the held-out run. It's labelled as such in code and its effect is reported separately from the frozen numbers, never merged into them.
- **The adversarial verifier check was partly self-graded.** The first 12 cases were written by the same author as the verifier's prompt. A later set of 8 was written blind by a separate session, with a control set of corrected claims — that control is what revealed the verifier was rejecting 8/8 *correct* claims too, which the flawed-claim result alone would have hidden.

## What I'd do next

1. **Coverage detection.** The system needs a notion of "this isn't in the corpus" before retrieving. This is the biggest gap.
2. **The gate rule separating fail-closed from thin answers** — small, and worth about +3 on the escalate lane.
3. **Prose rendering.** Atomic claims are verifiable but read as a list of facts. Turning them into a message a merchant would want to receive, without letting an unverified sentence slip in, is real work.
4. **Real data.** Every number here describes a corpus I built. That's a hard ceiling on what it demonstrates.

## Running it

```bash
pip install -r requirements.txt
pytest -q                          # 412 tests, fully offline
uvicorn demo.app:app --port 8000   # the console
```

Reproduce the results:

```bash
python eval/run_heldout.py && python eval/score_heldout.py
python eval/run_heldout_posthoc.py && python eval/score_posthoc.py
```

The console serves entirely from a pre-computed snapshot and works with no API key and no model reachable. Live mode, where you can run your own ticket, needs `GEMINI_API_KEY` and takes around 180 seconds — roughly a dozen sequential model calls.

## Stack

Python, FastAPI, SQLite-free (JSON fixtures), Gemini via `google-genai`, brute-force cosine retrieval over ~66 policy chunks, vanilla JS console. Every model call is disk-cached by content hash, which is what made a 40-ticket evaluation re-runnable on a free tier.
