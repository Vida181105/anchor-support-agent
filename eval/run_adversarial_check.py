"""Unit 3's adversarial check: ~10 hand-constructed diagnosis claims where
a claim cites a real, correctly-fetched evidence_id but subtly
misrepresents what it says. These are exactly the cases
src.schema.validate_diagnosis passes (the evidence_id is real and was
genuinely fetchable) - the question is whether src.verifier.verify_claim,
which actually reads the content, catches them.

Evidence content is fetched live through the real state tools / policy
chunker, not hand-transcribed, so a case can't accidentally test against
evidence that doesn't match what a real tool call would return.

"Caught" = verdict != SUPPORTED (either UNSUPPORTED or PARTIALLY_SUPPORTED
counts - both mean the verifier noticed something wrong). A SUPPORTED
verdict on one of these is a miss: the claim is deliberately wrong, and
the verifier said it was fine.

Run: python eval/run_adversarial_check.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.chunker import load_policy_chunks  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.state_tools import get_disputes, get_merchant_state, get_settlement_schedule  # noqa: E402
from src.verifier import verify_claim  # noqa: E402

POLICY = {c["evidence_id"]: c["content"] for c in load_policy_chunks()}


def build_cases() -> list[dict]:
    m2 = get_merchant_state("merchant_2")["content"]
    m6 = get_merchant_state("merchant_6")["content"]
    m5 = get_merchant_state("merchant_5")["content"]
    m7 = get_merchant_state("merchant_7")["content"]

    m9_dispute = get_disputes("merchant_9")[0]
    m14_dispute = get_disputes("merchant_14")[0]
    m1_schedule = get_settlement_schedule("merchant_1")

    return [
        {
            "id": 1,
            "category": "date arithmetic (self-contained in the evidence)",
            "claim": {
                "text": "The merchant had a 10-day window to respond to this dispute and "
                "submitted their evidence 5 days before the deadline.",
                "evidence": ["state:merchant_9.disputes[0]"],
            },
            "evidence_content": {"state:merchant_9.disputes[0]": m9_dispute["content"]},
            "why_wrong": "opened_date 2026-08-10 to response_deadline 2026-08-17 is 7 days, not "
            "10. evidence_submitted_date 2026-08-15 is 2 days before the deadline, not 5. Both "
            "numbers are computable purely from this one evidence item's own dates.",
        },
        {
            "id": 2,
            "category": "threshold value misstated",
            "claim": {
                "text": "The reserve was triggered because merchant_2's trailing 30-day "
                "chargeback rate reached 2.1%, well above the 1% policy threshold.",
                "evidence": ["state:merchant_2"],
            },
            "evidence_content": {"state:merchant_2": m2},
            "why_wrong": "trigger_metric.value_pct is 1.4, not 2.1. The qualitative conclusion "
            "(above threshold) happens to still be true, but the specific number is fabricated.",
        },
        {
            "id": 3,
            "category": "policy rule generalized beyond its stated scope",
            "claim": {
                "text": "Per policy, any document rejected with the DOC_MISMATCH reason code "
                "can be resolved by submitting a GST registration certificate instead.",
                "evidence": ["doc:kyc-documents#c6"],
            },
            "evidence_content": {"doc:kyc-documents#c6": POLICY["doc:kyc-documents#c6"]},
            "why_wrong": "The policy scopes this fix specifically to the case where a bank "
            "account is held in a director's name rather than the company's - not to every "
            "DOC_MISMATCH rejection, which can have other causes (e.g. submitting a personal "
            "document when a business one was required) that a GST certificate wouldn't fix.",
        },
        {
            "id": 4,
            "category": "numeric range distortion",
            "claim": {
                "text": "Card refunds settle on the customer's statement within 2-3 business "
                "days per policy.",
                "evidence": ["doc:refund-timelines#c2"],
            },
            "evidence_content": {"doc:refund-timelines#c2": POLICY["doc:refund-timelines#c2"]},
            "why_wrong": "Policy states 5-7 business days for card refunds, not 2-3.",
        },
        {
            "id": 5,
            "category": "date off-by-one, contradicted by the same evidence item",
            "claim": {
                "text": "The batch captured on 2026-09-15 is scheduled to settle on "
                "2026-09-17, three business days later.",
                "evidence": ["state:merchant_1.settlement_schedule"],
            },
            "evidence_content": {"state:merchant_1.settlement_schedule": m1_schedule["content"]},
            "why_wrong": "next_payout.date in the SAME evidence item is 2026-09-18, not "
            "2026-09-17. This is the most self-contained case: the contradiction is entirely "
            "within one field of the one cited evidence item.",
        },
        {
            "id": 6,
            "category": "direct contradiction of a stated rule, softened in phrasing",
            "claim": {
                "text": "Since merchant_14 won this dispute, they are entitled to a partial "
                "refund of the chargeback fee, prorated for the successful outcome.",
                "evidence": ["state:merchant_14.disputes[0]", "doc:chargebacks-disputes#c7"],
            },
            "evidence_content": {
                "state:merchant_14.disputes[0]": m14_dispute["content"],
                "doc:chargebacks-disputes#c7": POLICY["doc:chargebacks-disputes#c7"],
            },
            "why_wrong": "Policy states the fee does not vary by outcome and is flatly "
            "non-refundable. 'Partial, prorated refund' is a softer-sounding invention with no "
            "support in either the policy or the dispute record.",
        },
        {
            "id": 7,
            "category": "count miscounted",
            "claim": {
                "text": "The mandate was paused after only 2 failed retry attempts, one short "
                "of the standard 3-retry policy.",
                "evidence": ["state:merchant_6"],
            },
            "evidence_content": {"state:merchant_6": m6},
            "why_wrong": "retry_attempts has exactly 3 entries (2026-08-28, 2026-08-30, "
            "2026-09-01), not 2.",
        },
        {
            "id": 8,
            "category": "causal misattribution contradicted by the same evidence",
            "claim": {
                "text": "Merchant_5's international payments approval is delayed because their "
                "KYC has not yet cleared, and per policy this must complete before the 3-5 day "
                "approval clock even starts.",
                "evidence": ["state:merchant_5", "doc:international-payments#c2"],
            },
            "evidence_content": {
                "state:merchant_5": m5,
                "doc:international-payments#c2": POLICY["doc:international-payments#c2"],
            },
            "why_wrong": "state:merchant_5 shows kyc.status='verified' (already cleared) and "
            "ineligibility_reason='business_category_excluded' - KYC timing is not the cause at "
            "all, contradicting the claim directly.",
        },
        {
            "id": 9,
            "category": "duration overstatement (conclusion right, process wrong)",
            "claim": {
                "text": "The platform retried delivery for the full 24-hour window specified by "
                "policy before marking the event undelivered.",
                "evidence": ["state:merchant_7", "doc:webhooks#c2"],
            },
            "evidence_content": {"state:merchant_7": m7, "doc:webhooks#c2": POLICY["doc:webhooks#c2"]},
            "why_wrong": "The 6 logged attempts span roughly 8.6 hours (14:02 to 22:38 on "
            "2026-09-05), not the full 24-hour policy window, even though the final_status "
            "('undelivered') the claim leads to is correct either way.",
        },
        {
            "id": 10,
            "category": "premature-completion implication from partial data",
            "claim": {
                "text": "Merchant_2's full reserve balance will be released back to them by "
                "early October 2026.",
                "evidence": ["state:merchant_2"],
            },
            "evidence_content": {"state:merchant_2": m2},
            "why_wrong": "release_schedule shows only 18,500 + 21,300 = 40,300 releasing by "
            "2026-10-02, out of a held_balance of 412,000 - the bulk (372,200) doesn't release "
            "until 2026-12-22. 'Full balance by early October' is false.",
        },
        # --- bonus stress cases beyond the required ~10: the first 10 all
        # got caught, which is worth being suspicious of rather than just
        # reporting - these two are deliberately harder (much smaller
        # numeric drift; an inferential leap rather than a factual
        # misstatement) to actually probe for where the verifier's
        # judgment breaks down, not to pad a clean number.
        {
            "id": 11,
            "category": "BONUS (stress test): small numeric drift, plausible as rounding",
            "claim": {
                "text": "The chargeback rate that triggered the reserve was approximately 1.5%.",
                "evidence": ["state:merchant_2"],
            },
            "evidence_content": {"state:merchant_2": m2},
            "why_wrong": "trigger_metric.value_pct is 1.4, not 1.5 - a small enough drift that "
            "it could plausibly be waved through as 'approximately' correct rather than flagged.",
        },
        {
            "id": 12,
            "category": "BONUS (stress test): inferential overreach, not a bare factual error",
            "claim": {
                "text": "Since two of the three retry attempts failed with INSUFFICIENT_FUNDS, "
                "the customer's card was likely permanently closed.",
                "evidence": ["state:merchant_6"],
            },
            "evidence_content": {"state:merchant_6": m6},
            "why_wrong": "INSUFFICIENT_FUNDS means exactly what it says (not enough balance at "
            "the time of the charge) and has no logical connection to a card being closed - the "
            "evidence doesn't state or imply anything about the card's status at all. This is an "
            "unsupported inferential leap, not a misstated number, which is a different kind of "
            "error for the verifier to catch than the first 10 cases.",
        },
    ]


def main() -> None:
    llm = LLMClient()
    cases = build_cases()

    results = []
    print(f"{'#':<3} {'category':<62} verdict            caught?")
    for case in cases:
        result = verify_claim(case["claim"], case["evidence_content"], llm)
        results.append(result)
        is_caught = result["verdict"] != "SUPPORTED"
        mark = "YES" if is_caught else "MISS"
        print(f"{case['id']:<3} {case['category']:<62} {result['verdict']:<18} {mark}")
        if not is_caught:
            print(f"    >>> MISSED: claim was '{case['claim']['text']}'")
            print(f"    >>> why it's actually wrong: {case['why_wrong']}")
            print(f"    >>> verifier's reason: {result['reason']}")

    print()
    core = results[:10]
    bonus = results[10:]
    core_caught = sum(r["verdict"] != "SUPPORTED" for r in core)
    print(f"Core set: caught {core_caught} / {len(core)}")
    if bonus:
        bonus_caught = sum(r["verdict"] != "SUPPORTED" for r in bonus)
        print(f"Bonus stress cases: caught {bonus_caught} / {len(bonus)}")

    out = [
        {**{k: v for k, v in case.items() if k != "evidence_content"}, "verdict_result": result}
        for case, result in zip(cases, results)
    ]
    out_path = Path(__file__).resolve().parent / "adversarial_check_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
