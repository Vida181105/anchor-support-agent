"""Re-run of the adversarial + control sets under the two fixes:
derived-fact evidence (src/derived_facts.py) and atomic claims.

NOT A BLIND MEASUREMENT. These 8 cases were blind exactly once - the
original run in eval/adversarial_blind_results.json. Both fixes were
designed after seeing where that run failed, and the rewrites below were
written with the `flaw` fields open. The after-numbers show the fixes work
on the cases that exposed the problem. They are not evidence that the
fixes generalise; only a fresh, unseen set could show that.

What changes between before and after:
  - claims are atomic: one checkable fact each, no merchant names, no
    framing, citing only the evidence for that one fact
  - time-relative claims cite derived:... facts instead of asking the
    verifier to do calendar arithmetic it structurally cannot do

Each case keeps the same underlying fact (and, for the flawed set, the
same flaw). A real diagnosis would split each original bundled claim into
several atomic ones; here each case carries the single atomic claim (two,
for adv_8) that holds the fact at issue, so before/after stays comparable.

Run: python eval/run_adversarial_atomic.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.llm import LLMClient  # noqa: E402
from src.verifier import verify_claim  # noqa: E402
from run_adversarial_blind import resolve_evidence_content  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLIND_PATH = ROOT / "eval" / "adversarial_blind.json"
BEFORE_FLAWED = ROOT / "eval" / "adversarial_blind_results.json"
BEFORE_CORRECTED = ROOT / "eval" / "adversarial_control_results.json"

# Flawed claims, re-expressed atomically. Each preserves its original flaw
# exactly - only the bundling and the evidence citation change.
ATOMIC_FLAWED = {
    "adv_1": [
        {
            "text": "The expected settlement date for the oldest unsettled batch is 2026-09-17.",
            "evidence": ["derived:merchant_1.expected_settlement_date"],
        }
    ],
    "adv_2": [
        {"text": "The rolling reserve percentage is 10.", "evidence": ["state:merchant_2.reserve.percentage"]}
    ],
    "adv_3": [
        {
            "text": "The threshold for early reserve removal is a chargeback rate under 1% for "
            "60 consecutive days.",
            "evidence": ["doc:rolling-reserves#c5"],
        }
    ],
    "adv_4": [
        {
            "text": "A partial hold displays on the dashboard as a payout of zero.",
            "evidence": ["doc:account-review-holds#c5"],
        }
    ],
    "adv_5": [
        {
            "text": "The rejection reason code on the second address-proof submission was DOC_MISMATCH.",
            "evidence": ["state:merchant_12.kyc.documents[2].rejection_reason"],
        }
    ],
    "adv_6": [
        {
            "text": "Bank-side settlement delays typically resolve within 3-5 additional business days.",
            "evidence": ["doc:settlement-cycles#c6"],
        }
    ],
    "adv_7": [
        {
            "text": "A chargeback fee is credited back to the merchant when the merchant wins the dispute.",
            "evidence": ["doc:chargebacks-disputes#c5"],
        }
    ],
    "adv_8": [
        {
            "text": "The mandate charge was declined because the customer had revoked the mandate.",
            "evidence": [
                "state:merchant_15.mandates[0].attempted_charges[0].failure_code",
                "doc:subscription-mandates#c5",
            ],
        }
    ],
}

# Corrected claims, atomic. Same facts as the Stage B corrections, with the
# business name, framing and bundled second facts removed, and the
# time-relative one citing a derived fact.
ATOMIC_CORRECTED = {
    "adv_1": [
        {
            "text": "The expected settlement date for the oldest unsettled batch is 2026-09-18.",
            "evidence": ["derived:merchant_1.expected_settlement_date"],
        }
    ],
    "adv_2": [
        {"text": "The rolling reserve percentage is 15.", "evidence": ["state:merchant_2.reserve.percentage"]}
    ],
    "adv_3": [
        {
            "text": "The threshold for early reserve removal is a chargeback rate under 0.5% for "
            "60 consecutive days.",
            "evidence": ["doc:rolling-reserves#c5"],
        }
    ],
    "adv_4": [
        {
            "text": "A partial hold displays on the dashboard as a reduced next payout amount "
            "rather than a payout of zero.",
            "evidence": ["doc:account-review-holds#c5"],
        }
    ],
    # Second attempt. My first rewrite read "The rejection reason code on
    # the second address-proof submission was DOC_ILLEGIBLE" and came back
    # PARTIALLY_SUPPORTED, because it still carried two facts (the code,
    # and that this was the second submission) while citing evidence for
    # only the first - a bare "DOC_ILLEGIBLE" string can't establish
    # ordinal position. The verifier was right and my rewrite was wrong;
    # see FIRST_ATTEMPT_NOTES.
    "adv_5": [
        {
            "text": "The rejection reason code was DOC_ILLEGIBLE.",
            "evidence": ["state:merchant_12.kyc.documents[2].rejection_reason"],
        }
    ],
    "adv_6": [
        {
            "text": "Bank-side settlement delays typically resolve within 1-2 additional business days.",
            "evidence": ["doc:settlement-cycles#c6"],
        }
    ],
    "adv_7": [
        {
            "text": "A chargeback fee is non-refundable even when the merchant wins the dispute.",
            "evidence": ["doc:chargebacks-disputes#c5"],
        }
    ],
    # adv_8 is the clearest case for atomicity: the original fused a true
    # state fact, a true policy mechanism, and a false causal link between
    # them. Atomically, the two true facts stand and the false link is
    # simply not asserted.
    "adv_8": [
        {
            "text": "The attempted mandate charge failed with failure code DO_NOT_HONOR.",
            "evidence": ["state:merchant_15.mandates[0].attempted_charges[0].failure_code"],
        },
        {
            "text": "A revoked mandate can produce a generic decline for up to 24 hours before "
            "the platform receives confirmation from the bank.",
            "evidence": ["doc:subscription-mandates#c5"],
        },
    ],
}


# Recorded rather than quietly overwritten: two results in the first pass
# of this re-run were not what they appeared to be, and both are worth
# keeping visible.
FIRST_ATTEMPT_NOTES = {
    "adv_5": (
        "First rewrite was 'The rejection reason code on the second address-proof submission "
        "was DOC_ILLEGIBLE' -> PARTIALLY_SUPPORTED, reason: 'does not explicitly verify that "
        "this was the second submission'. That was a correct catch: the rewrite still bundled "
        "two facts while citing evidence for one. Fixed by making the claim genuinely atomic, "
        "which is applying the stated rule, not loosening it."
    ),
    "adv_8": (
        "Second claim initially returned UNSUPPORTED with reason 'verifier error, failing "
        "closed: Exceeded 5 retries on 429s' - a free-tier rate-limit artifact, not a verdict. "
        "Re-run once quota recovered: SUPPORTED."
    ),
}


def run_set(llm, claims_by_id: dict) -> dict:
    out = {}
    for case_id, claims in claims_by_id.items():
        verdicts = []
        for claim in claims:
            evidence_content = {eid: resolve_evidence_content(eid) for eid in claim["evidence"]}
            verdicts.append({"claim": claim, "verdict": verify_claim(claim, evidence_content, llm)})
        out[case_id] = verdicts
    return out


def main() -> None:
    llm = LLMClient()
    flaws = {r["id"]: r["flaw"] for r in json.loads(BLIND_PATH.read_text(encoding="utf-8"))}
    before_flawed = {r["id"]: r["verdict"] for r in json.loads(BEFORE_FLAWED.read_text(encoding="utf-8"))}
    before_corrected = {
        r["id"]: r["verdict"] for r in json.loads(BEFORE_CORRECTED.read_text(encoding="utf-8"))
    }

    after_flawed = run_set(llm, ATOMIC_FLAWED)
    after_corrected = run_set(llm, ATOMIC_CORRECTED)

    def caught(verdicts):  # flawed case: caught if any atomic claim is not SUPPORTED
        return any(v["verdict"]["verdict"] != "SUPPORTED" for v in verdicts)

    def all_supported(verdicts):  # corrected case: passes only if every claim is SUPPORTED
        return all(v["verdict"]["verdict"] == "SUPPORTED" for v in verdicts)

    print("=" * 78)
    print("FLAWED SET - before (bundled, narrow evidence) vs after (atomic + derived)")
    print("=" * 78)
    for case_id in ATOMIC_FLAWED:
        b = before_flawed[case_id]
        a = after_flawed[case_id][0]["verdict"]
        print(f"\n{case_id}: {b['verdict']}  ->  {a['verdict']}")
        print(f"  before reason: {b['reason']}")
        print(f"  after  reason: {a['reason']}")

    n_before_caught = sum(1 for cid in ATOMIC_FLAWED if before_flawed[cid]["verdict"] != "SUPPORTED")
    n_after_caught = sum(1 for cid, v in after_flawed.items() if caught(v))
    print(f"\nFlawed caught: {n_before_caught}/8 before -> {n_after_caught}/8 after")

    print("\n" + "=" * 78)
    print("CORRECTED SET - before (bundled, narrow evidence) vs after (atomic + derived)")
    print("=" * 78)
    for case_id in ATOMIC_CORRECTED:
        b = before_corrected[case_id]
        verdicts = after_corrected[case_id]
        after_str = ", ".join(v["verdict"]["verdict"] for v in verdicts)
        print(f"\n{case_id}: {b['verdict']}  ->  {after_str}")
        print(f"  before reason: {b['reason']}")
        for v in verdicts:
            print(f"  after  reason: {v['verdict']['reason']}")

    n_before_supported = sum(
        1 for cid in ATOMIC_CORRECTED if before_corrected[cid]["verdict"] == "SUPPORTED"
    )
    n_after_supported = sum(1 for cid, v in after_corrected.items() if all_supported(v))
    print(f"\nCorrected fully SUPPORTED: {n_before_supported}/8 before -> {n_after_supported}/8 after")

    out = {
        "note": "NOT BLIND - fixes were designed after seeing these cases fail.",
        "first_attempt_notes": FIRST_ATTEMPT_NOTES,
        "flawed": {
            cid: {
                "flaw": flaws[cid],
                "before": before_flawed[cid],
                "after": [v["verdict"] for v in after_flawed[cid]],
                "atomic_claims": ATOMIC_FLAWED[cid],
            }
            for cid in ATOMIC_FLAWED
        },
        "corrected": {
            cid: {
                "before": before_corrected[cid],
                "after": [v["verdict"] for v in after_corrected[cid]],
                "atomic_claims": ATOMIC_CORRECTED[cid],
            }
            for cid in ATOMIC_CORRECTED
        },
    }
    out_path = ROOT / "eval" / "adversarial_atomic_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
