"""Stage B: reads eval/adversarial_blind.json's `flaw` fields (only after
Stage A's blind pass is complete - see run_adversarial_blind.py), builds a
corrected version of each of the 8 claims citing the SAME evidence, and
runs the verifier over those too.

Run: python eval/run_adversarial_control.py
Requires eval/adversarial_blind_results.json to already exist (Stage A).
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
BLIND_RESULTS_PATH = ROOT / "eval" / "adversarial_blind_results.json"

# Corrected text for each id, citing the identical evidence list as the
# original flawed claim - only the claim's TEXT changes, built from the
# `flaw` field's explanation of what the evidence actually shows.
CORRECTED_TEXT = {
    "adv_1": "Kavya Handloom Exports' oldest unsettled batch was captured on 2026-09-15, "
    "and under the standard T+3 business-day cycle, that batch is scheduled to settle on "
    "2026-09-18 — the payout is proceeding on the normal schedule, not running behind.",
    "adv_2": "Trailblazer Holiday Co. currently has 15% of each day's settlement withheld "
    "under its active rolling reserve, per the standard reserve terms.",
    "adv_3": "Per policy, Trailblazer Holiday Co.'s reserve is eligible for early removal "
    "once its trailing chargeback rate has stayed below 0.5% for 60 consecutive days.",
    "adv_4": "Because Suraksha Home Appliances' hold is scoped to only six flagged "
    "transactions, policy says the dashboard should show the September 18 payout as a "
    "reduced amount rather than ₹0 until the review clears.",
    "adv_5": "Anjali Tiffin Services' second address-proof submission was rejected again "
    "with a DOC_ILLEGIBLE code, meaning the resubmitted scan was still unreadable.",
    "adv_6": "Coastal Spice Traders' September 14 settlement is showing as "
    "settled-but-not-cleared on the bank side; per policy this typically resolves within "
    "1-2 additional business days.",
    "adv_7": "Since Zenith Fitness Equipment won dispute disp_14_01, the ₹500 chargeback "
    "fee already deducted on the September 16 payout is non-refundable and will not be "
    "credited back, even though the disputed transaction amount itself was returned.",
    "adv_8": "The mandate charge attempt on StudyStream Learning's account failed with a "
    "DO_NOT_HONOR decline. Separately, per policy, a DO_NOT_HONOR-style generic decline can "
    "occur for up to 24 hours after a mandate revocation, before the platform receives "
    "confirmation from the bank.",
}


def main() -> None:
    llm = LLMClient()
    raw = json.loads(BLIND_PATH.read_text(encoding="utf-8"))  # now safe to read `flaw`
    blind_results = {r["id"]: r for r in json.loads(BLIND_RESULTS_PATH.read_text(encoding="utf-8"))}

    control_results = []
    for item in raw:
        evidence_ids = item["claim"]["evidence"]
        corrected_claim = {"text": CORRECTED_TEXT[item["id"]], "evidence": evidence_ids}
        evidence_content = {eid: resolve_evidence_content(eid) for eid in evidence_ids}
        verdict = verify_claim(corrected_claim, evidence_content, llm)
        control_results.append(
            {
                "id": item["id"],
                "corrected_text": corrected_claim["text"],
                "verdict": verdict,
            }
        )
        print(f"{item['id']} (corrected): {verdict['verdict']} - {verdict['reason']}")

    out_path = ROOT / "eval" / "adversarial_control_results.json"
    out_path.write_text(json.dumps(control_results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")

    # ---- combined report ----
    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)

    flawed_caught = 0
    print("\n--- Flawed claims (blind pass) ---")
    for item in raw:
        r = blind_results[item["id"]]
        caught = r["verdict"]["verdict"] != "SUPPORTED"
        flawed_caught += caught
        print(f"{item['id']}: {r['verdict']['verdict']} ({'caught' if caught else 'MISSED'})")
        print(f"  reason: {r['verdict']['reason']}")
        print(f"  actual flaw: {item['flaw']}")

    print(f"\nFlawed claims caught (verdict != SUPPORTED): {flawed_caught} / {len(raw)}")

    control_supported = sum(1 for r in control_results if r["verdict"]["verdict"] == "SUPPORTED")
    print(f"Corrected claims returning SUPPORTED: {control_supported} / {len(control_results)}")
    false_rejections = [r for r in control_results if r["verdict"]["verdict"] != "SUPPORTED"]
    if false_rejections:
        print("\n--- FALSE REJECTIONS (corrected/accurate claim not marked SUPPORTED) ---")
        for r in false_rejections:
            print(f"{r['id']}: {r['verdict']['verdict']} - {r['verdict']['reason']}")


if __name__ == "__main__":
    main()
