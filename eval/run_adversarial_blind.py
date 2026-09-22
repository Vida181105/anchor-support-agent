"""Stage A of the blind adversarial check: runs src.verifier.verify_claim
over eval/adversarial_blind.json WITHOUT ever reading the `flaw` field.

Deliberately structured so the blind pass is mechanically incapable of
being influenced by the answer key: `_load_claims_blind` strips `flaw`
out of each record before anything else in this file ever touches it, so
there's no path by which this script's behavior could depend on it.

Run: python eval/run_adversarial_blind.py
Writes eval/adversarial_blind_results.json (blind verdicts only - no
flaw field, no analysis, no corrected claims - that's Stage B).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.chunker import load_policy_chunks  # noqa: E402
from src.derived_facts import get_derived_facts  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.state_tools import get_merchant_state  # noqa: E402
from src.verifier import verify_claim  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLIND_PATH = ROOT / "eval" / "adversarial_blind.json"
POLICY_CHUNKS = {c["evidence_id"]: c["content"] for c in load_policy_chunks()}


def _load_claims_blind() -> list[dict]:
    """Loads only {id, merchant_id, claim} - `flaw` is discarded immediately
    and never enters this process's working data.
    """
    raw = json.loads(BLIND_PATH.read_text(encoding="utf-8"))
    return [{"id": r["id"], "merchant_id": r["merchant_id"], "claim": r["claim"]} for r in raw]


def _navigate(obj, path: str):
    """Walk a dotted path with optional [i] indices, e.g.
    'settlement_schedule.oldest_unsettled_batch_captured_date' or
    'mandates[0].attempted_charges[0].failure_code', against a parsed
    JSON object.
    """
    import re

    for part in path.split("."):
        m = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)((?:\[\d+\])*)$", part)
        if not m:
            raise ValueError(f"unparseable path segment: {part!r}")
        key, indices = m.group(1), m.group(2)
        obj = obj[key]
        for idx in re.findall(r"\[(\d+)\]", indices):
            obj = obj[int(idx)]
    return obj


def resolve_evidence_content(evidence_id: str):
    """Fetch the REAL content for one evidence_id, live from the corpus -
    handles whole-tool-shaped ids (state:merchant_N, doc:slug#cN),
    finer-grained dotted-path ids that no single existing state tool
    returns directly (state_tools.py only exposes whole sub-objects; the
    blind set cites individual fields within them), and derived:... facts
    computed by src/derived_facts.py.
    """
    if evidence_id.startswith("doc:"):
        return POLICY_CHUNKS[evidence_id]

    if evidence_id.startswith("derived:"):
        merchant_id = evidence_id[len("derived:") :].split(".", 1)[0]
        facts = {f["evidence_id"]: f["content"] for f in get_derived_facts(merchant_id)}
        return facts[evidence_id]

    assert evidence_id.startswith("state:")
    rest = evidence_id[len("state:") :]
    if "." not in rest:
        merchant_id = rest
        return get_merchant_state(merchant_id)["content"]

    merchant_id, path = rest.split(".", 1)
    full = get_merchant_state(merchant_id)["content"]
    return _navigate(full, path)


def main() -> None:
    llm = LLMClient()
    claims = _load_claims_blind()

    results = []
    for item in claims:
        claim = item["claim"]
        evidence_content = {eid: resolve_evidence_content(eid) for eid in claim["evidence"]}
        verdict = verify_claim(claim, evidence_content, llm)
        results.append(
            {
                "id": item["id"],
                "merchant_id": item["merchant_id"],
                "claim_text": claim["text"],
                "evidence": claim["evidence"],
                "evidence_content": evidence_content,
                "verdict": verdict,
            }
        )
        print(f"{item['id']}: {verdict['verdict']} - {verdict['reason']}")

    out_path = ROOT / "eval" / "adversarial_blind_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
