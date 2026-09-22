"""Grounding verifier: a separate pass that checks whether a claim's TEXT
is actually supported by the CONTENT of the evidence it cites. This is
exactly the semantic check src.schema.validate_diagnosis explicitly does
NOT do - that one only confirms an evidence_id was fetched at all, never
whether the evidence it points to says what the claim says it says.

Architecturally separate from the composer (src/agent.py), not a "be
accurate" instruction folded into its prompt: this module has its own
prompt template, its own model call, and runs as a distinct pass over an
already-produced diagnosis. Delete this file and src/agent.py still runs
end to end (just ungrounded-in-content, which is the entire reason this
module exists) - that's the actual test for "architecturally separate,"
not a comment claiming it.

The verifier sees ONLY a claim's text and the full content of the
evidence it cites - never the ticket, never the merchant_id, never any
other claim, never root_cause unless root_cause is the thing being
checked. Anything else would let it infer the "intended" answer from
surrounding context instead of checking the specific text against the
specific evidence, which would defeat the point of a second, independent
pass. tests/test_verifier.py asserts this directly by inspecting exactly
what text reaches the model, not just by trusting this paragraph.
"""

from __future__ import annotations

import json
from typing import Any

from src.llm import LLMClient

VERIFIER_MODEL = "gemini-3.5-flash-lite"

VERDICTS = ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED")

VERDICT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "reason": {"type": "string", "description": "One short sentence."},
    },
    "required": ["verdict", "reason"],
}

_PROMPT_TEMPLATE = """\
You are a strict fact-checker. You are given a CLAIM and the EVIDENCE it \
cites. Decide whether the evidence actually supports the claim's text.

Judge only the literal content shown below - no outside knowledge, no \
assumptions about what is "probably" true, no benefit of the doubt.

- SUPPORTED: every material fact in the claim is directly stated in, or \
unambiguously and directly implied by, the evidence.
- PARTIALLY_SUPPORTED: some of the claim is backed by the evidence, but \
part of it overstates, generalizes beyond, or is simply absent from the \
evidence (for example: a specific date or number is off, a threshold is \
described as met when the evidence doesn't show that, or a rule is \
applied more broadly than the evidence actually states it).
- UNSUPPORTED: the evidence does not support the claim, contradicts it, \
or the claim's key fact is not present in the evidence at all.

CLAIM:
{claim_text}

EVIDENCE:
{evidence_block}

Return your verdict now.\
"""


def _format_evidence_block(evidence_content: dict[str, Any]) -> str:
    parts = []
    for evidence_id, content in evidence_content.items():
        parts.append(f"[{evidence_id}]\n{json.dumps(content, ensure_ascii=False, indent=2, default=str)}")
    return "\n\n".join(parts)


def verify_claim(
    claim: dict, evidence_content: dict[str, Any], llm: LLMClient, model: str = VERIFIER_MODEL
) -> dict:
    """Check ONE claim against ONLY the content of the evidence it cites.

    `evidence_content` may contain more entries than this claim cites
    (the caller typically has one dict for the whole diagnosis) - only
    the ids in `claim["evidence"]` are ever included in what's sent to
    the model.

    Never raises. Any failure - malformed model output, an invalid verdict
    value, a network/rate-limit error - is caught here and returned as an
    UNSUPPORTED verdict with the reason explaining why. Fail closed, never
    fail open: an exception must never silently mean "assume it's fine."
    """
    if not claim["evidence"]:
        return {"verdict": "UNSUPPORTED", "reason": "claim cites no evidence at all"}

    missing = [eid for eid in claim["evidence"] if eid not in evidence_content]
    if missing:
        return {"verdict": "UNSUPPORTED", "reason": f"evidence content unavailable for {missing}"}

    cited_content = {eid: evidence_content[eid] for eid in claim["evidence"]}
    prompt = _PROMPT_TEMPLATE.format(
        claim_text=claim["text"],
        evidence_block=_format_evidence_block(cited_content),
    )

    try:
        result = llm.generate_turn(
            [{"role": "user", "text": prompt}],
            model=model,
            response_json_schema=VERDICT_JSON_SCHEMA,
            temperature=0.0,
        )
        parsed = json.loads(result["text"])
        if parsed.get("verdict") not in VERDICTS:
            raise ValueError(f"model returned an invalid verdict: {parsed.get('verdict')!r}")
        return {"verdict": parsed["verdict"], "reason": parsed.get("reason", "")}
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure fails closed
        return {"verdict": "UNSUPPORTED", "reason": f"verifier error, failing closed: {exc}"}


def verify_diagnosis(
    diagnosis: dict, evidence_content: dict[str, Any], llm: LLMClient, model: str = VERIFIER_MODEL
) -> dict:
    """Run the grounding verifier over a full diagnosis: root_cause first,
    then every entry in claims.

    If root_cause comes back UNSUPPORTED, the entire diagnosis is
    rejected - this returns a fixed rejection diagnosis routed to
    escalate, not the original diagnosis with an emptied-out root_cause.
    Otherwise, returns a copy of `diagnosis` with any UNSUPPORTED claim
    (root_cause is never UNSUPPORTED at this point) removed from `claims`,
    plus a `_verification` key recording every verdict.

    `_verification` follows this project's existing `_`-prefixed
    convention for "not part of the core contract" - but unlike
    corpus/README.md's `_derivation` (which src.evidence.make_evidence
    structurally strips before anything can leak), nothing currently
    strips `_verification` automatically. It's a naming convention here,
    not an enforced boundary - a caller that wants it hidden must drop it
    itself.
    """
    root_cause = diagnosis["root_cause"]
    root_verdict = verify_claim(root_cause, evidence_content, llm, model)

    if root_verdict["verdict"] == "UNSUPPORTED":
        return _rejected_diagnosis(diagnosis, root_verdict)

    verdicts = [{"claim": root_cause["text"], "is_root_cause": True, **root_verdict}]
    kept_claims = []
    for claim in diagnosis["claims"]:
        verdict = verify_claim(claim, evidence_content, llm, model)
        verdicts.append({"claim": claim["text"], "is_root_cause": False, **verdict})
        if verdict["verdict"] != "UNSUPPORTED":
            kept_claims.append(claim)

    result = dict(diagnosis)
    result["claims"] = kept_claims
    result["_verification"] = {"root_cause_verdict": root_verdict, "verdicts": verdicts}
    return result


def _rejected_diagnosis(diagnosis: dict, root_verdict: dict) -> dict:
    return {
        "category": diagnosis.get("category", "unknown"),
        "risk_class": diagnosis.get("risk_class", "informational"),
        "claims": [],
        "root_cause": {
            "text": f"Root cause could not be verified against its cited evidence: {root_verdict['reason']}",
            "evidence": [],
        },
        "recommended_action": "escalate",
        "identity_status": diagnosis.get("identity_status"),
        "_verification": {
            "root_cause_verdict": root_verdict,
            "verdicts": [
                {"claim": diagnosis["root_cause"]["text"], "is_root_cause": True, **root_verdict}
            ],
        },
    }
