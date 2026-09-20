"""The structured diagnosis output every agent run produces.

Output is a structured object, never prose: a claim is text plus the
evidence_ids that back it, decided *before* any sentence is written.
`render_prose()` turns claims into a message at the very end - prose is
never generated first and cited after, because that ordering is exactly
how ungrounded text gets in.
"""

from __future__ import annotations

RISK_CLASSES = ("money_movement", "informational")

# Matches eval/README.md's three lanes exactly, so a diagnosis's
# recommended_action is directly comparable to eval/heldout.json's
# correct_routing without translation.
ROUTING_LANES = ("auto_resolve", "draft_for_human", "escalate")

IDENTITY_STATUSES = ("CONFIRMED", "UNCORROBORATED", "MISMATCH", "UNIDENTIFIABLE")

# The JSON schema handed to the model for response_json_schema. Deliberately
# flat and simple: a nested/optional-heavy schema is where structured
# output tends to degrade in practice.
DIAGNOSIS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "risk_class": {
            "type": "string",
            "enum": list(RISK_CLASSES),
            "description": (
                "'money_movement' if the ticket concerns settlements, refunds, reserves, "
                "or holds - anything about whether or when money moves. 'informational' "
                "for everything else (policy explanations, KYC document guidance, generic "
                "how-does-this-work questions), even if the merchant sounds urgent about it."
            ),
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "evidence"],
            },
        },
        "root_cause": {"type": "string"},
        "recommended_action": {"type": "string", "enum": list(ROUTING_LANES)},
    },
    "required": ["category", "risk_class", "claims", "root_cause", "recommended_action"],
}


class UngroundedClaimError(ValueError):
    """A claim cited an evidence_id that was never actually gathered."""


def validate_diagnosis(diagnosis: dict, known_evidence_ids: set[str]) -> None:
    """Raise if the model's output violates the output contract.

    This is a *grounding* check, not a verifier: it confirms every cited
    evidence_id is one the loop actually fetched (so a citation can't be
    hallucinated out of thin air). It says nothing about whether the
    CONTENT of that evidence really supports the claim's text - that
    semantic check is the Phase 3 verifier's job, not this one's.
    """
    if diagnosis["risk_class"] not in RISK_CLASSES:
        raise ValueError(f"invalid risk_class: {diagnosis['risk_class']!r}")
    if diagnosis["recommended_action"] not in ROUTING_LANES:
        raise ValueError(f"invalid recommended_action: {diagnosis['recommended_action']!r}")

    for claim in diagnosis["claims"]:
        for evidence_id in claim["evidence"]:
            if evidence_id not in known_evidence_ids:
                raise UngroundedClaimError(
                    f"claim {claim['text']!r} cites evidence_id {evidence_id!r}, "
                    f"which was never gathered by any tool call in this run"
                )


def render_prose(diagnosis: dict) -> str:
    """Render claims into a message. Called after the diagnosis exists,
    never before - prose is a view of the claims, not their source.
    """
    lines = [claim["text"] for claim in diagnosis["claims"]]
    body = " ".join(lines)
    citations = sorted({eid for claim in diagnosis["claims"] for eid in claim["evidence"]})
    if citations:
        body += "\n\nSources: " + ", ".join(citations)
    return body
