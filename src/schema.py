"""The structured diagnosis output every agent run produces.

Output is a structured object, never prose: a claim is text plus the
evidence_ids that back it, decided *before* any sentence is written.
`render_prose()` turns claims into a message at the very end - prose is
never generated first and cited after, because that ordering is exactly
how ungrounded text gets in.

Claims are ATOMIC. Each one asserts exactly one checkable fact and cites
only the evidence for that fact. This came out of the blind adversarial
check, where every *correct* claim still came back PARTIALLY_SUPPORTED:
the composer had bundled a business name, explanatory framing and several
facts into a single claim citing one narrow field, so the verifier - which
by design sees only the claim and its cited evidence - was right to say
the evidence didn't cover all of it. The fix is to stop asking one claim
to carry more than one fact, not to teach the verifier to wave framing
through.

Everything that isn't a checkable fact - the merchant's name, connective
tissue, tone - belongs in `render_prose`, which runs after verification
and can be handed context the verifier deliberately never sees.
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
            "description": (
                "Atomic claims. Each entry asserts EXACTLY ONE checkable fact and cites "
                "only the evidence for that one fact. Split anything compound into "
                "separate claims."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "One checkable fact, stated plainly. No merchant or business "
                            "name, no dates or amounts that aren't in the cited evidence, "
                            "no explanatory framing, no 'and'-joined second fact. Write "
                            "'The settlement cycle is T+3.' - not 'Kavya Handloom Exports "
                            "is on a T+3 cycle, so their payout is on track.'"
                        ),
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "The evidence_id(s) that establish this one fact, and nothing "
                            "else. Any claim about elapsed time, a due date, or whether a "
                            "window has passed must cite a derived:... fact - never "
                            "compute dates yourself."
                        ),
                    },
                },
                "required": ["text", "evidence"],
            },
        },
        # root_cause is claim-shaped, not a bare string: the Phase 3
        # verifier's rule is "if the root_cause claim is unsupported,
        # reject the whole diagnosis," which is only checkable if
        # root_cause actually cites the evidence it rests on, same as any
        # other claim. A bare string had nothing to verify against.
        "root_cause": {
            "type": "object",
            "description": (
                "The single underlying cause, held to the same atomic standard as a claim: "
                "one checkable assertion, citing only the evidence that establishes it."
            ),
            "properties": {
                "text": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text", "evidence"],
        },
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

    for claim in [*diagnosis["claims"], diagnosis["root_cause"]]:
        for evidence_id in claim["evidence"]:
            if evidence_id not in known_evidence_ids:
                raise UngroundedClaimError(
                    f"claim {claim['text']!r} cites evidence_id {evidence_id!r}, "
                    f"which was never gathered by any tool call in this run"
                )


def render_prose(diagnosis: dict, merchant_name: str | None = None) -> str:
    """Render verified claims into a message. Called after the diagnosis
    exists, never before - prose is a view of the claims, not their source.

    This is where the framing that claims are forbidden from carrying goes
    back in. `merchant_name` is the clearest example: naming the business
    is right in a customer-facing message and wrong inside a claim, because
    the verifier sees only a claim and its cited evidence - a bare
    `reserve.percentage` of 15 can never establish *whose* reserve it is,
    so a claim asserting the name is unverifiable by construction. Adding
    it here costs nothing, because prose is rendered after verification has
    already passed on the facts.
    """
    all_claims = [diagnosis["root_cause"], *diagnosis["claims"]]
    lines = [claim["text"] for claim in all_claims if claim["text"]]
    body = " ".join(lines)

    if merchant_name and body:
        body = f"For {merchant_name}: {body}"

    citations = sorted({eid for claim in all_claims for eid in claim["evidence"]})
    if citations:
        body += "\n\nSources: " + ", ".join(citations)
    return body
