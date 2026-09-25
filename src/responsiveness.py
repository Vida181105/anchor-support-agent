"""Responsiveness check: was the question that was asked the question that
got answered?

This exists because of a measured failure, not a hypothetical one. On the
40 held-out tickets, all six false auto-resolves were diagnoses whose
claims were faithfully grounded in the evidence they cited - the verifier
passed them, correctly - but which answered a different question than the
merchant asked. heldout_002 was asked how to change a settlement bank
account and replied that the settlement cycle is T+3: true, correctly
cited, verified, and useless. src/verifier.py cannot catch that, by
design: it never sees the ticket, so it has no idea what was asked.

So this is the orthogonal half. Two checks, each blinded to the thing that
would let it rubber-stamp:

    src/verifier.py      sees the evidence, never the question
                         -> can judge TRUTH, cannot judge RELEVANCE
    src/responsiveness.py sees the question, never the evidence
                         -> can judge RELEVANCE, cannot judge TRUTH

Neither can wave something through on the strength of the other's
information. A verifier that saw the ticket could infer the intended
answer and grade toward it; a responsiveness check that saw the evidence
could start reasoning about whether the answer is correct and conflate the
two axes. Keeping them blind in opposite directions is the point, and it
is enforced by the function signatures: check_responsiveness takes two
strings and literally cannot read anything else.

Fail closed: any error, malformed output or invalid verdict is
NON_RESPONSIVE, never a pass.
"""

from __future__ import annotations

import json

from src.llm import LLMClient

RESPONSIVENESS_MODEL = "gemini-3.5-flash-lite"

VERDICTS = ("RESPONSIVE", "PARTIALLY_RESPONSIVE", "NON_RESPONSIVE")

RESPONSIVENESS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "reason": {"type": "string", "description": "One short sentence."},
    },
    "required": ["verdict", "reason"],
}

_PROMPT_TEMPLATE = """\
You are checking whether an answer is RESPONSIVE to a question - whether \
the question that was asked is the question that got answered.

You are NOT checking whether the answer is true. You have not been shown \
any evidence and cannot verify anything. Assume every factual statement \
below is correct. Judge only whether it addresses what this person \
actually asked.

- RESPONSIVE: it addresses the question asked. It may be brief or \
incomplete, but what it says is on the question.
- PARTIALLY_RESPONSIVE: it addresses part of the question, or answers a \
closely adjacent question while leaving the one actually asked \
unaddressed.
- NON_RESPONSIVE: it answers a different question. The person reading it \
would feel it missed the point entirely.

A ticket may be messy, misspelled, or ask more than one thing. Judge \
against what the person was evidently trying to find out.

TICKET:
\"\"\"
{ticket_body}
\"\"\"

PROPOSED ROOT CAUSE:
\"\"\"
{root_cause_text}
\"\"\"

Return your verdict now.\
"""


def check_responsiveness(
    ticket_body: str,
    root_cause_text: str,
    llm: LLMClient,
    model: str = RESPONSIVENESS_MODEL,
) -> dict:
    """Judge whether `root_cause_text` answers what `ticket_body` asked.

    Takes two plain strings rather than the ticket and diagnosis objects,
    so there is no route by which evidence content, merchant state or
    sibling claims could reach the prompt - the isolation is a property of
    the signature, not of the caller remembering to be careful.

    Never raises. Any failure returns NON_RESPONSIVE with the reason.
    """
    if not root_cause_text or not root_cause_text.strip():
        return {"verdict": "NON_RESPONSIVE", "reason": "no root cause text to assess"}
    if not ticket_body or not ticket_body.strip():
        return {"verdict": "NON_RESPONSIVE", "reason": "no ticket text to assess against"}

    prompt = _PROMPT_TEMPLATE.format(
        ticket_body=ticket_body, root_cause_text=root_cause_text
    )
    try:
        result = llm.generate_turn(
            [{"role": "user", "text": prompt}],
            model=model,
            response_json_schema=RESPONSIVENESS_JSON_SCHEMA,
            temperature=0.0,
        )
        parsed = json.loads(result["text"])
        if parsed.get("verdict") not in VERDICTS:
            raise ValueError(f"invalid verdict: {parsed.get('verdict')!r}")
        return {"verdict": parsed["verdict"], "reason": parsed.get("reason", "")}
    except Exception as exc:  # noqa: BLE001 - deliberately broad, fail closed
        return {
            "verdict": "NON_RESPONSIVE",
            "reason": f"responsiveness check error, failing closed: {exc}",
        }


def check_diagnosis_responsiveness(
    ticket: dict, diagnosis: dict, llm: LLMClient, model: str = RESPONSIVENESS_MODEL
) -> dict:
    """Convenience wrapper: pulls exactly the two fields the check is
    allowed to see out of a ticket and a diagnosis, and nothing else.
    """
    return check_responsiveness(
        ticket.get("body", ""),
        (diagnosis.get("root_cause") or {}).get("text", ""),
        llm,
        model,
    )
