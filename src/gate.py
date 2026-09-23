"""The routing gate: a deterministic function over a finished diagnosis.

No LLM. Same inputs always produce the same outcome, and every decision
carries the name of the rule that produced it, so a routing call can be
explained after the fact without re-running anything.

Inputs it reads: risk_class, identity_status, the per-claim verifier
verdicts, and the root-cause verification outcome. Nothing else - notably
not the model's own `recommended_action`, which is the composer's opinion
and not evidence. The gate decides; the composer suggests.

Four outcomes:
  AUTO_RESOLVE            - answer the merchant directly
  DRAFT_FOR_HUMAN         - correct answer, human sends it
  REQUEST_IDENTIFICATION  - we don't know who is asking; ask them
  ESCALATE                - a human specialist needs to look at this

REQUEST_IDENTIFICATION is deliberately not a flavour of ESCALATE. When a
merchant can't be identified there is nothing for a specialist to
investigate yet - the missing thing is their identity, and the correct
response is to ask for it. Collapsing the two would both overload the
escalation queue and describe the situation wrongly.

Everything the gate can tune lives in src/config.py. Anything unexpected -
a missing field, an unknown enum, a malformed verification record -
escalates. Fail closed: the gate never guesses its way to a permissive
outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import AUTO_RESOLVE_MIN_CLAIMS, AUTO_RESOLVE_MIN_SUPPORTED_RATIO
from src.schema import IDENTITY_STATUSES, RISK_CLASSES

AUTO_RESOLVE = "AUTO_RESOLVE"
DRAFT_FOR_HUMAN = "DRAFT_FOR_HUMAN"
REQUEST_IDENTIFICATION = "REQUEST_IDENTIFICATION"
ESCALATE = "ESCALATE"

OUTCOMES = (AUTO_RESOLVE, DRAFT_FOR_HUMAN, REQUEST_IDENTIFICATION, ESCALATE)

# Conservatism order, used when a multi-cause ticket produces more than one
# diagnosis: the whole ticket takes the most conservative lane any single
# cause demands. ESCALATE is the most conservative (a specialist looks at
# it); REQUEST_IDENTIFICATION outranks DRAFT because it declines to answer
# at all. In practice identity is a property of the ticket rather than of
# one cause, so the identity rules fire for every sub-diagnosis at once and
# the ordering between those two rarely decides anything - it is defined
# here so the aggregation is total rather than ambiguous.
_CONSERVATISM = {
    AUTO_RESOLVE: 0,
    DRAFT_FOR_HUMAN: 1,
    REQUEST_IDENTIFICATION: 2,
    ESCALATE: 3,
}

# Rule names. Returned on every decision for the audit trail.
RULE_MALFORMED_FAIL_CLOSED = "malformed_diagnosis_fail_closed"
RULE_UNIDENTIFIABLE_ASK = "unidentifiable_request_identification"
RULE_MISMATCH_DATA_ACCESS_RISK = "identity_mismatch_data_access_risk"
RULE_ROOT_CAUSE_REJECTED = "root_cause_rejected_by_verifier"
RULE_UNCORROBORATED_DISCLOSES_ACCOUNT_DATA = "uncorroborated_identity_discloses_account_data"
RULE_CLAIM_STRIPPED = "claim_stripped_as_unsupported"
RULE_INSUFFICIENT_SUPPORT_RATIO = "insufficient_supported_claim_ratio"
RULE_TOO_FEW_CLAIMS = "too_few_surviving_claims"
RULE_CLEAN_AUTO_RESOLVE = "clean_auto_resolve"

# Risk classes flagged as involving money movement, for the stricter bar.
_MONEY_MOVEMENT = "money_movement"

# Evidence id prefixes that mean the answer discloses something about a
# specific merchant's account, rather than quoting published policy.
# `derived:` counts: a derived fact is computed from that merchant's own
# state, so "your batch was captured 2 business days ago" reveals account
# activity just as surely as the raw field does.
_ACCOUNT_DATA_PREFIXES = ("state:", "derived:")


@dataclass(frozen=True)
class GateDecision:
    outcome: str
    rule: str
    reason: str
    risk_flags: tuple[str, ...] = ()
    inputs: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "rule": self.rule,
            "reason": self.reason,
            "risk_flags": list(self.risk_flags),
            "inputs": self.inputs,
        }


def _verification_summary(diagnosis: dict) -> dict:
    """Pull the gate-relevant facts out of a diagnosis' `_verification`
    record, or report that verification did not run.

    The distinction between "verification ran and found nothing wrong" and
    "verification did not run" is load-bearing. When the verifier is off
    (the ablation), the gate simply has fewer signals; it must not treat
    absent verdicts as failed verdicts. Doing so would push every
    ablation ticket toward DRAFT_FOR_HUMAN and make the measured
    verifier-vs-no-verifier delta an artifact of the gate rather than a
    property of the verifier.
    """
    verification = diagnosis.get("_verification")
    if not verification:
        return {"ran": False}

    verdicts = verification.get("verdicts", [])
    claim_verdicts = [v for v in verdicts if not v.get("is_root_cause")]
    root = verification.get("root_cause_verdict", {}) or {}

    supported = sum(1 for v in claim_verdicts if v.get("verdict") == "SUPPORTED")
    stripped = [v for v in claim_verdicts if v.get("verdict") == "UNSUPPORTED"]

    return {
        "ran": True,
        "root_cause_verdict": root.get("verdict"),
        "claim_count": len(claim_verdicts),
        "supported_count": supported,
        "supported_ratio": (supported / len(claim_verdicts)) if claim_verdicts else None,
        "stripped_count": len(stripped),
    }


def _account_data_citations(diagnosis: dict) -> set[str]:
    """Evidence ids in the response that disclose this account's own data.

    Reads the surviving claims AND the root cause, because both are
    rendered into the message the merchant would receive and both appear
    in its Sources list - the risk is what gets sent, not which field of
    the diagnosis object it came from. Stripped claims are already gone
    from `claims` by this point and correctly don't count: they are never
    disclosed.
    """
    cited: set[str] = set()
    parts = list(diagnosis.get("claims", []))
    root_cause = diagnosis.get("root_cause")
    if isinstance(root_cause, dict):
        parts.append(root_cause)
    for part in parts:
        for evidence_id in part.get("evidence", []):
            if isinstance(evidence_id, str) and evidence_id.startswith(_ACCOUNT_DATA_PREFIXES):
                cited.add(evidence_id)
    return cited


def evaluate(diagnosis: dict) -> GateDecision:
    """Route one diagnosis. Never raises: any failure escalates."""
    try:
        return _evaluate(diagnosis)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, fail closed
        return GateDecision(
            outcome=ESCALATE,
            rule=RULE_MALFORMED_FAIL_CLOSED,
            reason=f"Gate could not evaluate this diagnosis, escalating: {exc}",
            risk_flags=("gate_error",),
        )


def _evaluate(diagnosis: dict) -> GateDecision:
    identity_status = diagnosis.get("identity_status")
    risk_class = diagnosis.get("risk_class")

    if identity_status not in IDENTITY_STATUSES:
        raise ValueError(f"unknown identity_status: {identity_status!r}")
    if risk_class not in RISK_CLASSES:
        raise ValueError(f"unknown risk_class: {risk_class!r}")

    verification = _verification_summary(diagnosis)
    surviving_claims = len(diagnosis.get("claims", []))
    inputs = {
        "identity_status": identity_status,
        "risk_class": risk_class,
        "surviving_claims": surviving_claims,
        "verification": verification,
    }

    # --- 1. identity gates, before anything about the diagnosis itself ---
    if identity_status == "UNIDENTIFIABLE":
        return GateDecision(
            outcome=REQUEST_IDENTIFICATION,
            rule=RULE_UNIDENTIFIABLE_ASK,
            reason=(
                "The merchant could not be identified. The missing thing is their identity, "
                "so the correct next step is to ask who they are - not to escalate."
            ),
            inputs=inputs,
        )

    if identity_status == "MISMATCH":
        return GateDecision(
            outcome=ESCALATE,
            rule=RULE_MISMATCH_DATA_ACCESS_RISK,
            reason=(
                "The ticket's claimed merchant_id does not match the merchant identified from "
                "a cited reference or business name. Answering would risk disclosing one "
                "merchant's account data to another. This is a data-access risk, not a "
                "diagnostic failure - the diagnosis may well be correct for the wrong account."
            ),
            risk_flags=("data_access_risk",),
            inputs=inputs,
        )

    # --- 2. verification gates ---
    if verification["ran"] and verification["root_cause_verdict"] == "UNSUPPORTED":
        return GateDecision(
            outcome=ESCALATE,
            rule=RULE_ROOT_CAUSE_REJECTED,
            reason=(
                "The verifier found the root cause unsupported by its own cited evidence. "
                "With the central finding rejected there is no diagnosis left to send."
            ),
            risk_flags=("ungrounded_root_cause",),
            inputs=inputs,
        )

    # --- 3. rules that forbid AUTO_RESOLVE but allow a human to send it ---
    # ticket_081's defence. Identity resolution alone cannot catch a ticket
    # that claims one merchant_id while describing another's situation when
    # the body names no id and no business name (see tests/test_identity.py).
    # The residual protection is procedural, and it keys on what the answer
    # would DISCLOSE rather than on what the ticket is about: if identity
    # was never independently corroborated and the response would quote
    # this account's own data back to whoever wrote in, a human checks it
    # first. An answer built only from published policy discloses nothing
    # account-specific and can still auto-resolve, even if it was wrong
    # about who asked.
    if identity_status == "UNCORROBORATED":
        disclosing = _account_data_citations(diagnosis)
        if disclosing:
            return GateDecision(
                outcome=DRAFT_FOR_HUMAN,
                rule=RULE_UNCORROBORATED_DISCLOSES_ACCOUNT_DATA,
                reason=(
                    "Identity rests only on the merchant_id submitted with the ticket, with no "
                    "reference or business name corroborating it, and the answer cites this "
                    f"account's own data ({', '.join(sorted(disclosing)[:3])}"
                    f"{' and others' if len(disclosing) > 3 else ''}). Sending it would "
                    "disclose one merchant's account data to someone we have not verified, so "
                    "a human checks it first."
                ),
                risk_flags=("uncorroborated_identity", "account_data_disclosure"),
                inputs={**inputs, "disclosing_evidence": sorted(disclosing)},
            )

    if verification["ran"] and verification["stripped_count"] > 0:
        return GateDecision(
            outcome=DRAFT_FOR_HUMAN,
            rule=RULE_CLAIM_STRIPPED,
            reason=(
                f"{verification['stripped_count']} claim(s) were found unsupported by their "
                "cited evidence and stripped. The remaining answer may be incomplete in ways "
                "the merchant would notice, so a human reviews it."
            ),
            risk_flags=("stripped_claim",),
            inputs=inputs,
        )

    min_claims = AUTO_RESOLVE_MIN_CLAIMS[risk_class]
    if surviving_claims < min_claims:
        return GateDecision(
            outcome=DRAFT_FOR_HUMAN,
            rule=RULE_TOO_FEW_CLAIMS,
            reason=(
                f"Only {surviving_claims} claim(s) survived, below the minimum of "
                f"{min_claims} required to auto-resolve a {risk_class} ticket."
            ),
            inputs=inputs,
        )

    if verification["ran"] and verification["supported_ratio"] is not None:
        threshold = AUTO_RESOLVE_MIN_SUPPORTED_RATIO[risk_class]
        if verification["supported_ratio"] < threshold:
            return GateDecision(
                outcome=DRAFT_FOR_HUMAN,
                rule=RULE_INSUFFICIENT_SUPPORT_RATIO,
                reason=(
                    f"{verification['supported_count']}/{verification['claim_count']} claims "
                    f"came back fully SUPPORTED (ratio "
                    f"{verification['supported_ratio']:.2f}), below the "
                    f"{threshold:.2f} required for a {risk_class} ticket. The rest were only "
                    "partially supported."
                ),
                inputs=inputs,
            )

    return GateDecision(
        outcome=AUTO_RESOLVE,
        rule=RULE_CLEAN_AUTO_RESOLVE,
        reason=(
            "Identity is corroborated, the root cause holds against its evidence, no claim "
            "was stripped, and the supported-claim bar for this risk class is met."
        ),
        inputs=inputs,
    )


def evaluate_all(diagnoses: list[dict]) -> GateDecision:
    """Route a ticket that produced more than one diagnosis.

    The whole ticket takes the most conservative lane any single cause
    demands - a ticket bundling an auto-resolvable question with one that
    needs a specialist is not half-answered automatically.
    """
    if not diagnoses:
        return GateDecision(
            outcome=ESCALATE,
            rule=RULE_MALFORMED_FAIL_CLOSED,
            reason="No diagnosis was produced for this ticket, escalating.",
            risk_flags=("gate_error",),
        )

    decisions = [evaluate(d) for d in diagnoses]
    worst = max(decisions, key=lambda d: _CONSERVATISM[d.outcome])
    if len(decisions) == 1:
        return worst

    return GateDecision(
        outcome=worst.outcome,
        rule=worst.rule,
        reason=(
            f"{len(decisions)} causes were diagnosed for this ticket; the most conservative "
            f"lane any one of them required was {worst.outcome}. {worst.reason}"
        ),
        risk_flags=worst.risk_flags,
        inputs={"per_cause": [d.as_dict() for d in decisions]},
    )
