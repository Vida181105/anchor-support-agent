# KYC Document Requirements and Rejection Reasons

## Required documents by business type

Sole proprietorships require a PAN card and one address proof (utility bill
or bank statement, dated within 60 days). Private limited companies require
a Certificate of Incorporation, PAN of the company, and identity/address
proof of at least one authorized signatory. Partnerships require a
partnership deed in addition to the sole-proprietorship requirements for
each partner named on the deed.

## Common rejection reasons and their labels

Documents are rejected with a short internal reason code that is shown to
the merchant, but the code does not always spell out the underlying issue
in plain language. `DOC_MISMATCH` means the name or address on the
submitted document does not match the business name declared at signup —
this is the single most common rejection reason and is frequently caused by
a merchant submitting a personal document when a business document was
required, or vice versa. `DOC_EXPIRED` means the document's own validity
window (for address proofs specifically) has passed. `DOC_ILLEGIBLE` means
the scan or photo quality was too low for the verification system to read
key fields, even if the document itself is valid.

## Resubmission

A rejected document can be resubmitted at any time through the dashboard's
KYC section; there is no cooldown period. However, resubmitting the exact
same file that was already rejected will be rejected again for the same
reason — the system does not re-run a different check on an unchanged file.

## KYC status and settlement eligibility

An account with incomplete or rejected KYC is capped at a lower transaction
volume limit and is not eligible for T+1 settlement regardless of account
age. Once KYC is fully verified, the volume cap lifts within 24 hours, but
the T+1 eligibility clock (see settlement cycles) starts counting from the
KYC completion date, not the original signup date.

## Re-verification triggers

Completed KYC can be flagged for re-verification if the business changes
its registered address, changes its authorized signatory, or if the account
is escalated during an account review. Re-verification uses the same
document requirements as initial KYC and follows the same rejection codes.

## Document ownership mismatches

A specific and frequently confusing case: when a business's bank account is
held in the name of a director rather than the registered company name,
address-proof documents tied to that bank account are rejected as
`DOC_MISMATCH` even though the document itself is genuine. The correct fix
is to submit a document that shows the company name directly, such as a
GST registration certificate, rather than resubmitting bank-linked proof.

## Changing business type

Converting a registered entity type (for example, a sole proprietorship
incorporating as a private limited company) is treated the same as a
re-verification trigger: it counts as a change to the business's registered
structure, so KYC must be redone in full against the document set required
for the new entity type, rather than layered on top of the original
sole-proprietorship KYC.
