# Ford 2024-A loan-state and target decision

- Decision date: 2026-08-23
- Implementation revision: `aa537c0`
- Dataset: Ford Credit Auto Owner Trust 2024-A, 2025 reports
- ABS schema: 3.1

## Decision

Accept one derived research target: first 60-or-more-day delinquency or
charge-off in the next three reports. A cutoff is eligible only when the loan is
reported, nonterminal, has a known delinquency state below 60 days, and has
three future trust reports. Charge-off is observed through
`zeroBalanceCode=4` or positive `chargedoffPrincipalAmount`; serious delinquency
is observed through `currentDelinquencyStatus>=60`.

Any other reported zero-balance code is a competing event. A missing loan state
inside the horizon is censored, never a default or prepayment. This rule is a
loan-performance research label for an already-originated securitized pool; it
is not an applicant-default, approval, or adverse-action label.

Reject two candidate targets:

- Standalone prepayment: Schedule AL `zeroBalanceCode=1` combines prepaid and
  matured loans.
- Ultimate net loss: `chargedoffPrincipalAmount` and `recoveredAmount` exist,
  but twelve reports do not observe a complete post-charge-off recovery horizon.

## Reproduction

The audit was run from revision `aa537c0` in an isolated worktree against the
existing ignored cache:

```console
uv run --locked python -m quantcredit.audit --cache ../quantcredit/data/sec
```

The command verifies every byte and checksum before returning aggregate JSON.
It emits no source rows or asset identifiers and returns no partial result after
an invariant failure.

Aggregate result:

| Measure | Count |
| --- | ---: |
| Documents / report periods | 12 |
| Snapshots | 408,052 |
| Stable loan identities | 38,224 |
| Duplicate snapshot keys | 0 |
| Retained immutable contradictions | 0 |
| Contiguous observed transitions | 369,668 |
| Distinct observed transition types | 41 |
| Disappearances between reports | 8,586 |
| Reappearances after an absent report | 160 |

Three-report target classifications across all 458,688 loan-cutoff positions:

| Classification | Count |
| --- | ---: |
| Positive | 2,855 |
| Negative with complete follow-up | 287,209 |
| Competing zero-balance event | 18,724 |
| Missing follow-up | 0 |
| Right-censored | 88,887 |
| Ineligible at cutoff | 61,013 |

The zero missing-follow-up target count does not turn disappearance into an
event. Under this panel, otherwise eligible cutoffs either have all three future
states or encounter a reported competing terminal state before disappearance.
Reappearances remain explicit continuity evidence.

Observed snapshot states were 373,328 current, 21,664 at 1–29 days, 2,684 at
30–59 days, 500 at 60–89 days, 209 at 90+ days, 427 with missing delinquency,
8,912 with zero-balance code 1, 61 with code 3, and 267 with code 4.

## Exact source pins

The tracked manifest owns the official URLs in addition to these byte counts
and SHA-256 digests:

| Report period | Accession | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `2025-01-31` | `0002014176-25-000006` | `136232097` | `a09207114e580864113b571de95d4cf676336b94cc6ac95b37743e8819ad8a8d` |
| `2025-02-28` | `0002014176-25-000010` | `133526572` | `7a14b1815f2b041949fcd30d185c9a7322863e5a69ba96a4fab60668346076b8` |
| `2025-03-31` | `0002014176-25-000016` | `131101190` | `1494536e48ed7b4d383413bd5602c964205a1a5f88318892417f0c5d53ae3f8d` |
| `2025-04-30` | `0002014176-25-000020` | `128423769` | `b3c6ab7e1706e1e3fffad78c13d6efe7e08b912c202423f52c19dc18135c6667` |
| `2025-05-31` | `0002014176-25-000024` | `125732176` | `136ab3546a252b374a1c628c215c246c75a37f6a8a0d26c08eb3a79a77e12130` |
| `2025-06-30` | `0002014176-25-000028` | `122908754` | `b735f0038e1e99195cef269bb0e461c5654f6ede3645b07a44afbb09e4903ca3` |
| `2025-07-31` | `0002014176-25-000033` | `120127298` | `5ff26621cb684bc55b057fa34c41f03d0b0a1a9477c02c99c9fb724273aeefbb` |
| `2025-08-31` | `0002014176-25-000037` | `117180390` | `5a084ef31e834bed368549ba9f5e568f4478087cb7930433553caf41bfd9d5e1` |
| `2025-09-30` | `0002014176-25-000042` | `114296060` | `8f2dfe6c97a39245862cc9160414826827cedff1d427bac65b04221bb633ca41` |
| `2025-10-31` | `0002014176-25-000047` | `111425666` | `8ae0f8b77b8f11ae01063050251e4d3f820a65d02612f603ae64ff0c43e614ed` |
| `2025-11-30` | `0002014176-25-000051` | `108609415` | `2bd17434b9064d6a16716083f0807f58a3742c11eb7e7731af7801aa24150c15` |
| `2025-12-31` | `0002014176-26-000002` | `106254720` | `459d3c90d0714f5df478e0a05b19d0e682021a608f55ab09a0866d89f14466fb` |

## Limits and reopening conditions

This is one Ford securitization trust-year. It cannot establish applicant-level
creditworthiness, rejected-applicant behavior, cross-originator validity, or a
production probability of default. Repeated cutoffs from one loan are dependent
observations; Stage 2 must split by prediction time and keep loans/cohorts from
leaking future facts.

Reopen the target decision only when one of these facts changes:

- a longer checksum-pinned panel materially changes horizon coverage or event
  prevalence;
- an official field or independently validated rule separates prepayment from
  contractual maturity;
- seasoned recovery observations support a declared net-loss horizon; or
- another pinned issuer supports an issuer-held-out generalization claim.

Schema meanings follow SEC Release 33-9638 and the ABS 3.1 technical
specification. The tracked source manifest, parser, target function, and this
record jointly own reproducibility; visible notebook output does not.
