# ATHAR component status (2026-11-05)

An earlier manuscript listed three ATHAR parts as "designed, not built": (a) the agentic layer,
(b) chain-head signing, (c) a human-in-the-loop review dashboard. This file records what the code
actually contained **before** branch `claude/athar-completion`, and what that branch adds. Paths
are relative to `greenpolice_odm_dataset/`.

## Before this branch (established from the code on `main`, commit d911ed7)

| Part | Status | Evidence in the code |
|---|---|---|
| Hash-chained evidence log | **Built** | `07_edge_benchmark.py` → `ForensicAuditLogger` (`log_detection`, `verify_chain`; verify-before-append on resume). `11_evidence_pipeline.py` writes `run_<UTC>_<manifest>/evidence_audit.jsonl` with frame records and `evidence_object` records (status `CANDIDATE - requires human review`). Tested: `tests/test_forensic_hardening.py`, `tests/test_fusion.py`. |
| (b) Chain-head signing | **Built** | `tools/sign_chain_head.py` (`keygen` / `sign` / `verify`, Ed25519; pins record count, first and last hash, whole-file SHA-256; trusted fingerprint required). `11 --signing_pubkey` puts the key fingerprint in `run_manifest.json`. Tested: `tests/test_forensic_hardening.py::test_signed_head`, `tests/test_tamper_experiment.py` (H3 tamper experiment; the official 200-trial run is **not done** and waits for the targets freeze, HANDOFF §16). |
| End-to-end run verifier | **Partial** | The chain and the head could each be checked (`07 --verify_chain`, `sign_chain_head.py verify`). Nothing checked a whole run folder in one place: manifest hash against what the records cite, review crops against their recorded SHA-256, source media re-hash (only inside `tools/tamper_experiment.py`, on synthetic logs). No RFC 3161 timestamping (HANDOFF §15 "next step"). |
| (c) Human-in-the-loop review | **Partial (CSV only)** | 11 writes `review.csv` with empty `reviewer_decision / reviewer_class / reviewer_name / reviewer_note` columns, plus `review_crops/`. No UI, and nothing records decisions in a tamper-evident way: a CSV edited by hand is neither chained nor signed. |
| (a) Agent layer (local LLM triage + report draft) | **Missing** | No LLM, Ollama or agent code anywhere in the repo (grep for `ollama`, `llm`, `agent`: only unrelated hits). |

## Added on `claude/athar-completion`

| Part | Where | What it does |
|---|---|---|
| Independent chain implementation | `athar/chain.py` | Same hashing convention as 07, so logs pass in both directions (tested). Also checks `record_index`, which 07 does not. No third-party dependencies. |
| Run verifier CLI | `athar/verify.py` (`python -m athar.verify <run_dir> --trusted_fingerprint sha256:…`) | One command checks the evidence chain, the signed head against a trusted key, the manifest (hash cited by every record, folder suffix, signing-key fingerprint equal to the head's), raw crops, source media (`--media_root`), the review chain and its anchors, and the review head. Exit code 1 on any failure. |
| Review chain | `athar/review.py` | `review_decisions.jsonl` is a separate hash chain. Its `review_session_open` record pins the evidence log (count, last hash, file SHA-256, head fingerprint). Each decision pins the evidence record it decides. Decisions are never edited: a changed mind adds a new record. The review can be signed with the reviewer's own key using the existing `sign_chain_head.py`. A session refuses to start on a broken evidence chain (or an untrusted head, when a key is given) and stops if the evidence log changes mid-session. |
| Reviewer dashboard | `athar/dashboard.py` (`python -m athar.dashboard <run_dir> --reviewer "<name>"`) | Local web app using only the standard library (wsgiref). Binds to loopback only, rejects non-loopback `Host` headers, needs a CSRF token on every POST, loads nothing from outside, and serves crops only from `review_crops/` after re-hashing them. The queue is in triage order. |
| Agent layer | `athar/agent.py` (`python -m athar.agent triage|draft <run_dir>`) | Triage is deterministic code (tier-1, then multi-view confirmed, then score). The report draft takes its facts from human-approved decisions only. The LLM writes the narrative paragraph only, and any number in it that is not in the facts is flagged. Backends: `StubBackend` (deterministic) and `OllamaBackend` (loopback only, proxy bypassed, temperature 0, fixed seed). Each draft's SHA-256, backend, model and prompt hash are logged into the review chain. |
| Tests | `tests/test_athar_{custody,review,agent}.py`, `tests/conftest.py` | 27 tests. They run on a **real** `11_evidence_pipeline.py` run built with the fake pixel detector from `test_fusion.py` (CPU, no weights, no data). |
| CI | `.github/workflows/ci.yml`, `requirements-ci.txt`, `pytest.ini` | Ubuntu and Windows × Python 3.10 and 3.11, CPU only. Markers `gpu`, `needs_data` and `needs_torch` are deselected in CI. |

## Still missing or not verified

- **RFC 3161 trusted timestamping** of signed heads: not built. Without it, the signing time is only the signer's clock.
- **Hardware-token signing** (PIV / HSM): not built. Keys are PEM files kept outside the repo.
- **Reviewer identity is self-declared** in the dashboard (there is no login). It becomes binding only when the reviewer signs the review chain's head with their own key. That step is a CLI command and is not in the UI.
- **`*_boxed.png` review crops are not hash-pinned by 11.** Only `*_raw.png` is. The dashboard therefore shows the raw crop, and each decision records the boxed crop's SHA-256 as it was at decision time.
- **Video evidence:** the media re-hash cannot check frames referenced as `<file>#<frame>`. They are reported as "not available", not as passed.
- **Ollama with a real model has not been run.** Only the HTTP path was tested, against a fake server on 127.0.0.1. Narrative quality and how well the number guard catches hallucinations on a real model are unmeasured, and the guard only checks numbers, not names or claims.
- **No WORM storage.** The chain makes tampering detectable, not impossible. HANDOFF §15 applies unchanged.
- **The dashboard has not been tested in a real browser.** Tests call the WSGI app in-process. A live smoke run (curl against 127.0.0.1) passed.
- Opening the dashboard or the agent `draft` command appends a `review_session_open` record even when no decision follows. This is intended, because it logs who opened the evidence, but it makes the review chain grow.
