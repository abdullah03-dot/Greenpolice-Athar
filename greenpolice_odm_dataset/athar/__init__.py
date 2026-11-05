"""
ATHAR completion layer (2026-11): chain-of-custody verification, human-in-the-loop review and a
local-only agent that drafts (never finalises) the evidence report.

  athar.chain      hash-chained JSONL, byte-compatible with 07's ForensicAuditLogger
  athar.verify     one CLI that verifies a whole 11 run folder (chain, signed head, manifest, crops,
                   media, review decisions)            python -m athar.verify <run_dir> ...
  athar.review     review session: every human decision is a record in a hash chain anchored to the
                   evidence chain
  athar.dashboard  local-only (127.0.0.1) web app for the reviewer, stdlib only
  athar.agent      triage + report drafting behind an LLM interface (Ollama on localhost, or a
                   deterministic stub)

Nothing here needs a GPU, model weights or network access.
"""
