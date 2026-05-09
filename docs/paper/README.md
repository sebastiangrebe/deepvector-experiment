# `docs/paper/` — Research write-up

This directory contains the technical write-up of the DeepVector experiment.

## Contents

- **`deepvector_writeup.md`** — Full paper draft. Workshop / arXiv-level. ~7,000 words. Markdown with LaTeX-in-Markdown for equations and standard Markdown tables for results.

## How to read

- §1–§3 for the question, background, and methodology.
- §4 for the results, broken into four phases (pooled baselines, MaxSim discrimination, frozen ceiling, hybrid rerankers).
- §5 for interpretation; §6 for limitations (exhaustive); §7 for next experiments; §8 one-paragraph conclusion.
- All numbers cite specific JSON files in `data/results/` so each can be independently verified by running:
  ```python
  json.load(open('data/results/<file>.json'))['summary']
  ```

## How the paper was drafted

All experiments were run between Phase 2.5 (MaxSim discrimination), Phase 2.6 (frozen ceiling), and Phase 2.7 (hybrid rerankers) on a single Lambda Cloud H100 PCIe instance. Numbers were extracted verbatim from the committed `data/results/*.json` files; no value in the paper is approximated.

Web-verified citations are listed in §References. Items that could not be confidently verified are marked `[CITATION NEEDED]` for resolution at review.

## Status

First draft. Pending one final read by author before commit. Open for review thereafter.
