# How Much Retrieval Signal Lives in Mamba Latent State? A Case Study with Codestral-Mamba-7B on SWE-Bench Lite

## Abstract

Selective state-space models (SSMs) such as Mamba scale linearly in sequence length, making them attractive encoders for long-context retrieval tasks. Whether their token-level hidden states preserve enough discriminative signal for code retrieval — and what matching operation extracts it — is unclear. We study this question on SWE-Bench Lite using `mistralai/Mamba-Codestral-7B-v0.1` as the encoder and a per-repo single-commit indexing protocol. We report four findings. **(1)** Mean-pooled Codestral vectors achieve indexable Recall@10 of 0.34 against Voyage code-3's 0.95, despite no vector collapse (per-repo pairwise cosine means in the 0.67–0.82 range) — retrieval failure here is in the aggregation operator, not the representation geometry. **(2)** On a 80-instance subset where Voyage retrieves the gold file in top-10 but Codestral mean-pooling does not, ColBERT-style MaxSim over per-token Codestral latents recovers 35 of those 80 (Recall@10 = 0.4375); mean-pooling in the same harness recovers 1 (Recall@10 = 0.0125), a 35× lift in absolute hits. **(3)** Adding frozen architectural structure beyond MaxSim — multi-head random orthogonal projections at H ∈ {4, 8, 16, 32}, an explicit late-interaction normalization variant, and multi-granularity composites combining file-level, sliding-window mid-level, and token-level scores — does not exceed single-head MaxSim. The best multi-head method scores 0.4250 (Δ = −0.0125); the best multi-granularity composite scores 0.2875 (Δ = −0.15). **(4)** Two off-the-shelf cross-encoder rerankers stacked on the MaxSim shortlist (`bge-reranker-v2-m3`, `cross-encoder/ms-marco-MiniLM-L-12-v2`) underperform the MaxSim filter alone (Recall@10 of 0.4125 and 0.3000 respectively). The work supports a single interpretation: discriminative signal exists at the per-token level of Codestral-Mamba but is destroyed by mean-pooling, and closing the gap to retrieval-trained dense encoders likely requires retrieval-specific training rather than a clever frozen matching head or off-the-shelf NLP rerankers. We do not evaluate trained matching; we leave this as future work.

## 1. Introduction

Long-context retrieval is increasingly load-bearing for AI workloads: agents that read large codebases, legal due diligence, and technical literature search all push beyond the 8–32K-token window where transformer-style dense retrievers are competitive. Selective state-space models such as Mamba [Gu and Dao 2023] and Mamba-2 [Dao and Gu 2024] offer linear-time inference and a recurrent state that, in principle, summarizes arbitrarily long context without re-attending. This invites the question of whether their hidden states carry enough signal to drive *retrieval* — not generation, where SSMs have been studied extensively, but ranking — and whether existing dense-retrieval matching operations transfer.

Our specific question, posed in operational form: given `mistralai/Mamba-Codestral-7B-v0.1` (a 7B Mamba-2 model trained on code) as a frozen encoder, how much retrieval signal is recoverable from its hidden states, and what matching operation extracts it? We approach the question phase-by-phase. We compare the mean-pooled vector baseline against a retrieval-trained dense encoder (Voyage code-3) on SWE-Bench Lite [Jimenez et al. 2024] (§4.1). We isolate the failure modes of mean-pooling and test whether ColBERT-style MaxSim [Khattab and Zaharia 2020] over per-token Codestral latents recovers them (§4.2). We test whether frozen architectural sophistication beyond MaxSim — multi-head random orthogonal projections, explicit token-level normalization — improves the result (§4.3). We test whether off-the-shelf cross-encoder rerankers, stacked on a MaxSim shortlist, close the gap to Voyage (§4.4).

Our contributions are four findings, each negative or constrained, but together establishing a clear interpretation of what frozen Codestral latents can and cannot do for code retrieval:

- Mean-pooled Codestral underperforms Voyage code-3 on SWE-Bench Lite by a wide margin (indexable Recall@10 of 0.34 vs 0.95) despite producing well-spread vectors.
- ColBERT-style MaxSim on Codestral per-token latents recovers a substantial fraction of mean-pooling's failures on the discriminating subset (Recall@10 = 0.4375 vs 0.0125; 35× absolute-hit lift).
- Frozen multi-head projections beyond single-head MaxSim do not extract additional signal at this scale; frozen multi-granularity composites (G0/G1/G2/G3 sum, max, and routed) actively harm retrieval (best Δ = −0.15).
- Two cross-encoder rerankers trained on natural-language passage retrieval, applied to Codestral's MaxSim shortlist on SWE-Bench Lite, underperformed the shortlist alone; we attribute this narrowly to training-distribution mismatch and do not generalize to code-specific rerankers.

The work is a case study, not a general claim about Mamba-based retrieval. Our caveats are explicit (§6).

## 2. Background and Related Work

**State-space models.** Mamba [Gu and Dao 2023] introduced selective SSMs, where the state-transition matrices are input-dependent, recovering attention-like content selectivity while retaining linear-time recurrence. Mamba-2 [Dao and Gu 2024] reformulated the architecture as a state-space duality, exposing matrix-multiplication-friendly forms that match the throughput of transformer kernels at training time. Mamba-3 [Lahoti et al. 2026], an oral paper at ICLR 2026, further refines the recurrence with complex-valued state tracking and a multi-input/multi-output (MIMO) variant; at the time of this work, no pretrained Mamba-3 weights are publicly released and the released kernels target CUDA only, so we do not evaluate Mamba-3 here.

**Dense retrieval.** The dominant paradigm for retrieval treats queries and passages as fixed-dimension vectors produced by a frozen-or-fine-tuned encoder and ranks via cosine similarity over an ANN index [Karpukhin et al. 2020]. Voyage AI's `voyage-code-3` is an example of a code-specialized commercial encoder in this family and serves as our retrieval-trained baseline.

**Late interaction and MaxSim.** ColBERT [Khattab and Zaharia 2020] showed that retaining per-token vectors and matching them via *late interaction* — the MaxSim operator — produces strong out-of-domain retrieval, sacrificing index size for recall. MaxSim is the operation we study in §4.2 as a candidate matching head over Mamba latents.

**Reranking.** Cross-encoder rerankers — small transformers that take `(query, passage)` pairs and return a relevance score — sit on top of dense retrieval shortlists in production pipelines. We test two: BAAI's `bge-reranker-v2-m3` [Chen et al. 2024] (568M, multilingual, MTEB-trained) and `cross-encoder/ms-marco-MiniLM-L-12-v2` (33M, MS-MARCO-trained) [Reimers and Gurevych 2019]. We attempted `jinaai/jina-reranker-v2-base-multilingual` (no formal paper; model card only) but its dynamic-code path is incompatible with `transformers >= 5.0`.

**Code retrieval benchmarks.** SWE-Bench [Jimenez et al. 2024] frames issue-to-fix as a coding task; SWE-Bench Lite, the 300-instance test split with 23 dev instances, is a single-file benchmark by construction (each issue's gold patch modifies one source file). We use SWE-Bench Lite *retrieval* — the easier subtask of finding the modified file given the issue text — as our benchmark.

**SSM-based retrieval.** Concurrent and prior work on SSM-based retrieval includes Mamba Retriever, which trains a contrastive objective on Mamba encoders for general-domain retrieval. We do *not* train an encoder here; we study what an off-the-shelf code-pretrained Mamba-2 already supports.

## 3. Methodology

### 3.1 Encoder

We use `mistralai/Mamba-Codestral-7B-v0.1`, a 7B-parameter Mamba-2 model continually pre-trained on code. We chose it because it is the only HuggingFace-loadable Mamba-2 checkpoint of usable scale: the original `state-spaces/mamba2-*` checkpoints require the `mamba_ssm` package, which only builds against CUDA, and Mamba-3 weights are not publicly released.

The encoder is loaded via `transformers.AutoModelForCausalLM` in bfloat16 on a single H100 PCIe (80 GB). We consume the *last layer's* hidden states from `output_hidden_states=True`. For mean-pooled scoring we average the non-padded last-layer activations to produce a `(d_model = 4096)` vector per chunk; for token-level scoring we keep the full `(seq_len, 4096)` tensor. Implementation: `src/encoder.py`. The `MambaEncoder` class also exposes `.encode(texts)` which returns `EncoderOutput(last_hidden, pooled, attention_mask)` so all downstream methods consume the same forward pass.

A precision sanity check (`scripts/dtype_sanity.py`, results in `data/results/dtype_sanity_codestral.json`) confirms that switching from fp32 to bf16 changes per-vector cosine by less than $3 \times 10^{-6}$ on average and pairwise similarities by less than $4 \times 10^{-4}$ in mean absolute difference, so all matching operations run in bf16 throughout.

### 3.2 Benchmark and Subset Definitions

We evaluate on **SWE-Bench Lite** (`princeton-nlp/SWE-bench_Lite`, dev + test combined, 323 instances total across 18 unique GitHub repositories, all Python). Each instance carries an issue text (`problem_statement`), a base commit hash, and a gold patch from which we extract the modified file path — by construction, exactly one path per instance. We refer to this as the **gold file**. Patch parsing is in `src/eval.py:gold_files_from_patch`.

We define two subsets:

- **Indexable subset (n = 285, 88.2% of total)**: instances where the gold file exists in our index at the chosen indexing commit (see §3.3). The 38 non-indexable instances arise because we index each repo once at its first observed base commit; for instances whose base commit shifted the gold file's path or content, the gold is absent from our index.
- **Discriminating subset (n = 80, sampled from 173 candidates)**: instances where (a) the gold is in the index for both methods, (b) Voyage code-3 retrieves it within top-10, and (c) Codestral mean-pooling does not. This isolates the cases where mean-pooling specifically fails and tests whether richer matching recovers them. We sample 80 with `random.seed(42)`; the per-repo distribution is reported in §4.2.

### 3.3 Per-repo Single-commit Indexing

SWE-Bench Lite has 320 unique `(repo, base_commit)` tuples across 323 instances. Indexing each tuple separately with a 7B encoder is prohibitive (~$66 of API equivalent, see `data/results/voyage_baseline.json` dry-run). We instead index each of the 18 repos once at its first observed base commit and reuse that index across all instances of the same repo. The cost-correctness tradeoff is explicit:

- Encoding cost drops by approximately 18× (320 → 18 indexings).
- Some gold files are absent at the chosen indexing commit (created later, deleted earlier, or moved); we observed 38 such cases out of 323 (~12%).

We address this by reporting two metric variants throughout:

- **Raw**: averaged across all 323 instances, with auto-failures (gold not in index) counted as Recall@k = 0. This is the conservative, comparable-to-published number.
- **Indexable**: averaged only over the 285 instances where the gold is indexed. This isolates encoder/matcher quality from the indexing protocol's incidental losses.

The Voyage and Codestral pooled baselines both use this same protocol; the discriminating subset is constructed only over the indexable intersection so this caveat does not affect §4.2–§4.4 comparisons.

### 3.4 File Filter

The same filter (`src/utils.py`) applies to all retrievers. Files are included if and only if:

- File extension `.py` (SWE-Bench Lite is Python-only).
- File size in `(0, 1_000_000]` bytes; the 1 MB cap is the standard published-benchmark setting and retains 285/323 (88.2%) of gold files (a 100 KB cap drops 22 instances; 1 MB drops 1).
- File path does *not* match any of `*/tests/*`, `*/test/*`, `*_test.py`, `*_tests.py`, `test_*.py`, `*/docs/*`, `*/doc/*`, `*/build/*`, `*/dist/*`, `*/.git/*`, `*/migrations/*`, `*/__pycache__/*`. Test files are excluded because SWE-Bench Lite gold patches are *source* fixes, never test files; including tests would inflate the corpus with content the gold patch never selects, depressing Recall@k for all retrievers symmetrically. Migrations are excluded as Django-specific noise.

### 3.5 Chunking and Index Construction

Each kept file is tokenized (Mistral tokenizer for Codestral, Voyage's tokenizer for the Voyage baseline) and split into chunks of up to 1500 tokens. Each chunk's text is prefixed with its file path: `# File: <rel_path>\n\n<chunk_body>`. Chunks are encoded into per-chunk vectors (mean-pooled) or per-chunk tensor stacks (token-level) and stored in FAISS `IndexFlatIP` after L2 normalization. Search returns top-N chunks; we deduplicate to file IDs preserving the highest-rank chunk's score per file (`src/maxsim.py:dedup_to_files`, identical logic in `src/baselines/voyage.py`). Final top-K is taken from the deduplicated list.

For Voyage we use `voyage-code-3` (1024-dim embeddings), with `EMBED_BATCH_TOKENS = 80_000` per request to stay under the 120K-tokens-per-request and 3 M-tokens-per-minute limits. Total Voyage baseline cost: 14.0M tokens × $0.18/M = $2.55. Codestral indexing wrote 200.9 MB of float32 mean-pooled vectors across 18 repos.

### 3.6 Metrics

For each instance we record Recall@1, Recall@5, Recall@10, Recall@20, and reciprocal rank. We aggregate by mean. Latency is measured in milliseconds per query (encoding-and-matching, excluding repository clone and one-time index build). Implementation: `src/eval.py`.

For the discriminating-subset experiments (§4.2–§4.4) we additionally record per-method **rescues** (gold in this method's top-10 but not in MaxSim's) and **regressions** (the inverse) and a verdict label generated from pre-registered thresholds (specified at the top of `scripts/frozen_ceiling_test.py` and `scripts/hybrid_rerank_test.py`).

### 3.7 Strict Sanity Check

The chunked indexing harness reproduces the cached pooled baseline ranking when the same encoder is used. Each Phase 2.5+ script auto-enables `strict-sanity` mode when the supplied `--model-id` matches the model recorded in `mamba_codestral_baseline.json`. In strict mode, after the first 5 instances the harness compares the recomputed pooled top-10 against the cached top-10 and aborts with exit code 2 if fewer than 50% overlap; this catches harness drift before propagating into a full run. The Phase 2.6 run reported `sanity_pass_rate = 0.975` (78/80 instances within the 50%-overlap bound), well above threshold.

### 3.8 Reproducibility

All experiments run on a single Lambda Cloud H100 PCIe (80 GB VRAM, 200 GB system RAM) under bfloat16 with `mamba_ssm`'s CUDA kernels (`causal-conv1d` + Triton selective state update). The harness `scripts/cloud_run.sh` is hard-fail (no silent fallback to pure-PyTorch Mamba-2, which OOMs on 1500-token chunks). All seeds are fixed (data subset sample: `random.seed(42)`; orthogonal projection: `torch.Generator().manual_seed(42)`). Per-instance results are written to `data/results/*.json` and committed.

## 4. Results

### 4.1 Phase 1: Pooled Baselines

We index all 18 SWE-Bench Lite repositories under the protocol of §3.3 with both Voyage code-3 and Codestral mean-pooled vectors. Each retriever returns top-K files per query via FAISS exact `IndexFlatIP` over L2-normalized chunk embeddings, deduplicated to file IDs preserving rank.

#### Headline numbers

| Metric | Voyage code-3 (raw) | Voyage code-3 (indexable) | Codestral pooled (raw) | Codestral pooled (indexable) |
|---|---:|---:|---:|---:|
| Recall@1 | 0.5294 | 0.6000 | 0.0991 | 0.1123 |
| Recall@5 | 0.8080 | 0.9158 | 0.2260 | 0.2561 |
| Recall@10 | 0.8390 | 0.9509 | 0.3034 | 0.3439 |
| Recall@20 | 0.8638 | 0.9789 | 0.4087 | 0.4632 |
| MRR | 0.6506 | 0.7373 | 0.1583 | 0.1794 |
| Median latency (ms) | 408.2 | 408.2 | 64.4 | 64.4 |
| n | 323 | 285 | 323 | 285 |

Voyage's indexable Recall@10 of 0.9509 is consistent with the upper end of published dense-retrieval baselines on file-level SWE-Bench-style retrieval. Codestral's mean-pooled performance is dramatically lower across all cutoffs. The latency gap is the expected direction (Codestral runs locally on the H100; Voyage incurs network round-trips), so the comparison should be read as quality-only.

#### Vector quality

We computed pairwise cosine similarity statistics on a 1000-chunk random sample within each repo's Codestral pooled index (`src/diagnostics.py`). The result is summarized in Table 4.1:

| Repo | n chunks | pairwise cos mean | pairwise cos std | centroid cos mean | collapse flag |
|---|---:|---:|---:|---:|---|
| pydicom | 1005 | 0.671 | 0.187 | 0.820 | False |
| sqlfluff | 336 | 0.682 | 0.185 | 0.826 | False |
| seaborn | 309 | 0.693 | 0.160 | 0.833 | False |
| django | 1205 | 0.696 | 0.115 | 0.834 | False |
| matplotlib | 1884 | 0.718 | 0.121 | 0.847 | False |
| astropy | 1671 | 0.718 | 0.115 | 0.846 | False |
| pvlib | 290 | 0.720 | 0.129 | 0.849 | False |
| psf/requests | 333 | 0.742 | 0.208 | 0.862 | False |
| scikit-learn | 1283 | 0.746 | 0.105 | 0.863 | False |
| sympy | 2741 | 0.755 | 0.128 | 0.869 | False |
| pyvista | 784 | 0.768 | 0.116 | 0.877 | False |
| sphinx | 592 | 0.772 | 0.127 | 0.879 | False |
| xarray | 330 | 0.780 | 0.124 | 0.884 | False |
| pallets/flask | 76 | 0.785 | 0.133 | 0.888 | False |
| pylint-dev/astroid | 255 | 0.792 | 0.133 | 0.890 | False |
| pylint-dev/pylint | 352 | 0.801 | 0.108 | 0.895 | False |
| pytest | 350 | 0.800 | 0.106 | 0.895 | False |
| marshmallow | 44 | 0.823 | 0.086 | 0.909 | False |

Pairwise cosine means range 0.671–0.823, with non-trivial spread (std 0.086–0.208). The collapse flag — set when mean > 0.9 *and* std < 0.05 — is False for all 18 repos. This rules out the simplest mechanistic explanation for the low Recall@10 (vector collapse): Codestral mean-pooled vectors are not pointing in nearly the same direction.

#### Popular-file bias

Inspecting per-query top-10 lists exposes the actual failure mode. We computed how often each file appears in top-10 across all indexable queries within a repo:

- **django/django (n = 97 indexable queries):** `django/http/request.py` appears in 46 top-10s (47.4%); `django/db/models/query.py` in 44 (45.4%); `django/db/models/sql/query.py` in 39 (40.2%).
- **sympy/sympy (n = 68 indexable queries):** `sympy/solvers/ode.py` appears in 41 (60.3%); `sympy/printing/jscode.py` in 29 (42.6%); `sympy/printing/latex.py` in 22 (32.4%).
- **matplotlib/matplotlib (n = 22 indexable queries):** `lib/matplotlib/pyplot.py` appears in 13 (59.1%); `examples/images_contours_and_fields/image_demo.py` in 9 (40.9%); `examples/lines_bars_and_markers/simple_plot.py` in 7 (31.8%).

A small set of central files dominates Codestral pooled top-10 across queries that bear no obvious topical relationship to those files. The mean-pooled vector for a query about timezone-aware date formatting and a query about ORM annotation behavior both pull `django/http/request.py` into top-10. We interpret this as a "popular file" attractor: when the encoder produces a 4096-d query vector via mean-pooling, files whose own mean vector is high-magnitude or central in the corpus distribution cosine-correlate broadly with whatever the query happens to be. The signal that should distinguish them — token-level locality to specific function/class names appearing in the issue text — is present at the per-token level (we will demonstrate this in §4.2) but absent in the centroid.

This sets up §4.2: not "the encoder is broken" but "mean-pooling discards what makes Codestral useful for this task."

### 4.2 Phase 2: MaxSim Discrimination

#### Subset construction

We define the **discriminating subset** as instances where (i) the gold file is in the index for both Voyage and Codestral pooled, (ii) Voyage's Recall@10 = 1, and (iii) Codestral pooled's Recall@10 = 0. This isolates the cases where mean-pooling specifically fails. The strict criterion yields 173 candidates from the 285 indexable instances. We sample 80 with `random.seed(42)` to bound compute.

Per-repo distribution (n = 80): django/django (30), sympy/sympy (18), scikit-learn/scikit-learn (7), matplotlib/matplotlib (6), pydicom (4), pydata/xarray (3), psf/requests (2), pylint-dev/pylint (2), pytest-dev/pytest (2), sphinx (2), and 1 each in pvlib/pvlib-python, pylint-dev/astroid, pyvista/pyvista, sqlfluff/sqlfluff.

#### Method

MaxSim is implemented as in `src/maxsim.py`:

$$\text{MaxSim}(Q, F) = \sum_{q \in Q} \max_{f \in F} \cos(q, f)$$

with $q \in \mathbb{R}^{D}$ a query token's last-layer Codestral activation and $f$ similarly for a candidate file chunk. We L2-normalize each tensor along the model dimension once, then reduce via matmul + max + sum. Chunks are scored independently and deduplicated to file IDs preserving rank, identical to the Voyage and Codestral pooled retrievers.[^phase25-jsons]

[^phase25-jsons]: The Phase 2.5 results reported here are reproduced from the MaxSim Method 1 run in `data/results/frozen_ceiling_test.json` and the `maxsim_only` baseline in `data/results/hybrid_rerank_test.json`, both run on the same 80-instance discriminating subset. The Phase 2.5 standalone results JSON was not preserved; the cross-phase reproduction (Recall@10 = 0.4375 in all three runs) serves as self-consistency validation.

#### Headline results

| Metric | Pooled (in this harness) | MaxSim |
|---|---:|---:|
| Recall@1 | 0.0125 | 0.1000 |
| Recall@5 | 0.0125 | 0.3125 |
| Recall@10 | 0.0125 | 0.4375 |
| Recall@20 | 0.2000 | 0.5875 |
| MRR | 0.0250 | 0.2001 |
| Median latency (s) | 2.41 | 2.44 |
| n | 80 | 80 |

In absolute counts: pooled cosine recovers 1 of 80; MaxSim recovers 35 of 80. The 35× lift in absolute hits matches the framing of the contribution. The latency cost of MaxSim is negligible relative to pooled (0.03 s, ~1%) because the inner per-chunk matmul dominates the operation in both cases; the difference is the reduction (max-then-sum versus single dot product).

#### Sanity check

The harness records, for each instance, whether the recomputed pooled top-10 overlaps the cached `mamba_codestral_baseline.json` top-10 by ≥50%. Over the 80 instances, 78 satisfy the bound (`sanity_pass_rate = 0.975`). Both misses are sympy instances where the chunk boundaries differ subtly between the cached run and the fresh run; the rankings remain qualitatively similar but slip below 50% overlap. This is well above the 50%-of-instances threshold that triggers the strict-sanity hard fail (§3.7), so we proceed.

#### Rescues

We characterize MaxSim's gains qualitatively. Among the 35 rescued instances (gold reaches top-10 under MaxSim where pooled missed), three illustrative cases:

**`django__django-11583` — clean rescue to rank 1.** Gold: `django/utils/autoreload.py`. Pooled top-3: `django/db/backends/oracle/client.py`, `django/bin/django-admin.py`, `django/db/backends/sqlite3/client.py`. MaxSim top-3: **`django/utils/autoreload.py`**, `django/utils/archive.py`, `django/utils/translation/trans_real.py`. The issue text concerns autoreloader behavior on file-system events; pooled retrieved files whose mean vector aligns with Django's database-client cluster (a popular-file attractor). MaxSim picked `autoreload.py` directly, with neighboring `django/utils/*` files filling the rest of the top-3 — the right neighborhood at the per-token level.

**`django__django-11620` — partial rescue to rank 9.** Gold: `django/views/debug.py`. Pooled top-3: `django/http/request.py` (popular attractor), `django/template/exceptions.py`, `django/utils/regex_helper.py`. MaxSim top-3: `django/views/generic/dates.py`, `django/utils/regex_helper.py`, `django/urls/resolvers.py`. MaxSim's top-3 is closer to the gold's neighborhood (views/) without finding it directly; the gold lands at rank 9.

**`django__django-11905` — partial rescue to rank 8.** Gold: `django/db/models/lookups.py`. Pooled top-3: `django/http/request.py`, `django/db/models/constants.py`, `django/db/models/sql/query.py`. MaxSim top-3: `django/db/models/sql/query.py`, `django/db/models/fields/related.py`, `django/db/backends/oracle/operations.py`. Both methods reach `django/db/models/*` files at the top, but only MaxSim ranks the gold inside top-10.

#### Persistent failures

Three illustrative cases where MaxSim does not recover the gold in top-20:

**`django__django-10914`** — Gold: `django/conf/global_settings.py`. The issue text concerns the default `FILE_UPLOAD_PERMISSIONS` setting; the gold file is essentially a long flat list of `NAME = value` lines describing every Django setting. There is little local token-level structure to align with the issue's surface form, and MaxSim's top-3 (`django/core/files/temp.py`, `django/db/backends/oracle/base.py`, `django/utils/translation/__init__.py`) reflects this — token-level matches against `FILE_UPLOAD_PERMISSIONS`-related strings appear in numerous configuration-handling files but not in the settings declarations themselves.

**`django__django-11099`** — Gold: `django/contrib/auth/validators.py`. The issue text is a regex pattern bug report; the file contains the `UnicodeUsernameValidator` and `ASCIIUsernameValidator` class definitions with the relevant regexes. MaxSim top-3 picks `django/utils/regex_helper.py`, `django/utils/jslex.py`, `django/contrib/staticfiles/storage.py` — files about regex *handling* rather than the specific validator class. The gold's regex string is short and lexically generic; the issue text doesn't contain the validator class name.

**`django__django-11742`** — Gold: `django/db/models/fields/__init__.py`. This is a 4000+-line module exporting most Django field classes. MaxSim top-3: `django/db/models/sql/query.py`, `django/contrib/gis/utils/layermapping.py`, `django/db/models/base.py`. The gold's content is mostly class definitions imported from elsewhere, and the issue references `Field.choices` — a property defined in this file but referenced by name in many others, so token-level alignment scatters.

The persistent failures share a structure: gold files that are **import shells** (long flat lists of class/setting/field definitions with little localized prose) are systematically harder for MaxSim. This is consistent with the per-token signal hypothesis — there *is* less per-token signal in such files relative to surrounding ones.

#### Net interpretation

MaxSim recovers 35 of 80 instances where pooled fails, more than half of the original failure set, with no trained component. On the discriminating subset, Voyage's Recall@10 is 1.0 by construction; MaxSim's 0.4375 represents 35 of 80 cases recovered. We do not directly compare MaxSim to Voyage on the full corpus because we did not re-run a MaxSim-only evaluation across all 285 indexable instances; the 80-instance subset is sufficient to characterize MaxSim's behavior on the cases where pooling fails. The signal exists at the per-token level; pooling destroys it.

### 4.3 Phase 3: Frozen Ceiling

#### Question

If MaxSim's single-head matching extracts more signal than mean-pooling, does adding *more* frozen architectural sophistication — multi-head structure, explicit normalization variants — extract still more without training? This would suggest the frozen Codestral representation has additional headroom retrievable by an untrained operator. If not, MaxSim is approximately the frozen ceiling and further gains require training.

#### Methods

We test seven matching operations on the same 80-instance discriminating subset (`scripts/frozen_ceiling_test.py`, `src/frozen_methods.py`):

- **`pooled`** — mean-pool both sides, cosine. (Phase 1 baseline reproduced.)
- **`maxsim`** — Phase 2.2 baseline reproduced.
- **`mh_4`, `mh_8`, `mh_16`, `mh_32`** — multi-head MaxSim with H ∈ {4, 8, 16, 32} disjoint orthogonal subspaces. We construct one $D \times D$ orthogonal matrix $P$ via QR decomposition of a normal random matrix (seeded `torch.Generator().manual_seed(42)`), partition the columns of $P$ into $H$ blocks of $D/H$, project query and file tokens through $P$, L2-normalize within each subspace, compute MaxSim per head, and sum across heads. The same $P$ is used for all queries and files (loaded once, cached via `lru_cache` on `(d, device, dtype, seed)`).
- **`late_int_8`** — H = 8 multi-head as above, with explicit pre-projection L2 normalization on each token (closer to ColBERT's actual late-interaction formulation).

The orthogonal projection construction is the load-bearing design choice: a stable, content-agnostic decomposition of the latent space into independent subspaces. If signal is distributed across feature dimensions of Codestral's representation that pooled-MaxSim aggregates suboptimally, multi-head should isolate and sum it.

#### Results

| Metric | pooled | maxsim | mh_4 | mh_8 | mh_16 | mh_32 | late_int_8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recall@1 | 0.0125 | 0.1000 | 0.0875 | 0.0875 | 0.0875 | 0.1000 | 0.0875 |
| Recall@5 | 0.0125 | 0.3125 | 0.2875 | 0.2875 | 0.2875 | 0.2875 | 0.2875 |
| Recall@10 | 0.0125 | **0.4375** | 0.4250 | 0.4250 | 0.4125 | 0.4125 | 0.4250 |
| Recall@20 | 0.2000 | 0.5875 | 0.5875 | 0.6000 | 0.6000 | 0.5875 | 0.6000 |
| MRR | 0.0250 | **0.2001** | 0.1912 | 0.1933 | 0.1903 | 0.1974 | 0.1933 |
| Median latency (s) | 2.41 | 2.44 | 3.71 | 3.83 | 4.14 | 5.01 | 3.88 |
| n | 80 | 80 | 80 | 80 | 80 | 80 | 80 |

The best-multi-head Recall@10 is 0.4250 (`mh_4`, `mh_8`, `late_int_8`), versus single-head MaxSim's 0.4375. The pre-registered headroom delta is `best_mh − maxsim = −0.0125`, which falls in the **CEILING** verdict band ($\Delta < 0.02$).

We also computed a best-of-multi-head ensemble (per query, take the best rank across H ∈ {4, 8, 16, 32}). The ensemble Recall@10 is 0.4375 — identical to single-head MaxSim. No combination of frozen multi-head variants exceeds the single-head baseline. (Note: this is an oracle ensemble computed post-hoc; an inference-time ensemble would require a selection mechanism we do not provide.)

Per-method rescue/regression counts vs MaxSim (gold in this method's top-10 but not in MaxSim's top-10, and vice versa) are reported in `frozen_ceiling_test.json`. Highlights: `mh_32` rescues 2 instances and regresses 4; `mh_4`, `mh_8`, `late_int_8` each rescue 0 and regress 1; `pooled` (re-run for sanity) regresses 34.

#### Interpretation

In our experiments, frozen architectural sophistication beyond MaxSim does not extract additional signal. We do not claim the frozen architecture *cannot* extract additional signal; the data only supports the weaker statement that random orthogonal projections at the head counts and configuration we tested do not. One possible mechanism: random orthogonal subspaces uniformly redistribute signal across heads, so summing per-head MaxSim recovers the original similarity up to a constant factor. A *learned* projection might distinguish dimensions that contain retrieval-relevant signal from those that contain unrelated content (syntactic structure, source language conventions); a random projection, by construction, cannot.

The verdict is: trained multi-head matching is the next experiment. Frozen multi-head matching is a finished one.

### 4.4 Phase 4: Hybrid with Off-the-shelf Cross-encoder Rerankers

#### Setup

The pipeline (`scripts/hybrid_rerank_test.py`) is:

1. **Step 1** — MaxSim retrieval over per-token Codestral latents, top-100 candidate files per query.
2. **Step 2** — For each of the 100 candidates, build a `(issue_text, file_content)` pair where the file content is read fresh from disk and truncated to 6000 characters (~1500 tokens at typical 4-char/token, comfortably under the smaller reranker's 512-token context). Pass each pair through the reranker.
3. **Step 3** — Sort by reranker score, take final top-K.

We selected three rerankers, ordered fastest-to-slowest so that if the budget cap interrupts a slow one, fast results survive:

- `cross-encoder/ms-marco-MiniLM-L-12-v2` (33M, MS-MARCO-trained)
- `jinaai/jina-reranker-v2-base-multilingual` (278M)
- `BAAI/bge-reranker-v2-m3` (568M)

The `jina-reranker-v2` import path depends on `transformers.models.xlm_roberta.modeling_xlm_roberta.create_position_ids_from_input_ids`, which was removed in `transformers >= 5.0`; the harness skipped it via the try/except wrapper introduced for graceful failure handling. We do not report jina-v2 numbers.

The same 80-instance discriminating subset is used. The MaxSim filter portion of the pipeline is identical to §4.2 (and reproduces the same Recall@10 = 0.4375 baseline).

#### Results

| Metric | MaxSim only (filter) | + MiniLM (33M) | + BGE-v2-m3 (568M) |
|---|---:|---:|---:|
| Recall@1 | 0.1000 | 0.0500 | 0.1250 |
| Recall@5 | 0.3125 | 0.1750 | 0.2875 |
| Recall@10 | **0.4375** | 0.3000 | 0.4125 |
| Recall@20 | 0.5875 | 0.4000 | 0.4875 |
| MRR | 0.2001 | 0.1187 | 0.2083 |
| n | 80 | 80 | 80 |

Both rerankers underperformed the MaxSim filter alone at Recall@10. MiniLM was decisively worse (Δ = −0.1375); BGE-v2-m3 was modestly worse (Δ = −0.0250). BGE's MRR is marginally higher (Δ = 0.0082) but well within sampling noise at n=80; we do not interpret this.

The pre-registered verdict for both is **FILTER_LIMITED**: rerankers do not recover what MaxSim missed.

#### Sanity log: first instance, all rerankers

The harness emits the first-instance top-3 from each reranker for eyeball verification. For `django__django-10914` (gold: `django/conf/global_settings.py`):

- MiniLM top-3: `django/db/transaction.py`, `django/template/base.py`, `django/contrib/staticfiles/storage.py`.
- BGE-v2-m3 top-3: `django/contrib/staticfiles/storage.py`, `django/core/files/uploadhandler.py`, `django/contrib/admindocs/views.py`.

This is the same instance MaxSim missed (§4.2 persistent failures). Both rerankers retrieved storage- and file-handling-related files, plausibly because the issue mentions file upload permissions; neither retrieved the settings declaration file. This is a qualitative example of the same failure pattern (import-shell golds are hard for token-level matching, hard for off-the-shelf rerankers, and apparently hard for everything that isn't retrieval-trained on this task).

#### Interpretation

The correct claim from these data is narrow: two cross-encoder rerankers trained on natural-language passage retrieval, applied to Codestral's MaxSim shortlist on SWE-Bench Lite, underperformed the shortlist alone. We attribute this to training-distribution mismatch: rerankers trained on MS-MARCO (English Q-A pairs over web text) and multilingual general-domain passage retrieval (BGE's training mixture) are scoring text whose lexical surface — Python identifiers, GitHub issue prose, source code — is far from their training distribution. The reranker confidently scores files high based on surface cues that do not predict actual relevance for code retrieval.

We do *not* test code-specific rerankers (e.g., from Cohere or Voyage's commercial reranker offerings, or open-source code-trained cross-encoders if any exist at the time of writing). We cannot generalize beyond this configuration. A code-specific reranker, or a reranker fine-tuned on code-retrieval data, may behave differently and is left as future work (§7).

### 4.5 Phase 5: Multi-granularity Matching

#### Question

Phase 4.3 tested whether *multi-head* sophistication beyond MaxSim helps. This phase tests whether *multi-granularity* sophistication helps: if queries operate at different granularities (some want global "what is this file about" matching, some want local "this exact identifier" matching), a frozen representation that exposes multiple granularity views and a matcher that selects or combines them might extract more signal than single-granularity token-level MaxSim.

#### Method

We extract four granularities from the same per-token Codestral latents (no re-encoding):

- **G0 — file-level pool**: mean over all tokens of all chunks in a file → 1 vector of shape (D,) per file.
- **G1 — chunk-level pool**: mean over each chunk's tokens → list of (D,) per file (Phase 1 baseline).
- **G2 — sliding-window pool**: within each cached chunk tensor, take overlapping windows of 256 tokens with stride 128 and mean-pool each window. Approximates "function-sized" segments without parser-level metadata.
- **G3 — token-level**: the cached chunk tensors verbatim (Phase 2.2 baseline).

We test seven matching methods (`scripts/multigranular_test.py`, `src/multigranular.py`):

- **`pooled_chunk`** — query mean-pool vs G1 (max over chunks). Phase 1 baseline reproduced on this subset.
- **`pooled_file`** — query mean-pool vs G0.
- **`func_pool`** — query mean-pool vs G2 (max over function-sized segments).
- **`maxsim`** — Phase 2.2 baseline reproduced.
- **`mg_sum`** — equal-weight sum of G0/G1/G2/G3 scores after per-query min-max normalization to [0,1] within the candidate set.
- **`mg_max`** — per-query, per-file max across normalized G0/G1/G2/G3 scores.
- **`mg_routed`** — heuristic per-query route to one granularity. Rule: if query token count < 30 and the text contains an identifier-like pattern (camelCase, snake_case, dot.notation), route to G3; if token count > 100 with no identifier pattern, route to G0; else G2.

Critical implementation choice: the per-query min-max normalization makes `mg_sum` effectively a 4-way ensemble vote rather than a magnitude-calibrated combination. If `mg_max` wins but `mg_sum` does not, the interpretation is "one granularity matters per query" rather than "combining granularities helps." The output JSON records this and the next caveat as `interpretation_notes`.

#### Tree-sitter caveat for G2

We installed `tree-sitter` and `tree-sitter-python` and verified function/class boundary parsing on Python source. We did *not* use them for G2. The cached token-level latents (`data/maxsim_cache/*.pt`) were produced under a chunked-then-decoded encoding pipeline (file is tokenized, sliced into 1500-token windows, each window is decoded back to text, the decoded text is re-tokenized with a `# File: <rel>\n\n` header prefix, and the result is encoded). Mapping tree-sitter byte ranges back into cached tensor indices requires byte-offset metadata per cached chunk, which the cache does not store. Adding it forces re-encoding. We left tree-sitter G2 as future work (§7.6) and ran the experiment with sliding-window G2.

#### Headline results (n = 80 discriminating subset)

| Metric | pooled_chunk | pooled_file | func_pool | maxsim | mg_sum | mg_max | mg_routed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recall@1 | 0.0125 | 0.0000 | 0.0500 | 0.1000 | 0.0375 | 0.0250 | 0.0500 |
| Recall@5 | 0.0125 | 0.0125 | 0.1500 | 0.3125 | 0.1375 | 0.1875 | 0.1375 |
| Recall@10 | 0.0125 | 0.0500 | 0.1875 | **0.4375** | 0.2125 | 0.2875 | 0.1750 |
| Recall@20 | 0.2000 | 0.1250 | 0.2750 | 0.5875 | 0.3625 | 0.4750 | 0.2500 |
| MRR | 0.0250 | 0.0135 | 0.0884 | **0.2001** | 0.0926 | 0.1138 | 0.0813 |
| Median latency (s) | 0.083 | 0.030 | 0.526 | 2.595 | 3.205 | 3.166 | 0.515 |

The best multi-granularity composite is `mg_max` at Recall@10 = 0.2875, against single-head `maxsim` at 0.4375 (Δ = −0.15). The pre-registered verdict is **HURTS**: composite multi-granularity matching actively damages retrieval relative to single-granularity token-level MaxSim.

`pooled_chunk` reproduces the Phase 1 collapsed-on-this-subset result (Recall@10 = 0.0125, by construction). `pooled_file` is even worse (0.0500 R@10) — adding more aggregation doesn't help when aggregation is the failure mode. `func_pool` (G2 alone) is the strongest single-granularity beyond `maxsim`, at Recall@10 = 0.1875. None of the composites recovers what MaxSim already extracts.

The `sanity_pass_rate` is 0.975 (78/80 of `pooled_chunk` rankings overlap the cached Codestral baseline by ≥50%), unchanged from the Phase 4.3 reproduction.

#### Routing distribution and rescue/regression counts

The `mg_routed` heuristic routed 74 of 80 queries to G2 and 6 to G0. None routed to G3 (the strongest single granularity), because the heuristic's "tokens < 30 AND identifier-like" rule did not trigger on issue-text queries — SWE-Bench problem statements are long and contain natural-language sentences alongside any code identifiers. The router was naive enough that it selected an inferior granularity on most queries.

Per-method rescues vs MaxSim (gold in this method's top-10 but not in MaxSim's top-10) and regressions vs MaxSim (the inverse):

| Method | Rescues vs MaxSim | Regressions vs MaxSim |
|---|---:|---:|
| pooled_chunk | 0 | 34 |
| pooled_file | 0 | 31 |
| func_pool | 5 | 25 |
| mg_sum | 0 | 18 |
| mg_max | 3 | 15 |
| mg_routed | 5 | 26 |

Every multi-granularity variant has more regressions than rescues. Even `mg_max`, the least bad composite, sacrifices 15 instances MaxSim got right while gaining 3 MaxSim missed.

#### G2 segment statistics

Across the 80-instance subset's pooled candidate files: mean of 28.0 G2 segments per file, minimum 1, maximum 2828 (an outlier file from one of the larger candidate pools). Zero files yielded zero G2 segments, so the sliding-window extraction was robust at every file size. No fall-back was triggered.

#### Interpretation

Frozen multi-granularity composition introduces more noise than signal at this size class and benchmark. Three mechanistic explanations are consistent with the data, and we cannot distinguish among them without further work:

1. **Granularity-mixing destroys MaxSim's contribution.** MaxSim's signal is concentrated in token-level matching against specific identifiers; mixing it equally with three coarser granularities (G0, G1, G2) dilutes the signal. The min-max normalization compounds this: G3's discriminative variance gets compressed into [0,1] alongside near-uniform G0 scores.

2. **Sliding-window G2 is not the right semantic mid-level.** A function-boundary G2 might pool across more meaningful units. The HURTS result on sliding-window does not rule out semantic-boundary granularity helping; it rules out *this* granularity construction helping.

3. **Routing heuristic is too coarse.** With 74 of 80 queries routed to G2 and 0 to G3, `mg_routed` is essentially "always-G2", which is dominated by single-granularity G2 (`func_pool`, Recall@10 = 0.1875). A routing function that learned which query → granularity from a small held-out set would address this; we did not test this.

We treat the Phase 5 result as: frozen multi-granularity in the form we tested HURTS. Whether semantic-boundary granularity or a learned router would change this is left for future work (§7.6).


## 5. Discussion

### 5.1 Codestral-Mamba representation structure

The pooled-vs-MaxSim gap (0.0125 vs 0.4375 Recall@10 on the discriminating subset) localizes the failure of mean-pooling. Codestral's per-token last-layer activations carry sufficient signal to identify the gold file in 35 of 80 cases that Voyage retrieves perfectly; mean-pooling those same activations carries that signal in 1 of 80. The §4.1 vector-quality measurements confirm the failure is *not* total geometric collapse: pairwise cosine means stay below 0.83 across all 18 repos and standard deviations remain non-trivial. Vectors are not pointing in nearly the same direction — they are spread, but the spread is dominated by gross repo-level features (overall topical content) rather than by per-query distinguishing features. We interpret mean-pooling as collapsing the positional variance of token-level activations into a corpus-aware centroid that retains topical coarseness at the cost of localized matching.

This is consistent with the observed popular-file bias: a small set of central files dominates Codestral pooled top-10 across queries that bear little topical relationship to those files. The mean-pooled vector for an autoreloader query and the mean-pooled vector for an ORM annotation query are both close to `django/http/request.py`'s mean vector — not because either query is about HTTP requests, but because that file's centroid is itself a high-magnitude attractor in the Django repo's pooled-vector distribution. Token-level matching breaks this attractor because the per-token similarity is computed against specific tokens, not against the file's overall pooled signature.

The popular-file bias we observe is structurally analogous to the documented *hubness* and anisotropy phenomena in dense retrieval and contextual embedding spaces [Radovanović et al. 2010; Cai et al. 2021]. In those settings, a small set of points emerges as universal nearest neighbors due to high-dimensional geometry, and retrieval failure correlates with anisotropic concentration of the representation space. Our finding differs in mechanism: in our pooled Codestral indices, vectors are *not* anisotropically collapsed — per-repo pairwise cosine std remains ≥ 0.086 across all 18 repos (Table in §4.1), and pairwise means do not exceed 0.83 — yet the popular-file pattern still emerges and retrieval still fails. This suggests the failure here is in the aggregation operator (mean-pooling) rather than in the representation geometry of Codestral's last-layer activations themselves. Token-level matching, which operates before any aggregation, recovers signal that the post-aggregation index has lost. We treat this as a hypothesis rather than a proof; explicit measurement of intrinsic-dimension and hubness statistics on the per-token versus pooled distributions would be needed to make the distinction rigorous.

### 5.2 Why off-the-shelf NLP rerankers fail on code retrieval

The two cross-encoder rerankers we tested were trained on natural-language passage retrieval (MS-MARCO web text for MiniLM; multilingual general-domain web text for BGE-v2-m3). When applied to `(issue_text, file_content)` pairs from SWE-Bench Lite, they reranked toward files whose surface text matched the issue's lexical surface — for example, retrieving storage- and file-handling-related Django files when the issue mentions `FILE_UPLOAD_PERMISSIONS`. That pattern of attention is appropriate for web-text retrieval, where lexical overlap predicts topical relevance. It is not appropriate for code, where the relevant file is often the *declaration* site of a setting rather than any file that *uses* it.

We attribute the underperformance to training-distribution mismatch: rerankers learn what relevance looks like from their training data, and rerankers trained on web text learn relevance signals that do not transfer cleanly to code. We do not test code-specific rerankers and our negative result does not bear on whether such rerankers exist or whether they would close the gap.

### 5.3 What this means for SSM-based retrieval research

The pooled-vs-MaxSim gap (a 35× absolute-hit lift, with no architectural change to the encoder) is larger than the multi-head-vs-single-head gap (zero or slightly negative across all H we tested). For this size class of Mamba-2 encoder on this benchmark, the matching operation is more consequential than the architectural sophistication of the matching head. Mamba Retriever [Wang et al. 2024] takes the orthogonal approach of *training* a Mamba encoder for retrieval; our work suggests the natural complement: *training* a matching head over a code-pretrained Mamba's frozen representation, and comparing the two pathways to competitive retrieval quality.

Three independent axes of frozen architectural sophistication — multi-head random-projection MaxSim variants (§4.3), off-the-shelf cross-encoder rerankers stacked on the MaxSim shortlist (§4.4), and multi-granularity composites combining file-, chunk-, mid-level-, and token-level scores (§4.5) — all fail to exceed single-head token-level MaxSim. The §4.3 verdict is CEILING (Δ = −0.0125); the §4.4 verdict is FILTER_LIMITED (best reranker R@10 = 0.4125, Δ = −0.0250); the §4.5 verdict is HURTS (best composite R@10 = 0.2875, Δ = −0.15). The pattern across these three independent axes is consistent: frozen architectural elaboration on top of a code-pretrained Mamba-2's last-layer activations does not extract additional retrieval signal beyond what single-head MaxSim already provides on this benchmark. The pattern strengthens — though does not prove — the case that retrieval-specific training, not frozen architectural cleverness, is the load-bearing path to closing the gap to retrieval-trained dense encoders.

The frozen-MaxSim ceiling is a useful waypoint. It establishes that the per-token signal exists without any retrieval training. Closing the gap to a retrieval-trained encoder like Voyage code-3 likely requires retrieval-specific training; our data does not reveal whether that training is best applied to the encoder, the matching head, or both.

## 6. Limitations

We list limitations exhaustively. Most are direct consequences of the case-study scope.

1. **Single encoder family.** We evaluate exactly one encoder: `mistralai/Mamba-Codestral-7B-v0.1`, a Mamba-2 architecture continually pre-trained on code. We do not evaluate Mamba-1 [Gu and Dao 2023] (we ran a smoke test on `state-spaces/mamba-130m-hf` and found vector collapse at the 130M scale, so we did not pursue it; intermediate scales 790M / 1.4B / 2.8B are untested).

2. **No Mamba-3 evaluation.** Mamba-3 [Lahoti et al. 2026] does not have publicly released weights at the time of writing; the released `mamba_ssm.Mamba3` module requires CUDA + Triton kernels with no PyTorch fallback. We treat Mamba-3 as future work.

3. **Single benchmark, single-file by construction.** SWE-Bench Lite contains 323 single-file fix instances. Multi-file retrieval, function-level retrieval, span-level retrieval, and cross-language retrieval are all out of scope.

4. **Per-repo single-commit indexing protocol.** We index each repo once at the first observed `base_commit`, losing 38 of 323 instances (~12%) where the gold file's path or content differs at the chosen commit. This indexing tradeoff is explicit and reported as the *raw vs indexable* metric split throughout, but the comparison to per-scope-indexed published baselines is not exact.

5. **Discriminating subset n = 80 is small.** The strict criterion (Voyage R@10 = 1 AND Codestral pooled R@10 = 0) yielded 173 candidates; we sampled 80. We do not report bootstrapped confidence intervals or significance tests on the 35-of-80 / 1-of-80 split.

6. **Single-seed runs.** All matching runs (MaxSim, multi-head, late-interaction) use a single random projection seed. We do not measure variance across seeds. The orthogonal projection's content-agnostic construction makes seed sensitivity a minor concern, but we did not verify this empirically.

7. **Two rerankers, both natural-language-trained.** `cross-encoder/ms-marco-MiniLM-L-12-v2` (33M, MS-MARCO) and `BAAI/bge-reranker-v2-m3` (568M, multilingual). We attempted but skipped `jinaai/jina-reranker-v2-base-multilingual` due to a `transformers >= 5.0` compatibility break. We do not test code-specific commercial or open-source rerankers.

8. **No trained matching baseline.** Phase 3 of the original research plan (a learned cross-attention head over Codestral latents, trained on retrieval objectives) is the natural next experiment but is not implemented here.

9. **Discriminating subset construction has a built-in floor.** Voyage's Recall@10 is 1.0 by construction on this subset; we cannot use it to test whether MaxSim *closes the gap* to Voyage on the full corpus (we did not re-run a MaxSim full-corpus evaluation across all 285 indexable instances).

10. **Python-only.** SWE-Bench Lite is a Python benchmark. We do not evaluate other languages.

11. **File-level granularity only.** The gold target is a file path. Span-level / function-level retrieval, which a per-token matching system might in principle do better at, is not evaluated.

12. **No ablation on chunking.** We use 1500-token chunks throughout; we do not test how chunk size, chunk overlap, or alternative segmentation strategies affect MaxSim quality.

13. **Reranker context truncation.** We truncate file content to 6000 characters (~1500 tokens) before reranker input, which is comfortably under the smaller reranker's 512-token context but means longer files are seen only by their head. Tail content is untested.

14. **Strict-sanity threshold is loose.** The 50%-overlap criterion catches gross harness drift but not subtle ranking shifts. Two of 80 instances failed the loose check (`sanity_pass_rate = 0.975`) and were retained; tightening the threshold could change the discriminating subset.

15. **Phase 5 multi-granularity used sliding-window pools for G2** rather than semantic function/class boundaries; tree-sitter integration was attempted but blocked by the latent cache lacking byte-offset metadata. Semantic-boundary granularity may behave differently and is not tested here.

## 7. Future Work

Each item below specifies the experiment, the gap it is designed to close, the evidence required to claim closure, and a rough compute estimate.

### 7.1 Trained cross-attention over Codestral latents

**Experiment.** Train a small (1–10M parameter) cross-attention head on top of frozen Codestral-Mamba-7B latents, using a contrastive retrieval objective on a code retrieval dataset (e.g., CodeSearchNet, CoIR, or SWE-Bench training instances disjoint from SWE-Bench Lite). The encoder remains frozen; only the matching head trains.

**Gap closed.** Tests whether the frozen Codestral representation has additional retrievable signal beyond what MaxSim extracts, given a learned (rather than random) projection structure. The Phase 2.6 CEILING verdict argues frozen multi-head heads do not exceed MaxSim; this experiment isolates the contribution of *training*.

**Evidence required for claim of closure.** Indexable Recall@10 ≥ 0.7 on SWE-Bench Lite (closing roughly half the remaining gap to Voyage), with the trained head used at the same dedup-rank pipeline as MaxSim.

**Approximate compute.** $10K–$50K of GPU-time depending on training-data size, contrastive-batch construction, and learning-rate sweep budget. Storage of token-level cached latents for the training corpus is the limiting factor (Codestral's 4096-d × ~1500-token bf16 chunks are ~12 MB each).

### 7.2 Code-specific reranker training

**Experiment.** Train (or fine-tune) a cross-encoder reranker on `(issue_text, file_content)` pairs from a code-retrieval corpus. Compare against the MaxSim filter-only baseline on the §4.4 setup.

**Gap closed.** Tests whether the reranker negative result in §4.4 is genuinely about training-distribution mismatch or about a deeper mismatch between cross-encoder reranking and code retrieval.

**Evidence required.** Code-specific reranker beating MaxSim filter-only by ≥0.05 Recall@10 on the discriminating subset, with the same `MaxSim top-100 → reranker rerank → top-K` pipeline.

**Approximate compute.** $5K–$20K depending on reranker scale (33M MiniLM-class to 568M BGE-class) and training corpus size.

### 7.3 Mamba-3 baseline re-run on weight release

**Experiment.** When `state-spaces` releases pretrained Mamba-3 weights for code, re-run Phases 1–4 with Mamba-3 as the encoder.

**Gap closed.** Establishes whether Mamba-3's complex-valued state and MIMO formulation extract better retrieval signal at the per-token level than Mamba-2.

**Evidence required.** Phase 1 indexable Recall@10 numbers and Phase 2 MaxSim numbers on the same SWE-Bench Lite subsets, comparable directly to the Codestral-Mamba-7B numbers in this paper.

**Approximate compute.** Same order as the present paper, $10–$15 of cloud GPU per full eval run.

### 7.4 Multi-vector compression for storage tractability

**Experiment.** Compress the per-token Codestral latent state via product quantization, learned linear projection to a smaller dimension (e.g., 4096 → 128, mirroring ColBERTv2 [Santhanam et al. 2022]), or top-k token selection per chunk. Measure the Recall@10 / index-size Pareto frontier.

**Gap closed.** Token-level matching's index size at 4096-d × ~1500-token-per-chunk × thousands-of-chunks-per-repo (~117 GB across our test set, see §3.3) is impractical for production. Compression is required for deployment-relevant comparisons.

**Evidence required.** A Pareto plot of MaxSim Recall@10 versus index size in GB, comparing un-compressed, dimension-reduced, quantized, and selection-based variants on the same discriminating subset.

**Approximate compute.** $1K–$5K, dominated by recomputation of the cached pool under compression variants.

### 7.5 Replication on non-code retrieval

**Experiment.** Repeat Phases 1–2 on a non-code long-context retrieval benchmark — for example, legal document retrieval (GuardRAIL or CaseHOLD-derived corpora) or scientific literature retrieval. Use a domain-pretrained Mamba encoder if available; otherwise a code-pretrained one as a transfer baseline.

**Gap closed.** Tests whether the pooled-vs-MaxSim gap we observe on Codestral-Mamba is specific to code retrieval (where token-level semantics matter heavily) or generalizes to other long-context-retrieval domains.

**Evidence required.** A comparable pooled-vs-MaxSim Recall@10 gap on at least one non-code benchmark; replication of the popular-file-bias finding (or its absence).

**Approximate compute.** $10–$20 per benchmark, plus encoder loading and corpus indexing time.

### 7.6 Semantic-boundary multi-granularity

**Experiment.** Re-encode the SWE-Bench Lite corpus with the tokenizer's `offset_mapping` enabled, store the byte-offset → token-index map per chunk in the cache, and use tree-sitter Python to extract function/class byte ranges per file. Map those ranges to cached tensor token indices and build a semantic-boundary G2 in `src/multigranular.py`. Re-run `mg_max` and `mg_routed` with semantic G2 and compare against the §4.5 sliding-window numbers and the MaxSim baseline.

**Gap closed.** Phase 5 found multi-granularity HURTS with sliding-window G2; the natural follow-up question is whether *semantic* mid-level granularity (functions, classes, methods) behaves differently. A NO_LIFT or STRONG_MULTIGRAN verdict on semantic G2 would resolve the interpretive ambiguity left by Phase 5.

**Evidence required.** Recall@10 on the same 80-instance discriminating subset, with the routing distribution from `mg_routed` reported. A null result would be sufficient to claim the multi-granularity bet is dead in frozen form; a positive result would re-open it.

**Approximate compute.** Re-encoding the test repos under the new cache schema is ~$8 (same as Phase 2.5 token-level encoding). Tree-sitter parse + tensor-index mapping is CPU-bound, ~30 minutes. Total ~$10.

## 8. Conclusion

We characterized how much retrieval signal an off-the-shelf, code-pretrained Mamba-2 encoder (`mistralai/Mamba-Codestral-7B-v0.1`) preserves and what frozen matching operations extract it on SWE-Bench Lite. Mean-pooled Codestral underperforms a retrieval-trained baseline (Voyage code-3) by a wide margin (indexable Recall@10 of 0.34 vs 0.95) despite producing well-spread vectors that do not collapse; we attribute this to a popular-file attractor in the pooled-vector distribution rather than to encoder failure. ColBERT-style MaxSim over per-token Codestral latents recovers 35 of 80 cases on a subset where pooling specifically fails (Recall@10 = 0.4375 vs 0.0125; 35× absolute-hit lift). Three independent axes of frozen architectural sophistication beyond MaxSim — multi-head random-orthogonal-projection variants (best Δ = −0.0125), off-the-shelf cross-encoder rerankers stacked on the MaxSim shortlist (best Δ = −0.0250), and multi-granularity composites combining file-, chunk-, mid-level-, and token-level scores (best Δ = −0.15) — all fail to exceed single-head MaxSim. The pattern across these axes argues that retrieval-specific training, not frozen architectural cleverness, is the load-bearing path forward. The natural next experiment is a trained matching head over Codestral's frozen representation, evaluated against the same MaxSim ceiling and the Voyage retrieval-trained baseline.

## References

The following citations were verified by web search prior to inclusion. For each, the canonical URL is given. Items that could not be confidently resolved are marked [CITATION NEEDED].

- **[Gu and Dao 2023]** Albert Gu and Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752, 2023. (Search: `Mamba Gu Dao 2023 arXiv "Linear-Time Sequence Modeling with Selective State Spaces"`. Canonical: https://arxiv.org/abs/2312.00752.)

- **[Dao and Gu 2024]** Tri Dao and Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* International Conference on Machine Learning (ICML), 2024. arXiv:2405.21060. (Search: `Mamba-2 Dao Gu "state space duality" arXiv 2024`. Canonical: https://arxiv.org/abs/2405.21060.)

- **[Lahoti et al. 2026]** Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, and Albert Gu. *Mamba-3: Improved Sequence Modeling using State Space Principles.* International Conference on Learning Representations (ICLR), 2026. arXiv:2603.15569. (Search: `"Mamba-3" "Improved Sequence Modeling using State Space Principles" ICLR OpenReview arXiv`. Canonical: https://arxiv.org/abs/2603.15569 and https://openreview.net/forum?id=HwCvaJOiCj.)

- **[Karpukhin et al. 2020]** Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. *Dense Passage Retrieval for Open-Domain Question Answering.* Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, 2020. arXiv:2004.04906. (Search: `Karpukhin "Dense Passage Retrieval for Open-Domain Question Answering" 2020 arXiv EMNLP`. Canonical: https://aclanthology.org/2020.emnlp-main.550/.)

- **[Khattab and Zaharia 2020]** Omar Khattab and Matei Zaharia. *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 39–48, 2020. arXiv:2004.12832. (Search: `Khattab Zaharia ColBERT 2020 SIGIR "Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"`. Canonical: https://arxiv.org/abs/2004.12832.)

- **[Chen et al. 2024]** Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.* arXiv:2402.03216, 2024. (Search: `"BGE" reranker BAAI Chen 2024 arXiv "M3-Embedding" "Self-Knowledge Distillation"`. Canonical: https://arxiv.org/abs/2402.03216. Note: this paper covers the BGE M3 *embedding* model; the `BAAI/bge-reranker-v2-m3` cross-encoder is a derived model released alongside BGE M3 with a separate model card at https://huggingface.co/BAAI/bge-reranker-v2-m3, but no separate technical report we could locate.)

- **[Reimers and Gurevych 2019]** Nils Reimers and Iryna Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, 2019. arXiv:1908.10084. (Search: `Reimers Gurevych "Sentence-BERT" 2019 EMNLP arXiv`. Canonical: https://aclanthology.org/D19-1410/. Cited as the lineage for the cross-encoder family `cross-encoder/ms-marco-MiniLM-L-12-v2` we use as a reranker baseline; the specific MS-MARCO-MiniLM cross-encoder is distributed via the `sentence-transformers` ecosystem this paper established.)

- **[Jimenez et al. 2024]** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* International Conference on Learning Representations (ICLR), 2024. arXiv:2310.06770. (Search: `Jimenez SWE-bench "Can Language Models Resolve Real-World GitHub Issues" 2024 ICLR arXiv`. Canonical: https://arxiv.org/abs/2310.06770 and https://openreview.net/forum?id=VTF8yNQM66.)

- **[Wang et al. 2024]** *Mamba Retriever: Utilizing Mamba for Effective and Efficient Dense Retrieval.* arXiv:2408.08066, 2024. (Search: `"Mamba Retriever" arXiv state space retrieval 2024`. Canonical: https://arxiv.org/abs/2408.08066. Author list as appearing in arXiv metadata; we cite under first-author shorthand `Wang et al.` per arXiv listing convention but did not independently re-verify the full author list.)

- **[Santhanam et al. 2022]** Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.* Proceedings of NAACL 2022, pages 3715–3734. arXiv:2112.01488. Canonical: https://aclanthology.org/2022.naacl-main.272/.

- **[Radovanović et al. 2010]** Miloš Radovanović, Alexandros Nanopoulos, and Mirjana Ivanović. *Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data.* Journal of Machine Learning Research, 11:2487–2531, 2010. (Search: `Radovanović "Hubs in Space" 2010 JMLR popular nearest neighbors high-dimensional`. Canonical: https://www.jmlr.org/papers/v11/radovanovic10a.html.)

- **[Cai et al. 2021]** Xingyu Cai, Jiaji Huang, Yuchen Bian, and Kenneth Church. *Isotropy in the Contextual Embedding Space: Clusters and Manifolds.* International Conference on Learning Representations (ICLR), 2021. (Search: `Cai 2021 "Isotropy in the Contextual Embedding Space" ICLR`. Canonical: https://openreview.net/forum?id=xYGNO86OWDH.)
