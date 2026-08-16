# Design Decisions

For every major choice below: what it is, what else was considered, and why
this one won. Written to be re-read and explained from memory, not just
submitted.

## Chunking Strategy

**Chosen**: two strategies are implemented and both are switchable via
config — fixed-size chunking (220 words, 40-word overlap) as the default,
and embedding-similarity-based semantic chunking as an alternative.

**Alternatives considered**: (1) fixed-size only — simplest, fully
deterministic, but cuts chunks mid-thought whenever a concept boundary
doesn't land on a word-count multiple; (2) semantic-only — respects the
document's actual topic boundaries by splitting where consecutive-sentence
embedding similarity drops below a per-document percentile threshold, but
depends on embedding quality and produces a less predictable chunk-size
distribution; (3) LLM-based chunking (ask an LLM to propose split points) —
rejected outright as disproportionate cost/latency for a corpus this size,
and non-deterministic across runs.

**Why fixed-size is the default**: on this sample corpus, fixed-size
chunking is simpler to reason about, cheaper (no embedding pass required at
chunk time), and the eval harness didn't show semantic chunking winning
enough retrieval accuracy to justify the extra complexity and
non-determinism. Both remain fully implemented and swappable via
`ChunkingConfig.strategy` — this is a corpus-dependent decision, not a
universal one, and a denser or more narrative corpus could tip it toward
semantic chunking.

## Hybrid Retrieval (RRF Fusion)

**Chosen**: dense (FAISS) and sparse (BM25) retrieval run independently,
each returning its own top-20 candidates, combined via Reciprocal Rank
Fusion (RRF) implemented from scratch rather than calling a library.

**Alternatives considered**: (1) dense-only — the Phase 1 baseline;
concretely failed on exact-term queries (e.g. "What is the time complexity
of BFS?" ranked a Bellman-Ford chunk above the chunk stating BFS's own
complexity, because `all-MiniLM-L6-v2` places topically-related
graph-algorithm terms close together in embedding space); (2) weighted
raw-score combination (`α · dense_score + (1-α) · bm25_score`) — rejected
because cosine similarity (bounded, roughly `[0,1]`) and BM25 scores
(unbounded, corpus-statistics-dependent) live on incomparable scales, so any
fixed or normalized weighting is an arbitrary calibration choice that can
flip rankings on outliers; (3) a reranker alone, skipping fusion — rejected
because reranking only reorders whatever candidates retrieval already
surfaced, so a genuinely dense-only candidate pool with a BM25-favored
chunk missing entirely can't be recovered by reranking after the fact.

**Why RRF won**: RRF uses only rank position (`1 / (k + rank)`, `k=60` from
Cormack et al. 2009), not raw scores, sidestepping the scale-calibration
problem entirely. The ablation study confirms it's close to a free win on
this corpus: MRR improved from 0.824 (dense-only) to 0.880 (dense+BM25)
with retrieval latency essentially unchanged (RRF's own overhead is
negligible compared to running two retrievers).

## Reranker

**Chosen**: a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks
the top-20 RRF-fused candidates down to a final top-5, in a "retrieve wide,
rerank narrow" two-stage pattern.

**Alternatives considered**: (1) no reranker — cheaper and simpler, and
still the right choice for latency-sensitive paths; (2) reranking the
entire corpus directly with the cross-encoder instead of a first-stage
retriever — rejected because a cross-encoder scores one `(query, chunk)`
pair per transformer forward pass, which doesn't scale past a few dozen
candidates, let alone a full corpus; (3) a larger/more accurate
cross-encoder — would likely improve accuracy further but at proportionally
higher per-candidate latency, not evaluated for this corpus size.

**Why it won, with real numbers, not a guess**: the ablation study measured
the reranker adding the single largest accuracy jump of any component (MRR
+0.066, P@1 +0.140, dense+BM25 → dense+BM25+reranker) at a real, non-trivial
cost: ~306ms added to retrieval latency, ~98% of the total retrieval-path
time on this corpus (~12ms/candidate × 20 candidates on CPU, no GPU). That
tradeoff is worth it for an interactive Q&A tool where 300ms is invisible to
a user, and explicitly not a free win — a latency-sensitive application
should measure this exact tradeoff on its own corpus before enabling it by
default.

## LLM-as-Judge Prompt Design

**Chosen**: one Claude call per query scores both faithfulness (is every
claim grounded in the retrieved context?) and answer relevance (does the
answer address the query?) on a 1-5 scale each, using the same model as
generation (`claude-sonnet-5`), with output forced through a `submit_scores`
tool call rather than parsed from free text.

**Alternatives considered**: (1) two separate judge calls, one per
dimension — rejected as double the latency/token cost for judging the exact
same `(query, answer, context)` triple, with no accuracy benefit since the
dimensions aren't in tension; (2) a stronger/separate judge model (e.g.
Opus) instead of reusing the generation model — would reduce same-model
self-preference bias (a model rating its own generations tends to score
them slightly more favorably than an independent judge would), at the cost
of a second model to justify and extra spend; deliberately not chosen for
this project, and the bias risk is disclosed here rather than hidden;
(3) free-text scoring parsed with a regex (`"Score: 4"` style) — rejected
because output-format drift silently breaks a regex parser, whereas a
malformed tool call is a hard, visible API error.

**Why this combination won**: it's the cheapest design that still produces
reliable, structured, two-dimensional scores per query, with every tradeoff
(model choice, call count, parsing method) picked for a stated reason rather
than by default.

## Eval Metrics and Ground Truth

**Chosen**: Precision@k, Recall@k, and Mean Reciprocal Rank, computed from
43 hand-labeled `(query, chunk_id)` pairs, each with a one-sentence
justification for why that chunk is correct.

**Alternatives considered**: (1) single-label ground truth (exactly one
correct chunk per query) vs. multi-label (a query can have several correct
chunks) — the data model (`EvalExample.relevant_chunk_ids: list[str]`)
supports multi-label, but nearly every example in this set has exactly one
entry; chunk overlap (40-word overlap between adjacent fixed-size chunks)
means some content is genuinely retrievable from two chunks, which is a
known, disclosed simplification rather than a hidden one; (2) NDCG or other
graded-relevance metrics — not used, since binary relevance (a chunk either
answers the query or it doesn't) matched how the labels were actually
constructed; introducing graded relevance would have meant inventing scores
with no principled basis.

**A real bug this caught**: running the harness against the first draft of
labels surfaced two queries labeled to a chunk that, on inspection, never
actually contained the word "Bellman-Ford" — it was only the overlap tail of
a sentence that named the algorithm in the *previous* chunk. This is
exactly the kind of error hand-labeling is prone to, and exactly why the
harness runs against real chunk text rather than being trusted blindly: the
eval harness isn't just a report generator, it's a check on the labels
themselves.

## Embedding Model

**Chosen**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~80MB).

**Alternatives considered**: larger sentence-transformers models (e.g.
`all-mpnet-base-v2`, 768-dim) generally score higher on retrieval
benchmarks but are slower to embed and search; not evaluated here since
MiniLM's speed on CPU and small footprint fit a low-hundreds-of-pages
corpus without a measured accuracy problem to justify the tradeoff.

**Why it won**: fast enough for CPU-only iteration during development, and
the ablation study measures pipeline-level tradeoffs (BM25, reranking) that
matter more at this corpus scale than embedding model size does. Swappable
via `EmbeddingConfig.model_name` if a larger corpus later shows a measured
recall gap.

## Vector Index Type

**Chosen**: FAISS `IndexFlatIP` (exact search) over L2-normalized vectors,
equivalent to exact cosine similarity.

**Alternatives considered**: approximate indexes (IVF, HNSW) trade recall
for speed at large scale, but at a few thousand chunks (low-hundreds-of-pages
corpus), exact flat search is already sub-millisecond per query — there's no
latency problem for an approximate index to solve, and it would only add
tuning surface (nlist, nprobe, ef_search) with no measured benefit.

**Why it won**: correct by construction (no approximation error) and simpler
to reason about; the right point to revisit is corpus size, not a default
preference for either approach.

## BM25 Tokenizer

**Chosen**: a simple regex tokenizer (`[A-Za-z0-9]+`, lowercased) rather
than a linguistically-aware tokenizer.

**Alternatives considered**: a code-aware tokenizer that preserves
programming notation (e.g. keeping `O(log n)` intact) — rejected as
unnecessary complexity for a notes corpus, where most of BM25's exact-match
value comes from whole-word acronyms and identifiers (`BFS`, `LRU`,
`AVL`) rather than notation fragments; stemming/lemmatization (e.g. via
NLTK) — rejected to avoid an extra NLP dependency and model download for a
benefit not measured to matter on this corpus.

**Why it won**: cheap, dependency-free, and sufficient — BM25's job in this
pipeline is to catch exact terms dense embeddings blur past, and simple
alphanumeric tokenization does that.

## Citation-Forcing Generation Prompt

**Chosen**: the generation system prompt requires every factual claim to be
immediately followed by a `[chunk_id]` citation copied verbatim from the
provided context, with an explicit instruction never to invent a chunk_id.

**Alternatives considered**: citing at the end of the answer only (one
citation list per response) — rejected because it can't attribute individual
claims to individual chunks, which is exactly what faithfulness scoring
needs; asking for citations in a different format (e.g. numbered
footnotes `[1]`) — rejected in favor of the chunk_id itself, since it lets
both the citation parser and a human reader jump straight from a claim to
the exact retrieved text without an indirection table.

**Why it won**: an unmatched or fabricated citation is itself a measurable
faithfulness signal (the judge and the citation parser both see the raw
`chunk_id` string), which a footnote-style scheme would have hidden behind
an extra mapping step.

## Config-Driven Ablation Design

**Chosen**: `RetrievalConfig.mode` (`"dense"` | `"hybrid"`) and
`RerankerConfig.enabled` (`bool`) are the only two switches needed to
reproduce all three ablation configs — there is one `RagPipeline.retrieve()`
implementation, not three.

**Alternatives considered**: a separate pipeline class or code path per
ablation config — rejected early, since it would have meant re-implementing
(and re-testing) retrieval logic three times, with configs silently drifting
apart from each other over time as the codebase changed.

**Why it won**: `run_ablation.py` is ~150 lines that construct three
`PipelineConfig` objects and run the same `EvalHarness` against each — the
ablation study is nearly free to run and rerun precisely because retrieval
was built as one switchable implementation from Phase 2 onward, not bolted
on afterward.
