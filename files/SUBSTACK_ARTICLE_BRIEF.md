# LLMevallab — Substack Article Brief

> Purpose of this doc: a single reference you can pull from while drafting Substack articles. It captures what the project actually does today, the concepts worth teaching, real numbers from your own runs, and three fully-scoped article ideas.

---

## 1. The one-liner (at three depths)

- **Elevator pitch:** "I built a system that takes documents in different languages, has multiple LLMs extract/translate/summarise them, and then scientifically measures which model is actually best — on quality, cost, and speed — instead of guessing."
- **One paragraph:** LLMevallab is a model-agnostic LLM evaluation framework. It ingests multilingual documents, runs them through a pipeline (extract → translate → summarise), and lets you swap in Gemini, Claude, GPT, Llama, or DeepSeek with zero code changes. It then scores every model's output with industry-standard NLP metrics (BLEU, ROUGE-L, BERTScore) and tracks token usage, cost, and latency — producing a fair, side-by-side benchmark report. It's being built in public, in phases, moving toward a public web app where anyone can bring their own API key and run the same benchmark.
- **Why it exists:** You (Akash) are a senior engineer who wanted to *really* understand how LLM evaluation works in production — not just call an API and eyeball the output — so you built the evaluation harness yourself, learning the metrics, the failure modes, and the engineering patterns along the way. ~60-70% of the code is written with Claude as a pair-programmer, which is itself part of the story.

---

## 2. The journey so far (Phases = natural article structure)

| Phase | Status | What it delivered |
|---|---|---|
| **Phase 1** — Single-model prototype | ✅ Done (v0.1.0, May 2026) | Gemini-only pipeline: extract entities/dates/deadlines, translate German→English (EuroParl corpus), summarise, and score with BLEU/ROUGE/BERTScore |
| **Phase 2** — Multi-model benchmark platform | ✅ Done (v0.1.6, July 2026) | Added Claude + OpenAI-compatible providers (GPT, Llama, DeepSeek via OpenRouter), a config-driven model catalog, token/cost tracking, retry logic, and a `BenchmarkRunner` that runs every model on the *same* documents and produces a comparison table |
| **Phase 2b** — Harden & ship (production honesty) | ✅ Done (v0.2.1, July 2026) | Discovered the evaluator could silently score the wrong data. Fixed it with a `RunManifest` system: every run now writes a provenance record (dataset hash, ground truth hash, doc IDs, config snapshot), the evaluator verifies the hash before computing a single metric, and a mismatch raises a hard error instead of producing garbage scores. Added task-specific truncation limits, GitHub Actions CI, and a RUNBOOK. |
| **Phase 3** — Local analysis dashboard | 📋 Planned (spec updated) | A Streamlit dashboard: one command to launch locally, pick models and task, run the benchmark, get a side-by-side quality + cost + latency comparison with charts and CSV/JSON export. No deployment, no BYOK plumbing — keys live in `.env` as they always have. |
| **Side quest** — Position bias experiment | ✅ Done, standalone | Reproduced the "Lost in the Middle" paper (Liu et al. 2023) — tested whether Claude and Gemini get worse at finding an answer buried in the *middle* of 20 documents vs. at the edges |

This phase structure is a gift for storytelling: it's a natural "before / after" or "how I built X in Y stages" arc. Phase 2b in particular has a narrative arc all its own: *discover a silent failure, fix it so it becomes impossible, then gate Phase 3 on having done so*.

---

## 3. Key concepts you can teach (grouped by theme)

You've been keeping two living "learning logs" (`docs/learning/genai_nlp_concepts.md` and `docs/learning/software_design_patterns.md`) that are basically pre-written article material. Highlights:

### A. LLM Evaluation & NLP concepts
- **Tokens & tokenization** — why cost and context limits are measured in sub-word chunks, not characters
- **BLEU** — n-gram overlap metric for translation; brittle to valid paraphrasing
- **ROUGE-L** — longest common subsequence metric for summarisation
- **BERTScore** — embedding-based semantic similarity; catches meaning even when wording differs completely
- **Hypothesis vs. Reference pairing** — the unglamorous but critical plumbing of evaluation (matching by `doc_id`, skipping missing data honestly)
- **Temperature & determinism** — why extraction/eval pipelines run at `temperature: 0.1`, not creative-writing temperatures
- **Structured JSON extraction from LLMs** — prompting for parseable output, and defensively stripping markdown fences before `json.loads()`
- **Position bias / "Lost in the Middle"** — accuracy drops when the right answer is buried mid-context; a U-shaped curve; direct relevance to anyone building RAG
- **Token-based pricing** — input vs. output tokens are billed at different (often 4-10x) rates
- **Cost / quality / latency trade-offs** — "highest BLEU" is not "best model"; a model 40x more expensive for +0.01 BERTScore may not be worth it
- **Multi-model benchmark design** — the *controlled comparison* principle: same docs, same prompts, same metrics, or the comparison is meaningless
- **Eval validity: truncation limits matter** — if you truncate a news article to 2000 chars and the summary reference covers the whole article, your ROUGE score is measuring an unfair contest. Raising the summarisation limit to 8000 chars is not a cosmetic change — it changes the difficulty of the task and the meaning of the score.
- **Reproducibility in ML evaluation** — a score is only meaningful if you can re-produce it. That requires knowing exactly: which documents, which slice of ground truth, which config. If any of those change without you knowing, you're comparing different experiments.
- **Hard fail vs. silent wrong answer** — the most dangerous bugs in an eval system aren't crashes; they're plausible-looking wrong numbers. The design principle: if you can't verify correctness, refuse loudly.
- **LLM-as-Judge & COMET** — named as the next metrics to add (state-of-the-art for MT and general judgment) — good "what's next" material

### B. Software architecture & engineering patterns (all with real code in the repo)
- **Abstract Base Class / Interface Segregation** — `BaseDocumentProcessor` defines `extract/translate/summarise`; every provider must implement all three
- **Dependency Inversion Principle** — the orchestrator only knows about the abstract processor, never a concrete `GeminiProcessor`
- **Factory Method** — `build_processor(model_key, ...)` decides which concrete class to instantiate from a config key
- **Registry pattern** — dict-based lookup for metrics (`bleu`, `rouge`, `bertscore`) and task→metric mapping
- **Adapter pattern** — `OpenAICompatibleProcessor` adapts one wire format to cover GPT, Llama, DeepSeek, and dozens of OpenRouter-hosted models with a single class
- **Facade pattern** — `BenchmarkRunner.run()` hides load→build→orchestrate→evaluate→aggregate behind one call
- **Strategy pattern** — `PipelineTask` enum routes which pipeline steps run (translation vs. summarisation vs. full)
- **Decorator pattern** — `retry_with_backoff` wraps API calls with exponential backoff (via `tenacity`)
- **Config-driven design / Open-Closed Principle** — adding a new model is a YAML block, not new Python code
- **DTOs via Pydantic** — every pipeline artifact is a typed, validated, JSON-serialisable object
- **Provenance / Run Manifest pattern** — every run writes a `RunManifest` DTO (dataset hash, ground truth hash, doc IDs, config snapshot, results path). Downstream tools read the manifest, not a pointer file. This makes re-evaluation reproducible and tamper-evident. The pattern generalises to any data pipeline where "what exactly did this run see?" matters.
- **Content-addressable verification** — SHA-256 hashing the ground truth *for the exact doc_id subset* (not the whole file) so that swapping or modifying the reference file is detected before a single metric runs. The hash is computed at write time and re-verified at read time: two different moments, same value = trustworthy.
- **Fail-fast vs. fail-silently** — the evaluator had a choice: ignore a hash mismatch and produce plausible scores, or raise `RuntimeError` before touching any metric. The Phase 2b choice was always the latter. Great teaching example of *where* to put a guard and *how hard* to make it fail.
- **Per-step error isolation** — one failed API call doesn't kill the whole batch; failures become `None` and get logged, not raised
- **Contract testing** — one test suite runs against *all* providers (Gemini, Claude, OpenAI-compatible) to guarantee they're truly interchangeable
- **Test pyramid for LLM systems** — unit tests (mocked APIs) → contract tests (interface compliance) → one e2e test with mocked providers, so the whole suite runs without spending API credits
- **CI for ML projects** — GitHub Actions running `pytest` on push/PR; API-dependent tests excluded (non-deterministic, cost money, need secrets); virtual environment cached on the lockfile hash. Different from web-app CI because the "live system" is a paid external API, not a local server.

### C. Product & platform thinking (from the Phase 3 spec)
- **BYOK (Bring Your Own Key) architecture** — you're building a platform, not an API reseller; keys are session-only, never persisted
- **Cost preview before running** — estimating `$` before a benchmark runs, not after
- **Honest scope limits as a feature** — explicitly stating "text PDFs: yes, scanned PDFs: not yet" instead of over-promising
- **Static curated benchmark slices** — shipping a fixed, versioned 30-doc sample instead of downloading from HuggingFace at runtime (reliability + reproducibility for a public demo)
- **Hosting trade-offs** — why Gradio + Hugging Face Spaces beats Vercel/Netlify for this workload (serverless timeouts kill multi-model runs)

---

## 4. Concrete proof points — real numbers from your own runs

Use these instead of hypotheticals; they're from your actual `outputs/reports/` and `configs/config.yaml`.

**BLEU vs. BERTScore disagreeing on the same document (translation, Gemini 2.5 Flash on EuroParl):**

| doc_id | BLEU | BERTScore | What it shows |
|---|---|---|---|
| europarl_de-en_0000 | 1.00 | 0.978 | Exact match — both metrics agree |
| europarl_de-en_0001 | 0.153 | 0.849 | Model paraphrased correctly — BLEU punishes it, BERTScore doesn't |
| europarl_de-en_0002 | 0.099 | 0.865 | Same pattern |
| europarl_de-en_0003 | 0.079 | 0.824 | Same pattern |

This is a genuinely great teaching example: "here's a real translation my pipeline produced that a human would call *correct*, but BLEU scored it 0.08/1.0." It makes the abstract "BLEU is brittle" claim concrete.

**Cost spread across your model catalog (per 1M tokens):**

| Model | Input $ | Output $ | Relative cost |
|---|---|---|---|
| gemini-2.5-flash | $0.075 | $0.30 | 1x (baseline) |
| deepseek-v3 | $0.14 | $0.28 | ~1x |
| llama-3.3-70b (OpenRouter) | $0.13 | $0.40 | ~1.3x |
| gpt-4o-mini | $0.15 | $0.60 | ~2x |
| claude-sonnet-4-6 | $3.00 | $15.00 | **~40-50x** |

Great hook: "The most expensive model in my catalog costs 50x more per document than the cheapest. Was it 50x better? Here's what I found."

**Position bias experiment (Lost in the Middle, expected pattern from your own README):**

| Model | Start | Middle | End | Drop (edge→middle) |
|---|---|---|---|---|
| Claude Haiku | 75% | 45% | 70% | -30pp — clear U-curve |
| Gemini 2.5 Flash | 70% | 55% | 65% | -15pp — milder |

Great hook for RAG builders: "If your RAG system buries the right chunk in the middle of the context, you might be losing 15-30 percentage points of accuracy — and it depends which model you use."

---

## 5. Interesting narrative angles (the "human" story, not just the tech)

- **Learning in public with receipts** — the two `docs/learning/*.md` files are literally a dated log of concepts as you encountered them. That's a rare, authentic artifact most technical writers don't have. Phase 2b added a third: a CI pipeline tutorial written as you built the thing.
- **Claude wrote 60-70% of this code** — worth addressing directly: what did you write vs. review vs. direct? What did you catch that the AI got wrong? This is *very* on-trend for Substack right now (AI-assisted engineering).
- **You didn't just call an API and ship it** — you built the *measurement infrastructure* first. Most people building "an LLM app" skip evaluation entirely. This is the differentiator.
- **The honesty of "not wired into the main pipeline yet"** — the position bias experiment is explicitly a side experiment, not integrated. That kind of transparency about scope builds trust with readers.
- **The bug you found in your own pipeline** — Phase 2b exists because you realised your evaluator could silently score *a completely different dataset* from the one used for inference. Admitting this, fixing it systematically, and making it impossible to repeat is a stronger engineering story than never having the bug. Most ML codebases have this exact flaw and don't know it.
- **Production honesty as a phase** — you named it explicitly. "Phase 2b — Harden & Ship" was a deliberate pause before Phase 3 to make the numbers trustworthy. That's unusual product discipline for a side project, and it shows.
- **Phase 3 hasn't happened yet** — you can write the "here's my plan and why" article *before* building it, then a follow-up "here's what actually happened" — a natural two-part or three-part series.

---

## 6. Top 4 Article Ideas

### 🥇 Article 1 — "The Metrics Lie: What Benchmarking 5 LLMs Taught Me About Evaluating AI"
**Best for:** broad technical audience, AI-curious readers, anyone who's used an LLM API and wondered "but how do I know if it's actually good?"

**Hook:** Open with the real BLEU vs. BERTScore table above — a translation that's *clearly* correct scoring 0.08/1.0 on the "official" metric.

**Outline:**
1. The problem: "It looks right" is not a benchmark
2. Meet the metrics: BLEU (word overlap), ROUGE-L (structural overlap), BERTScore (meaning overlap) — explained with your own before/after examples, not textbook ones
3. The reveal: run the same document through Gemini, Claude, GPT-4o-mini, Llama, and DeepSeek; show the actual comparison table (cost, latency, quality) from your `BenchmarkRunner`
4. The cost twist: the 40-50x price gap and whether it bought better scores
5. What's still missing: LLM-as-Judge and COMET as the next frontier, and why more metrics ≠ more truth
6. Takeaway for readers: a mini-checklist for evaluating any LLM feature they ship

**Why it works:** teaches something genuinely useful (most engineers don't know BLEU vs BERTScore), is grounded in your real data, and doubles as a soft demo of the project.

---

### 🥈 Article 2 — "I Designed an LLM Platform Using 10 Classic Software Patterns (So Adding a New Model Takes One YAML Line)"
**Best for:** software engineers / senior engineers who care about architecture, less about ML specifically — a "system design meets AI" crossover piece.

**Hook:** "Adding Llama, DeepSeek, or GPT to my evaluation platform required editing one YAML file and zero Python. Here's the architecture that makes that possible."

**Outline:**
1. The trap most "LLM wrapper" projects fall into: hardcoding one provider's SDK everywhere
2. The fix: an abstract `BaseDocumentProcessor` + Dependency Inversion — walk through the actual `extract/translate/summarise` interface
3. Factory Method + config catalog: how a model goes from "idea" to "usable" via one YAML block
4. Adapter pattern in the wild: how one `OpenAICompatibleProcessor` class quietly covers GPT, Llama, DeepSeek, and dozens of OpenRouter models
5. Facade pattern: how `BenchmarkRunner.run()` hides six moving pieces behind one call
6. Testing LLM systems without burning API credits: the test pyramid (unit → contract → mocked e2e)
7. Takeaway: these are the same patterns from your CS textbooks — here's what they look like solving a 2026 problem

**Why it works:** differentiates you as a *systems* thinker, not just someone calling `openai.chat.completions.create()`. Strong LinkedIn/Substack crossover appeal for senior engineers.

---

### 🏅 Article 4 — "I Was Silently Evaluating the Wrong Data. Here's How I Made It Impossible to Happen Again."
**Best for:** ML engineers, data scientists, anyone who has ever reported an evaluation metric in a meeting — the most immediately useful piece for a working engineer.

**Hook:** "My LLM evaluation pipeline had a quiet bug. The evaluator would happily score *whatever was the last run*, not the run you intended. The ground truth file could be completely different from what was used during inference. The scores would still look perfectly reasonable. I had no idea."

**Outline:**
1. The silent failure: show the exact code path that let this happen — `latest_translation.txt` pointer + ground truth from config = no link between inference and evaluation
2. Why this is insidious: the scores are not obviously wrong; they just measure something you didn't intend
3. The fix: `RunManifest` — a provenance record written alongside every results file, covering dataset hash, ground truth hash, doc IDs, and config snapshot
4. Content-addressable ground truth: why we hash the *subset* (the exact doc_ids processed), not the whole file — catching the "right doc_ids, swapped file" case
5. The evaluator's new contract: it reads the manifest, re-verifies the hash, and raises a `RuntimeError` before touching a single metric on mismatch
6. Task-specific truncation as the same class of bug: if you truncate CNN articles to 2000 chars and the reference summary covers the full article, what are you even measuring?
7. CI as the final gate: a green badge on `main` means the manifest, hash verification, and truncation logic all pass on every push
8. Generalisation: any data pipeline that separates "produce" from "consume" has this class of bug. The manifest pattern is the fix.

**Why it works:** every senior engineer has shipped or reviewed code with this exact flaw. Naming it clearly, showing the fix, and offering a generalised pattern makes this immediately shareable. It's also an honest "I made a mistake" piece, which performs well on Substack.

---

### 🥉 Article 3 — "Building an LLM Benchmark Platform in Public: 3 Phases, 1 AI Pair-Programmer, and a Lot of Trial and Error"
**Best for:** "build in public" audience — founders, indie hackers, engineers considering a portfolio/side project — the most personal, narrative-driven piece.

**Hook:** "I'm a senior engineer who wanted to actually understand LLM evaluation instead of nodding along in meetings. So I built the whole pipeline myself — with Claude writing 60-70% of the code. Here's the honest account, mistakes included."

**Outline:**
1. Why I started: the gap between "I use LLMs at work" and "I understand how to evaluate them"
2. Phase 1: the humble single-model prototype (Gemini + EuroParl) — what I learned building the *interface* before the implementation
3. Phase 2: going multi-model — the moment cost tracking and retries became non-optional, not nice-to-haves
4. The side quest: reproducing an actual research paper ("Lost in the Middle") just to see if it held up on models I actually use
5. Working with Claude as a pair programmer: what it got right, what I had to catch, how the "learning log" habit emerged
6. What's next: Phase 3's BYOK web app — the plan, the security model, and why I'm shipping an "ugly MVP" fast instead of polishing in private
7. Where to follow along / try it yourself

**Why it works:** this is the article that makes people *care about you*, not just the tech. It's the natural companion/intro piece to Articles 1 and 2 — could even be published first as the "why should I read the next two" hook, or last as the "here's the full story" wrap-up.

---

## 7. Suggested visuals to pull in

- `docs/assets/pipeline_overview.png` — the existing architecture diagram, good for Article 2 or 3
- A simple bar chart: BLEU vs. BERTScore per document (Article 1) — easy to generate from `outputs/reports/report_translation_*.json`
- A cost-vs-quality scatter plot: x = cost per doc, y = BERTScore, one point per model (Article 1 or 3)
- The position-bias "U-curve" line chart: accuracy vs. gold-document position, one line per model (Article 1, bonus section)
- A simple before/after code snippet: "adding a new model" as one YAML block (Article 2)

---

## 8. Suggested publishing order

1. **Article 3** (the story) — hooks readers emotionally, sets context, low technical bar. Introduces the project and the AI pair-programmer angle.
2. **Article 4** (the silent failure / manifest) — strongest shareable hook for engineers ("I was measuring the wrong thing"). Ride the momentum from Article 3 while the project is fresh. Most immediately useful for working ML engineers.
3. **Article 1** (the metrics deep-dive) — cashes in with real data; most SEO-friendly. Now readers already trust the numbers because Article 4 explained how the pipeline was hardened.
4. **Article 2** (the architecture piece) — for the subset who want to go deeper technically. Doubles as a portfolio artifact for your resume/LinkedIn.

Each article ends with a link/teaser to the next. Articles 1 and 4 are natural cross-links (Article 1 shows the scores; Article 4 explains why you can trust them). All four point back to the GitHub repo.

> **Note on timing:** Articles 3 and 4 can be written and published *now* (Phase 2b is done). Articles 1 and 2 benefit from Phase 3 data — real multi-model benchmark scores from the BYOK app. Consider publishing 3 → 4 now, then 1 → 2 after Phase 3 ships.
