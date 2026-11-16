# Cactus Compute "Needle" family (Needle v1, Needle 2, Needle 3) — research notes as of 16 Nov 2026

**How to read the tags.** Every finding carries two tags:
- Source type: **[CO]** = company-reported (Cactus Compute blog, model card, README, or Cactus staff comments); **[IND]** = independent (third-party user, HN commenter, Google's dataset card, arXiv metadata); **[SEMI]** = hosted by a third party but written by or with Cactus (the Raspberry Pi blog post).
- Read type: **[DR]** = I read the primary page in full on 16 Nov 2026; **[SS]** = taken from a search-engine summary only, page not opened.

**Access notes.** arxiv.org, huggingface.co, cactuscompute.com, github.com, news.ycombinator.com and www.raspberrypi.com were all fetched directly (via curl; the WebFetch tool could not resolve arxiv.org and huggingface.co). Benchmark charts on cactuscompute.com are drawn by JavaScript, so exact Needle 3 numbers were read out of the site's chart data arrays in its script bundle, not from visible text. docs.cactuscompute.com, liquid.ai and marktechpost.com were not needed/not fetched. The GitHub star count could not be parsed from the GitHub page; only the company's own claim is given.

**Three corrections to the brief's framing (all [DR]):**
1. arXiv 2607.18363 is not "the Needle 2 paper". Its title is "A Controlled Study of Attention-Only Transformers" (submitted 20 Jul 2026, 10 pages). It never mentions Needle or tool calling. It is the scientific basis Cactus cites for the architecture. — [arXiv abs](https://arxiv.org/abs/2607.18363), [arXiv HTML](https://arxiv.org/html/2607.18363)
2. "Single 14 MB binary with engine + weights baked in" describes Needle 2 only. Needle 3 ships a separate engine (under 1 MB per platform) plus a weights file `needle3.cact` of 8–29 MB. — [Supported devices post](https://cactuscompute.com/blog/needle-supported-devices)
3. Needle 3 is not smaller than Needle 2. The full Needle 3 is 121M parameters / 29 MB; it is a "ladder" whose smallest practical slice (4 layers) is 29M parameters / about 8 MB. — [Needle 3 page](https://cactuscompute.com/needle)

---

## 1. Architecture of each Needle version (parameters, layers, tokenizer, context, input/output format, free text)

### Takeaway
The three versions are three different designs under one brand name ("Simple Attention Network"): v1 is a 26M encoder-decoder with literally no feed-forward layers; Needle 2 is a 45M, 27-layer decoder-style stack that adds a cheap "Hadamard MLP" and hashed n-gram lookup tables ("engram"); Needle 3 is a 121M, 20-layer version of that stack trained so any depth from 2 to 20 layers is a usable model. All take "user text + JSON tool schemas" in and emit a JSON function call out; from Needle 2 on, output is grammar-constrained and there is no free-text chat.

### Cited Findings

**Needle v1 (released 12 May 2026)**
- 26M parameters; encoder-decoder; "pure attention (no FFN)". Encoder: 12 layers, grouped-query attention (8 heads / 4 KV heads), RoPE positions, gated residuals. Decoder: 8 layers, masked self-attention plus cross-attention to the encoder. Width (d_model) 512. Vocabulary 8,192 SentencePiece BPE tokens. Norm "ZCRMSNorm". Precision bfloat16 with INT4 quantisation-aware training. Tied input/output embeddings. [CO][DR] — [HF model card Cactus-Compute/needle](https://huggingface.co/Cactus-Compute/needle)
- Design rationale in the company's words: "Tool calling is fundamentally retrieval-and-assembly (match query to tool name, extract argument values, emit JSON), not reasoning. Cross-attention is the right primitive for this, and FFN parameters are wasted at this scale." "The entire model is just attention and gating, no MLPs anywhere." Described as "an experimental run for single-shot function calling." [CO][DR] — [Needle v1 blog, 12 May 2026](https://cactuscompute.com/blog/needle)
- Input format: a query string plus a JSON list of tools, e.g. `tools='[{"name":"get_weather","parameters":{"location":"string"}}]'`; output: `[{"name":"get_weather","arguments":{"location":"San Francisco"}}]`. No mention of constrained decoding or a confidence score in v1. [CO][DR] — [HF model card](https://huggingface.co/Cactus-Compute/needle)

**Needle 2 (released 9 Aug 2026)**
- 45M parameters, of which 35M are "matmul-active" and 8M are engram tables read by lookup. "27-layer, 512-wide network" with four parallel residual "lanes" (multi-lane hyper-connections). [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Components: "a Hadamard MLP in place of the FFN, GQA attention, engram key-value memory, and multi-lane hyper-connections." The Hadamard MLP uses a fixed Walsh-Hadamard transform with learned diagonals instead of dense projections. The engram is "hashed n-gram tables that are read a few rows per token." So Needle 2 is no longer strictly attention-only. [CO][DR] — [HF model card Cactus-Compute/needle2](https://huggingface.co/Cactus-Compute/needle2); [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Context/memory: "Attention uses a 256-token sliding window so the KV cache is bounded no matter how long a session runs, and the system prompt and tool declarations are pinned as permanent sinks" so tools cannot be evicted. [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Output: one JSON object per turn with `type`, `function_calls`, `reasoning`, `confidence`, plus speed/RAM counters. "A byte-level grammar compiled from your schemas constrains every token." The `reasoning` line "is generated unconstrained; only the call itself is grammar-constrained." [CO][DR] — [HF needle2 card](https://huggingface.co/Cactus-Compute/needle2)
- Free text: "A request no declared tool can serve is refused with the empty call `[]`. That is the whole contract for off-topic input; there is no free-text fallback." However the same card says a final step "may answer in plain text from the results: `"type": "respond"`." [CO][DR] — [HF needle2 card](https://huggingface.co/Cactus-Compute/needle2)
- Tool retrieval: "Five or fewer declared tools render directly. Above that, retrieval engages... only the five highest-scoring tools enter the context... An unselected tool is unreachable, not merely unlikely." [CO][DR] — [HF needle2 card](https://huggingface.co/Cactus-Compute/needle2)
- Optional "system facts" turn (`date`, `locale`, `device`, `battery`, `network`, `location`, `user`, `assistant`): "instructions placed there do not steer the model." [CO][DR] — [HF needle2 card](https://huggingface.co/Cactus-Compute/needle2)

**Needle 3 (blog posts dated 17–18 Sep 2026; Show HN posted 18 Sep 2026 00:11 UTC)**
- "29-121M Laddered Simple Attention Networks, CQ2 quantisation." Sub-network sizes: 2 layers = 25M params, 4L = 29M, 8L = 52M, 16L = 98M, 20L = 121M. [CO][DR] — [Needle 3 page](https://cactuscompute.com/needle); [Intelligence Ladders post](https://cactuscompute.com/blog/intelligence-ladders)
- From the weights-file header: "20 layers, width 768, 12 query heads over 2 KV heads, context 8192, a 1024-token local window, four residual lanes, an int8 KV cache trained at a window of 256." [CO][DR] — [.cact format post](https://cactuscompute.com/blog/cact-format)
- Engram: "70.8M of its 121M parameters live there, and a token reads 30 rows of them and multiplies none." Five engram sites, each a table of 110,592 rows × 128. [CO][DR] — [Blog index summary of engram post](https://cactuscompute.com/blog); [.cact format post](https://cactuscompute.com/blog/cact-format)
- Hadamard MLP: "25.6K parameters per layer instead of 4.7M, and about 0.21M multiply-adds per token instead of 4.7M." [CO][DR] — [Blog index summary of Hadamard MLP post](https://cactuscompute.com/blog)
- Tokenizer: "a plain SentencePiece BPE"; logits are an "8192-way" vector; tokenizer is embedded in the weights file (111 KB). The model emits special tokens `<think>` and `<tool_call>`. [CO][DR] — [Porting Needle 3 post](https://cactuscompute.com/blog/porting-needle); [.cact format post](https://cactuscompute.com/blog/cact-format)
- Three jobs: tool calls, structured extraction (schema declared as a single "tool"), and text embeddings (new in v3). Inputs: "Text prompts, plus tool definitions or an extraction schema." Outputs: "Structured JSON with tool calls or extractions." [CO][DR] — [Needle 3 page](https://cactuscompute.com/needle)
- No free text: "A final `"type": "respond"` with empty function_calls ends the loop; the answer is the tool results... No free text is generated." Listed under "Mistakes to avoid": "Expecting free-text answers." [CO][DR] — [Needle Python docs](https://cactuscompute.com/blog/needle-python-docs)
- Constrained decoding: "Ranges, enums, patterns, lengths and item counts are compiled into the decode grammar, so an invalid value is unrepresentable rather than merely discouraged." Hard caps: "a numeric literal stops at twenty characters, an array stops after sixty-four items." [CO][DR] — [Designing tools post](https://cactuscompute.com/blog/designing-tools-for-needle)
- Grounding rule: "a call contains only values evidenced by the request." A required argument with no evidence and no default causes the call to be withheld. Numbers are strict: "'a bit warmer' does not evidence any number at all." [CO][DR] — [Designing tools post](https://cactuscompute.com/blog/designing-tools-for-needle)
- "Triggers": developer-written regular expressions attached to a tool; a match forces a call to that tool, bypassing the confidence floor. [CO][DR] — [Needle 3 page](https://cactuscompute.com/needle)
- Multi-turn loop: `agent.run(query, max_steps=8, max_new_tokens=512)` executes calls and feeds JSON results back; "Later arguments may depend on earlier results." One toolset per agent session. [CO][DR] — [Needle Python docs](https://cactuscompute.com/blog/needle-python-docs)

**The arXiv paper (20 Jul 2026)**
- Authors: Henry Ndubuaku, Karen Mosoyan, Jakub Mroz, Noah Cylich, Satyajit Kumar, Parkirat Sandhu, Roman Shemet, Justin H Lee (Cactus Compute, Inc.). It pretrains attention-only decoders against standard transformers "matched separately for parameter count, training FLOPs, and depth (2 to 48 layers), for up to 105B tokens at 6M to 87M parameters." [IND for metadata / CO for content][DR] — [arXiv 2607.18363](https://arxiv.org/abs/2607.18363)
- Headline result: deleting feed-forward layers in place costs 0.47 nats at matched depth; "at matched parameters the difference is 0.006 nats (0.27 percent of loss)." "attention-only models are better on context-grounded answers and worse where knowledge must come from weights." [CO][DR] — [arXiv 2607.18363](https://arxiv.org/abs/2607.18363)
- The paper's own caveats: "At matched parameters the SAN pays ≈2× FLOPs per token at 2048 context; iso-FLOP favors the FFN... We claim regime, not superiority." "MMLU-class benchmarks are at chance throughout." "Of the eight predictions fixed before their measurements, two were falsified outright." [CO][DR] — [arXiv HTML](https://arxiv.org/html/2607.18363)

### Inferences
- "Simple Attention Network" is a brand that has drifted: v1 = encoder-decoder with zero MLPs; v2/v3 = decoder-style stacks with a (very cheap) MLP substitute and a large lookup-table memory. A teacher should not say "Needle has no feed-forward layers" about v2/v3 without qualification.
- In Needle 3 most parameters (about 59%) are lookup-table rows, not matrix weights. That explains how a "121M" model is cheap per token: Cactus counts only 50M as "matmul-active" (chart data on the [Needle 3 page](https://cactuscompute.com/needle)).
- The paper tests pure attention-only decoders; the shipped Needle 2/3 include extra parts (Hadamard MLP, engram, multi-lane residuals) that the paper's abstract does not evaluate. The HF needle2 card's phrase "See the paper for the design and ablations" overstates what the paper covers, judged by its abstract and limitations section.
- Plain-language summary for a non-specialist: Needle does not "know" things; it copies. It matches the user's words to a tool name and copies spans of the user's words into argument slots, under a grammar that makes malformed JSON impossible.

### Gaps
- Needle 3 docs conflict on tool retrieval. The Python docs say "Needle 3 reads the complete toolset in its prompt... It has no retrieval head" ([Python docs](https://cactuscompute.com/blog/needle-python-docs)); the tool-design post says "Above that [five tools], retrieval engages... only the five closest tools enter the context" ([Designing tools](https://cactuscompute.com/blog/designing-tools-for-needle)); the porting post mentions "retrieval works while the release ships without a contrastive head" ([blog index](https://cactuscompute.com/blog)). Unresolved which behaviour ships.
- Needle 2 "plain text respond" versus Needle 3 "No free text is generated": whether Needle 2 really emitted prose was not verified.
- Needle 2's exact vocabulary size and layer breakdown (which of the 27 layers are attention vs. other) were not found. v1's maximum context length was not found.
- The v1 architecture write-up (`docs/simple_attention_networks.md`) was not read; the old `cactus-compute/needle` repo is now a one-line redirect stub.

---

## 2. Training: Gemini distillation, data pipeline, dataset size, fine-tuning, cost

### Takeaway
Only v1 is explicitly described as "distilled from Gemini": Gemini generated synthetic function-calling examples that the small model was trained on (the teacher's weights were never touched). Needle 2 and 3 are trained from scratch on much larger "proprietary" synthetic corpora (153B and 360B tokens) whose generator is not named. Developers can fine-tune locally with LoRA or on Cactus's paid platform; no dollar training cost is published.

### Cited Findings
- v1: "Pretrained on 200B tokens across 16 TPU v6e (27 hours). Post-trained on 2B tokens of synthesized function-calling data (45 minutes). Dataset synthesized via Gemini with 15 tool categories (timers, messaging, navigation, smart home, etc.)" [CO][DR] — [Needle v1 blog](https://cactuscompute.com/blog/needle)
- v1 card: "We distilled Gemini 3.1 into a 26m parameter" model; weights open "as well as the dataset generation." [CO][DR] — [HF needle card](https://huggingface.co/Cactus-Compute/needle). Third-party coverage names the teacher as "Gemini 3.1 Flash Lite". [IND][SS] — [NYU Shanghai RITS summary](https://rits.shanghai.nyu.edu/ai/cactus-releases-needle-a-26m-distilled-model-for-on-device-tool-calling)
- What "distilled" means here: Cactus's founder on HN, replying to a terms-of-service objection: "the distillation process did not access the weights." Asked for the dataset: "We have so far released the dataset generation code." [CO][DR] — [HN thread 48111896, May 2026](https://news.ycombinator.com/item?id=48111896)
- ToS criticism: an HN commenter noted "distilling Gemini is explicitly against the ToS: 'You may not use the Services to develop models that compete with the Services'." Another asked whether Google's anti-distillation defences could have degraded the data. [IND][DR] — [HN thread 48111896](https://news.ycombinator.com/item?id=48111896)
- Needle 2: "pretrained on a proprietary 115B-token corpus and post-trained on 38B tokens with compact reasoning traces and careful dataset distribution design. For scale: LFM2.5-230M was pretrained on 19 trillion tokens, roughly 120× Needle's total." Training corpus is "consumer device actions—smart home, mobile, wearables, TV, car—plus structured extraction." [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Needle 2 was "trained from the ground up" (not pruned from a larger model); when a commenter suggested LLM-synthesised training data, staff replied "Your thinking is correct haha." [CO][DR] — [HN thread 49246804, Aug 2026](https://news.ycombinator.com/item?id=49246804)
- Needle 3: "Training: 360B tokens of proprietary structured dataset." Staff on HN: "data was honestly most of the work... we spend way more time on the generation pipeline than on the model." [CO][DR] — [Needle 3 page](https://cactuscompute.com/needle); [HN thread 49748553, Sep 2026](https://news.ycombinator.com/item?id=49748553)
- Quantisation is in the training loop: "we never quantize post-hoc: Needle 2 trains against Cactus Quants from pretraining through post-training – weights, activations, and KV cache alike." [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Needle 3 "ladder" training: each step, with probability 0.8 the full model trains normally; otherwise a random depth from 2 to L−1 is trained with a loss pulling it toward the full model's predictions (self-distillation). In an 18-layer ablation over 3,000 steps, the ladder model's validation loss was 1.958 vs 2.037 without. [CO][DR] — [Intelligence Ladders post](https://cactuscompute.com/blog/intelligence-ladders)
- Fine-tuning, local path (Needle 3): "LoRA adapters of rank 16 on the five attention projections of every layer", base frozen; `pip install "cactus-compute[train]"`; JAX on CPU, CUDA or Apple Metal; "On an M5 Max a step takes 0.71 seconds against 2.90 on CPU"; exports at 4-bit, not 2-bit. Data is JSONL with `query`, `tools`, `answers`, optional `reasoning`. [CO][DR] — [Fine-tuning Needle post](https://cactuscompute.com/blog/finetuning-needle)
- Fine-tuning, hosted path ("Cactus Platform"): trains "the full model, every depth from 2 layers up", mixes in Cactus's original data, retrains the confidence head, returns 2-bit files; "100 to 10,000 examples per run"; uses an API key and a plan "allowance". [CO][DR] — [Fine-tuning Needle post](https://cactuscompute.com/blog/finetuning-needle)
- Data-size guidance: "a few hundred clean examples measurably improve which tool gets picked. Argument grounding moves later and needs more, on the order of thousands of examples." Synthetic data generator `cactus generate-data` needs an OpenRouter API key. Include off-topic examples (about one in eight) or "the tuned model calls a tool on everything." [CO][DR] — [Fine-tuning Needle post](https://cactuscompute.com/blog/finetuning-needle)
- v1 fine-tuning: a local web "playground" that "generates data via Gemini, trains, evaluates, bundles result." [CO][DR] — [HF needle card](https://huggingface.co/Cactus-Compute/needle)
- Independent cost opinion: HN commenters: "At <50M parameters, training costs are completely trivial"; "the expensive part wasn't the training itself but the data curation and evaluation." [IND][DR] — [HN thread 49246804](https://news.ycombinator.com/item?id=49246804)

### Inferences
- "Distillation" in v1 means behavioural imitation: ask a big model to produce (request, tool list, correct call) triples, then train the small model on them. It is not logit-matching or weight transfer.
- Cactus dropped "Gemini" from the Needle 2/3 messaging entirely; whether Gemini still generates the data is unknown. The ToS criticism of v1 is a plausible reason, but that is my speculation.
- The local LoRA route produces a larger (4-bit) and less capable artefact (no confidence score) than the hosted route. The best-quality fine-tune therefore depends on Cactus's commercial platform, which matters when judging "open".
- 16 TPU v6e for 27 hours (v1 pretraining) is a few hundred TPU-hours — small by LLM standards, but no price is given.

### Gaps
- No dollar figure for training any version. No hardware/time figure for Needle 2 or Needle 3 training.
- Which model(s) generated the Needle 2/3 corpora, and how examples were validated, is not disclosed. The 153B/360B-token corpora are not released; only v1's generation code was.
- Cactus Platform pricing was not found.
- Number of training examples (as opposed to tokens) is not published for any version.

---

## 3. Compression (CQ2 / Cactus Quants), the single binary, RAM, speed

### Takeaway
"CQ2" stores each weight as a 2-bit code plus a shared scale, about 2.125 bits per weight, and the model is trained with that rounding in place so nothing is lost afterwards (company claim). Needle 2 packed model + tokenizer + engine into one 14 MB executable using about 28 MB of RAM; company-claimed speed on a Raspberry Pi 5 is 500+ tokens/s decode, while the one hosted demonstration measured about 250–315 tokens/s and 76–149 ms per command.

### Cited Findings
- How Cactus Quants works: weights are quantised "in groups of 128"; each group is first rotated by a Walsh-Hadamard matrix so it "looks Gaussian", then split into one fp16 length and a direction whose coordinates snap to a small Lloyd-Max codebook. "the cost per weight is b + 1/8: 2.125 bits at CQ2, 4.125 at CQ4." [CO][DR] — [.cact format post, 18 Sep 2026](https://cactuscompute.com/blog/cact-format)
- Not everything is 2-bit: "Every matrix the blocks multiply through is CQ2; the embedding, which is also the output head, the lane maps and the confidence head keep 4 bits; everything elementwise stays half precision." Needle 2 card: "mixed bits per weight averaging 2bit." [CO][DR] — [.cact format post](https://cactuscompute.com/blog/cact-format); [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Weights are never decompressed: "the 2-bit codes are expanded inside vector registers, fused into integer dot products, so resident memory stays at blob size and the arithmetic path is int8 end to end." Activations and KV cache are 8-bit ("CQ-W2A8"). [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2); [.cact format post](https://cactuscompute.com/blog/cact-format)
- "Lossless" claim: "The 2bit model you deploy is the model that was trained. That is what fits 45M parameters into 14MB with nothing lost on our benchmarks." [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Why 2-bit rather than 4-bit (founder on HN): "A lot of Needle 1 users deployed on microcontrollers, we couldn't just throw them under the bus only t[o] chase benchmarks, so its a price we are willing to pay." [CO][DR] — [HN thread 49246804](https://news.ycombinator.com/item?id=49246804)
- Needle 2 packaging: "a single dependency-free C++ binary that probes the CPU at startup and picks its kernels, with the model, tokenizer, and grammar compiler sealed inside. One artifact runs from Cortex-M to x86 to WebAssembly." Kernel tiers: "SDOT, NEON, AVX2, RISC-V vectors, wasm SIMD, or scalar." [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Needle 3 packaging changed: "Each platform folder holds an engine that loads needle3.cact at start; the engine carries no weights." Engine "under 1 MB each"; "The shipped 20-layer needle3.cact is 29 MB; the ladder runs down to 8 MB at four layers and smaller still at two." [CO][DR] — [Supported devices post](https://cactuscompute.com/blog/needle-supported-devices)
- RAM: Needle 2 "RAM is a deterministic 28MB ceiling" because of the sliding window. Needle 3 example response shows `"peak_ram_mb": 28.5`. [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2); [Needle 3 page](https://cactuscompute.com/needle)
- RAM measured in the Pi demo: "Needle's native session stays around 28MB. In this Python demo, the complete process peaked between 43MB and 46.4MB, including the interpreter." [SEMI][DR] — [Raspberry Pi news, 22 Sep 2026](https://www.raspberrypi.com/news/turn-text-input-into-actions-with-needle-a-14mb-function-calling-llm/)
- Speed, v1: "6000 tok/s prefill and 1200 tok/s decode on consumer devices" (device not named). [CO][DR] — [Needle v1 blog](https://cactuscompute.com/blog/needle)
- Speed, Needle 2 (company): "500 tokens/sec decode speed on a Raspberry Pi 5"; "800+ tok/s Pi5 prefill"; "400–1,500 tokens/sec on VR devices like Meta Quest 3S and Apple Vision Pro"; "300–700 on sub-$200 phones such as the Samsung A-Series." [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Speed, Needle 2 (Pi demo, Raspberry Pi 5 8GB, CPU only, cactus-needle 2.0.7), six prompts: prefill 461–488 tok/s; decode 248–314 tok/s; wall-clock per `complete()` call 76–149 ms (e.g. "Turn the LED on." → 78 ms; off-topic "What is the capital of France?" → empty call in 92 ms). [SEMI][DR] — [Raspberry Pi news](https://www.raspberrypi.com/news/turn-text-input-into-actions-with-needle-a-14mb-function-calling-llm/)
- Speed, Needle 3: "400-4k tokens/s decode and 1-10k tokens/s prefill on a Raspberry Pi 5" (range spans the 20-layer to 2-layer slices). [CO][DR] — [Needle 3 page](https://cactuscompute.com/needle)
- Compute per token (company's own FLOP accounting): Needle 2 = 70 MFLOPs/token; LFM2.5 230M = 460; FunctionGemma 270M = 540; Apple FM ≈ 6,000. Needle 3 20L = 100; 16L = 83; 8L = 48; 4L = 30. [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2); [Needle 3 page chart data](https://cactuscompute.com/needle)
- Grammar as a speed trick: "the engine computes output scores only for candidate rows, skipping up to 98% of the vocabulary projection on structural tokens." [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Microcontrollers: blog intro says "Needle runs on newer microcontrollers like ESP32-S3"; later in the same post the fit is stated as "MCU-class parts with external RAM, such as ESP32-P4 with 32MB of PSRAM, or STM32H7 and NXP i.MX RT boards with SDRAM", shipped "as a static library for Cortex-M4, M7, and M55." The HF card says: "others have reported running it on an ESP32-S3 in about 11MB." [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2); [HF needle2 card](https://huggingface.co/Cactus-Compute/needle2)
- Independent microcontroller reports: an HN user posted "https://github.com/memovai/mimimodel i made it run on esp32s3, and it works!!!" Another noted a Pi Pico 2 "has 520k of RAM", so "I don't see this doing anything useful on a Pico." As of Aug 2026 Cactus said an ESP32 guide was still being written. [IND/CO][DR] — [HN thread 49246804](https://news.ycombinator.com/item?id=49246804)
- Browser (WebAssembly) speed reported by HN users for Needle 2: "310 tok/sec on my browser" and "~4 tok/s" (with V8 JIT disabled). [IND][DR] — [HN thread 49246804](https://news.ycombinator.com/item?id=49246804)
- Wider use of the format: Cactus also published a 120B-parameter Nemotron checkpoint at "1.418 effective bits per weight", 22.06 GB (11 Nov 2026). [CO][DR] — [Cactus blog index](https://cactuscompute.com/blog)

### Inferences
- Arithmetic check on the 14 MB claim: 45M weights × 2.125 bits ≈ 12 MB, plus 4-bit embeddings, half-precision small tensors, tokenizer and engine code ≈ 14 MB. The claim is internally consistent.
- The only quasi-independent Pi 5 measurement (about 300 tok/s decode) is roughly 60% of the company's "500+ tok/s" headline. Cause unknown (short outputs, Python overhead, thread count). Because a tool call is only tens of tokens, the practically meaningful figure is the 76–149 ms end-to-end latency.
- "Lossless 2-bit" means "no loss relative to the same model's own training precision", not "as good as a full-precision model of the same size would be"; no full-precision Needle is published for comparison except the Needle 3 half-precision master checkpoint.
- The Raspberry Pi post is a guest post ("our friends from Cactus Compute show how..."), so its numbers are not an arms-length benchmark even though hosted on raspberrypi.com.

### Gaps
- No tokens/s or latency figure on any microcontroller was found, from the company or anyone else. The ESP32-S3 "about 11MB" report has no speed or accuracy data.
- No phone latency figures beyond the company's 300–700 tok/s range; no named phone model with a measurement.
- No independent replication of the "nothing lost at 2 bits" claim. Cactus's own 11 Nov 2026 post "Why quantization can break multi-step workflows" suggests quantisation effects are an open topic; it was not read.
- The "/quants" page on cactuscompute.com returned almost no text; no standalone Cactus Quants spec beyond the .cact post was read.

---

## 4. Benchmarks: Mobile Actions and others; exact scores; who reported them

### Takeaway
Every accuracy number for Needle is company-reported; I found no independent replication on a public benchmark. On Google's Mobile Actions eval split (961 rows, strict exact match), Cactus reports Needle 2 at about 63.7% (roughly level with FunctionGemma 270M at 64.0%, below LFM2.5-230M at 69.1%) and the full Needle 3 at 86.0% (above every small baseline it tested, below DeepSeek V4 Flash at 88.4%). No Mobile Actions score for Needle v1 was found. Independent hands-on reports are far less flattering.

### Cited Findings

**What Mobile Actions is**
- Published by Google on Hugging Face as `google/mobile-actions`, licence CC-BY-4.0, English. "The dataset contains conversational traces designed to train lightweight models (such as FunctionGemma 270M) to translate natural language instructions into executable function calls for Android OS system tools." Each row has a tool list, a user message and the expected `tool_calls`; rows are tagged "train" or "eval". [IND][DR] — [HF dataset card google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions)
- Cactus uses the "eval split, 961 rows" and scores "ordered strict exact match: a row passes only if the function names, the call order, and every argument value match." [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)

**Needle 2 launch tables (9 Aug 2026).** Needle measured through the shipped 2-bit binary with tool retrieval on and the 256-token window; baselines at f16 under vLLM; Apple FM on-device. [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)

| Mobile Actions (961 rows) | Accuracy | Name acc. | 1-call | 2-call |
|---|---|---|---|---|
| LFM2.5 230M | 69.1 | 93.0 | 76.1 | 55.0 |
| FunctionGemma 270M | 64.0 | 87.3 | 73.0 | 46.2 |
| Needle 2 | 63.7 | 98.3 | 71.3 | 48.4 |
| Apple FM (~3B) | 57.6 | 94.2 | 64.5 | 43.8 |

- DroidCall test split (200 rows): FunctionGemma 17.5; Needle 2 17.0; LFM2.5 230M 11.0. All three score 0.0 on the 24 two-call rows. [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Seal-Tools in-domain (700 rows): Needle 2 32.6; LFM2.5 230M 26.9; FunctionGemma 16.3. Out-of-domain (654 rows): Needle 2 28.7; LFM 17.0; FunctionGemma 15.6. Needle 2 on rows needing 4+ calls: 14.6 / 15.4. [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- BFCL v4 single-turn (3,641 rows), overall: Apple FM 61.7; LFM2.5 230M 60.8; FunctionGemma 46.1; Needle 2 42.6 (last). Irrelevance detection (correctly not calling): LFM 77.7; FunctionGemma 72.1; Needle 2 60.8; Apple FM 28.3. Well-formed rate: Needle 2 93.4 (lowest of the four; FunctionGemma 100.0). [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Cactus's stated caveats: baselines stay at f16 (favours baselines); "Needle is trained specifically for agentic tool calling and nothing else, while every baseline is a general language model" (favours Needle). "There is no clean way to level both at once, so we do not try." [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Needle 2 fine-tuned (chart data): DroidCall 17.0 → 74.5 (DeepSeek V4 Flash 58.5); Seal-Tools in-domain 33.0 → 56.0 (DeepSeek 50.1); out-of-domain 29.4 → 56.7 (DeepSeek 51.5); Mobile Actions 63.9 → 84.6 (DeepSeek 87.4). "ahead of DeepSeek V4 Flash... on three of the four benchmarks." [CO][DR] — [Needle 2 blog chart](https://cactuscompute.com/blog/needle-2)

**Needle 3 launch chart (Sep 2026).** Needle slices run through the shipped CQ2 binary "with the confidence gate on"; baselines f16 under vLLM; DeepSeek via cloud API. Values read from the page's chart data. [CO][DR] — [Needle 3 page](https://cactuscompute.com/needle)

| Model (params) | Mobile Actions | DroidCall | BFCL v4 | DSTC8 F1 | SNIPS gold F1 | SNIPS 7-way F1 |
|---|---|---|---|---|---|---|
| Needle3-4L (29M) | 11.7 | 21.0 | 19.5 | 6.9 | 7.7 | 4.3 |
| Needle3-8L (52M) | 36.8 | 36.5 | 28.2 | 15.3 | 16.6 | 10.1 |
| Needle3-16L (98M) | 80.7 | 40.0 | 41.3 | 28.5 | 23.5 | 19.2 |
| Needle3-20L (121M) | 86.0 | 47.0 | 50.2 | 40.7 | 30.2 | 24.7 |
| Needle 2 (45M) | 63.5 | 17.0 | n/a | n/a | n/a | n/a |
| LFM2.5 230M | 69.3 | 11.5 | 46.3 | 53 | 27 | 22 |
| FunctionGemma 270M | 65.1 | 16.5 | 46.6 | 27 | 29 | 14 |
| LFM2.5 350M | 72.8 | 32.5 | 59.1 | 20 | 34 | 29 |
| Qwen3.5 0.8B | 76.0 | 28.0 | 56.8 | 49 | 35 | 34 |
| LFM2.5 1.2B | 82.4 | 35.5 | 62.0 | 48 | 43 | 38 |
| Apple FM (~3B) | 57.6 | n/a | n/a | n/a | n/a | n/a |
| DeepSeek V4 Flash (cloud) | 88.4 | 60.5 | 77.2 | 80 | 69.4 | 66.7 |

- Headline claim: "beats models 10× its size on mobile tool calls and matches models 2-3x its size on extraction." [CO][DR] — [Needle 3 page](https://cactuscompute.com/needle)
- Needle 3 fine-tuning chart, "base and tuned both scored with forced calls" (confidence gate off). DroidCall base → tuned for 2L/4L/8L/16L/20L: 20.5→56.5, 34.5→62.5, 42.5→68, 49.5→69, 52→70 (DeepSeek 60.5). Mobile Actions base → tuned: 0.9→66.7, 19→79.1, 45.2→83.7, 80.2→84.6, 86.7→84.5 (DeepSeek 88.4). [CO][DR] — [Needle 3 page chart](https://cactuscompute.com/needle)
- Claim as worded: "fine-tuning on DroidCall lifts every subnetwork by 18 to 36 points, and from 4 layers up the tuned subnetwork passes DeepSeek V4 Flash." [CO][DR] — [Needle 3 page](https://cactuscompute.com/needle)
- 2-layer base model: "at 2 layers the base model's confidence head withholds almost every call, so its gated score is zero... Two layers of a generalist is not a product." [CO][DR] — [Intelligence Ladders post](https://cactuscompute.com/blog/intelligence-ladders)
- Effect of grounding: "the same 121M model moved by close to twenty points on a mobile benchmark once every argument was grounded in the request instead of predicted freely." [CO][DR] — [Designing tools post](https://cactuscompute.com/blog/designing-tools-for-needle)
- v1: only a qualitative claim: "it beats FunctionGemma-270M, Qwen-0.6B, Granite-350M, LFM2.5-350M on single-shot function calling, those models have more scope/capacity and excel in conversational settings." No table. [CO][DR] — [Needle v1 blog](https://cactuscompute.com/blog/needle)

**Independent data points**
- One HN user fine-tuned for "a tool-calling interface to a Runescape database site" (230 test cases): FunctionGemma fine-tune (BF16) 209/230 correct tool shape (90.9%), 196/230 exact arguments (85.2%); Needle 3 fine-tuned (W4A8) 74/230 (32.2%) and 47/230 (20.4%); Needle 2 fine-tuned (W4) 59/230 (25.7%) and 43/230 (18.7%). "still not quite capable enough for my target compared to FunctionGemma." [IND][DR] — [HN thread 49748553](https://news.ycombinator.com/item?id=49748553)
- Another HN user ran Needle 3 on a home-assistant test suite: with household context passed in, "5 of 27 single-turn requests right, versus 8 of 27 without" (the user notes Claude wrote the integration and the interpretation). [IND][DR] — [HN thread 49748553](https://news.ycombinator.com/item?id=49748553)
- An HN user reported Needle 3 was "very bad" for labelling; "MNLI was better." [IND][DR] — [HN thread 49748553](https://news.ycombinator.com/item?id=49748553)
- Raspberry Pi post: 6 of 6 demo prompts produced the intended call or correct refusal; Eben Upton quote: "Needle 2 is rather excellent." Not a benchmark. [SEMI][DR] — [Raspberry Pi news](https://www.raspberrypi.com/news/turn-text-input-into-actions-with-needle-a-14mb-function-calling-llm/)
- Secondary coverage states the benchmark and speed claims "remain Cactus Compute's measurements." [IND][SS] — [runtimewire](https://runtimewire.com/article/cactus-needle-2-14mb-agent-model-tiny-devices)

### Inferences
- The Needle 3 fine-tuning headline is selective. On DroidCall the tuned 4L+ slices do exceed DeepSeek (62.5–70 vs 60.5). On Mobile Actions, by Cactus's own chart, no tuned slice does (best 84.6 vs 88.4), and fine-tuning the full 20L model slightly lowered its score (86.7 → 84.5). The HN title "can match DeepSeek V4 Flash" holds for one 200-row benchmark after fine-tuning on that benchmark's own training data.
- "Beats models 10× its size" is true on Mobile Actions only (86.0 vs LFM2.5 1.2B at 82.4). On BFCL v4 Needle 3 (50.2) trails LFM2.5 350M (59.1), Qwen3.5 0.8B (56.8) and LFM2.5 1.2B (62.0); on DSTC8 extraction it trails LFM2.5 230M (40.7 vs 53).
- Numbers shift between Cactus's own posts: Needle 2 on Mobile Actions is 63.7 (launch table), 63.9 (launch fine-tune chart) and 63.5 (Needle 3 chart); FunctionGemma 64.0 vs 65.1; LFM2.5 230M 69.1 vs 69.3; DeepSeek 87.4 vs 88.4. Differences are small but show the harness changed; treat the last digit as noise.
- Two scoring modes are mixed in Needle 3 material: "confidence gate on" (scatter chart, e.g. DroidCall 20L = 47.0) and "forced calls" (fine-tune chart, 20L = 52). Comparing across charts is unsafe.
- Baselines are un-fine-tuned general checkpoints at f16; Needle is a specialist. Fine-tuned baselines (as the independent Runescape test did for FunctionGemma) could reverse the picture. Cactus did not publish fine-tuned-baseline comparisons.
- The 4L (29M, about 8 MB) base model is weak out of the box (11.7% on Mobile Actions); the "8 MB" headline size is only useful after fine-tuning.
- DroidCall and Seal-Tools absolute scores are low for all small models (11–33% for Needle 2-class models), so "trading wins" happens at a low level of reliability.

### Gaps
- No Mobile Actions number for Needle v1 anywhere. No Needle 3 Seal-Tools results. No Needle 2 extraction results.
- The number of distinct tools in Mobile Actions and the train-split size are not stated on the dataset card as read (its size tag reads "10M<n<100M", which looks like a mis-set tag). From memory, not verified today: Google reported FunctionGemma at roughly 58% on Mobile Actions before and 85% after fine-tuning; if right, Cactus's 64–65% baseline is for the un-tuned model and a tuned FunctionGemma would match tuned Needle.
- No independent run of Needle on Mobile Actions, DroidCall, Seal-Tools or BFCL was found. No entry on the official BFCL leaderboard was checked.
- Who made DroidCall/Seal-Tools and their designs were not researched here.
- Whether Mobile Actions (or look-alike synthetic data) was in Needle's training corpus is not stated; Cactus says base models were "measured on benchmarks it never saw."
- No third-party review article with its own measurements was found; aggregator pages (mer.vin, stork.ai, pyshine, developersdigest, mindstudio) appear to repeat vendor numbers [SS].

---

## 5. Limits and criticisms

### Takeaway
Needle is a pattern-matcher for direct commands over a small, carefully described toolset (ideally five tools or fewer). It is weak at implied meaning, relative values, unit conversion, multi-step chains and non-English input, and users repeatedly showed confident wrong calls in the public demos. Cactus's answer is a layered safety net (grammar, confidence score, grounding gates, regex triggers) plus fine-tuning.

### Cited Findings

**Number of tools**
- "Keep the toolset small. Five or fewer tools render directly... Every extra tool in a turn is a chance to misroute." For big catalogues Cactus recommends two passes rather than one. [CO][DR] — [Designing tools post](https://cactuscompute.com/blog/designing-tools-for-needle)
- Staff on HN: "the model tends to work better with fewer tools." [CO][DR] — [HN thread 49748553](https://news.ycombinator.com/item?id=49748553)

**Multi-step, multi-call, multi-turn**
- Staff: "the model is pretty good at simpler tool calls and parallel ones, but can struggle with implied references and multistep reasoning... for some things you just have to have a model that is properly sized." [CO][DR] — [HN thread 49748553](https://news.ycombinator.com/item?id=49748553)
- Staff: "Implications and relations are hard for the model to understand (things like go to the living room, then the kitchen, and back)... Reasoning isn't true reasoning in the way general LLMs do it, it is more like grounding." [CO][DR] — [HN thread 49748553](https://news.ycombinator.com/item?id=49748553)
- Benchmarks: Needle 2 scores 48.4% on two-call Mobile Actions rows vs 71.3% on one-call rows; 0.0% on DroidCall two-call rows; about 15% on Seal-Tools rows needing 4+ calls; 22.5–30% on BFCL parallel categories. [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Independent: "Follow-ups mostly broke. 'take off the last one' removed the whole list." "It can't use household context... Passing them anyway made results worse." [IND][DR] — [HN thread 49748553](https://news.ycombinator.com/item?id=49748553)
- Local fine-tuning data: "multi-turn lines are skipped." [CO][DR] — [Fine-tuning Needle post](https://cactuscompute.com/blog/finetuning-needle)
- v1 was "single-shot" only; HN commenters asked how it would chain `web_search → summarize_url → send_email`; one replied "This isn't for agentic loops." [CO/IND][DR] — [HN thread 48111896](https://news.ycombinator.com/item?id=48111896)

**Argument accuracy and wrong/hallucinated calls (independent user reports)**
- v1: prompt "i need to contact my boss i will be late" produced `set_timer` "20 minutes" for one user; for another it produced `send_email` to an invented address "boss@company.com"; with context added it emitted three near-duplicate `send_email` calls. [IND][DR] — [HN thread 48111896](https://news.ycombinator.com/item?id=48111896)
- Needle 2 demo: "Make it a little warmer in here" → `set_thermostat` temperature 65, mode "cool", reasoning "'warmer' implies need for cooling" (another user saw confidence 0 for it). "Make the living room dark" → lights on. "HN" / "potato" / "I'm hungover" → `lock_door` "front door" at confidence 0. "turn on the tv" → `lock_door` with door "tv" (confidence 0.0158). "5° warmer" → sets temperature to 5°F. A "degrees Fahrenheit" description did not make it convert "70 degrees Celsius". [IND][DR] — [HN thread 49246804](https://news.ycombinator.com/item?id=49246804)
- Needle 3 demo: "I need a wee" → plays music; "sleepy time" → starts the vacuum in the bedroom; "it's too cold" → thermostat down; "warm the house" → lights to "warm brightness"; "My car crashed I need help" → `play_music` with "confidence": 1; after Cactus added an emergency tool, "call nine one one" → `call_contact` with invented placeholder number "9101". [IND][DR] — [HN thread 49748553](https://news.ycombinator.com/item?id=49748553)
- Needle 3 test suite: "It copies numbers instead of converting them. '25 minute timer' becomes duration_seconds: 25, and 'twelve minutes' becomes 120. The first one comes with 100% confidence." "'wake me up at 6 30' became a 630-second timer." "take the paper towels off the list" became an add. [IND][DR] — [HN thread 49748553](https://news.ycombinator.com/item?id=49748553)
- Cactus's response pattern: it edited the demo's tool presets and triggers after complaints ("We updated them just now"; "presets updated for you now"). [CO][DR] — [HN thread 49748553](https://news.ycombinator.com/item?id=49748553)

**Refusal when no tool fits; confidence**
- Design: off-topic requests return an empty call; confidence is "the minimum of two signals: a calibrated post-hoc head... and the decoding probability of the call tokens"; the engine withholds calls scoring under 0.1 into `suppressed_calls`. Suggested routing: act at ≥0.7, confirm in the middle band, else refuse; "a light switch tolerates 0.5, a transfer wants 0.9 and a confirmation anyway." [CO][DR] — [Needle confidence post](https://cactuscompute.com/blog/needle-confidence)
- Extra rule-based "grounding gates" withhold calls when the request negates the verb, reports someone else's command, lacks a required number, etc. [CO][DR] — [Needle confidence post](https://cactuscompute.com/blog/needle-confidence)
- Counter-evidence: "It never declined anything with our full tool set. Background chatter became note_save 'blue one' at 0.99 confidence." [IND][DR] — [HN thread 49748553](https://news.ycombinator.com/item?id=49748553)
- Needle 2 BFCL irrelevance score 60.8 (below FunctionGemma 72.1 and LFM2.5 77.7): by Cactus's own table it wrongly called a tool on roughly 4 in 10 irrelevant prompts. [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- HN critique of calibration evidence: "that's not the same thing as reporting a benchmark across a test set... It doesn't tell me with what reliability the confidence will be below a given threshold when it should be." Another: "'do not lock the door'" produced the same lock call. [IND][DR] — [HN thread 49246804](https://news.ycombinator.com/item?id=49246804)
- Locally fine-tuned models lose the score: "Local fine-tuning does not train it... `Cactus(weights=...)` reports confidence as None." [CO][DR] — [Fine-tuning Needle post](https://cactuscompute.com/blog/finetuning-needle)
- Triggers trade safety for recall: a triggered call "ships even below the confidence floor." [CO][DR] — [Needle 3 page](https://cactuscompute.com/needle)

**Languages**
- Founder on HN (Needle 2): "7 for now, English, German, Polish, French, Dutch, Latin, Italin." A user: "Tested with german and this kinda works, but confidence suffers"; another was "quite impressed" with German commands. [CO/IND][DR] — [HN thread 49246804](https://news.ycombinator.com/item?id=49246804)
- Company docs: "Non-English text fragments into roughly 1.7 times more tokens (measured on Spanish), which taxes both quality and the context budget." "Non-English deployments should also treat the score with caution; correct Spanish calls have been measured at 0.0." [CO][DR] — [Fine-tuning Needle post](https://cactuscompute.com/blog/finetuning-needle); [Needle confidence post](https://cactuscompute.com/blog/needle-confidence)

**Scope and general criticisms**
- Outside its training distribution: "Needle was not trained for general function calling... The gap concentrates where its training data has never been: Java, JavaScript, and the parallel multi-call categories." [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- "Is it better than regexes?": "14MB can fit a lot of regexes for fixed sentences"; "needing very precise sentences to work at which point you may as well do regexes"; a commenter proposed a BM25 + classifier baseline; staff: "we might look into creating a baseline like this." [IND/CO][DR] — [HN 49246804](https://news.ycombinator.com/item?id=49246804); [HN 49748553](https://news.ycombinator.com/item?id=49748553)
- Over-claiming: "The 'DeepSeek 4 Flash grade' claim seems far fetched"; "the growing number of dubious claims that a tiny model beats LLMs will make any useful innovation be overlooked... tell me the anti use case clearly." [IND][DR] — [HN thread 49748553](https://news.ycombinator.com/item?id=49748553)
- Use-case scepticism on the Raspberry Pi post: "I fail to see the utility of using an LLM to handle deterministic logic... a solution looking for a problem." [IND][DR] — [Raspberry Pi news comments](https://www.raspberrypi.com/news/turn-text-input-into-actions-with-needle-a-14mb-function-calling-llm/)
- Maintenance friction: "the .cact format is tied to the engine version, so an archive exported by an older package will not load." An open HF discussion (15 Nov 2026) is titled "Please preserve runtime wheels required by published releases." [CO/IND][DR] — [Fine-tuning post](https://cactuscompute.com/blog/finetuning-needle); [HF org page](https://huggingface.co/Cactus-Compute)
- Early portability report (Needle 2): "I tried using it on my x86_64 desktop. Could not use it. Then I tried... raspberry pi 4. Did not work." [IND][DR] — [HN thread 49246804](https://news.ycombinator.com/item?id=49246804)
- Telemetry: the Python package sends "Anonymous usage counts (function name, package version, OS, a random install id; never prompts, outputs or data)" unless `CACTUS_TELEMETRY=0`; the Needle 2 web sandbox states "Prompts, tool definitions and responses are saved for evaluation." [CO][DR] — [Python docs](https://cactuscompute.com/blog/needle-python-docs); [Needle 2 blog](https://cactuscompute.com/blog/needle-2)

### Inferences
- The confidence score is necessary but not sufficient: independent reports include wrong calls at confidence 0.99–1.0. For anything consequential (locks, payments, emergency calls) a human confirmation step or a larger fallback model is needed, as Cactus's own docs imply.
- Many Needle 3 "features" (triggers, verb-polarity rules, number-evidence gates, date checks) are hand-written rules wrapped around the network. Shipped behaviour is model plus rules; benchmark gains from v2 to v3 cannot be attributed to the neural network alone ("close to twenty points" came from grounding).
- "The toolset is the product" (Cactus's phrase) shifts the burden to the developer: good results require narrow tools, user-vocabulary enum names, defaults, triggers and a test suite. That is closer to building a classic intent/slot system than to dropping in an LLM.
- Demo failures were mostly indirect phrasings ("it's too cold"); direct commands ("turn all the lights off") generally worked. The fair summary is "robust to wording of direct commands, not to implication".

### Gaps
- No published false-accept/false-reject curves for the confidence score.
- No systematic multilingual evaluation; the "7 languages" list is one HN comment (including "Latin", possibly a slip).
- Maximum practical number of tools before accuracy degrades is not quantified beyond "five or fewer".
- No independent adversarial or safety evaluation found.

---

## 6. Licence, developer usage, adoption signals

### Takeaway
Licensing moved from MIT (v1) to Apache 2.0 (Needle 2 and 3, including prebuilt runtime binaries). Developers use a pip package (`cactus-needle` for v2, `cactus-compute` for v3), a small C API with static libraries for 17 targets (including iOS, Android, watchOS, WebAssembly), or a CLI/HTTP server. Adoption signals are strong for attention (HN front page three times; about 185k HF downloads for Needle 3) and thin for verified production use (one named customer, Pebble, via a company-published testimonial).

### Cited Findings

**Licence**
- v1: HF card metadata `license: mit` and text "MIT"; Show HN: "Everything is MIT licensed." [CO][DR] — [HF needle card](https://huggingface.co/Cactus-Compute/needle); [HN 48111896](https://news.ycombinator.com/item?id=48111896)
- The v1 blog page as it reads today says: "Needle is licensed under Apache 2.0; Cactus Engine is available under its own source-available license." [CO][DR] — [Needle v1 blog](https://cactuscompute.com/blog/needle)
- Needle 2: `license: apache-2.0`. [CO][DR] — [HF needle2 card](https://huggingface.co/Cactus-Compute/needle2). Raspberry Pi post: weights and code "are released under Apache 2.0." [SEMI][DR] — [Raspberry Pi news](https://www.raspberrypi.com/news/turn-text-input-into-actions-with-needle-a-14mb-function-calling-llm/)
- Needle 3 / current repo: "Apache 2.0, including the prebuilt runtime binaries." [CO][DR] — [GitHub README, cactus-compute/cactus](https://github.com/cactus-compute/cactus)
- Secondary sources disagree on Needle 2's licence (MIT vs Apache 2.0). [IND][SS] — [mer.vin](https://mer.vin/news/needle-2-cactus-14mb-on-device-tool-calling-model/)

**How developers use it**
- Needle 3: `pip install -U cactus-compute` (Python 3.9+); "The package downloads and caches the engine and requested model weights on first use. Inference runs locally and does not require JAX." Decorate a Python function with `@cactus.tool`, then `cactus.Cactus(tools=[...]).run("...")`. [CO][DR] — [GitHub README](https://github.com/cactus-compute/cactus)
- Needle 2: `pip install cactus-needle`, `import needle`, `@needle.tool`, `needle.Needle(tools=[...])`; now on the repo's `legacy` branch with `cactus-compute<4`. [CO][DR] — [HF needle2 card](https://huggingface.co/Cactus-Compute/needle2); [GitHub README](https://github.com/cactus-compute/cactus)
- Three ways to declare tools: decorated function (docstring = description), `Annotated[..., Field(...)]` constraints, or raw JSON schema; OpenAI-style `{"type":"function","function":{...}}` wrappers accepted. Extraction via `cactus.extract(text, PydanticModel)`. [CO][DR] — [Python docs](https://cactuscompute.com/blog/needle-python-docs)
- C API: `cactus.h` exposes `cactus_load`, `cactus_init`, `cactus_complete`, `cactus_transcribe`, `cactus_embed`; "The native API owns one process-global model and conversation." CLI: `./cactus --model needle3.cact --tools tools.json --prompt "..."` or `--serve` (HTTP `POST /complete`). [CO][DR] — [Supported devices post](https://cactuscompute.com/blog/needle-supported-devices)
- Platforms: "Release 4.0.0 has 17 target folders": macOS arm64; Linux x86-64/arm64/armv7/riscv64/mipsel; Windows x64/arm64; Android arm64/armv7/riscv64; iOS and simulator; tvOS; watchOS; browser/Node WebAssembly; WASI Preview 2 component. Mobile targets get a static library + header, not a Swift/Kotlin SDK. [CO][DR] — [HF Cactus-Compute/cactus card](https://huggingface.co/Cactus-Compute/cactus)
- Same engine also runs Cactus's speech model Whistle (16.9 MB, Oct 2026) and vision model Mirage (22 MB, Nov 2026), so audio can go straight to a tool call. [CO][DR] — [Cactus blog index](https://cactuscompute.com/blog)
- Community ports: Cactus says "The first community port of Needle 3 landed within days of the release, written in Rust"; HN users mention a `needle-rs` npm package and an ESP32-S3 port (memovai/mimimodel). [CO/IND][DR] — [Porting Needle 3 post](https://cactuscompute.com/blog/porting-needle); [HN 49246804](https://news.ycombinator.com/item?id=49246804)

**Adoption signals**
- Hacker News: v1 Show HN (12 May 2026) 776 points, 211 comments; Needle 2 Show HN (10 Aug 2026) 537 points, 185 comments; Needle 3 Show HN (18 Sep 2026) 236 points, 93 comments. [IND][DR] — [HN submissions from cactuscompute.com](https://news.ycombinator.com/from?site=cactuscompute.com); [HN submissions from github.com/cactus-compute](https://news.ycombinator.com/from?site=github.com/cactus-compute)
- Hugging Face on 16 Nov 2026: `needle3` 185k downloads, 417 likes; `needle2` 1.7k downloads, 341 likes; org has 932 followers and 66 model repos. [IND][DR] — [HF org page](https://huggingface.co/Cactus-Compute)
- Company's site claims "312k downloads/mo" and "21k GitHub stars". [CO][DR] — [Needle 3 page](https://cactuscompute.com/needle)
- The original `cactus-compute/needle` repo is now a stub ("Cactus Needle has moved to cactus-compute/cactus") showing 11 stars and 1 commit. [IND][DR] — [github.com/cactus-compute/needle](https://github.com/cactus-compute/needle)
- Named production user (company-published): "Pebble... runs it locally in the Index 01 app to turn spoken requests into actions"; quote attributed to Eric Migicovsky: "We run Cactus Needle locally in the app, instead of relying on the cloud. The model's footprint is tiny and the performance never lets us down." [CO][DR] — [Needle 2 blog](https://cactuscompute.com/blog/needle-2)
- Raspberry Pi published a guest tutorial (22 Sep 2026) with a quote from CEO Eben Upton: "Needle 2 is rather excellent." [SEMI][DR] — [Raspberry Pi news](https://www.raspberrypi.com/news/turn-text-input-into-actions-with-needle-a-14mb-function-calling-llm/)
- Independent hobby/evaluation use: one developer "used Needle as a router" in a personal project (memento); another company was "evaluating small models embedded directly in the web app... needle is one of the most promising" but still needed to test. [IND][DR] — [HN 49246804](https://news.ycombinator.com/item?id=49246804); [HN 49748553](https://news.ycombinator.com/item?id=49748553)
- Company background: Cactus Compute is a Y Combinator S25 company; team of 8 listed on Hugging Face. [IND][DR] — [HN submissions list ("Launch HN: Cactus (YC S25)")](https://news.ycombinator.com/from?site=github.com/cactus-compute); [HF org page](https://huggingface.co/Cactus-Compute)

### Inferences
- HN interest fell with each release (776 → 537 → 236 points) while thread tone grew more critical; the Needle 3 thread is dominated by demo failures and "what is the real use case" questions.
- The "21k GitHub stars" figure is for the whole `cactus-compute/cactus` repo, which predates Needle (it was the general inference SDK launched in 2025), so it is not a Needle-specific signal.
- HF "downloads" for `needle3` are inflated by design: the pip package fetches weights from Hugging Face on first use (and CI runs would too), so 185k downloads does not mean 185k developers.
- The licence story is slightly muddled (v1 MIT on its card, Apache 2.0 on the updated blog; an earlier "source-available" engine). For Needle 3, weights, Python code and prebuilt binaries are Apache 2.0, but the best fine-tuning/2-bit quantisation pipeline is a hosted commercial service.
- "Mobile SDKs" in the conventional sense (Swift package, Kotlin/AAR, React Native) are not offered for Needle 3; integration on phones is via the C library.

### Gaps
- Actual GitHub star/fork counts for `cactus-compute/cactus` could not be parsed from the page; only the company's "21k" claim is available.
- PyPI download statistics were not checked (pypi.org / pypistats not on the allowed host list).
- Whether the C++ engine source is in the public repo, or only prebuilt binaries are Apache-licensed, was not verified. The earlier phrase "source-available license" for the engine suggests it was not fully open at v1.
- The Pebble deployment is confirmed only by Cactus's blog; no Pebble-side source was found. No other named shipping product was found.
- Cactus Platform pricing, funding, and revenue were not researched.
