# Liquid AI: tiny tool-calling models (LFM2.5-230M/350M, Nanos) and d1 decision models

Research date: 2026-11-16. Tags used on every finding:

- **[CR]** company-reported (Liquid AI or a named partner) / **[IND]** independent of Liquid
- **[DR]** direct-read of the primary page (fetched in full with curl) / **[SS]** search-summary only (page not fetched)

Access notes: liquid.ai, docs.liquid.ai, huggingface.co, arxiv.org and news.ycombinator.com were read directly. distillabs.ai, cactuscompute.com, Reddit and all other hosts were NOT readable from this session; anything from them is [SS]. Speed charts on the Hugging Face cards are images and could not be read; only numbers printed as text are reported.

Naming corrections to the brief, from primary sources:

- "Open d1" (7 Oct 2026) = **d1-3B** and **d1-omni-600M**.
- LFM3 (27 Oct 2026) shipped **d1-2B** and **d1-Omni-2B** (not branded "LFM3 d1-2B").
- "d1.2" (16 Nov 2026) = **d1.2** and **d1.2-ultra**, hosted API only, no open weights.
- There are two different things called "Decision Index" (see the d1 section).

---

## 1. LFM2.5-230M and LFM2.5-350M: architecture, size, speed, tool-call format, intended use

### Takeaway
These are ordinary generative chat models (14 and 16 layers, mostly gated short convolutions plus 6 grouped-query attention blocks, 32K context) post-trained to be good at tool calls and extraction. They emit a Pythonic call list between two special tokens, and are about 150 MB / 220 MB at 4-bit. Liquid explicitly says not to use them for math, code or creative writing.

### Cited Findings

Architecture and training
- LFM2.5-350M: 350M parameters, 16 layers (10 double-gated convolution blocks + 6 GQA blocks), 28T-token training budget, 32,768-token context, 65,536 vocabulary, knowledge cutoff mid-2024, 9 languages. [CR][DR] — [HF card LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M)
- LFM2.5-230M: 230M parameters, 14 layers (8 double-gated convolution blocks + 6 GQA blocks), 19T-token training budget, 32,768-token context, 65,536 vocabulary, cutoff mid-2024, 10 languages (adds Italian). [CR][DR] — [HF card LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)
- LFM2.5-350M was released 31 Mar 2026 as "an improved version of our 350M model with additional pre-training (from 10T to 28T tokens) and large-scale reinforcement learning". [CR][DR] — [Liquid blog, 350M](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind)
- LFM2.5-230M was released 25 Jun 2026 as "our smallest model yet". Pre-trained for 19T tokens "including a 32K context extension phase". Post-training has three stages: (1) supervised fine-tuning with distillation from LFM2.5-350M, (2) direct preference optimization, (3) multi-domain reinforcement learning. Liquid calls it a "lightweight post-training recipe designed to preserve flexibility for developers". [CR][DR] — [Liquid blog, 230M](https://www.liquid.ai/blog/lfm2-5-230m)
- The underlying LFM2-350M backbone in the technical report: 16 layers, d_model 1024, feed-forward dim 4608, 16 heads / 8 KV heads / head size 64, 6 attention blocks, convolution kernel 3. All layers use pre-norm RMSNorm; attention uses RoPE and QK-Norm; MLPs are SwiGLU. [CR][DR] — [arXiv 2511.23404, Table 1](https://arxiv.org/html/2511.23404)
- Why this shape: a hardware-in-the-loop architecture search (scored on real device latency/memory plus an internal 50+ evaluation suite that includes tool use) "repeatedly selects a minimal hybrid architecture where most blocks are inexpensive gated short convolution blocks, interleaved with a small minority of GQA blocks"; adding state-space or linear-attention operators did not improve quality and worsened device metrics. [CR][DR] — [arXiv 2511.23404, §2.1](https://arxiv.org/html/2511.23404)
- The tokenizer "includes special tokens for fill-in-the-middle training objectives, tool calling, and the ChatML chat template". [CR][DR] — [arXiv 2511.23404](https://arxiv.org/html/2511.23404)
- LFM2 post-training is three stages: SFT, length-normalised preference alignment, model merging; function calling is a named target. Tool use is roughly 10% of the SFT mixture for the dense models (table row "Tool use 10.1"; the table was read from flattened HTML, so treat the exact column attribution with some caution). [CR][DR] — [arXiv 2511.23404, §4](https://arxiv.org/html/2511.23404)

File sizes (GGUF, read from the Hugging Face file listing on 2026-11-16) [CR][DR]
- LFM2.5-230M: Q4_0 149.1 MB, Q4_K_M 153.4 MB, Q5_K_M 171.6 MB, Q6_K 191.0 MB, Q8_0 246.6 MB, F16 461.9 MB; plus a QAD-Q4_0 (quantization-aware distilled) file at 149.1 MB. — [LFM2.5-230M-GGUF](https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF)
- LFM2.5-350M: Q4_0 219.3 MB, Q4_K_M 229.3 MB, Q5_K_M 260.4 MB, Q6_K 293.4 MB, Q8_0 379.2 MB, F16 711.5 MB; QAD-Q4_0 219.3 MB. — [LFM2.5-350M-GGUF](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF)

Speed and memory
- LFM2.5-230M: 213 tok/s decode on Galaxy S25 Ultra and 42 tok/s on Raspberry Pi 5 (CPU, llama.cpp implied; flash-attention flag tuned per device). Liquid claims "the highest prefill and decode throughput in its class while keeping the smallest memory footprint", but the memory figure is only in a chart image. [CR][DR] — [Liquid blog, 230M](https://www.liquid.ai/blog/lfm2-5-230m)
- LFM2.5-350M headline: "313 tok/s decode on AMD CPU, 188 tok/s on Snapdragon Gen4. Runs under 1GB of memory". [CR][DR] — [HF card LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M)
- LFM2.5-350M partner speed table (1K prefill, 100 decode tokens) [CR][DR] — [Liquid blog, 350M](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind):

| Device | Backend | Framework | Prefill tok/s | Decode tok/s | Peak memory |
|---|---|---|---|---|---|
| AMD Ryzen AI Max 395+ | CPU | llama.cpp (Q4) | 2.9K | 313 | 434 MB |
| Galaxy S25 Ultra (Snapdragon 8 Elite) | NPU | RunAnywhere (Q4) | 2.8K | 15 | 169 MB |
| Galaxy S25 Ultra | GPU | RunAnywhere (Q4) | 5.3K | 62 | 81 MB |
| Apple A18 Pro | GPU | Mirai (bf16) | 1,953 | 73 | 945 (unit not printed) |
| Apple M1 Max | GPU | Mirai (bf16) | 10.4K | 328 | 940 MB |
| Apple M5 Max | GPU | Mirai (bf16) | 44.8K | 564 | 1 GB |
| iPhone 13 Mini | CPU | Cactus Engine (int8) | 496 | 88 | 56 MB |
| Google Pixel 6a | CPU | Cactus Engine (int8) | 208 | 42 | 328 MB |
| Raspberry Pi 5 | CPU | Cactus Engine (int8) | 200 | 30 | 300 MB |

- Server-side: LFM2.5-350M reaches 40.4K output tokens/s on a single H100 with SGLang at high concurrency ("over 3.5B tokens per day"). [CR][DR] — [Liquid blog, 350M](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind)

Tool-calling format (both models; identical wording on both cards) [CR][DR]
- Chat template is ChatML-like: `<|startoftext|><|im_start|>system ... <|im_end|>`, then `user`, `assistant`, `tool` turns. — [HF card LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)
- Tool definitions: "Provide the list of tools as a JSON object in the system prompt, or use `tokenizer.apply_chat_template()` with `tools=...`." The shipped 230M template renders them as `List of tools: [ {json}, ... ]` appended to the system prompt. — [HF card](https://huggingface.co/LiquidAI/LFM2.5-230M); [chat_template.jinja](https://huggingface.co/LiquidAI/LFM2.5-230M/raw/main/chat_template.jinja)
- Tool call: "By default, LFM2.5 writes Pythonic function calls (a Python list between `<|tool_call_start|>` and `<|tool_call_end|>` special tokens), as the assistant answer. You can override this behavior by asking the model to output JSON function calls in the system prompt." Example: `<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.` — [HF card](https://huggingface.co/LiquidAI/LFM2.5-350M)
- The list syntax allows several calls in one turn (`[f(...), g(...)]`); the llama.cpp guide notes "LFM2.5 may request several calls at once". — [docs: llama.cpp function calling](https://docs.liquid.ai/deployment/on-device/llama-cpp/function-calling.md)
- Tool result goes back as a `tool` role message; the model then writes a plain-text answer. — [HF card](https://huggingface.co/LiquidAI/LFM2.5-350M)
- Docs: add "Output function calls as JSON" to the system prompt for JSON; "Depending on the model, the chat template also wraps tool definitions in `<|tool_list_start|>`/`<|tool_list_end|>` and tool responses in `<|tool_response_start|>`/`<|tool_response_end|>`." — [docs: Tool Use](https://docs.liquid.ai/lfm/key-concepts/tool-use)
- In llama.cpp, start `llama-server` with `--jinja` and pass OpenAI-style `tools`; llama.cpp "ships a dedicated parser" and returns structured `tool_calls`, "There is nothing to parse on the client". Older builds fall back to a generic parser. `tool_choice` accepts auto/none/required/a named function. — [docs: llama.cpp function calling](https://docs.liquid.ai/deployment/on-device/llama-cpp/function-calling.md)
- Recommended sampling: temperature 0.1, top_k 50, repetition_penalty 1.05. — [HF card](https://huggingface.co/LiquidAI/LFM2.5-350M)
- LFM3 (Oct 2026) changed the format: tools go between `<tools>` and `</tools>`, calls are XML-like `<tool_call><function=name><parameter=arg>value</parameter></function></tool_call>`, "a superset of qwen3_coder". The Pythonic format is now documented as the "LFM2/LFM2.5" format. — [docs: Tool Use](https://docs.liquid.ai/lfm/key-concepts/tool-use)
- For large tool catalogues the docs describe "defer loading" plus a `search_tools` tool, with discovered definitions injected in an `<additional_tools>` block. The examples use LFM3-1.7B, and the docs say to "check the model's chat template for deferred tools and tool_reference support". — [docs: Tool Use](https://docs.liquid.ai/lfm/key-concepts/tool-use)

Recommended and non-recommended uses [CR][DR]
- 350M card: "We recommend using it for data extraction, structured outputs, and tool use. It is not recommended for knowledge-intensive tasks and programming." — [HF card LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M)
- 350M blog: "an ideal solution to power large-scale data extraction pipelines or lightweight on-device agentic pipelines. However, it is not recommended for tasks like math, code, and creative writing." — [Liquid blog, 350M](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind)
- 230M card: "We recommend using it for data extraction and lightweight on-device agentic pipelines. It is not recommended for reasoning-heavy workloads such as advanced math, code generation, or creative writing." — [HF card LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)
- Both cards: "We recommend fine-tuning LFM2.5 for your specific use case to achieve the best results." — [HF card](https://huggingface.co/LiquidAI/LFM2.5-230M)
- Robotics demo: a fine-tuned LFM2.5-230M ran on a Unitree G1 humanoid's onboard Jetson Orin as a "skill-selection layer", turning one spoken-style instruction into a sequence of tool calls to pre-trained skills. Liquid calls the behaviours "deliberately simple at this stage". [CR][DR] — [Liquid blog, 230M](https://www.liquid.ai/blog/lfm2-5-230m)

### Inferences
- Teaching point: these models "call tools" purely by generating text in a trained format. The two special tokens only delimit the call; correctness of the function name and arguments is not enforced by the model, so a parser (or constrained decoding) is still needed. Liquid's fine-tuning overview lists "constrained decoding for strict JSON or grammar-bound outputs" as a lever to try first.
- A Pythonic call is shorter in tokens than the equivalent JSON, which matters when decode speed is 30–200 tok/s; Liquid does not state this rationale in the pages read.
- The "188 tok/s on Snapdragon Gen4" headline on the 350M card does not match any row of the blog's partner table (15 tok/s NPU, 62 tok/s GPU via RunAnywhere). The 188 figure presumably comes from the llama.cpp CPU chart (an image). The numbers are from different runtimes and should not be mixed.
- At Q4 the weights alone are ~150 MB (230M) and ~220 MB (350M); reported peak memory ranges from 56 MB to ~1 GB depending on runtime and precision, so "RAM needed" is a property of the runtime as much as of the model.

### Gaps
- No text-form RAM figure for LFM2.5-230M, and no text-form per-device table for 230M beyond the two decode speeds; the charts are images.
- Liquid does not publish which RL environments or tool-call datasets were used in post-training LFM2.5.
- Whether LFM2.5-230M/350M templates support deferred tool loading was not verified.

---

## 2. Liquid Nanos "Tool" and "Extract" models: how they were made, what was reported, are they deprecated?

### Takeaway
The September 2025 Nanos were task-specific fine-tunes of LFM2 with benchmark claims made only on Liquid's own private test sets. As of November 2026 Liquid's docs mark LFM2-1.2B-Tool, LFM2-1.2B-Extract and LFM2-350M-Extract as deprecated in favour of the general LFM2.5 models.

### Cited Findings
- Liquid Nanos launched 25 Sep 2025 as "a family of 350M–2.6B parameter foundation models" occupying "RAM space of 100Mb to 2GB", described as "experimental task-specific" models. The launch list contains LFM2-Extract (350M and 1.2B), LFM2-350M-ENJP-MT, LFM2-1.2B-RAG, LFM2-1.2B-Tool, LFM2-350M-Math and community Luth-LFM2. [CR][DR] — [Liquid blog, Nanos](https://www.liquid.ai/blog/introducing-liquid-nanos-frontier-grade-performance-on-everyday-devices)
- LFM2-1.2B-Tool: based on LFM2-1.2B, "designed for concise and precise tool calling. The key challenge was designing a non-thinking model that outperforms similarly sized thinking models for tool use." Rationale: chain-of-thought adds waiting time on edge devices. Recommended greedy decoding (temperature 0). "The system prompt must provide all the available tools." Supports single- and multi-turn. [CR][DR] — [HF card LFM2-1.2B-Tool](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool)
- LFM2-1.2B-Tool format is the older, more heavily tokenised variant: tool list between `<|tool_list_start|>`/`<|tool_list_end|>`, Pythonic call between `<|tool_call_start|>`/`<|tool_call_end|>`, result between `<|tool_response_start|>`/`<|tool_response_end|>`. [CR][DR] — [HF card LFM2-1.2B-Tool](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool)
- Benchmark for the Tool model: "a proprietary benchmark that was specifically designed to prevent data contamination". Results are shown only as a chart image; no public benchmark (e.g. BFCL) is given. [CR][DR] — [HF card LFM2-1.2B-Tool](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool)
- LFM2-Extract: turns unstructured documents into JSON, XML or YAML; defaults to JSON with no system prompt; a schema in the system prompt is recommended; "intended for single-turn conversations"; greedy decoding strongly recommended. [CR][DR] — [HF card LFM2-350M-Extract](https://huggingface.co/LiquidAI/LFM2-350M-Extract)
- Extract training data "was primarily synthetic", varied across document types, domains, lengths, languages and density of relevant text. [CR][DR] — [HF card LFM2-350M-Extract](https://huggingface.co/LiquidAI/LFM2-350M-Extract)
- Extract evaluation: 5,000 documents, 100+ topics, five metrics (syntax score, format accuracy, keyword faithfulness, LLM-judge absolute score 1–5, LLM-judge relative score). Claims: "LFM2-350M-Extract outperforms Gemma 3 4B at this task, a model more than 11x its size"; LFM2-1.2B-Extract performs "on a level higher than Gemma 3 27B". Numbers are in a chart image. [CR][DR] — [HF card LFM2-350M-Extract](https://huggingface.co/LiquidAI/LFM2-350M-Extract); [Liquid blog, Nanos](https://www.liquid.ai/blog/introducing-liquid-nanos-frontier-grade-performance-on-everyday-devices)
- Deprecation: "This model is deprecated. Use LFM2.5-1.2B-Instruct for tool calling instead, which offers improved accuracy and follows the standard tool use format." [CR][DR] — [docs: LFM2-1.2B-Tool](https://docs.liquid.ai/lfm/models/lfm2-1.2b-tool.md)
- Deprecation table: LFM2-1.2B-Tool → LFM2.5-1.2B-Instruct; LFM2-1.2B-Extract → LFM2.5-1.2B-Instruct; LFM2-350M-Extract → LFM2.5-350M; LFM2-350M → LFM2.5-350M; LFM2-1.2B-RAG → LFM2.5-1.2B-Instruct. "Deprecated models stay downloadable but are no longer maintained." [CR][DR] — [docs: Deprecations](https://docs.liquid.ai/lfm/help/deprecations.md)
- The current "Liquid Nanos" docs page no longer lists a tool-calling Nano at all. It now covers extraction (vision extract models, Japanese PII), summarisation, retrieval/encoders and translation. [CR][DR] — [docs: Liquid Nanos](https://docs.liquid.ai/lfm/models/liquid-nanos.md)
- LFM2.5-350M vs its predecessor on the same public tests: BFCLv3 22.95 → 44.11, CaseReportBench (extraction) 11.67 → 32.45, IFBench 18.20 → 40.69. [CR][DR] — [Liquid blog, 350M](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind)

### Inferences
- Liquid's strategy shifted between Sep 2025 and 2026: from shipping separate task-specific checkpoints to shipping one general small model whose post-training already targets tool use and extraction, and telling users to fine-tune that themselves.
- Because the Nanos numbers were on private benchmarks shown as images, they cannot be compared with BFCL-style numbers for other models.

### Gaps
- **LFM2-350M-Tool could not be verified.** `huggingface.co/LiquidAI/LFM2-350M-Tool` returned HTTP 401 on 2026-11-16, the model does not appear in the Hugging Face API search for LiquidAI "Tool" models (only LFM2-1.2B-Tool and its GGUF do), the Nanos launch post names only the 1.2B Tool model, and the deprecation table has no 350M-Tool row. I cannot say whether it was removed, gated, or renamed.
- No numeric Tool-model benchmark results are available as text.

---

## 3. d1 decision models: what a "decision model" is, mechanism, sizes, latency, benchmarks, uses

### Takeaway
A d1 model is a language-model backbone run for exactly one forward pass: the prompt ends at an "answer slot" and the answer is a probability distribution read directly from the model's output there (over single-token option codes, or via pointer/box readouts), so nothing is generated and a malformed output is impossible. It is aimed at routing, guardrails, classification, scoring and reading values, and is explicitly not a way to produce free-form tool arguments.

### Cited Findings

Release timeline [CR][DR]
- ~29 Sep 2026: first hosted "d1" (text only); HN submissions titled "d1: Liquid AI's First Decision Model" are dated 47–48 days before 16 Nov. — [HN submissions from liquid.ai](https://news.ycombinator.com/from?site=liquid.ai)
- 5 Oct 2026: hosted d1 gains vision. — [Liquid blog, d1](https://www.liquid.ai/blog/d1-decision-model)
- 7 Oct 2026: open weights d1-3B and d1-omni-600M. — [Liquid blog, Open d1](https://www.liquid.ai/blog/d1-open)
- 15 Oct 2026: hosted d1-ultra, $0.042 per million input tokens. — [Liquid blog, d1-ultra](https://www.liquid.ai/blog/d1-ultra)
- 27 Oct 2026: LFM3 release includes open d1-2B and d1-Omni-2B. — [Liquid blog, LFM3](https://www.liquid.ai/blog/lfm3)
- 16 Nov 2026: hosted d1.2 ($0.02/M input tokens) and d1.2-ultra ($0.042/M), adding audio input and five new answer types. — [Liquid blog, d1.2](https://www.liquid.ai/blog/d1-2)
- Docs state: "d1.1, d1-ultra, and the earlier d1 are deprecated"; d1-3B is "deprecated - use d1-2B instead". — [docs: Decision models](https://docs.liquid.ai/lfm/models/decision-models); [docs index](https://docs.liquid.ai/llms.txt)

Definition
- "Decision models answer questions about a situation with a probability for each possible answer. The d1 decision model takes unstructured data (e.g., text, images, or both) and one or more questions as input, reads them in one forward pass, and returns the probabilities, without generating any tokens." [CR][DR] — [Liquid blog, d1](https://www.liquid.ai/blog/d1-decision-model)
- Input is a "state" (string, any JSON value, or none if media are the whole state) plus named questions. Output is `{"answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}`. [CR][DR] — [HF card d1-3B](https://huggingface.co/LiquidAI/d1-3B)
- Three core question types: **noul** (yes/no → P(yes)), **choice** (one of named options → choice, confidence, per-option probabilities), **score** (2–10 ordered levels → expected level, confidence, probabilities). Each question has `type`, `instructions`, `criteria`; this is called "the Decision Index schema". [CR][DR] — [HF card d1-3B](https://huggingface.co/LiquidAI/d1-3B)
- Typed answers added with d1-2B / d1-Omni-2B / d1.2: choice with `multiple` (each option gets its own P(yes)); `text`, `number`, `date` (a value read from the input, returned with its character span and a `found` probability); `number` with `range` (an estimate with median and p10–p90 interval, over at most 128 values); `box` / `boxes` (rectangle as fractions of the picture, up to 16 instances, with a count); `spans` (Liquid API only). "A value or a box is missing when found is under 0.5". [CR][DR] — [docs: Decision models](https://docs.liquid.ai/lfm/models/decision-models)
- "Values are read, not converted: number and date are parsed from the span." [CR][DR] — [docs: Decision models](https://docs.liquid.ai/lfm/models/decision-models)

Mechanism (read from the model code shipped in the Hugging Face repos) [CR][DR]
- d1-3B code: "A System One decision is a single forward pass that stops at the answer slot: everything the model sees is built here, and the answer is a softmax over its options' tokens." — [d1-3B prompt.py](https://huggingface.co/LiquidAI/d1-3B/raw/main/prompt.py)
- d1-3B code: "The model sees the state and the question once, and the answer is read off the logits at the answer slot. Nothing is decoded, so a schema violation is impossible and `tokens generated per decision` is zero." — [d1-3B runner.py](https://huggingface.co/LiquidAI/d1-3B/raw/main/runner.py)
- How options map to tokens in d1-3B: each choice label is assigned a distinct single-token code (A–Z, then 00–99, etc.); yes/no reads the tokens for "yes/Yes/YES" and "no/No/NO"; score reads the single-digit tokens. The option probability is a softmax over only those token ids ("the answer is a softmax over these and nothing else"), optionally passed through a stored calibration. — [d1-3B prompt.py](https://huggingface.co/LiquidAI/d1-3B/raw/main/prompt.py)
- So d1-3B has no separate classification head: it reuses the language model's vocabulary logits at one position. The class is literally `D1Model(Lfm2VlForConditionalGeneration)`. — [d1-3B modeling_d1.py](https://huggingface.co/LiquidAI/d1-3B/raw/main/modeling_d1.py)
- Multiple questions about one state share the computation: the state (and images) form a "trunk" read once, each question is a "branch", "packed with no padding". — [d1-3B runner.py](https://huggingface.co/LiquidAI/d1-3B/raw/main/runner.py)
- d1-omni-600M is different: it is built on a bidirectional encoder and does have a dedicated head. The card describes "a 381M shared trunk and decision head, a 94M vision encoder and a 112M audio encoder"; the code defines `DecisionHead(d, config.head_layers)` with `head_layers=2`. Text answers are calibrated with per-type temperatures stored in config.json. — [HF card d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M); [d1-omni-600M modeling_d1.py](https://huggingface.co/LiquidAI/d1-omni-600M/raw/main/modeling_d1.py)
- d1-2B changes attention so that, on the global-attention layers, the state can be read in both directions and each question sees the state and its own question but not other questions ("set" mode, `[state][q1]..[qN]`, default 16 questions per set). Convolution and sliding-window layers stay causal. — [d1-2B attention.py](https://huggingface.co/LiquidAI/d1-2B/raw/main/attention.py); [HF card d1-2B](https://huggingface.co/LiquidAI/d1-2B)
- d1-2B extraction uses "Parameter-free readouts over a row's state positions", i.e. a pointer to start/end token positions in the input, with position 0 meaning "nothing"; span caps are 512 tokens for text and 32 for number/date. — [d1-2B pointer.py](https://huggingface.co/LiquidAI/d1-2B/raw/main/pointer.py)
- d1-2B boxes: "a presence and four edges, each a distribution over the same grid of fractions of the picture"; an edge is the median of its distribution; for `boxes`, block k's presence is P(count ≥ k). — [d1-2B box.py](https://huggingface.co/LiquidAI/d1-2B/raw/main/box.py)
- "It reads its answers from a single forward pass, so there are no sampling parameters to set." — [HF card d1-2B](https://huggingface.co/LiquidAI/d1-2B)

Sizes and backbones [CR][DR]
- d1-omni-600M: 587M parameters, built on LFM2.5-Encoder-350M, SigLIP2 vision tower from LFM2.5-VL-450M, 17-layer FastConformer audio encoder; context 16,384 tokens (with images the text is cut to 896 tokens); text+image or text+audio (not both), audio ≤30 s and trained on English speaker-to-assistant requests. — [HF card d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M)
- d1-3B: 3.12B parameters, built on LFM2.5-VL-3B (decoder-only), 32,768 context. Made by averaging LFM2.5-2.6B with the VL-3B text backbone, fine-tuning several seeds/data mixes and merging. "Training on long inputs, shuffling answer options, and fixing shortcuts in the data made a bigger difference than more advanced techniques." — [HF card d1-3B](https://huggingface.co/LiquidAI/d1-3B); [Liquid blog, Open d1](https://www.liquid.ai/blog/d1-open)
- d1-2B: 1.92B parameters, built on LFM3-VL-2B; LFM3 backbone of 24 layers (13 gated short convolutions, 5 sliding-window attention, 6 global attention); 128K context; text, pictures and video. — [HF card d1-2B](https://huggingface.co/LiquidAI/d1-2B)
- d1-Omni-2B: 2.17B parameters, built on LFM3-Omni-2B, adds audio. — [docs index](https://docs.liquid.ai/llms.txt); [Liquid blog, LFM3](https://www.liquid.ai/blog/lfm3)
- GGUF sizes: d1-2B Q4_0 1,049.6 MB, Q4_K_M 1,099.6 MB, Q8_0 1,846.2 MB, plus vision projector 203–379 MB. d1-omni-600M Q8_0 407.2 MB plus projector 262.8 MB (no 4-bit file published). — [d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF); [d1-omni-600M-GGUF](https://huggingface.co/LiquidAI/d1-omni-600M-GGUF)
- Runtime limit: "typed answers run on the Liquid API, Transformers, and mlx-vlm. llama.cpp supports Noul, Choice, and Score only." — [docs: Decision models](https://docs.liquid.ai/lfm/models/decision-models)
- All open d1 models need `trust_remote_code=True` (they ship their own Python). — [HF card d1-3B](https://huggingface.co/LiquidAI/d1-3B)

Latency [CR][DR]
- Hosted d1 (Oct 2026): "A text decision takes 200 to 300 ms". — [Liquid blog, d1](https://www.liquid.ai/blog/d1-decision-model)
- d1-3B, one question / 3 questions / 3.4K-token state / 384 px image: RTX 4090 8 / 21 / 102 / 17 ms; Jetson AGX Thor 16 / 20 / 220 / 35 ms; Jetson AGX Orin 64 GB 26 / 35 / 560 / 83 ms; Jetson Orin Nano 50 / 73 / 1,640 / 202 ms; Apple M5 Pro 30 / 41 / 640 / 62 ms. — [Liquid blog, Open d1](https://www.liquid.ai/blog/d1-open)
- d1-2B: H100 5.7 ms one question, 7.6 ms three questions, 52 ms for a 5K-token state, 12.3 ms for a 384 px picture; AMD MI325X 3.8 ms; Apple M5 Pro 18.6 ms one question, 753 ms for a 5K-token state. — [HF card d1-2B](https://huggingface.co/LiquidAI/d1-2B)
- d1-omni-600M: "We don't report inference numbers ... it is an early research release". — [HF card d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M)

Benchmarks
- Decision Index v0.2.1 (public split), a leaderboard hosted as a third-party Hugging Face Space (`multimodalart/jev-decision-index`), with sub-scores Knowledge, Language, Retrieval, Tools, Arts. d1-3B 48.57 overall (Tools 74.5, the highest Tools score in the table); Winnow-12B 50.02; Decider 35B-A3B 47.11; Decider 4B 40.70; Decider 2B 28.97; d1-omni-600M 15.95 (Tools 15.1). Liquid ran "the official scorer (not a leaderboard submission)". [CR][DR] — [HF card d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M)
- d1-2B vs d1-3B: Decision Index public suite v0.3 51.1 vs 49.0; Decision Bench v4 85.2 vs 83.6; ImajevBench 75.5 vs 64.0; Liquid's own board of 100+ benchmarks 63.2 vs 58.8. On the same table d1-omni-600M scores 18.3 on Decision Index v0.3. [CR][DR] — [HF card d1-2B](https://huggingface.co/LiquidAI/d1-2B); [Liquid blog, LFM3](https://www.liquid.ai/blog/lfm3)
- Text "benchmarks as decisions" mean of 7: d1-3B 82.9, Decider 4B 81.1, d1-omni-600M 78.4, Decider 2B 77.1. d1-omni-600M has the top score on Civil Comments (95.8) and PAWS-X (79.5). [CR][DR] — [Liquid blog, Open d1](https://www.liquid.ai/blog/d1-open)
- A second, differently named index: "The Artificial Analysis Decision Index v2.1 evaluates decision models on six benchmarks, from decisions and extraction on resumes, receipts and research papers to tool calling and hard knowledge questions. It also scores the probability a model assigns to each answer". d1.2-ultra scores 75, "two points ahead of the previous best decision model". [CR][DR] — [Liquid blog, d1.2](https://www.liquid.ai/blog/d1-2)
- Calibration error across 13 benchmarks / 22,505 questions: d1.2-ultra 0.8 points, d1.2 1.2, Jev 1.14 2.7, Perplexity's Decider v1.3 3.7, GPT-6 Luna Decisions API 7.2. [CR][DR] — [Liquid blog, d1.2](https://www.liquid.ai/blog/d1-2)
- Hosted d1 vs frontier chat models on six applications (single run each, 5 Oct 2026): d1 "matches or beats GPT-6.1 Sol on four of them" and "costs 19x to 200x less". [CR][DR] — [Liquid blog, d1](https://www.liquid.ai/blog/d1-decision-model)
- A secondary write-up cautioned that no independent evaluation was available, and described the Decision Index as Liquid's own benchmark. [IND][SS] — [mer.vin](https://mer.vin/news/liquid-ai-d1-3b-a-3b-decision-model-that-answers-in-one-forward-pass/); the "Liquid's own" part is contradicted by [HF card d1-3B](https://huggingface.co/LiquidAI/d1-3B), which links the index to a third-party Space and lists competitor rows "from the public leaderboard"

Intended uses and limits [CR][DR]
- "routing and triage, moderation, intent and topic classification, voice-command routing, extraction checks, reranking, agent guardrails, and visual inspection. It is not a chat model and does not write text." — [HF card d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M)
- Docs add: "LLM-as-judge replacement, agent tool-call approval, and model routing or cascades." — [docs: Decision models](https://docs.liquid.ai/lfm/models/decision-models)
- Rule of thumb from the migration guide: "If the answer is one of N known options, or a value already written in the input, use a decision model. If the answer is a new string the model must compose, use an LLM." — [docs: Decision Model Guide](https://docs.liquid.ai/guides/decision-model-guide.md)
- Liquid's demo of action selection: "Web Agent: d1 operates a flight-search website from a one-sentence goal. At each step, it picks the next action among everything the page allows." — [Liquid blog, d1](https://www.liquid.ai/blog/d1-decision-model)
- In robotics, a fine-tuned d1-2B is used as a "skill monitor" alongside a generative orchestrator, to "double recovery success under disturbances". — [Liquid blog, LFM3](https://www.liquid.ai/blog/lfm3)
- Stated limits of d1-2B: "Knowledge-heavy questions favor larger models, and answers get less reliable past about 64K tokens of state. Estimates are reliable on counts and scores, less so on free quantities." "Calibrate any decision threshold on data from your own use." — [HF card d1-2B](https://huggingface.co/LiquidAI/d1-2B)
- Numeric caveat: for d1-omni-600M, "Avoid bfloat16, which changed the top answer on 0.8% of text and 1.7% of audio rows." — [HF card d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M)

### Inferences
- How this differs from generative tool calling, in plain terms: a generative model writes `set_timer(minutes=5)` token by token and may write something invalid. A decision model is handed the list of possible tools as options and returns "tool 3, 92%". It can pick which tool (choice), decide whether to act at all (noul), and, in d1-2B and later, copy an argument value that is literally present in the input (text/number/date). It cannot compose an argument that is not in the input.
- So d1 covers tool *selection*, gating and simple slot filling, not general function calling. A full agent would pair it with a generative model, which is exactly how Liquid uses it in the robotics example.
- The decoder-based d1 models are the old "verbalizer / constrained single-token classification" idea, productised with training for calibration and with shared-prefix batching. The 600M model is closer to a classic encoder-plus-head classifier.
- Latency is dominated by input length (one question 8 ms vs a 3.4K-token state 102 ms on the same GPU), since there is no output generation.
- For a microcontroller-class or phone-class device maker, the only d1 small enough to compare with 230M/350M generative models is d1-omni-600M, which Liquid labels experimental, publishes no speed numbers for, and which scores far below the 2–3B models (Decision Index 15.95 vs 48.57).
- Several competitor/benchmark names in Liquid's tables (Jev, Decider, Winnow, JPT, JevBench, ImajevBench) come from an ecosystem that post-dates my background knowledge; I report them as named and did not verify what each is.

### Gaps
- Who operates the "Decision Index" leaderboard, and whether the "Artificial Analysis Decision Index v2.1" is the same lineage as the Hugging Face "Decision Index v0.2.1/v0.3", was not established. The version numbers differ and Liquid's posts use both names.
- No independent replication of any d1 number was found.
- No CPU-only or phone latency for any open d1 model in text form; smallest reported hardware is Jetson Orin Nano (d1-3B, 50 ms).
- Hosted d1.2 / d1.2-ultra parameter counts are not disclosed.
- The d1-Omni-2B card was downloaded but not read in full.

---

## 4. Tool-calling benchmark numbers: Liquid's own vs competitors, and independent evaluations

### Takeaway
Liquid reports BFCLv3 of about 43–44 for both 230M and 350M, slightly above Granite 4.0-H-350M and well above Gemma 3 1B, but multi-turn agent scores (τ²-Bench) are low in absolute terms. Liquid does not compare against FunctionGemma 270M, Gemma 3 270M, Qwen3 0.6B, SmolLM or Cactus Needle anywhere I read, and I found no independent evaluation of the 230M model.

### Cited Findings

Liquid's table (same numbers on blog and model cards) [CR][DR] — [HF card LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)

| Model | BFCLv3 | BFCLv4 | τ²-Bench Telecom | τ²-Bench Retail | CaseReportBench | IFEval |
|---|---|---|---|---|---|---|
| LFM2.5-230M | 43.26 | 21.03 | 5.26 | 13.68 | 22.51 | 71.71 |
| LFM2.5-350M | 44.11 | 21.86 | 18.86 | 17.84 | 32.45 | 76.96 |
| LFM2-350M | 22.95 | 12.29 | 10.82 | 5.56 | 11.67 | 64.96 |
| Granite 4.0-H-350M | 43.07 | 13.28 | 13.74 | 6.14 | 12.44 | 61.27 |
| Granite 4.0-350M | 39.58 | 13.73 | 2.92 | 6.14 | 0.84 | 53.48 |
| Qwen3.5-0.8B (Instruct) | 35.08 | 18.70 | 12.57 | 6.14 | 13.83 | 59.94 |
| Qwen3.5-0.8B (Thinking) | 39.64 | 25.39 | 14.33 | 7.02 | 0.39 | 32.93 |
| Gemma 3 1B IT | 16.61 | 7.17 | 9.36 | 6.43 | 2.28 | 63.49 |

- Qwen3.5-0.8B in thinking mode beats both Liquid models on BFCLv4 (25.39 vs 21.86 / 21.03). [CR][DR] — [HF card LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M)
- LFM2.5-230M scores 5.26 on τ²-Bench Telecom, below LFM2-350M (10.82) and Gemma 3 1B (9.36). [CR][DR] — [HF card LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)
- For scale, Liquid's 1.7B LFM3 model reports BFCL V4 71.39 vs LFM2.5-2.6B 55.89, Qwen3.5 4B 55.12, Gemma 4 E4B 48.25. [CR][DR] — [Liquid blog, LFM3](https://www.liquid.ai/blog/lfm3)
- Decision-model "Tools" sub-score on Decision Index v0.2.1: d1-3B 74.5, Winnow-12B 71.0, Decider 4B 58.6, Decider 2B 42.4, d1-omni-600M 15.1. [CR][DR] — [HF card d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M)

Fine-tuned results from a partner (distil labs; Liquid-announced collaboration, so partner-reported rather than independent)
- Shell-command task (derived from the BFCL Gorilla file-system task, multi-turn, closed-book): base LFM2.5-350M 61.4% tool-call equivalence → 98.0% after SFT with LoRA; teacher GPT-oss-120B 97.03%. Reached 98.0% after one epoch. [CR-partner][DR] — [distil-lfm25-shellper card](https://huggingface.co/distil-labs/distil-lfm25-shellper)
- Smart-home task: base 63.2% → 96.7% tuned; teacher 92.11%. [CR-partner][DR] — [distil-lfm25-home-assistant card](https://huggingface.co/distil-labs/distil-lfm25-home-assistant)
- Liquid's summary: fine-tuning "across smart home, banking, and terminal-based use cases ... hitting over 95% accuracy throughout the dialog". [CR][DR] — [Liquid blog, 350M](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind)

Competitor-reported comparisons that include Liquid models (Cactus; not read directly)
- Cactus says the original 26M Needle beats FunctionGemma-270M, Qwen-0.6B, Granite-350M and LFM2.5-350M on single-shot function calling; per-task numbers are reportedly not printed. [competitor-reported][SS] — [Cactus blog](https://cactuscompute.com/blog/needle)
- A reading of Cactus's Needle 2 "Mobile-Actions" chart puts LFM2.5 (listed at 230M) at about 69% and FunctionGemma at about 65%. [competitor-reported][SS] — [mer.vin on Needle 2](https://mer.vin/news/needle-2-cactus-14mb-on-device-tool-calling-model/?amp=1)
- For Needle 3 (121M, Sep 2026), BFCL v4 is reported as Needle 3 50.2%, LFM2.5-350M 59.1%, LFM2.5-1.2B 62.0%, Qwen3.5-0.8B 56.8%. [competitor-reported][SS] — [Better Stack guide](https://betterstack.com/community/guides/ai/needle3-edge/); this LFM2.5-350M figure conflicts with Liquid's own BFCLv4 of 21.86 on [HF card LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M)

Independent evaluations
- BenchLM lists LFM2.5-230M's BFCL v4 as vendor-reported with zero independent runs. [IND][SS] — [benchlm.ai](https://benchlm.ai/models/lfm2-5-230m)
- A small-model agent benchmark reportedly had qwen3:0.6b and lfm2.5:1.2b tied at the top (Agent Score 0.880); it covers the 1.2B model, not 230M/350M, and no FunctionGemma entry. [IND][SS] — [ai-radar.it](https://ai-radar.it/article/valutazione-di-llm-piccoli-l-importanza-del-parsing-negli-agenti-locali)
- Hacker News: d1 submissions drew almost no discussion (1–7 points, no comment threads listed for d1, d1-open, d1-ultra; d1.2 had 2 points four hours after posting). LFM3 had 173 points / 67 comments and LFM2.5-24B-A2B 288 points / 81 comments. Comment contents were not read. [IND][DR] — [HN submissions from liquid.ai](https://news.ycombinator.com/from?site=liquid.ai)

### Inferences
- The two BFCLv4 figures for LFM2.5-350M (21.86 from Liquid, 59.1 via a Needle 3 write-up) differ by almost 3x. They are almost certainly different subsets or scoring modes of BFCL v4 (the full v4 includes agentic/web-search/memory categories that tiny models fail). Do not put them in one table without checking the harness.
- Base-model accuracy of ~61–63% on narrow multi-turn tasks rising to 96–98% after a LoRA fine-tune is the most decision-relevant number for a device maker: out of the box these models are not reliable enough for unattended action-taking, and Liquid itself recommends fine-tuning.
- The comparison set Liquid chose (Granite 350M, Qwen3.5-0.8B, Gemma 3 1B) omits the purpose-built sub-300M function-calling models, so Liquid's tables do not answer "230M vs FunctionGemma 270M".

### Gaps
- No Liquid-published or independent head-to-head of LFM2.5-230M/350M against FunctionGemma 270M, Gemma 3 270M, Qwen3 0.6B or SmolLM2/3 was found.
- The distil labs blog post itself (which may contain cross-model comparisons) could not be fetched.
- r/LocalLLaMA threads were not reachable and none surfaced in search.
- Cactus's benchmark pages were not read directly; all Needle numbers above are second-hand.

---

## 5. Fine-tuning path and licence

### Takeaway
Liquid's supported path is LoRA-first fine-tuning with its open-source "Liquid Finetune" repo (or TRL / Unsloth notebooks), training on the native Pythonic tool-call format; the LEAP SDK and bundling service are now deprecated. Weights are under the LFM Open License v1.0, which is Apache-like but withdraws the commercial licence from any legal entity with annual revenue of $10M or more.

### Cited Findings

Fine-tuning [CR][DR]
- Base checkpoints for fine-tuning are published: LFM2.5-230M-Base and LFM2.5-350M-Base. — [HF card LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)
- Model cards link Colab notebooks for: continued pre-training (Unsloth), SFT with LoRA (Unsloth, TRL), DPO with LoRA (TRL), GRPO with LoRA (Unsloth, TRL). — [HF card LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M)
- "Liquid Finetune is Liquid's open-source training stack": SFT, LoRA, DPO, GRPO; text, vision and MoE; local GPUs, SLURM, Kubernetes or Modal; dataset validation including tool-calling data; GGUF export and quantization. — [docs: Fine-tuning overview](https://docs.liquid.ai/lfm/fine-tuning/overview.md); [docs: Liquid Finetune](https://docs.liquid.ai/lfm/fine-tuning/liquid-finetune.md)
- Try cheaper levers first: correct chat template, default generation parameters, "constrained decoding for strict JSON or grammar-bound outputs", better tool descriptions. — [docs: Fine-tuning overview](https://docs.liquid.ai/lfm/fine-tuning/overview.md)
- Recipe: "Prepare 500 to 5,000 task examples"; "Train with LoRA for the first pass. A 1.2B LoRA run can take minutes to tens of minutes on a single modern GPU"; evaluate on a frozen held-out set. — [docs: Fine-tuning overview](https://docs.liquid.ai/lfm/fine-tuning/overview.md)
- "Train with the model's own chat template ... keep training formatting character-for-character identical to production." and "For tool calling, train on the native Pythonic tool-call format." — [docs: Fine-tuning overview](https://docs.liquid.ai/lfm/fine-tuning/overview.md)
- Worked example in the docs: "Fine-tuning LFM for a local Home Assistant", covering synthetic data and tool calling. — [docs index](https://docs.liquid.ai/llms.txt)
- Partner recipe: distil labs used SFT with LoRA, a 120B teacher, 4 epochs; accuracy plateaued after epoch 1. [CR-partner][DR] — [distil-lfm25-shellper card](https://huggingface.co/distil-labs/distil-lfm25-shellper)
- LEAP: "The LEAP SDK is deprecated ... new projects should use llama.cpp directly." The LEAP Model Bundling Service / `leap-bundle` is also deprecated: "Download GGUF files from Hugging Face and run them with llama.cpp". "The LEAP SDK was a wrapper around llama.cpp." — [docs: Deprecations](https://docs.liquid.ai/lfm/help/deprecations.md)
- No fine-tuning recipe is given on the open d1 cards; the d1-2B card says "For custom deployment and fine-tuning support, contact Liquid AI". — [HF card d1-2B](https://huggingface.co/LiquidAI/d1-2B)

Licence [CR][DR] — [LFM Open License v1.0 text](https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/LICENSE)
- Structure mirrors Apache 2.0: perpetual, worldwide, royalty-free copyright and patent grants; redistribution requires passing on the licence, marking modified files and keeping notices.
- "'Threshold' shall mean annual revenue of 10 million United States dollars ($10,000,000) or more."
- Section 5: "(a) The rights granted under this License for Commercial Use are conditioned upon You or Your Legal Entity not exceeding the Threshold. (b) Any Commercial Use of the Work or a Derivative Work by a Legal Entity that exceeds the Threshold is not licensed under this Agreement."
- "Legal Entity" includes all entities under common control (50%+ ownership), so a subsidiary is counted with its parent.
- "Commercial Use" = "any use of the Work for direct or indirect commercial advantage or monetary compensation."
- Qualified non-profits doing non-commercial or research work are exempt from the threshold.
- Termination: "This License will terminate automatically and immediately if You fail to comply ... You must cease all use of the Work and any Derivative Works and delete all copies in Your possession."
- The same licence file (10,574 bytes) is in the LFM2.5-350M and d1-3B repos; docs say d1-2B, d1-omni-2B and d1-omni-600M are "open-weight models under the LFM Open License". — [docs: Decision models](https://docs.liquid.ai/lfm/models/decision-models)
- Liquid's blog posts nonetheless describe the models as "Open-weight — Download, fine-tune, and deploy without restrictions". — [Liquid blog, 230M](https://www.liquid.ai/blog/lfm2-5-230m)

### Inferences
- For a device maker: a startup under $10M annual revenue (counting parents and affiliates) can ship LFM2.5-230M/350M or an open d1 model, including fine-tuned derivatives, at no charge. A company at or above $10M has no commercial licence under this document and must negotiate one with Liquid; internal commercial use counts too, since the definition covers "indirect commercial advantage".
- The licence text does not say what happens when a company crosses $10M after shipping; read literally, the commercial rights are "conditioned upon" staying under the threshold. This is a legal question, not something the sources settle.
- "Without restrictions" in the marketing copy is inconsistent with Section 5 of the licence; the licence text governs.
- "LEAP fine-tune tooling" in older write-ups has been superseded by Liquid Finetune; on-device deployment guidance is now plain llama.cpp plus GGUF.

### Gaps
- Commercial licence pricing or terms above the threshold are not published in the pages read.
- No Liquid-published LoRA hyperparameters specific to tool calling on 230M/350M (rank, learning rate) were found in text; they would be inside the Colab notebooks, which were not opened.
- Not verified whether Liquid Finetune supports training the d1 decision models.

---

## 6. Does Liquid respond to or mention Cactus Needle?

### Takeaway
No. Across every Liquid page read, the words "Needle" and "FunctionGemma" do not appear; Liquid mentions Cactus only as an inference-engine partner.

### Cited Findings
- A text search of the fetched Liquid pages (350M, 230M, Nanos, Open d1, LFM3 and d1.2 blog posts; model cards; deprecations and decision-guide docs) found zero occurrences of "Needle" or "FunctionGemma". [DR] — e.g. [Liquid blog, 230M](https://www.liquid.ai/blog/lfm2-5-230m); [Liquid blog, LFM3](https://www.liquid.ai/blog/lfm3)
- Liquid names "Cactus Compute" among local-AI ecosystem partners, and credits the Cactus Engine for running LFM2.5-350M on sub-$300 devices (iPhone 13 Mini 88 tok/s at 56 MB, Pixel 6a 42 tok/s, Raspberry Pi 5 30 tok/s). [CR][DR] — [Liquid blog, 350M](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind)
- The LFM3 post (27 Oct 2026) again thanks "Cactus" in its builder ecosystem list. [CR][DR] — [Liquid blog, LFM3](https://www.liquid.ai/blog/lfm3)
- In the other direction, Cactus reportedly uses LFM2.5-350M (and later LFM2.5 230M / 1.2B) as a comparison baseline for Needle. [competitor-reported][SS] — [Cactus blog](https://cactuscompute.com/blog/needle)

### Inferences
- The relationship is partner-and-competitor: Cactus's runtime is one of Liquid's showcased deployment paths, while Cactus's own Needle models are positioned against Liquid's smallest models. Liquid has not publicly answered Needle's claims in its blog, docs or model cards.

### Gaps
- Liquid's social media, Discord and any interviews were not searched; a response there cannot be ruled out.
- Cactus's own pages were not readable from this session.
