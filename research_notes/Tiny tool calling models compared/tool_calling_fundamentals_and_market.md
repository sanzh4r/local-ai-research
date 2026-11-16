# Tool calling fundamentals, market, and head-to-head evidence for tiny on-device models (as of 16 Nov 2026)

**How to read the tags.** Every finding carries: a date; `[CO]` = company-reported, `[IND]` = independent; `[DR]` = I read the page directly this session, `[SS]` = search-engine summary only (page not opened; treat as lower confidence). Items marked **BACKGROUND (unverified this session)** come from my prior knowledge (training cutoff mid-2026) and were not re-checked; they are kept out of "Cited Findings" and placed under "Inferences".

**Access notes.** Fetched directly via curl: cactuscompute.com, liquid.ai, arxiv.org (abstract pages), news.ycombinator.com. Failed: huggingface.co raw model card (HTTP 401, gated), marktechpost.com (403), raspberrypi.com (403). Better Stack, RuntimeWire and Mi&Bee reviews were not retrieved directly (hosts not on the allowlist; searches surfaced Mi&Bee's URL but only other reviewers' content in summaries).

**Important correction for the report writer.** arXiv 2607.18363 is *not* a "Needle 2 benchmark paper". Its title is "A Controlled Study of Attention-Only Transformers" (Ndubuaku et al., submitted 20 Jul 2026, 10 pages). The abstract is about whether feed-forward layers are necessary; it contains no comparison with FunctionGemma, LFM2.5 or Apple models. All head-to-head tables live in Cactus's blog/product pages, not in a peer-reviewed or arXiv paper (I read only the abstract page, not the PDF). — [arXiv 2607.18363](https://arxiv.org/abs/2607.18363) `[CO][DR]`

---

## 1. Mechanics: how tool calling works in small models

### Takeaway
Tool calling is "translate a messy sentence into a function name plus typed arguments"; the app, not the model, then runs the function. Tiny models make this workable by (a) narrowing the job to retrieval-and-copy rather than reasoning, (b) letting a grammar forbid any malformed output, (c) keeping the tool list short through embedding retrieval, and (d) attaching a confidence score so uncertain requests are confirmed or escalated to a bigger model.

### Cited Findings

**What it is / the problem framing**
- Cactus's framing (May 2026): "Tool calling is fundamentally retrieval-and-assembly (match query to tool name, extract argument values, emit JSON), not reasoning." — [Cactus, Needle blog, 12 May 2026](https://cactuscompute.com/blog/needle) `[CO][DR]`
- Cactus (Aug 2026): devices "already expose their abilities as functions with typed parameters, so the only hard part is mapping a messy sentence onto them: which function, and which arguments. Framed that way, the problem needs no world knowledge and no open-ended prose. That is why 45M parameters suffice whereas chat requires billions." — [Cactus, Needle 2 blog, 9 Aug 2026](https://cactuscompute.com/blog/needle-2) `[CO][DR]`
- An independent arXiv paper (Sep 2026) describes the two decisions every tool-calling assistant makes per request: "which tool to invoke, and whether any available tool applies" (selection vs abstention). — [Lee (Redrob), arXiv 2609.18672, 16 Sep 2026](https://arxiv.org/abs/2609.18672) `[IND][DR, abstract only]`

**How schemas are fed in, and what comes out**
- Needle 1 example (May 2026): tools are passed as a JSON string of name/description/parameters, e.g. `[{"name":"get_weather","description":"Get current weather for a city.","parameters":{"location":{"type":"string",...,"required":true}}}]`; output is a JSON list `[{"name":"get_weather","arguments":{"location":"San Francisco"}}]`. — [Cactus, Needle blog](https://cactuscompute.com/blog/needle) `[CO][DR]`
- Needle 3 (Sep 2026) returns one JSON object per turn with `function_calls`, a one-line `reasoning` trace (e.g. "'living room' -> room; 'dim' -> on true, brightness 30"), `confidence`, and performance telemetry (`prefill_tps`, `decode_tps`, `peak_ram_mb`). — [Cactus, Needle 3 page](https://cactuscompute.com/needle) `[CO][DR]`
- Needle 3 system turn takes environment *facts*, not instructions ("date: 2026-07-21 Tue 14:30; locale: en-US; device: phone; battery: 62%"); "Instructions placed there do not steer the model." — [Cactus, "How to Design Tools for Needle 3", 18 Sep 2026](https://cactuscompute.com/blog/designing-tools-for-needle) `[CO][DR]`
- FunctionGemma uses "a custom format rather than JSON" per a Google Developer Experts Medium post. — [Medium/GDE post](https://medium.com/google-developer-experts/on-device-function-calling-with-functiongemma-39f7407e5d83) `[IND][SS]` (the model card itself could not be fetched — HTTP 401)

**Constrained / grammar-guided decoding**
- Needle 2: "We compile grammar from the schema, preventing malformed JSON and invalid structures. This way, the model focuses its 45M parameters on choosing the right values." Decoding is "constrained by a byte-level grammar compiled from the declared schemas." — [Cactus, Needle 2 blog](https://cactuscompute.com/blog/needle-2) `[CO][DR]`
- Grammar as a *speed* optimisation too: "because the matcher knows which tokens are legal before the logits exist, the engine computes output scores only for candidate rows, skipping up to 98% of the vocabulary projection on structural tokens, and skips it entirely on steps whose output is already forced." — [Cactus, Needle 2 blog](https://cactuscompute.com/blog/needle-2) `[CO][DR]`
- Needle 3: "Ranges, enums, patterns, lengths and item counts are compiled into the decode grammar, so an invalid value is unrepresentable rather than merely discouraged." — [Cactus, tool design guide](https://cactuscompute.com/blog/designing-tools-for-needle) `[CO][DR]`
- Key limitation, independently stated: "constraining a decoder to a tool grammar repairs malformed output without improving the choice." (i.e. grammar fixes syntax, not which tool is picked) — [arXiv 2609.18672 abstract](https://arxiv.org/abs/2609.18672) `[IND][DR]`
- Well-formedness is not the bottleneck anyway: in Cactus's BFCL v4 single-turn run, well-formed rates were 93.4 (Needle 2), 94.2 (LFM2.5-230M), 95.0 (Apple FM), 100.0 (FunctionGemma), while overall accuracy ranged only 42.6–61.7. — [Cactus, Needle 2 blog](https://cactuscompute.com/blog/needle-2) `[CO][DR]`

**Why long prompts (many tool definitions) are costly; tool retrieval**
- Needle 2 uses a 256-token sliding attention window so the KV cache (working memory) is bounded; "the system prompt and tool declarations are pinned as permanent sinks so the one thing a tool-calling model must never forget — its tools — is structurally unable to be evicted." — [Cactus, Needle 2 blog](https://cactuscompute.com/blog/needle-2) `[CO][DR]`
- Needle 3: "Five or fewer tools render directly. Above that, retrieval engages: every schema is embedded once, each request is embedded, and only the five closest tools enter the context with the grammar rebuilt over them. An unselected tool is unreachable, not merely unlikely... Every extra tool in a turn is a chance to misroute." For big catalogues Cactus recommends two turns: one to pick the tool, one declaring only that tool. — [Cactus, tool design guide](https://cactuscompute.com/blog/designing-tools-for-needle) `[CO][DR]`
- Needle 3 doubles as an embedding model so "a query [can be] matched to the closest of hundreds of tools" locally. — [Cactus, Needle 3 page](https://cactuscompute.com/needle) `[CO][DR]`
- Independent measurement of retrieval-only routing over 70 Korean/English actions and 600 requests: character 3-gram BM25 selected 162/164 lexically matched requests but only 85/166 paraphrases; restricting to seven candidates raised paraphrase accuracy to a mean 0.825. Retrieval cannot abstain: no classifier on BM25 scores separated in-catalog from out-of-catalog above 0.697 AUC; a frozen multilingual-e5-base encoder reached 0.806. Conclusion: "Abstention, not selection, is where a neural component is required. A neural ranker improves every quality metric and is rejected on latency and memory rather than accuracy." — [arXiv 2609.18672 abstract](https://arxiv.org/abs/2609.18672) `[IND][DR]`

**Router pattern, hybrid local/cloud, confidence handoff**
- Needle 2: "Off-topic requests return an empty call, and every response carries a learned confidence score. Set your own confidence threshold: act above it, ask again, or escalate to the cloud below it." — [Cactus, Needle 2 blog](https://cactuscompute.com/blog/needle-2) `[CO][DR]`
- Needle 3 confidence = minimum of (a) a calibrated post-hoc head scoring prompt + finished call and (b) decode probability of the call tokens; engine withholds anything under 0.1; suggested three bands: >= 0.7 act, 0.1–0.7 confirm with user (or route to a bigger model), empty = refuse. "a light switch tolerates 0.5, a transfer wants 0.9 and a confirmation anyway." — [Cactus, "Leveraging Needle's Confidence", 18 Sep 2026](https://cactuscompute.com/blog/needle-confidence) `[CO][DR]`
- Caveats Cactus itself states: confidence is reported as `None` for fine-tuned weights (head is calibrated on the base model only); "correct Spanish calls have been measured at 0.0." — [same](https://cactuscompute.com/blog/needle-confidence) `[CO][DR]`
- Non-neural guardrails sit around the model: regex "triggers" that force a tool for matching phrasings, and "grounding gates" that withhold calls when the request negates the verb, quotes someone else's command, lacks a number for a required numeric field, etc. — [Cactus, tool design guide](https://cactuscompute.com/blog/designing-tools-for-needle); [confidence guide](https://cactuscompute.com/blog/needle-confidence) `[CO][DR]`
- Liquid's robot demo is a router/skill-selector pattern: LFM2.5-230M on a Unitree G1's onboard Jetson Orin "acts as a skill-selection layer: it takes a single natural-language instruction and decomposes it into a sequence of tool calls that invoke pre-trained low-level skills provided by NVIDIA's SONIC framework." — [Liquid AI, LFM2.5-230M blog, 25 Jun 2026](https://www.liquid.ai/blog/lfm2-5-230m) `[CO][DR]`
- A different mechanism, the "decision model": Liquid's d1 (5 Oct 2026) "reads [input and questions] in one forward pass, and returns the probabilities, without generating any tokens"; question types are yes/no ("Noul"), Choice (one label among many), Score. In the Web Agent demo "at each step, it picks the next action among everything the page allows." A text decision "takes 200 to 300 ms". At launch d1 is an API product (console.liquid.ai, Vercel, OpenRouter) priced $0.04 per million input tokens; open weights "for upcoming models" are promised "soon". — [Liquid AI, d1 blog, 5 Oct 2026](https://www.liquid.ai/blog/d1-decision-model) `[CO][DR]`
- HN practitioner view (May 2026 thread): "The key is to not run LLMs in loops... much more useful and reliable with one shot tooling... If multiple tools need to be called... I will chain those together deterministically in my code." — [HN thread 48111896](https://news.ycombinator.com/item?id=48111896) `[IND][DR]`

### Inferences
- **Analogy (mine):** a tool-calling model is a switchboard operator, not an expert. It hears "dim the lounge to 30", looks at the directory of extensions it was handed (tool schemas), plugs the call into `set_lights` and writes "lounge, 30" on the slip. It never needs to know how lights work. Grammar-constrained decoding is a paper form with tick boxes: the operator physically cannot write outside the boxes, but can still tick the wrong one.
- **Analogy (mine):** tool definitions are like reading the whole menu aloud before each order. On a phone or microcontroller every token of menu costs time, RAM (the KV cache) and battery, which is why systems pre-filter the menu to ~5 items by embedding similarity. The trade-off Cactus admits: a filtered-out tool is "unreachable", so retrieval errors become hard failures.
- The independent Korean-English study and Cactus's design converge on the same architecture: cheap retrieval for *selection*, a small neural component for *abstention/confidence*, grammar for *syntax*. That makes "is the tiny LLM needed at all vs BM25 + classifier + regex?" a legitimate open question (also raised in the Needle 3 HN thread, where a Cactus team member replied the two are "complimentary").
- Liquid d1 and Needle represent two answers to "how do you make a decision without generating prose": d1 scores a fixed set of options in one pass (classification-like, no arguments generated); Needle generates a grammar-constrained call (can fill free-form arguments). d1 as launched is cloud API, so it is not yet an on-device competitor; its role in a comparison is "router/decider", not "argument filler".
- **BACKGROUND (unverified this session):** GBNF is llama.cpp's grammar format; JSON-schema-to-grammar compilation and libraries such as Outlines/XGrammar/llguidance implement the same token-masking idea; "Pythonic" call formats (`get_weather(location="SF")`) are used by Liquid LFM2 models and some Llama variants because they are shorter than JSON; FunctionGemma uses dedicated control tokens (e.g. `<start_function_declaration>`, `<start_function_call>`, `<escape>`) rather than plain JSON; BFCL (Berkeley Function Calling Leaderboard) scores calls by AST match and v4 adds agentic/multi-turn categories. None of these specifics were re-verified here; other researchers covering the individual models should confirm.

### Gaps
- MCP on-device: I found no primary source describing MCP (Model Context Protocol) being served to or consumed by sub-500 MB models. Cactus's Needle 3 materials do not mention MCP in the pages I read. One search hit ("Best Small Local Models for Tool Calling 2026: MCP Tests", promptquorum.com) was not opened and is of unknown reliability.
- Exact FunctionGemma prompt template/special tokens: model card gated (401); only a search-summary "custom format" statement.
- LFM2.5's tool-call output format (Pythonic vs JSON) not verified from Liquid docs this session.

---

## 2. Why size matters: memory, storage, battery, latency, and "good enough" accuracy

### Takeaway
The argument for tiny is physical: most edge devices are sub-$200 with no GPU/NPU and tens of MB of spare RAM, and energy per token scales roughly with parameters exercised. But "good enough" is contested: out-of-the-box exact-match accuracy of all these models on public benchmarks is ~60–70% at best, so vendors lean on fine-tuning, confidence gating and confirmation UIs to make wrong actions rare.

### Cited Findings

**Footprints and speeds (all company-reported)**
- Needle 2 (9 Aug 2026): 45M params, 14 MB binary, 28 MB peak session RAM (a "deterministic 28MB ceiling"); 500+ tok/s decode and 800+ tok/s prefill on Raspberry Pi 5; 400–1,500 tok/s on Meta Quest 3S / Apple Vision Pro; 300–700 tok/s on sub-$200 phones (Samsung A-series); claimed to fit MCU-class parts with external RAM (ESP32-P4 with 32 MB PSRAM, STM32H7, NXP i.MX RT; static library for Cortex-M4/M7/M55). — [Cactus, Needle 2 blog](https://cactuscompute.com/blog/needle-2) `[CO][DR]`
- Needle 3 (17–18 Sep 2026): one weight set, any depth 2–20 layers = 25M–121M params; file ~8 MB at 4 layers to 29 MB at 20 layers (≈2.75 MB fixed + 1.3 MB per block at 2-bit); Pi 5 decode ≈4,000 tok/s at 2 layers down to ≈400 at 20; MFLOPs/token 30 (4L), 48 (8L), 83 (16L), 100 (20L). — [Cactus, Intelligence Ladders, 17 Sep 2026](https://cactuscompute.com/blog/intelligence-ladders); [supported devices guide](https://cactuscompute.com/blog/needle-supported-devices) `[CO][DR]`
- LFM2.5-230M (25 Jun 2026): "213 tok/s decode speed on Galaxy S25 Ultra, 42 tok/s on a Raspberry Pi 5"; ships via llama.cpp GGUF, MLX, vLLM, SGLang, ONNX. — [Liquid AI blog](https://www.liquid.ai/blog/lfm2-5-230m) `[CO][DR]`
- Compute per token per Cactus's own accounting: Needle 2 70 MFLOPs, LFM2.5-230M 460, FunctionGemma 270M 540 ("170M of its 270M parameters are a 262k-token embedding table"), Apple FM (~3B) ~6,000. — [Cactus, Needle 2 blog](https://cactuscompute.com/blog/needle-2) `[CO][DR]`
- Pi 5 decode gap implied by the two vendors' own numbers: Needle 2 500+ tok/s vs LFM2.5-230M 42 tok/s (different engines, quantisations and test setups; not a controlled comparison). — [Cactus](https://cactuscompute.com/blog/needle-2); [Liquid](https://www.liquid.ai/blog/lfm2-5-230m) `[CO][DR]`
- FunctionGemma is "about 288 MB" per one developer write-up. — [Medium/GDE post](https://medium.com/google-developer-experts/on-device-function-calling-with-functiongemma-39f7407e5d83) `[IND][SS]`

**The market-size argument for tiny**
- "there are more than 21 billion IoT devices against roughly 1.5 billion PCs... roughly four in five edge devices cost under $200. That is the hardware Needle targets: no GPU, no NPU, a few dozen MB of RAM." — [Cactus, Needle 2 blog](https://cactuscompute.com/blog/needle-2) `[CO][DR]` (Cactus gives no source for these counts)
- Energy: "On device silicon, moving a byte out of flash or DRAM costs orders of magnitude more than a multiply-accumulate... Even on a high-end phone, an always-on assistant lives inside a power budget; every MFLOP is milliwatt-hours." — [same](https://cactuscompute.com/blog/needle-2) `[CO][DR]`
- Download size as a product feature (Needle 3 HN thread, a developer building a browser product): "we'll pick the smallest model of sufficient quality because not having to download a 600MB bag of weight is a feature in itself"; also "WebGPU is a non-starter for production usage since the support is too limited". — [HN 49748553, Sep 2026](https://news.ycombinator.com/item?id=49748553) `[IND][DR]`

**What accuracy is "good enough" for actions**
- Cactus: "A tool call is an action. When a 121M model runs a lock, a payment or a message send without a person in the loop, the question is not only whether it is right on average but whether it knows when it is not." — [Cactus confidence guide](https://cactuscompute.com/blog/needle-confidence) `[CO][DR]`
- Google's own figures for FunctionGemma on Mobile Actions: base 58% → fine-tuned 85%; Google's guide says the model "requires fine-tuning for a specific task". — [Google DeepMind FunctionGemma page](https://deepmind.google/models/gemma/functiongemma/); [Google AI Edge fine-tune guide](https://ai.google.dev/gemma/docs/mobile-actions?hl=en) `[CO][SS]`
- Cactus: fine-tuning "lifts accuracy by 21 to 58 points and puts Needle 2 ahead of DeepSeek V4 Flash, a frontier cloud model, on three of the four benchmarks." Needle 3: fine-tuning lifts every depth 18–36 points on DroidCall; the 2-layer model "goes from unusable to 56.5". — [Needle 2 blog](https://cactuscompute.com/blog/needle-2); [Intelligence Ladders](https://cactuscompute.com/blog/intelligence-ladders) `[CO][DR]`
- Independent failure report on Needle 3 from an HN user running a ~36-scenario voice-assistant suite (results interpreted by an AI assistant, per the poster): "'25 minute timer' becomes duration_seconds: 25, and 'twelve minutes' becomes 120. The first one comes with 100% confidence"; "'take the paper towels off the list' became an add"; "'remind me in 20 minutes' became a timer". A Cactus team member replied the model "can struggle with implied references and multistep reasoning... for some things you just have to have a model that is properly sized." — [HN 49748553](https://news.ycombinator.com/item?id=49748553) `[IND][DR]`
- HN users on the Needle 2 web demo (Aug 2026): "the web demo is not particularly impressive. It really doesn't like anything I throw at it"; "the model struggles to have enough general intelligence... for arbitrary prompted tool calling. At this size, not surprising." Another asked for confidence-calibration results across a real test set, which were not published. — [HN 49246804](https://news.ycombinator.com/item?id=49246804) `[IND][DR]`
- One guide claims BFCL shows "1–3B models are the practical sweet spot for reliable single-turn tool use on edge devices." — [digitalapplied.com guide](https://www.digitalapplied.com/blog/small-language-models-on-device-agents-2026-guide) `[IND][SS]`, low-reliability blog, not opened.

### Inferences
- **Analogy (mine):** a wrong chat answer is a typo in a letter; a wrong tool call is the letter being posted to the wrong address. That asymmetry is why every vendor here adds a "pause and ask" layer (confidence bands, empty-call refusals, confirmation UI) rather than relying on raw accuracy.
- Sub-100 ms targets are plausible on paper: a typical tool call is a few dozen output tokens, so at Needle's claimed 500 tok/s decode on a Pi 5 a 30-token call is ~60 ms plus prefill, whereas at LFM2.5-230M's 42 tok/s it is ~0.7 s. This is my arithmetic from vendor numbers; no source I read states an explicit "<100 ms" requirement, and none reports end-to-end latency including speech-to-text.
- The 60–70% ceiling on zero-shot exact match across all three models means "tiny general tool caller" is not a shippable product by itself; the commercially meaningful comparison is fine-tuned-on-your-tools accuracy, which is exactly where public evidence is thinnest and most vendor-controlled (see section 4).
- The "100% confidence on a wrong timer duration" anecdote directly undercuts the confidence-gating safety story for at least some error types (argument copying errors look fluent, so both confidence signals agree).

### Gaps
- No source found giving OEM-stated hard budgets (e.g. "must fit in X MB on a watch SoC", "<100 ms"). The <100 ms figure in the brief is not sourced in anything I read.
- No independent energy/battery measurements for any of the three models.
- FunctionGemma RAM footprint (e.g. the ~550 MB figure sometimes quoted) not found in any source this session.
- Raspberry Pi's own article on Needle (listed on Cactus's blog index, 22 Sep 2026, "Cactus x Raspberry Pi", raspberrypi.com) could not be fetched (403); only its one-line description on Cactus's index was read: "a 14 MB function-calling model that turns plain English into local Python actions, entirely offline on the CPU". — [Cactus blog index](https://cactuscompute.com/blog) `[CO][DR]`

---

## 3. Use cases and buyers

### Takeaway
The pitched buyers are makers of screenless or cheap devices (wearables, smart-home hubs, robots, cars, AR glasses, budget phones) plus app developers who want offline "intents". Named production evidence is thin: one named customer quote for Needle (Pebble Index 01 ring app), one robotics demo for LFM2.5-230M (Unitree G1), and Google's own demo apps for FunctionGemma; I found no named automotive or appliance OEM deployment for any of the three.

### Cited Findings
- **Pebble (named deployment, Needle).** Cactus says Pebble "runs it locally in the Index 01 app to turn spoken requests into actions without depending on a network connection", with a quote attributed to Eric Migicovsky: "The Pebble Index Ring has no screen. So when you speak to it, the action just has to happen, every time, with or without internet connection. We run Cactus Needle locally in the app, instead of relying on the cloud." Note it runs in the companion *app* (phone), not on the ring itself. — [Cactus, Needle 2 blog, 9 Aug 2026](https://cactuscompute.com/blog/needle-2) `[CO][DR]` (not independently confirmed from Pebble's side)
- **Raspberry Pi** published an article about running Needle on a Pi 5 (22 Sep 2026), per Cactus's blog index. — [Cactus blog index](https://cactuscompute.com/blog) `[CO][DR]`
- **Unitree G1 humanoid (demo, LFM2.5-230M).** Liquid describes it as "an early look at ongoing work", running on the robot's NVIDIA Jetson Orin after "a quick fine-tune"; "the behaviors are deliberately simple at this stage." — [Liquid AI, 25 Jun 2026](https://www.liquid.ai/blog/lfm2-5-230m) `[CO][DR]`
- **Liquid's positioning for LFM2.5-230M:** "an ideal solution to power large-scale data extraction pipelines or lightweight on-device agentic workloads"; "we do not recommend it for reasoning-heavy workloads such as advanced math, code generation, or creative writing." Liquid's site lists target industries: Automotive, Consumer electronics, E-commerce, Financial services, Healthcare & Life sciences, Industrial & Robotics, Defense. — [Liquid AI blog](https://www.liquid.ai/blog/lfm2-5-230m) `[CO][DR]`
- **Liquid d1 use cases (cloud API as of Oct 2026):** SQL-embedded ticket filtering, code search, document filing, web-agent next-action choice, context compaction for coding agents (removes 52% of tokens), visual inspection of parts (85–97% accuracy on VisA, zero-shot). — [Liquid AI, d1 blog, 5 Oct 2026](https://www.liquid.ai/blog/d1-decision-model) `[CO][DR]`
- **Cactus's use-case list for Needle 3:** smart home ("Dim the bedroom and lock up" → two offline calls), robots/vacuums, phone assistants, wearables (notification → structured data), AR glasses (navigation/nearby search), automotive (climate, media, navigation, calls), computers, local semantic search. Supported targets include watchOS, tvOS, Android, iOS, RISC-V, and "Linux MIPS32el (Ingenic cameras, routers)". — [Cactus, Needle 3 page](https://cactuscompute.com/needle); [supported devices](https://cactuscompute.com/blog/needle-supported-devices) `[CO][DR]`
- **Needle training distribution** signals the intended buyers: "consumer device actions—smart home, mobile, wearables, TV, car—plus structured extraction"; Needle 1's synthetic data covered "15 tool categories (timers, messaging, navigation, smart home, etc.)". — [Needle 2 blog](https://cactuscompute.com/blog/needle-2); [Needle blog](https://cactuscompute.com/blog/needle) `[CO][DR]`
- **Voice stack:** Cactus shipped Whistle (2 Oct 2026), a 16.9 MB speech-to-text model that "loads beside Needle so one binary turns a clip straight into tool calls". — [Cactus blog index](https://cactuscompute.com/blog) `[CO][DR]`
- **Traction signals (Cactus-reported):** Needle 3 page displays "312k downloads/mo" and "21k GitHub stars". HN: Needle 1 Show HN 776 points (12 May 2026); Needle 2 537 points/185 comments (10 Aug 2026); Needle 3 236 points/93 comments (18 Sep 2026). — [Needle 3 page](https://cactuscompute.com/needle) `[CO][DR]`; [HN submissions from cactuscompute.com](https://news.ycombinator.com/from?site=cactuscompute.com) `[IND][DR]`
- **Developer/hobbyist demand seen on HN:** an ESP32-based Echo replacement; a tool-calling front end for a game-database site; a mapping/POI editing app (extract ~20 keys by voice); privacy-first desktop apps orchestrating many small constrained agents; Home Assistant recommended as the entry point for local voice. — [HN 49748553](https://news.ycombinator.com/item?id=49748553); [HN 49246804](https://news.ycombinator.com/item?id=49246804); [HN 48111896](https://news.ycombinator.com/item?id=48111896) `[IND][DR]`
- **FunctionGemma:** Google's fine-tuning workflow targets the Mobile Actions dataset, converts to `.litertlm`, and loads into the Google AI Edge Gallery app. — [Google AI Edge guide](https://ai.google.dev/gemma/docs/mobile-actions?hl=en) `[CO][SS]`

### Inferences
- The clearest product-market fit shown so far is "screenless device + companion app + must work offline" (Pebble ring). That is one data point, vendor-relayed.
- Automotive is claimed as a target by both Cactus and Liquid but neither names a car maker in what I read; treat automotive as aspiration, not evidence.
- Much visible adoption is hobbyist/indie (HN, Raspberry Pi), which generates downloads but not necessarily revenue; Cactus monetises via a fine-tuning platform ("Cactus Platform", "Fine-tune models" in site nav) and a source-available engine licence (Needle 1 post: model Apache 2.0, "Cactus Engine is available under its own source-available license").
- **BACKGROUND (unverified this session):** Liquid AI has previously announced enterprise/OEM relationships (e.g. automotive and consumer-electronics partners) and a LEAP edge platform; I did not verify any specific named partner this session, so none is listed as fact.

### Gaps
- No direct statements from OEMs (car makers, appliance makers, glasses makers) about what they need were found.
- No named commercial deployment of FunctionGemma or LFM2.5-230M/350M in a shipping product was found this session.
- Pebble's use of Needle is confirmed only via Cactus's blog quote.
- No revenue, pricing or customer-count data for Cactus; Liquid d1 price ($0.04/M input tokens) is the only price point found.

---

## 4. Head-to-head evidence: Needle vs LFM2.5-230M vs FunctionGemma (vs Apple FM)

### Takeaway
The only same-task tables covering all three models are Cactus's own (Aug 2026): zero-shot, the three trade wins, with LFM2.5-230M best on broad function calling (BFCL v4 single-turn 60.8 vs 46.1 FunctionGemma vs 42.6 Needle 2) and Needle 2 best on Seal-Tools while being 5–6x smaller. The one independent fine-tuned comparison I found (an HN user, 230 cases) had fine-tuned FunctionGemma far ahead of fine-tuned Needle 2/3 (85.2% vs ~19–20% exact arguments). No neutral third-party benchmark of all three exists that I could find.

### Cited Findings

**A. Cactus's Needle 2 evaluation (blog dated 9 Aug 2026)** — [Cactus, Needle 2 blog](https://cactuscompute.com/blog/needle-2) `[CO][DR]`

Test conditions as stated by Cactus: five public benchmarks; "ordered strict exact match: a row passes only if the function names, the call order, and every argument value match"; Needle 2 "measured end-to-end through the shipped C++ engine in its production configuration: CQ2-bit weights, tool retrieval on, and the 256-token sliding KV window"; "Baselines run the released checkpoints under vLLM at full context, and Apple FM runs on-device." All models zero-shot (base checkpoints).

| Benchmark (rows) | LFM2.5-230M (f16) | FunctionGemma 270M (f16) | Needle 2 (2-bit, 14 MB) | Apple FM (on-device, ~3B) |
|---|---|---|---|---|
| Mobile Actions (961) accuracy | **69.1** | 64.0 | 63.7 | 57.6 |
| – name accuracy | 93.0 | 87.3 | **98.3** | 94.2 |
| – 1-call / 2-call | 76.1 / 55.0 | 73.0 / 46.2 | 71.3 / 48.4 | 64.5 / 43.8 |
| DroidCall test (200) | 11.0 | **17.5** | 17.0 | not reported |
| Seal-Tools in-domain (700) | 26.9 | 16.3 | **32.6** | not reported |
| Seal-Tools out-of-domain (654) | 17.0 | 15.6 | **28.7** | not reported |
| BFCL v4 single-turn (3,641) overall | 60.8 | 46.1 | 42.6 | **61.7** |
| – Simple Python | 85.5 | 62.3 | 61.2 | 86.8 |
| – Simple Java / JavaScript | 48.0 / 56.0 | 38.0 / 44.0 | 29.0 / 32.0 | 67.0 / 66.0 |
| – Multiple | 78.5 | 60.0 | 57.0 | 84.0 |
| – Parallel / Parallel multiple | 64.0 / 51.5 | 36.5 / 30.5 | 30.0 / 22.5 | 65.0 / 52.0 |
| – Live simple / multiple | 45.0 / 47.8 | 33.7 / 25.2 | 36.8 / 27.9 | 70.5 / 45.9 |
| – Relevance / Irrelevance | 68.8 / 77.7 | 81.2 / 72.1 | 81.2 / 60.8 | 100.0 / 28.3 |
| – Well-formed rate | 94.2 | 100.0 | 93.4 | 95.0 |

- BFCL overall is "the collector's unweighted mean over all 13 raw categories" using the "Official BFCL v4 single-turn scorer". DroidCall 2-call accuracy was 0.0 for all three models (n=24).
- Cactus's own stated asymmetries: "Precision: the baselines stay at f16 deliberately... That skew favors the baselines. Scope: Needle is trained specifically for agentic tool calling and nothing else, while every baseline is a general language model... That skew favors Needle. There is no clean way to level both at once, so we do not try."
- Cactus on the BFCL loss: Needle's corpus is consumer device actions; BFCL's "general-purpose and enterprise API surfaces, including the Java and JavaScript SDK categories, sit entirely outside that distribution."
- Training-scale contrast: Needle 2 pretrained on 115B tokens + 38B post-training; "LFM2.5-230M was pretrained on 19 trillion tokens, roughly 120x Needle's total" (Liquid confirms "pre-trained for 19T tokens" — [Liquid blog](https://www.liquid.ai/blog/lfm2-5-230m) `[CO][DR]`).
- Fine-tuned Needle 2 vs zero-shot baselines is shown only as a chart (Figure 3); the text gives only "21 to 58 points" lift and "ahead of DeepSeek V4 Flash... on three of the four benchmarks". Per-benchmark fine-tuned numbers are not in the page text.

**B. Needle 3 (17–18 Sep 2026)** — [Cactus, Intelligence Ladders](https://cactuscompute.com/blog/intelligence-ladders) `[CO][DR]`
- Zero-shot, "with the confidence gate on", exact match: 4L/29M: Mobile Actions 11.7, DroidCall 21.0; 8L/52M: 36.8 / 36.5; 16L/98M: 80.7 / 40.0; 20L/121M (29 MB): 86.0 / 47.0. (Compare Needle 2: 63.7 / 17.0; FunctionGemma zero-shot per Cactus: 64.0 / 17.5; LFM2.5-230M: 69.1 / 11.0.)
- Needle 3 headline: "beats models 10x its size on mobile tool calls and matches models 2-3x its size on extraction"; baselines "at f16 under vLLM, DeepSeek V4 Flash through its cloud API". The per-model baseline numbers for Needle 3's Figure 2 are in an interactive chart I could not extract. — [Needle 3 page](https://cactuscompute.com/needle) `[CO][DR]`
- Search-summary only: on BFCL v4 Needle 3 "reaches 50.2%, behind LFM2.5 1.2B and 350M"; and a "tool-calling suite" mean of 63.5 (attributed confusingly to Needle 2), 65.1 FunctionGemma, 69.3 LFM2.5-230M. — [NYU Shanghai RITS summary page](https://rits.shanghai.nyu.edu/ai/cactus-needle-3-one-set-of-weights-an-8-29-mb-model-at-any-depth/) `[IND aggregator][SS]` — garbled attribution; verify before quoting.
- Cactus's design guide says "the same 121M model moved by close to twenty points on a mobile benchmark once every argument was grounded in the request instead of predicted freely" — i.e. a large share of the Needle 3 gain is engine-level grounding logic, not weights alone. — [tool design guide](https://cactuscompute.com/blog/designing-tools-for-needle) `[CO][DR]`

**C. Needle 1 (12 May 2026)** — "it beats FunctionGemma-270M, Qwen-0.6B, Granite-350M, LFM2.5-350M on single-shot function calling, [but] those models have more scope/capacity and excel in conversational settings." No table in the blog post. — [Cactus, Needle blog](https://cactuscompute.com/blog/needle) `[CO][DR]`

**D. Liquid's own numbers (25 Jun 2026)** — [Liquid AI, LFM2.5-230M blog](https://www.liquid.ai/blog/lfm2-5-230m) `[CO][DR]`
- Liquid does not benchmark against Needle or FunctionGemma. Its comparison set is LFM2.5-350M, LFM2-350M, Granite 4.0-(H-)350M, Qwen3.5-0.8B, Gemma 3 1B IT.
- Tool-use scores: LFM2.5-230M BFCLv3 43.26, BFCLv4 21.03, tau2-Bench Telecom 5.26, Retail 13.68. LFM2.5-350M: BFCLv3 44.11, BFCLv4 21.86, Telecom 18.86, Retail 17.84. Gemma 3 1B IT: BFCLv3 16.61, BFCLv4 7.17.
- **Conflict to flag:** Liquid reports LFM2.5-230M at 21.03 on "BFCLv4", Cactus reports the same model at 60.8 on "BFCL v4 single-turn". These are not contradictory if Liquid used the full v4 suite (which includes multi-turn/agentic categories where tiny models score near zero) and Cactus used only the 13 single-turn categories with an unweighted mean — but neither page states this explicitly, so the reconciliation is my inference.

**E. Google's own numbers** — FunctionGemma on Mobile Actions: 58% base, 85% fine-tuned. — [Google DeepMind](https://deepmind.google/models/gemma/functiongemma/) `[CO][SS]`. Cactus measured FunctionGemma zero-shot at 64.0 on the Mobile Actions eval split under strict ordered exact match (different harness/scoring from Google's 58%; unreconciled).

**F. Independent evidence**
- HN user (Needle 3 thread, Sep 2026), own tool-calling task for a game-database site, 230 test cases, all models fine-tuned by the user:

  | Model | Correct tool shape | Exact arguments |
  |---|---|---|
  | FunctionGemma fine-tune, BF16 | 209/230 (90.9%) | 196/230 (85.2%) |
  | Needle 3 fine-tuned W4A8 | 74/230 (32.2%) | 47/230 (20.4%) |
  | Needle 2 fine-tuned W4 | 59/230 (25.7%) | 43/230 (18.7%) |

  The user's summary: "a solid improvement over Needle 2... Unfortunately it's still not quite capable enough for my target compared to FunctionGemma." Single anonymous user, private dataset, unknown fine-tuning settings, domain (game database lookups) outside Needle's device-action training distribution. — [HN 49748553](https://news.ycombinator.com/item?id=49748553) `[IND][DR]`
- Independent reviewers' criticisms (all via search summaries, pages not opened):
  - Mervin Praison: "the only public benchmark is a single vendor chart", no published harness or third-party replication; fine-tuned weights return no confidence value, so teams that fine-tune lose the calibrated-gating feature. — [mer.vin](https://mer.vin/news/needle-2-cactus-14mb-on-device-tool-calling-model/) `[IND][SS]`
  - ai.thesatyajit.com: Needle 2 "ties or loses to a 230M model on three of four function-calling benchmarks"; on BFCL v4 "it lands in last place"; the fine-tuning lift is not broken down per benchmark in text; best reading is "a fine-tuned 14 MB model beats a zero-shot frontier model on your narrow domain". (Note: this domain name matches a Cactus co-author's name, Satyajit Kumar, so independence is uncertain.) — [ai.thesatyajit.com](https://ai.thesatyajit.com/articles/needle-finetune) `[independence unclear][SS]`
  - The Agent Post on Needle 1: "the evaluation methodology is thin." — [theagentpost.co](https://theagentpost.co/posts/review-needle) `[IND][SS]`
  - Another summary: FunctionGemma leads Needle 2 by 3.5 points and LFM2.5-230M leads by 18 on BFCL v4 (consistent with Cactus's table: 46.1−42.6=3.5; 60.8−42.6=18.2); Needle 2 "loses on the enterprise Java SDK tests by nineteen" (48.0−29.0=19 vs LFM). `[SS]`
  - Mi&Bee blog has a Needle 2 explainer at [blog.mickeyzzc.tech](https://blog.mickeyzzc.tech/en/posts/aihelper/needle2-on-device-tool-calling/) — URL surfaced, content not read.
- HN (Needle 1 thread, May 2026): a commenter reported Google's "Gemma4 edge models were promised to be great for agentic use, but have been really disappointing in all my tests. They fail at the most basic tool use scenarios." — [HN 48111896](https://news.ycombinator.com/item?id=48111896) `[IND][DR]`

### Inferences (methodological assessment — mine)
- **Who ran it:** every table containing all three models was produced by Cactus, the vendor of the smallest model. Liquid and Google have published nothing comparing against Needle.
- **Zero-shot vs fine-tuned mismatch:** Cactus's headline "ahead of DeepSeek V4 Flash" compares *fine-tuned* Needle against *zero-shot* baselines (including zero-shot FunctionGemma, which Google itself says needs fine-tuning and which gains 27 points on Mobile Actions when tuned). A like-for-like fine-tuned comparison is absent from Cactus's material, and the single independent one found went strongly against Needle.
- **System vs model:** Needle numbers are for model + bespoke engine (grammar, tool retrieval, grounding gates, verb-family/polar-pair heuristics, confidence floor), whereas baselines are raw checkpoints under vLLM, apparently without schema-compiled grammars or retrieval. That is fair as a "what you ship" comparison but not a pure model comparison; baselines given equivalent scaffolding would likely improve.
- **Benchmark selection:** Needle wins where the benchmark resembles its training distribution (consumer device actions, large candidate lists with retrieval) and loses on general API calling (BFCL). Mobile Actions is Google's own dataset built around FunctionGemma; Needle 3's jump from 63.7 to 86.0 on it within six weeks invites an overfitting/benchmark-targeting question that Cactus's pages do not address (no statement on whether Mobile Actions-like data entered training).
- **Tool count:** benchmarks mostly present a handful of tools; Needle's retrieval caps context at five tools. None of the published tables stress-tests 50–100+ tools for all three models (Seal-Tools' "large candidate tool lists" is the closest, and there Needle's retrieval helps).
- **Strict metric:** "ordered strict exact match" penalises harmless argument variations and call reordering, depressing all scores; relative ranks may shift under AST-match scoring.
- **Quantisation asymmetry cuts both ways:** baselines at f16 are flattered on accuracy but would be 2–4x smaller if shipped at 4–8 bit; the "5x to 70x smaller" claim compares Needle's 2-bit file to parameter counts/f16 sizes.
- Net reading: at zero-shot, LFM2.5-230M is the strongest general small tool caller of the three; Needle is the only one that fits in tens of MB and is competitive on device-style commands; FunctionGemma appears to fine-tune best on evidence from one independent user plus Google's own 58→85 figure. Confidence in any ranking is low because of the absence of neutral replication.

### Gaps
- No neutral third-party benchmark (academic or media lab) covering Needle 2/3, LFM2.5-230M/350M and FunctionGemma under identical harness, quantisation and fine-tuning was found.
- Better Stack and RuntimeWire reviews named in the brief: not found by search and hosts not directly fetchable; I cannot confirm they exist or what they say.
- Needle 3 vs baselines exact figures (interactive chart) and Needle 2 fine-tuned per-benchmark figures (chart) not extractable.
- No head-to-head involving Liquid d1 against any of these on tool selection.
- Needle 3 vs Apple FM numbers not found. Apple FM appears only in Cactus's Needle 2 tables (Mobile Actions 57.6; BFCL v4 single-turn 61.7).
- arXiv 2607.18363 PDF body not read (abstract only); it may contain architecture ablations but, per the abstract, not competitor benchmarks.

---

## 5. Outlook: will OS-bundled models absorb the niche?

### Takeaway
Evidence gathered this session is thin and mostly inferential. The data that exists cuts both ways: Apple's bundled ~3B model already matches or beats the tiny models on general function calling (BFCL single-turn 61.7) at zero download cost on supported iPhones, but it loses on device-command exact match (57.6 on Mobile Actions), over-calls on irrelevant requests (28.3 irrelevance), and does not exist on the sub-$200, no-NPU hardware that tiny models target.

### Cited Findings
- Apple FM (labelled "~3B", run on-device) in Cactus's test: BFCL v4 single-turn overall 61.7 (best of four), relevance 100.0 but irrelevance 28.3 (worst of four), Mobile Actions 57.6 (worst of four); compute ~6,000 MFLOPs/token vs 70 for Needle 2. — [Cactus, Needle 2 blog, Aug 2026](https://cactuscompute.com/blog/needle-2) `[CO][DR]`
- Apple's Foundation Models framework (announced WWDC 2025) gives developers a ~3B on-device model with built-in tool calling via a Swift API. — [digitalapplied.com guide](https://www.digitalapplied.com/blog/small-language-models-on-device-agents-2026-guide) `[IND][SS]`, low-reliability source. A Wikipedia snippet says Apple announced a third generation of models in June 2026 as "a family of five, with two on-device and three cloud models"; the search tool flagged that snippet as garbled. — [Wikipedia: Apple Intelligence](https://en.wikipedia.org/wiki/Apple_Intelligence) `[IND][SS]`, unverified.
- Platform-moat challenge put to Cactus on HN (Aug 2026): "How do you differentiate / compete against LiteRT (former TensorFlowLite) and LiteRT-LM? Google is heavily investing in this ecosystem because Android is where they have distribution moat. Wouldn't it be easier for me as a developer to build on top of LiteRT since it is relatively open ecosystem and I can pack large number of open models from HF directly?" — [HN 49246804](https://news.ycombinator.com/item?id=49246804) `[IND][DR]`
- Pro-market commenter views on HN: "the 'micro' sized LLM space is underappreciated... I foresee... a hierarchy of LLMs, with more competent models actively training smaller models to solve specific tasks very efficiently, and something like this could be the smallest layer in that stack"; "there's definitely a huge niche, a huge market for this, directly between the non-LLM driven tools... of yesteryear" and full LLMs. — [HN 49246804](https://news.ycombinator.com/item?id=49246804) `[IND][DR]`
- Sceptical views on HN: "what does a 28MB binary get you?... a 14MB binary using 28MB of RAM seems unnecessarily small and pretty arbitrary" (i.e. is ultra-tiny the right point on the curve when 140 MB would also fit most devices); and whether "8-30MB worth of regexs with accompanying algorithmic structure" would do as well on well-defined tasks. — [HN 49246804](https://news.ycombinator.com/item?id=49246804); [HN 49748553](https://news.ycombinator.com/item?id=49748553) `[IND][DR]`
- Cactus's counter-positioning: target is hardware with "no GPU, no NPU, a few dozen MB of RAM"; one artifact from "Cortex-M to x86 to WebAssembly"; platforms include watchOS, tvOS, MIPS cameras/routers, RISC-V, browsers. — [Needle 2 blog](https://cactuscompute.com/blog/needle-2); [supported devices](https://cactuscompute.com/blog/needle-supported-devices) `[CO][DR]`
- Liquid's trajectory points away from pure on-device open weights toward a paid decision API (d1 at $0.04/M input tokens, Oct 2026), while still promising open weights later. — [Liquid d1 blog](https://www.liquid.ai/blog/d1-decision-model) `[CO][DR]`
- Google itself ships an *open* tiny tool caller (FunctionGemma) alongside its OS-bundled models and says it "requires fine-tuning for a specific task". — [Google AI Edge guide](https://ai.google.dev/gemma/docs/mobile-actions?hl=en) `[CO][SS]`

### Inferences (mine)
- **For absorption:** on flagship phones the OS vendor's model is free, pre-installed (zero download, shared RAM), has privileged access to system intents, and — on Cactus's own numbers — is already as good as or better than tiny independents at general function calling. For "app intents on a recent iPhone/Pixel", an independent tiny model has to justify itself on latency, battery, cross-platform consistency or offline determinism, not accuracy.
- **Against absorption:** (1) OS models do not reach non-phone hardware (hubs, appliances, robots, MCUs, cameras, cheap Android, wearables' own SoCs), which is where the size advantage is decisive; (2) OEMs outside Apple/Google need a model they can fine-tune on their own tool vocabulary and ship identically across SoCs; OS models are generally not fine-tunable by third parties; (3) a ~3B model costs ~85x more compute per token (Cactus's estimate), which matters for always-on use; (4) Google's decision to release FunctionGemma as open weights suggests even the platform owner sees a need for a separate, fine-tunable tiny tier.
- **Analogy (mine):** OS-bundled models are like the built-in maps app on a phone: good enough that most phone apps will not ship their own, but irrelevant to a company building a sat-nav into a tractor. The independent tiny-model market is the tractor market: fragmented, hardware-diverse, lower-volume per customer, and sold as tooling (fine-tuning platform, engine licence) rather than as a model.
- Competitive threat to independents is as likely to come from "good-enough open weights from big labs" (FunctionGemma, future Gemma/Qwen tiny variants) plus commodity runtimes (LiteRT, llama.cpp) as from OS bundling; the single independent fine-tuned comparison found favoured FunctionGemma.
- **BACKGROUND (unverified this session):** Google exposes Gemini Nano to Android apps through AICore / ML Kit GenAI APIs on supported devices, with availability limited to higher-end hardware; Apple's Foundation Models framework supports guided (schema-constrained) generation and tool calling and is restricted to Apple Intelligence-capable devices. I could not verify the late-2026 state of either (function-calling support in AICore, device coverage, any fine-tuning/adaptor options).

### Gaps
- No analyst reports (Gartner, IDC, Counterpoint, etc.) or named commentator essays on the independent tiny tool-calling model market were found; the "views" above are HN commenters and my inference.
- Current (Nov 2026) capabilities and device coverage of Gemini Nano/AICore function calling and Apple's 2026 on-device models were not verified from primary sources (developer.android.com and developer.apple.com are not on the fetch allowlist; search returned nothing specific).
- No funding, revenue or customer-pipeline data found for Cactus Compute; no statement from Liquid on on-device revenue vs API revenue.
- No evidence found either way on whether OEMs prefer OS-bundled, big-lab open, or independent tiny models.
