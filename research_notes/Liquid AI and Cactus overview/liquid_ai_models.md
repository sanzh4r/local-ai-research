# Liquid AI models: science and architecture (as of 16 November 2026)

> **Method note for the report writer.** Direct page fetches of arxiv.org, liquid.ai and huggingface.co were blocked by this session's network policy, so nothing below was read from the full primary page. Every "Cited Finding" comes from web-search result summaries of the linked URL (primary URLs are linked wherever the search surfaced them). Items marked **[background]** are from the researcher's prior knowledge of the canonical papers/blogs and were NOT re-verified this session; treat them as probably right but check before quoting numbers. Nearly all benchmark and speed figures are **company-reported**; this is flagged per item.

## 1. What are LTC networks, CfC networks and Neural Circuit Policies? C. elegans inspiration; difference from RNNs/transformers; key papers

### Takeaway
"Liquid neural networks" are a 2018-2023 MIT/TU Wien/ISTA research line on small continuous-time recurrent networks whose neurons obey differential equations with an input-dependent ("liquid") time constant; CfC removed the expensive ODE solver with a closed-form approximation, and NCPs added a sparse four-layer wiring diagram loosely copied from the worm C. elegans. They are tiny (tens of neurons), aimed at time-series and robot control, and are a different thing from token-predicting transformers.

### Cited Findings
- **LTC paper.** "Liquid Time-constant Networks", Hasani, Lechner, Amini, Rus, Grosu; arXiv June 2020, AAAI proceedings (dated 2020 in BibTeX, proceedings volume commonly cited as 2021). The models "represent dynamical systems with varying (i.e., liquid) time-constants coupled to their hidden state, with outputs being computed by numerical differential equation solvers." — [arXiv 2006.04439](https://arxiv.org/pdf/2006.04439); [AAAI PDF](https://cdn.aaai.org/ojs/16936/16936-13-20430-1-2-20210518.pdf)
- **CfC paper.** "Closed-form continuous-time neural networks", Hasani et al., Nature Machine Intelligence 2022 (arXiv 2106.13898, 2021). Motivation: LTC expressivity/speed is bottlenecked by numerical ODE solvers. The authors derive a closed-form approximation of the LTC neuron-synapse interaction; resulting models are reported "between one and five orders of magnitude faster in training and inference" than ODE-based counterparts. — [arXiv 2106.13898](https://arxiv.org/abs/2106.13898); [ISTA record](https://research-explorer.ista.ac.at/record/12147)
- CfC "approximate[s] the closed-form solution to LTC-ODE. By replacing numerical integration through a fused solver with a single forward pass, CfC achieved the same accuracy as LTC at significantly less computational cost." — [comparative study, arXiv 2510.07578](https://arxiv.org/pdf/2510.07578)
- **NCP origins.** The 2018 precursor "Re-purposing Compact Neuronal Circuit Policies to Govern Reinforcement Learning Tasks" introduced NCPs as "a novel recurrent neural network instance with liquid time-constants, universal approximation capabilities and interpretable dynamics", inspired by the C. elegans nervous system; neuronal policies "perform as good as deep neural network policies with the advantage of realizing interpretable dynamics at the cell level." — [DeepAI mirror of the paper](https://deepai.org/publication/re-purposing-compact-neuronal-circuit-policies-to-govern-reinforcement-learning-tasks)
- **NCP flagship paper.** "Neural circuit policies enabling auditable autonomy", Lechner, Hasani, Amini, Henzinger, Rus, Grosu, Nature Machine Intelligence 2020. — cited in [arXiv 2510.07578](https://arxiv.org/pdf/2510.07578)
- **Code.** The `ncps` package implements LTC and CfC in PyTorch and TensorFlow; it describes NCPs as "designed sparse recurrent neural networks loosely inspired by the nervous system of the organism C. elegans." — [github.com/mlech26l/ncps](https://www.github.com/mlech26l/ncps)
- **Drone paper.** "Robust flight navigation out of distribution with liquid neural networks", Science Robotics (2023). — referenced in [arXiv 2510.07578](https://arxiv.org/pdf/2510.07578)
- Liquid neural networks are described as a type of time-continuous RNN introduced by MIT researchers in 2020. — [Constellation Research](https://www.constellationr.com/insights/news/liquid-ai-launches-non-transformer-genai-models-can-it-ease-power-crunch)

### Background (not re-verified this session; canonical sources linked)
- **[background] LTC equation.** Hidden state x(t) follows `dx/dt = -[1/tau + f(x, I, theta)] * x + f(x, I, theta) * A`, where f is a small neural network of state x and input I. The effective time constant `tau_sys = tau / (1 + tau*f(...))` therefore changes with the input: that is the "liquid" part. The state and time constant are provably bounded (stability). A standard RNN updates in discrete steps with fixed dynamics; a neural ODE has dx/dt = f(x,I) without the multiplicative input-dependent decay. — [arXiv 2006.04439](https://arxiv.org/pdf/2006.04439)
- **[background] CfC equation.** `x(t) = sigma(-f(x,I) t) * g(x,I) + [1 - sigma(-f(x,I) t)] * h(x,I)`: a time-dependent sigmoid gate blends two learned heads, so elapsed time t is an explicit input and no solver is needed; handles irregularly sampled data. — [arXiv 2106.13898](https://arxiv.org/abs/2106.13898)
- **[background] C. elegans.** The worm has 302 neurons with a fully mapped connectome; NCP wiring copies its sensory -> inter -> command -> motor layering with sparse, mostly feed-forward links plus recurrence among command neurons. The Nature MI 2020 lane-keeping demo used a convolutional vision front-end feeding an NCP of 19 neurons and roughly 253 synapses (control head ~1,000x smaller than comparable LSTM/CNN heads), with attention maps focused on the road horizon, which is the basis of the "auditable/interpretable" claim. — [Nature MI 2020 via ncps repo](https://www.github.com/mlech26l/ncps)
- **[background] Science Robotics 2023** (Chahine, Hasani, et al.): CfC/NCP-based drone policies trained in one environment (forest, summer) transferred to unseen environments/seasons better than LSTM/other baselines. — referenced in [arXiv 2510.07578](https://arxiv.org/pdf/2510.07578)
- **[background]** Liquid AI was spun out of MIT CSAIL in 2023 by Ramin Hasani (CEO), Mathias Lechner (CTO), Alexander Amini and Daniela Rus, i.e. the authors of these papers.

### Inferences
- Teaching framing: RNN = fixed-rule discrete update; LTC = neuron whose "reaction speed" is itself computed from the input, simulated by an ODE solver; CfC = same behaviour with a formula instead of a simulation; NCP = a wiring diagram (sparsity pattern) that can be filled with LTC or CfC neurons. Transformer = no recurrence at all, every token looks at every other token via attention.
- These networks' strengths (tiny size, continuous time, irregular sampling, causal interpretability, out-of-distribution robustness in control) are orthogonal to what makes language models good (scale, parallel training over long token sequences). This explains why the commercial language models do not use them directly (section 2).

### Gaps
- The "19 neurons / 253 synapses / 302-neuron worm" figures and the exact equations were not surfaced by this session's searches; they are from background knowledge and should be checked against the Nature MI 2020 paper and arXiv 2006.04439 (arxiv.org must be allowed in network settings to verify).
- No independent replication studies of LTC/CfC benchmark superiority were examined in depth; one 2025 comparative study (arXiv 2510.07578) exists but its conclusions were not read.

## 2. How do the commercial LFMs relate to that research? Architecture of LFM1, LFM2, LFM2.5, LFM3; how "liquid" are they; skeptical views

### Takeaway
The commercial LFMs are not LTC/CfC networks. LFM1 (Sept 2024) was closed and architecturally undisclosed; LFM2 (July 2025) onward is a decoder-only stack that looks like a transformer in which most self-attention layers are replaced by cheap gated short 1-D convolutions, with a minority of grouped-query attention layers retained; LFM3 (Oct 2026) adds sliding-window attention. The link to "liquid" research is conceptual (input-dependent gating / "linear input-varying" operators) plus branding and personnel, and the LFM2 technical report itself says the search selected no extra state-space or linear-attention operators.

### Cited Findings
**LFM1 (first generation, 30 Sept 2024)**
- Liquid AI debuted LFM-1B, LFM-3B and LFM-40B MoE on 30 Sept 2024, marketed as non-transformer models achieving "state-of-the-art performance at every scale" (company claim). — [Techmeme/VentureBeat](https://www.techmeme.com/240930/p29); [SiliconANGLE](https://siliconangle.com/2024/09/30/liquid-ai-debuts-new-lfm-based-models-seem-outperform-traditional-llms/)
- Architecture was described only abstractly: design levers of "token-mixing structure, channel-mixing structure, and featurization"; headline engineering claim was memory efficiency because a transformer's KV cache grows linearly with sequence length. — [HN thread 41698361](https://news.ycombinator.com/item?id=41698361)
- Liquid itself listed weaknesses: zero-shot code, precise numerical calculation, time-sensitive information, and no human-preference optimisation applied yet. — [Constellation Research](https://www.constellationr.com/insights/news/liquid-ai-launches-non-transformer-genai-models-can-it-ease-power-crunch)
- **Skeptical reception (independent, anecdotal):** no weights were released despite comparisons against open models; HN users reported the 40B gave "very authoritatively sounding wrong answers", suspected overfitting to MMLU-Pro, and judged the 3B "roughly on par with phi 3.5". — [Techmeme](https://www.techmeme.com/240930/p29); [HN 41699459](https://news.ycombinator.com/item?id=41699459)
- Reviewer Matthew Berman ("New Liquid Model - Benchmarks Are Useless") found the models underperformed in practical tests despite good benchmark scores. — [rosetta.to transcript page](https://rosetta.to/u/matthew_berman/new-liquid-model-benchmarks-are-useless?t=0)

**LFM2 (second generation, announced 10 July 2025; technical report arXiv 2511.23404, 28 Nov 2025)**
- Hybrid backbone: "gated short convolutions" plus "a small number of grouped query attention blocks", found via hardware-in-the-loop architecture search; conv blocks are the majority, GQA the minority, "no additional SSM or linear-attention operators". — [arXiv 2511.23404](https://arxiv.org/abs/2511.23404)
- Language models interleave ten double-gated short-convolution blocks with six GQA blocks in a 16-block pattern. — [Turing Post](https://www.turingpost.com/p/liquidhyena); [Liquid blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)
- LFM2-350M: 16 layers, 6 attention, conv kernel size 3 (other sizes may differ). — [emergentmind summary of the report](https://www.emergentmind.com/papers/2511.23404)
- Conv block = learned input-dependent multiplicative gates around a short depthwise 1-D convolution, mixing only nearby positions; feed-forward modules are SwiGLU; MoE experts are SwiGLU blocks. — [emergentmind](https://www.emergentmind.com/papers/2511.23404); [alphaxiv](https://www.alphaxiv.org/abs/2511.23404)
- Pre-norm RMSNorm, RoPE positions and QK-norm in attention are reported by a secondary Medium post (not verified against the paper). — [Medium/Coding Nexus](https://medium.com/coding-nexus/lfm2-breakthrough-small-models-that-outrun-giants-on-phones-and-laptops-e61813543cd8)
- Liquid frames the conv blocks as "Linear Input-Varying" operators: multiplicative gates + short convolutions give dynamic, input-driven weights. — [Liquid blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)
- Family: dense 350M/700M/1.2B/2.6B and MoE 8.3B total / 1.5B active, all 32K (32,768-token) context, pre-trained on 10-12T tokens. — [arXiv 2511.23404](https://arxiv.org/abs/2511.23404)
- "Not a pure state-space model like Mamba or RWKV (it still uses attention)"; built from short convolutions, a handful of attention layers and (for big models) MoE routing. — [InsiderLLM guide](https://insiderllm.com/guides/liquidai-lfm2-local-setup-guide/)

**LFM2.5 (Jan-Oct 2026): same architecture, more training**
- LFM2.5 gains came from a larger pre-training run (28T tokens for the 1.2B and 350M; ~34T for 2.6B) plus large-scale reinforcement learning, and later members extend context to 128K. — [Liquid blog LFM2.5-350M](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind); [aiweekly on 2.6B](https://aiweekly.co/alerts/liquid-ai-ships-lfm25-26b-agent-model-that-runs-on-device); [Pillitteri on 8B-A1B](https://pasqualepillitteri.it/en/news/3755/liquid-ai-lfm2-5-8b-a1b-on-device-moe)
- A Japanese explainer characterises LFM2.5 as "neither transformer nor Mamba" hybrid. — [lilting.ch](https://lilting.ch/articles/lfm-hybrid-architecture)

**LFM3 (27 Oct 2026): "tri-hybrid"**
- LFM3-1.7B-Base: 24 layers = 13 gated short convolutions + 5 sliding-window attention layers (256-token window) + 6 global attention layers; also head-wise gated attention, partial rotary position embeddings, sandwich normalisation. Search expanded to 14 device/backend combinations. 1.73B params, 256K context, 128,000-token vocabulary, 16 languages. Final 3.5T training tokens were a mid-training stage on curated data with more agentic traces/tool calls, which also extended context to 256K. — [aiweekly](https://aiweekly.co/alerts/liquid-ai-opens-six-model-lfm3-family-at-256k-context); [Unite.AI](https://www.unite.ai/liquid-ai-launches-lfm3-family-of-six-open-weight-2b-class-models/); [Local AI Zone deep dive](https://local-ai-zone.github.io/blog/lfm3-deep-dive.html) (all secondary)

### Background (not re-verified this session)
- **[background]** The July 2025 LFM2 blog gives the conv block as roughly `y = B(x) * (h conv (A(x) * x))` with linear gates A, B, and traces a lineage: LTC networks -> generalised "liquid" operators / Liquid-S4 -> linear input-varying (LIV) systems -> STAR neural-architecture search (Liquid paper, Dec 2024). It also stated LFM2 was distilled during pre-training from an internal LFM1-7B teacher, on ~75% English / 20% multilingual / 5% code data, and claimed 3x training efficiency vs. the previous generation. — [Liquid blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)

### Inferences
- **How liquid is it?** Honest answer for a lay audience: an LFM2/LFM3 has no differential equations, no ODE solver, no continuous time, no recurrent hidden state and no C. elegans wiring. What survives from the research is one idea, that the operator applied to the input should itself depend on the input (multiplicative gating), realised as a gated convolution. Gated convolutions for language are not unique to Liquid (cf. Dauphin et al. gated conv LMs, 2017; Hyena).
- **How different from a transformer?** Same outer skeleton (decoder-only, residual blocks, norm, SwiGLU MLP, RoPE, autoregressive next-token training, BPE-style tokenizer). The difference is the sequence-mixing sublayer: in LFM2, 10 of 16 blocks use a kernel-3 local convolution (cost O(n), no KV cache for those layers) and only 6 use global attention. "Hybrid conv-attention LM" is more accurate than "non-transformer".
- LFM1's "non-transformer" claims were never independently auditable (closed weights, no architecture paper); LFM2 was the first generation with open weights and a technical report.
- The fact that an architecture search across hardware converged on "mostly convs + some standard attention" and rejected SSM/linear-attention operators is itself evidence that the exotic continuous-time machinery does not pay off on current edge CPUs for language.

### Gaps
- Could not read the LFM2 technical report body: exact per-size layer counts (700M/1.2B/2.6B), GQA head/group counts, search-space contents, distillation details and whether the report discusses LTC/CfC lineage explicitly are unverified.
- LFM1 architecture remains undisclosed publicly as far as found.
- LFM3 details come only from secondary coverage of the 27 Oct 2026 release; no LFM3 technical report was found.
- The actual r/LocalLLaMA threads debating "is it really liquid" were not surfaced by search; skeptical evidence here is HN + reviewers.

## 3. Full catalogue of released models (dated)

### Takeaway
From July 2025 Liquid has shipped open-weight models in three generations (LFM2, LFM2.5, LFM3) spanning 230M to 24B parameters across text, vision, audio, retrieval and task-specific "Nanos"; many LFM2-era models are now officially deprecated in favour of LFM2.5 equivalents, and LFM3 (27 Oct 2026) is the current frontier. Dates below marked (agg.) come from third-party release trackers and conflicted in a few cases.

### Cited Findings
**Generation 1 (closed, API/playground only)**
- LFM-1B, LFM-3B, LFM-40B MoE: 30 Sept 2024; text; weights not released. — [Techmeme](https://www.techmeme.com/240930/p29)

**LFM2 text (open weights)**
- LFM2-350M, LFM2-700M, LFM2-1.2B: 10 July 2025; text; 32K context. — [Liquid blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models); [Robot Report](https://www.therobotreport.com/liquid-ai-releases-on-device-foundation-model-lfm2/)
- LFM2-2.6B: later dense addition, 32K context (exact date not found). — [arXiv 2511.23404](https://arxiv.org/abs/2511.23404)
- LFM2-2.6B-Exp: experimental checkpoint on LFM2-2.6B trained with pure reinforcement learning for instruction following, knowledge, math; now deprecated -> LFM2.5-2.6B. — [HF model card](https://huggingface.co/LiquidAI/LFM2-2.6B-Exp); [Liquid deprecations](https://docs.liquid.ai/lfm/help/deprecations)
- LFM2-8B-A1B: 7 Oct 2025 (agg.); MoE, 8.3B total / 1.5B active per token; deprecated -> LFM2.5-8B-A1B. — [theaiengineering timeline](https://theaiengineering.com/releases/liquid-ai); [Liquid docs](https://docs.liquid.ai/lfm/models/lfm2-8b-a1b)
- LFM2-24B-A2B: a "Scaling Up the LFM2 Architecture" release exists (HN thread), 24B total / 2B active; date not confirmed (background: about Feb 2026). — [HN 47957648](https://news.ycombinator.com/item?id=47957648)

**LFM2 multimodal / retrieval / Nanos**
- LFM2-VL-450M and LFM2-VL-1.6B: 12 Aug 2025; image+text -> text; SigLIP2 NaFlex image encoders on LFM2; 450M deprecated -> LFM2.5-VL-450M. — [Liquid blog](https://www.liquid.ai/blog/lfm2-vl-efficient-vision-language-models); [VentureBeat](https://venturebeat.com/ai/liquid-ai-wants-to-give-smartphones-small-fast-ai-that-can-see-with-new-lfm2-vl-model)
- LFM2-VL-3B: 22 Oct 2025 (agg.); built on LFM2-2.6B. — [theaiengineering](https://theaiengineering.com/releases/liquid-ai)
- LFM2-Audio-1.5B: 1 Oct 2025 (agg.); end-to-end audio+text in, audio+text out; extends LFM2-1.2B. — [theaiengineering](https://theaiengineering.com/releases/liquid-ai)
- LFM2-ColBERT-350M: 28 Oct 2025; multilingual late-interaction retriever on LFM2 backbone; deprecated -> LFM2.5-ColBERT-350M. — [Liquid deprecations](https://docs.liquid.ai/lfm/help/deprecations); [arXiv 2511.23404](https://arxiv.org/abs/2511.23404)
- Liquid Nanos (LFM2 models further pre-/post-trained for a single task): collection includes LFM2-1.2B-RAG, LFM2-1.2B-Tool, LFM2-1.2B-Extract, LFM2-350M-PII-Extract-JP; LFM2-350M-Math exists (ONNX community build seen). — [HF Liquid Nanos collection](https://huggingface.co/collections/LiquidAI/liquid-nanos); [arXiv 2511.23404](https://arxiv.org/abs/2511.23404)

**LFM2.5 (2026)**
- LFM2.5-1.2B family (Base, Instruct, Japanese, VL, Audio): launch early Jan 2026 (LFM2.5-Audio-1.5B listed 5 Jan 2026; Gigazine coverage 7 Jan 2026); 28T training tokens; NPU partners AMD, Qualcomm, Nexa AI. — [Liquid blog](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai); [Gigazine](https://gigazine.net/gsc_news/en/20260107-lfm2-5-on-device-ai); [llm-releases](https://www.llm-releases.com/labs/liquid-ai)
- LFM2.5-1.2B-Thinking: 20 Jan 2026; reasoning model; ~900 MB on a phone, 32K context. — [Liquid blog](https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb); [MarkTechPost](https://www.marktechpost.com/2026/01/20/liquid-ai-releases-lfm2-5-1-2b-thinking-a-1-2b-parameter-reasoning-model-that-fits-under-1-gb-on-device/)
- LFM2.5-350M: 31 Mar 2026; 28T-token pre-training + large-scale RL. — [Liquid blog](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind)
- LFM2.5-8B-A1B: 28 May 2026; MoE 8.3B/1.5B active; reasoning-only; 128K context. — [Liquid blog](https://www.liquid.ai/blog/lfm2-5-8b-a1b); [Pillitteri](https://pasqualepillitteri.it/en/news/3755/liquid-ai-lfm2-5-8b-a1b-on-device-moe)
- LFM2.5-1.2B-JP-202606 (Japanese text) plus Japanese audio model: 5 June 2026. — [Washinmura PR mirror](https://aeo.washinmura.jp/ai/liquid-ai%E6%A0%AA%E5%BC%8F%E4%BC%9A%E7%A4%BE/en/news/2026-06-05-liquid-ai-%E6%97%A5%E6%9C%AC%E8%AA%9E%E3%81%A7%E6%9C%80%E9%AB%98%E6%B0%B4%E6%BA%96%E3%81%AE%E6%80%A7%E8%83%BD%E3%82%92%E6%8C%81%E3%81%A4%E5%B0%8F%E8%A6%8F%E6%A8%A1%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%E3%81%A8%E9%9F%B3%E5%A3%B0%E3%83%A2%E3%83%87%E3%83%AB%E3%82%92%E5%85%AC)
- LFM2.5-230M: 25 June 2026 (agg.); smallest model, base for fine-tuning in agentic workflows. — [Liquid blog](https://liquid.ai/blog/lfm2-5-230m); [llm-releases](https://www.llm-releases.com/labs/liquid-ai)
- LFM2.5-2.6B: 4 Aug 2026 per Liquid blog (one aggregator says 28 May); 2.69B params; 128K context; ~34T tokens; agent workloads; LFM Open License v1.0. — [Liquid blog](https://www.liquid.ai/blog/lfm2-5-2-6b); [VentureBeat](https://venturebeat.com/technology/no-cloud-no-gpus-no-problem-liquid-ais-new-model-lfm2-5-2-6b-brings-powerful-ai-agents-to-devices-as-small-as-a-raspberry-pi)
- LFM2.5-VL-3B: Aug 2026 (12 Aug per Unite.AI; 6 Aug per another tracker); 3.1B; SigLIP2 400M NaFlex encoder + LFM2.5-2.6B backbone. — [Unite.AI](https://unite.ai/liquid-ai-ships-lfm2-5-vl-3b-for-faster-vision-language-ai-on-the-edge/); [Liquid blog](https://www.liquid.ai/blog/lfm2-5-vl-3b)
- LFM2.5-VL-450M and LFM2.5-ColBERT-350M exist as the recommended replacements for their LFM2 versions (dates not found). — [Liquid deprecations](https://docs.liquid.ai/lfm/help/deprecations)
- LFM2.5-24B-A2B: Oct 2026 (13 or 15 Oct depending on source); MoE 24B total / 2B active; "largest and final" LFM2.5 model. — [Liquid blog](https://www.liquid.ai/en/blog/lfm2-5-24b-a2b); [ThursdAI](https://thursdai.news/companies/liquid-ai)

**LFM3 (27 Oct 2026): six ~2B open-weight models from one base, LFM3-1.7B-Base**
- LFM3-1.7B: text -> text; 1.73B; 256K context; 16 languages; post-trained for agent harnesses (Hermes Agent, Pi, OpenClaw); LFM Open License v1.0.
- LFM3-VL-2B: text+image+video (2 fps) -> text; 1.9B; new in-house 164M vision encoder.
- LFM3-Audio-2B: text/audio -> text/speech; 18 languages.
- LFM3-Omni-2B: text+image+video+audio -> text+speech; Liquid's first omni model.
- d1-2B and d1-Omni-2B: "decision" models specialised from VL and Omni, returning typed/calibrated answers rather than prose.
- Sources: [Unite.AI](https://www.unite.ai/liquid-ai-launches-lfm3-family-of-six-open-weight-2b-class-models/); [aiweekly](https://aiweekly.co/alerts/liquid-ai-opens-six-model-lfm3-family-at-256k-context); [VentureBeat](https://venturebeat.com/technology/liquid-ais-new-lfm3-models-put-browser-agents-voice-commands-and-image-recognition-onto-enterprise-controlled-devices); [NYU Shanghai RITS](https://rits.shanghai.nyu.edu/ai/liquid-ai-opens-six-lfm3-models-on-one-1-7b-backbone/); [TechTimes](https://www.techtimes.com/articles/329588/20261028/liquid-ai-releases-lfm3-six-open-weight-models-fuse-text-vision-audio-one-2b-backbone.htm)

**Licence**
- LFM Open License v1.0: use, modification, redistribution incl. commercial use free for organisations under US$10M annual revenue; larger companies need a commercial licence. Described variously as "built on Apache 2.0" and "not a standard Apache 2.0 licence". — [VentureBeat](https://venturebeat.com/technology/no-cloud-no-gpus-no-problem-liquid-ais-new-model-lfm2-5-2-6b-brings-powerful-ai-agents-to-devices-as-small-as-a-raspberry-pi); [iadecider](https://iadecider.com/articles/liquid-ai-review); [The Rundown](https://www.therundown.ai/tools/lfm-2-5)
- One count puts the ecosystem at "3,100+ variants" (quantisations/formats). — [hokai.io](https://hokai.io/hub/companies/liquid-ai)

### Inferences
- Naming convention: `LFM<gen>-<total params>[-A<active params>B]` for MoE (8B-A1B = 8B total, ~1B active); suffixes -VL, -Audio, -ColBERT, -Extract/-RAG/-Tool/-Math, -Thinking, -JP.
- "Open weights", not "open source" in the OSI sense, because of the revenue cap.
- The Nano names in the brief ("LFM2-350M-Extract", "LFM2-350M-ENJP-MT") were not directly confirmed by search this session; background knowledge says the 25 Sept 2025 Nanos launch included LFM2-350M-Extract, LFM2-1.2B-Extract, LFM2-350M-ENJP-MT, LFM2-1.2B-RAG, LFM2-1.2B-Tool and LFM2-350M-Math, with PII-Extract-JP added later.

### Gaps
- No robotics- or time-series-specific released foundation model was found in any search (the liquid-network robotics work remains research papers, not catalogue products). The "d1" decision models are the nearest non-generative product.
- Missing exact dates: LFM2-2.6B, LFM2-24B-A2B, LFM2.5-VL-450M, LFM2.5-ColBERT-350M, Nanos launch date; context lengths for VL/Audio/ColBERT/Nanos models; per-model licence for LFM3 non-text models.
- Date conflicts between aggregators and Liquid's blog (LFM2.5-2.6B, LFM2.5-VL-3B, LFM2.5-24B-A2B); Liquid's blog dates should be preferred.
- Hugging Face org page could not be fetched, so the catalogue may omit models (e.g. LFM2.5-Audio variants, other language-specific fine-tunes).

## 4. Performance claims: benchmarks, speed/memory, training efficiency; independent vs company

### Takeaway
Liquid's consistent claim is "same or better quality than the next size class up, at roughly 2x the CPU speed"; nearly every number is company-run, and the limited independent evidence is mixed: a third-party fine-tuning study found LFM2-1.2B beating some larger models, while forum anecdotes report coherence problems and Liquid itself says the models are not for coding or knowledge-heavy work.

### Cited Findings
**Company claims**
- LFM2 launch: "2x faster decode and prefill than Qwen3 on CPU" (headline version: 200% faster than Qwen3 and Gemma 3; the page is internally inconsistent on whether Gemma 3 is included). — [Liquid blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)
- Technical report: "up to 2x faster prefill and decode on CPUs compared to similarly sized models". — [arXiv 2511.23404](https://arxiv.org/abs/2511.23404)
- LFM2-1.2B competitive with Qwen3-1.7B (47% more params); LFM2-700M beats Gemma 3 1B IT; from Liquid's internal eval suite. — [Liquid blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)
- Report caveat: a "loose" score was used for Gemma 3 because strict scoring gave it low performance; LFM2-700M outperforms Qwen3-0.6B in all four settings of that comparison. — [arXiv 2511.23404 HTML](https://arxiv.org/html/2511.23404v1)
- LFM2.5-1.2B-Instruct: GPQA 38.89, MMLU-Pro 44.35, IFEval 86.23, IFBench 47.33, Multi-IF 60.98. — [Liquid blog](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- LFM2.5-1.2B-Thinking: ~239 tok/s decode on an AMD CPU, ~82 tok/s on a mobile NPU, under 1 GB (about 900 MB on phone); matches or exceeds Qwen3-1.7B on most reasoning benchmarks with 40% fewer params. — [Liquid blog](https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb); [MarkTechPost](https://www.marktechpost.com/2026/01/20/liquid-ai-releases-lfm2-5-1-2b-thinking-a-1-2b-parameter-reasoning-model-that-fits-under-1-gb-on-device/)
- LFM2.5-8B-A1B: 253 tok/s on Apple M5 Max. — [Pillitteri](https://pasqualepillitteri.it/en/news/3755/liquid-ai-lfm2-5-8b-a1b-on-device-moe)
- LFM2.5-2.6B: "competitive with models nearly four times its size"; Liquid does not recommend it for agentic coding or knowledge-heavy tasks; pitched as running on a Raspberry Pi. — [Liquid blog](https://liquid.ai/blog/lfm2-5-2-6b); [VentureBeat](https://venturebeat.com/technology/no-cloud-no-gpus-no-problem-liquid-ais-new-model-lfm2-5-2-6b-brings-powerful-ai-agents-to-devices-as-small-as-a-raspberry-pi)
- LFM2.5-24B-A2B: competitive with Qwen3.6-35B-A3B on 5 of 8 agentic/tool-use benchmarks; leads all three instruction-following benchmarks (IFEval 96.75); larger models still lead on GPQA Diamond and SWE-Bench Verified; 184 tok/s on M5 Max. — [Liquid blog](https://www.liquid.ai/en/blog/lfm2-5-24b-a2b)
- LFM3-1.7B: averages 49.37 across Liquid's 29-benchmark table vs 47.24 for Qwen3.5-9B; beats LFM2.5-2.6B with 35% fewer params; WebArena-Verified 38.42 vs 3.02 for LFM2.5-2.6B; decodes 1.8-2.8x faster than the next model on three on-device setups; ~1 GB at 4-bit; model card says not recommended for agentic coding or complex multi-step reasoning. — [eesel.ai](https://www.eesel.ai/blog/lfm3); [aiweekly](https://aiweekly.co/alerts/liquid-ai-opens-six-model-lfm3-family-at-256k-context); [Data Phoenix](https://dataphoenix.info/news/liquid-ai-lfm3-models-local-multimodal-ai)
- LFM3-VL-2B: average over 62 vision benchmarks 71.4 vs 57.6 for LFM2.5-VL-3B (1.9B vs 3.1B params). — [aiweekly](https://aiweekly.co/alerts/liquid-ai-opens-six-model-lfm3-family-at-256k-context)
- LFM3-Audio-2B: UTMOSv2 3.83 vs 3.22 for LFM2.5-Audio-1.5B; time-to-first-token as low as 50 ms. — [Startup Fortune](https://startupfortune.com/liquid-ai-releases-lfm3-models-small-enough-to-run-on-a-phone/)

**Independent / third-party**
- AMD Ryzen NPU via FastFlowLM (third-party runtime): LFM2.5-1.2B sustains ~52 tok/s at 16K context and ~46 tok/s at full 32K. — [FastFlowLM HF card](https://huggingface.co/FastFlowLM/LFM2.5-1.2B-Thinking-NPU2)
- distil labs fine-tuning study: LFM2-1.2B outperformed "several larger competitors including SmolLM2-1.7B and gemma-3-1b-it" after fine-tuning; Qwen3-1.7B ranked 4th on fine-tuned average rank (4.44 +/- 1.60). — [distil labs](https://www.distillabs.ai/learn/qwen3-1-7b-vs-gemma-3-1b-for-classification/)
- HN on LFM2-24B-A2B: one user said the LFM models they tried "all seemed to be suffering from serious coherence issues"; others preferred Gemma for short chats and Qwen for long contexts/code. — [HN 47957648](https://news.ycombinator.com/item?id=47957648)
- Coverage of LFM3 and LFM2.5 repeatedly notes all figures are self-reported with no independent evaluations yet. — [TechTimes](https://www.techtimes.com/articles/329588/20261028/liquid-ai-releases-lfm3-six-open-weight-models-fuse-text-vision-audio-one-2b-backbone.htm)

### Background (not re-verified)
- **[background]** July 2025 blog: 3x training efficiency vs. LFM1; knowledge distillation from LFM1-7B throughout pre-training. — [Liquid blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)

### Inferences
- Pattern across generations: strongest on instruction following, tool calling, extraction and speed; weakest on stored knowledge, coding and long multi-step reasoning. This is what one expects when most layers only see a 3-token neighbourhood and total parameters are small: little capacity for memorised facts, but adequate for format-following and routing.
- Quality gains from LFM2 -> LFM2.5 came from training (2.3-3x more tokens, RL), not architecture, which weakens the story that the architecture is the differentiator for quality; the architecture's defensible advantage is speed/memory per parameter.
- Comparisons often pit LFMs against older or differently-budgeted baselines chosen by Liquid, and benchmark suites change between releases (10, 29, 62 benchmarks), so cross-release numbers are not comparable.
- Liquid's ~24-34T token pre-training budgets are very large for 1-3B models; "small model beats bigger model" claims are partly a training-compute story.

### Gaps
- No full benchmark tables (per-model MMLU/GSM8K/IFEval vs Qwen/Gemma/Llama/Phi/SmolLM) could be extracted without access to liquid.ai/arXiv; the report writer should not reproduce a comparison table from these notes.
- No independent, controlled CPU tokens-per-second replication of the "2x" claim was found.
- No phone-specific (Snapdragon/Galaxy/iPhone) tok/s tables were retrieved this session, though they exist in the blogs and technical report.
- No Phi-specific or Llama 3.2-specific head-to-head numbers were found.

## 5. Supported runtimes and formats

### Takeaway
Liquid ships its weights in the mainstream local-inference formats (GGUF for llama.cpp, MLX, ONNX, plus server stacks) and wraps them in its own LEAP SDK for iOS/Android; ExecuTorch support was not confirmed by sources found this session.

### Cited Findings
- Formats/runtimes listed as supported from release day for LFM2.5: llama.cpp via GGUF, MLX on Apple silicon, vLLM, SGLang, ONNX and CoreML; not every model exists in every configuration. — [The Rundown](https://www.therundown.ai/tools/lfm-2-5); [hokai.io](https://hokai.io/hub/companies/liquid-ai)
- Official GGUF repos exist (e.g. LiquidAI/LFM2-8B-A1B-GGUF, LFM2-2.6B-Exp-GGUF); community ONNX builds exist (onnx-community/LFM2-350M-Math-ONNX). — [HF GGUF repo](https://huggingface.co/LiquidAI/LFM2-8B-A1B-GGUF); [HF Nanos collection](https://huggingface.co/collections/LiquidAI/liquid-nanos)
- LEAP (Liquid Edge AI Platform): cross-platform SDK for running small models on mobile/embedded; iOS and Android; works with Liquid's and other open small models; free developer licence at launch, enterprise features to be commercial. — [VentureBeat](https://venturebeat.com/ai/finally-a-dev-kit-for-designing-on-device-mobile-ai-apps-is-here-liquid-ais-leap)
- Apollo: at launch a free iOS app for chatting with LEAP-compatible models offline; a later profile calls it a hosted access layer (descriptions conflict). — [VentureBeat](https://venturebeat.com/ai/finally-a-dev-kit-for-designing-on-device-mobile-ai-apps-is-here-liquid-ais-leap); [hokai.io](https://hokai.io/hub/companies/liquid-ai)
- NPU paths: AMD, Qualcomm and Nexa AI partnerships for LFM2.5; FastFlowLM publishes AMD Ryzen NPU builds. — [Liquid blog](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai); [FastFlowLM](https://huggingface.co/FastFlowLM/LFM2.5-1.2B-NPU2)
- Hosted API exists for larger models; LFM2.5-24B-A2B pricing reported as $0.05/$0.70 per M tokens by one listing and $0.10/$1.00 by Unite.AI (conflict). — [benchmarklist](https://benchmarklist.com/models/liquid-lfm-2.5-24b-a2b/); [Unite.AI](https://www.unite.ai/?p=484993)

### Background (not re-verified)
- **[background]** The July 2025 LFM2 blog and the technical report benchmarked on both llama.cpp (GGUF) and ExecuTorch, and LFM2 support was merged into upstream llama.cpp and Hugging Face transformers in July 2025; LEAP launched 15 July 2025. — [Liquid blog](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)

### Inferences
- Because LFM2 is a custom block type, each runtime needed explicit support (a new architecture in llama.cpp, not just a new GGUF file); this is why availability "day one" in llama.cpp/MLX/ONNX is something Liquid advertises.

### Gaps
- ExecuTorch: no source found this session confirms it (one passing reference to "ExecuTorch-style runtimes"); background says it was a launch benchmark runtime. Verify on liquid.ai docs.
- LEAP SDK's current inference backends, supported platforms beyond iOS/Android, and how it relates to Cactus were not established.

## 6. Why does this architecture suit edge devices?

### Takeaway
On a phone or laptop CPU the costly parts of a transformer are attention's quadratic prompt processing and its ever-growing key-value cache; LFM2 replaces most attention layers with kernel-3 convolutions that cost a fixed, tiny amount per token and need almost no cache, and the layer mix was chosen by measuring real latency and memory on target devices rather than by theory.

### Cited Findings
- The architecture was selected by hardware-in-the-loop search (measuring on-device latency/memory), yielding mostly gated short convolutions with a few GQA layers and up to 2x faster CPU prefill and decode. — [arXiv 2511.23404](https://arxiv.org/abs/2511.23404)
- Gated short convolutions mix only nearby positions with a short depthwise 1-D kernel (size 3 in LFM2-350M). — [emergentmind](https://www.emergentmind.com/papers/2511.23404)
- Liquid's stated memory argument since LFM1: a transformer's KV cache grows linearly with sequence length. — [HN 41698361](https://news.ycombinator.com/item?id=41698361)
- LFM3 extended the search to 14 device/backend combinations and added sliding-window (256-token) attention layers between local convs and global attention. — [aiweekly](https://aiweekly.co/alerts/liquid-ai-opens-six-model-lfm3-family-at-256k-context)
- Footprints: LFM2.5-1.2B-Thinking ~900 MB on phone; LFM3-1.7B ~1 GB at 4-bit; MoE models activate only 1.5-2B of 8-24B parameters per token. — [Liquid blog](https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb); [Data Phoenix](https://dataphoenix.info/news/liquid-ai-lfm3-models-local-multimodal-ai); [Liquid docs](https://docs.liquid.ai/lfm/models/lfm2-8b-a1b)

### Inferences
- Mechanism for teaching: (a) prefill, a conv layer processes an n-token prompt in O(n) with small constants and cache-friendly sequential memory access, whereas attention is O(n^2); with 10 of 16 layers convolutional, most of the stack escapes the quadratic term; (b) decode, a kernel-3 conv layer needs only the last couple of activations as state, versus keys and values for every past token in an attention layer, so the KV cache is roughly 6/16 the size of an all-attention model of equal depth, and GQA shrinks it further by sharing key/value heads; (c) convolutions and small matmuls are the operations CPUs, NPUs and their kernels (XNNPACK, NEON, Accelerate) are best optimised for, and they quantise well; (d) MoE trades storage (cheap on-device) for compute (scarce), keeping per-token work at 1.5-2B-parameter level.
- The few global attention layers are kept because pure local mixing cannot do long-range lookup (copying a name from 5,000 tokens back); the search evidently found about a third of layers enough.
- Edge suitability is thus an engineering co-design result, not a consequence of "liquid" continuous-time dynamics.
- Trade-off: fewer global-mixing layers and small parameter counts plausibly explain the observed weakness on knowledge-heavy, coding and long multi-step reasoning tasks (section 4).

### Gaps
- No measured KV-cache or peak-RAM comparison against an equal-size transformer was retrieved; the 6/16 cache ratio is the researcher's arithmetic, not a published figure.
- Energy-per-token (battery) measurements were not found.
