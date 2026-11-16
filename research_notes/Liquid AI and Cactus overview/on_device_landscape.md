# On-Device / Edge / Local AI Landscape (as of 16 Nov 2026): Situating Liquid AI and Cactus Compute

> **Method caveat for the report writer (read first).** Page fetches were blocked in this research session (liquid.ai, cactuscompute.com, docs.cactuscompute.com, ycombinator.com and others did not resolve from the sandbox). Every cited finding below is therefore taken from **search-result summaries/snippets** of the linked page, not from a full read of the page. Quotes are as surfaced by search and may be lightly truncated. Where a claim is vendor-authored (most Cactus and Liquid claims are), it is marked "vendor claim". Items I know from background knowledge but could not source in this session are placed under Inferences or Gaps and labeled "unsourced background". Nothing here was independently benchmarked.

## 1. What is the stack for running AI locally on phones/laptops/embedded devices?

### Takeaway
The stack is commonly described in 4-6 layers: silicon (CPU/GPU/NPU) -> inference runtime/kernels -> model format + quantization -> small models -> SDK/platform/orchestration (incl. local-vs-cloud routing, fleet/model management) -> applications. Liquid AI's center of gravity is the **model** layer (with LEAP reaching up into tooling/SDK); Cactus's center of gravity is the **runtime + SDK + routing** layer.

### Cited Findings
- One analyst-style framework splits the "ambient computing" stack into four layers: silicon (NPU chips), on-device runtime (inference frameworks), orchestration (local-vs-cloud routing and context management), and application (user-facing features); it proposes orchestration/routing as the value layer. — [liveinthefuture.org](https://liveinthefuture.org/stories/ambient-ai-smart-devices-platform-layer)
- Cactus describes its own architecture in three layers: kernels (low-level compute routines optimized for CPU and NPU), a graph layer, and an engine (vendor description). — [Cactus docs v1.9](https://docs.cactuscompute.com/v1.9/)
- Model format layer: llama.cpp's GGUF is described in 2026 guides as "the most widely supported format"; llama.cpp is called the most popular open-source local inference engine (86K+ GitHub stars per one guide). — [Edge LLM Runtime Stack 2026](https://edgeaistack-site.pages.dev/blog/edge-llm-runtime-stack-2026/); [Cactus comparison page](https://cactuscompute.com/compare/best-on-device-llm-framework)
- Quantization layer examples: Cactus reports INT8 and (Feb 2026) INT4 / "lossless quantisation" in its release timeline; an LFM3 model card notes "NVFP4 GGUF supported in llama.cpp b12088". — [Cactus docs v1.9](https://docs.cactuscompute.com/v1.9/); [LiquidAI/LFM3-1.7B-GGUF on Hugging Face](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/main)
- ExecuTorch uses PyTorch's native quantization and is positioned as PyTorch-native for mobile/embedded with "Meta-scale production reliability" (guide characterization). — [Edge LLM Runtime Stack 2026](https://edgeaistack-site.pages.dev/blog/edge-llm-runtime-stack-2026/)
- Hardware layer is now a co-design target for model builders: Liquid says it used "proprietary automated hardware-in-the-loop architecture search" for LFM3; for LFM2 the search targeted CPUs on phones/laptops; LFM3's search reportedly spanned 14 hardware combinations including Qualcomm Hexagon NPU and AMD Ryzen AI laptops. — [Liquid AI on X](https://x.com/liquidai/status/2115128333054964049); [Liquid AI LFM3 blog](https://www.liquid.ai/blog/lfm3)
- Silicon/hardware partners named at LFM3 launch (27 Oct 2026): Ambarella, AMD, Arm, Intel (via OpenVINO), NVIDIA, Qualcomm (via Qualcomm AI Hub and GenieX); plus OpenRouter and Vercel on the distribution side. — [Liquid AI LFM3 blog](https://www.liquid.ai/blog/lfm3)
- OS-vendor layer: Apple at WWDC 2026 introduced **Core AI**, described in Apple's material as "a brand new framework ... designed to be the best way to bring and run models on-device in your apps", with ahead-of-time compilation and PyTorch conversion tools; coverage says it succeeds Core ML (2017). — [Apple WWDC26 session 382](https://developer.apple.com/videos/play/wwdc2026/382/?time=1271); [Callstack](https://callstack.com/blog/on-device-ai-after-wwdc-2026-whats-new); [letsdatascience](https://letsdatascience.com/news/apple-unveils-core-ai-for-on-device-generative-models-1142390f)
- A less-discussed operational layer is model delivery/versioning: "version fragmentation is the central operational challenge of on-device AI" because "the model" is a bundle of artifacts that drift across devices. — [tianpan.co, Apr 2026](https://tianpan.co/blog/2026-04-19-on-device-llm-inference-model-update-propagation)

### Inferences
- Suggested layer map for the report (synthesis of the above; layer assignments are mine):
  1. **Hardware/silicon**: Apple silicon (ANE/GPU), Qualcomm Snapdragon/Hexagon NPU, AMD Ryzen AI, Intel (OpenVINO), Arm, NVIDIA Jetson, Ambarella, MediaTek (unsourced background), Raspberry Pi-class boards.
  2. **Kernels / inference runtimes**: llama.cpp, ExecuTorch, MLX, LiteRT / LiteRT-LM, ONNX Runtime, Core ML -> Core AI, MLC (unsourced background), TensorRT Edge-LLM, Cactus engine, Nexa SDK.
  3. **Model formats + quantization**: GGUF (llama.cpp), .pte (ExecuTorch, unsourced background), MLX weights, ONNX, LiteRT/.litertlm (unsourced background), Core ML/Core AI packages, Cactus's own converted weights (it publishes "Cactus-Compute/LFM2-*" repos on HF); INT8/INT4/NVFP4 etc.
  4. **Models**: Gemma, Phi, Qwen, Llama/MobileLLM, SmolLM, Apple AFM, Liquid LFM2/2.5/3.
  5. **SDK / platform / orchestration**: Cactus SDKs + hybrid routing, Liquid LEAP, RunAnywhere, Nexa SDK, Ollama / LM Studio (desktop), Google AI Edge/MediaPipe, Apple Foundation Models framework.
  6. **Apps**: Liquid's Apollo app, end-developer apps; OEM system features (Siri/Apple Intelligence, Gemini Nano).
- Liquid spans layers 4 and 5 (and touches 1 through hardware-in-the-loop design); Cactus spans layers 2, 3 and 5. The overlap zone is layer 5 (developer SDK/deployment tooling).

### Gaps
- No single neutral, authoritative taxonomy of the stack was found; the layer names above are synthesized from a blog framework plus vendor docs.
- MLC LLM, ONNX Runtime GenAI, MediaPipe LLM Inference API and Ollama were not individually researched with sources this session (search budget); their placement above is from background knowledge.

## 2. Small-model builders: how do Gemma, Phi, Qwen, Llama/MobileLLM, SmolLM, Apple and Liquid LFM compare in approach and openness?

### Takeaway
Big-tech small models are mostly given away under permissive (MIT/Apache) or custom-but-free terms to feed their own ecosystems; Liquid is the notable venture-backed independent whose small models are open-weight but under a **revenue-thresholded custom license** (free commercial use below $10M revenue), and whose differentiator is a non-standard, hardware-searched architecture optimized for CPU/NPU speed rather than a shrunk transformer.

### Cited Findings
**Liquid AI (LFM family)**
- LFM2 is designed for edge use; its technical report claims up to 2x faster CPU inference than similarly sized models (vendor claim). — [arXiv 2511.23404 overview](https://www.alphaxiv.org/overview/2511.23404)
- LFM2.5 launched 5 Jan 2026, extending LFM2 with more pretraining and RL post-training. — [latent.space wiki profile](https://latent.space/wiki/liquid-ai-7ebae2f8/)
- LFM2-24B-A2B (MoE) is Liquid's largest LFM2 model, pitched "from cloud to AI PC" with a partner ecosystem. — [Liquid AI blog](https://www.liquid.ai/blog/lfm2-24b-a2b-from-cloud-to-ai-pc)
- LFM2.5-350M ("No Size Left Behind") is available on Hugging Face, LEAP and Liquid's Playground. — [Liquid AI blog](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind)
- **LFM3 launched 27 Oct 2026**: six open-weight ~2B-class models (text, vision, audio, omni, decision) trained from one shared base, LFM3-1.7B-Base; native support for llama.cpp, MLX and vLLM across Apple, AMD, NVIDIA, Qualcomm hardware. — [MarkTechPost](https://www.marktechpost.com/2026/10/27/liquid-ai-lfm3-open-weight-2b-multimodal-models/); [Unite.AI](https://www.unite.ai/liquid-ai-launches-lfm3-family-of-six-open-weight-2b-class-models/); [Liquid AI LFM3 blog](https://www.liquid.ai/blog/lfm3)
- Mercedes-Benz and Shopify reportedly partnered on LFM3 pre-release testing/evaluation. — [Unite.AI](https://www.unite.ai/liquid-ai-launches-lfm3-family-of-six-open-weight-2b-class-models/)
- License: LFM3 weights ship under the "LFM Open License v1.0"; commercial use is allowed for companies under $10M annual revenue; above that a separate commercial deal is needed; non-profit research is exempt from the cap; no public price above the threshold. — [HF LICENSE file](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE); [FourWeekMBA](https://fourweekmba.com/ai-liquid-ai-opens-lfm3-free-to-firms-under-10m-in-revenue/); Liquid's license page (last updated 15 Oct 2026) says business use can count as commercial even when not directly revenue-generating — [Liquid docs](https://docs.liquid.ai/lfm/help/model-license)
- Scale (vendor-derived, via a profile checked Aug 2026): 42.2M model downloads, 56 LFMs shipped, 3,300+ variants. — [yespress profile](https://yespress.io/liquid-ai)
- The European Open Source AI Index maintains an openness entry for LFM. — [osai-index.eu](https://osai-index.eu/model/lfm)

**Google (Gemma)**
- Gemma 3 270M: 270M-parameter model in pre-trained and instruction-tuned versions; IFEval 51.2% (instruction-tuned); under the custom Gemma Terms of Use (use/modification/distribution allowed subject to conditions). — [VentureBeat](https://venturebeat.com/ai/google-unveils-ultra-small-and-efficient-open-source-ai-model-gemma-3-270m-that-can-run-on-smartphones)
- FunctionGemma: a Gemma 3 270M variant tuned for function calling; fine-tuning raised "Mobile Actions" accuracy from 58% to 85%; targets phones and Jetson Nano-class devices; Gemma Terms of Use. — [Google blog](https://blog.google/technology/developers/functiongemma/); [VentureBeat](https://venturebeat.com/technology/google-releases-functiongemma-a-tiny-edge-model-that-can-control-mobile)
- Gemma 4 (E2B/E4B) shipped alongside LiteRT-LM on 7 Apr 2026; one source says Gemma 4 E2B/E4B are Apache-2.0 while Gemma 3n E4B is under "Gemma terms". — [Edge LLM Runtime Stack 2026](https://edgeaistack-site.pages.dev/blog/edge-llm-runtime-stack-2026/); [Turing Post SLM list](https://www.turingpost.com/p/slmslist) (license claim flagged unverified; sources conflict — check the model card)

**Microsoft (Phi)**
- Phi-4-mini (3.8B): MIT license, ~3 GB VRAM, 128K context. — [tinyweights.dev](https://tinyweights.dev/posts/best-small-language-models-2026/); background at [Wikipedia: Phi](https://en.wikipedia.org/wiki/Phi_(language_model))

**Alibaba (Qwen)**
- Qwen3 small models (0.6B-8B) are Apache 2.0; Qwen3.5 small models also named as Apache 2.0 for newer deployments. — [Turing Post SLM list](https://www.turingpost.com/p/slmslist)
- Qwen3-0.6B is a de facto cross-runtime benchmark model (used in the ExecuTorch paper and the Apple-silicon bench repo). — [arXiv 2605.08195](https://arxiv.org/pdf/2605.08195); [apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench)

**Hugging Face (SmolLM)**
- SmolLM3-3B: Apache-2.0, July 2025, positioned as a "fully open model". — [Turing Post SLM list](https://www.turingpost.com/p/slmslist)

**Meta (Llama small / MobileLLM)**
- Llama 3.2 1B is one of three reference models (with Qwen3 0.6B and Phi-4 Mini) in Meta's ExecuTorch benchmark paper. — [arXiv 2605.08195](https://arxiv.org/pdf/2605.08195)

**Apple**
- Apple's on-device model exposed via the Foundation Models framework is reported as a ~3B-parameter model ("AFM 3 Core") accessed through a native Swift API; closed weights, OS-bundled. — [Callstack](https://callstack.com/blog/on-device-ai-after-wwdc-2026-whats-new) (model name/size is from secondary coverage; treat as unverified)
- After WWDC 2026, the Foundation Models framework reportedly offers one Swift API spanning on-device, Private Cloud Compute, and third-party models (Claude, Gemini), and gained image input. — [appbot.co](https://appbot.co/blog/apple-wwdc-2026-ai-foundation-model-update/); [MacRumors](https://macrumors.com/2026/06/09/apple-outlines-major-ai-and-developer-tool-updates)

### Inferences
- Openness spectrum (most to least permissive): SmolLM / Qwen / Phi (Apache/MIT) -> Gemma (custom terms; possibly Apache for Gemma 4, unverified) -> Meta Llama (custom community license, unsourced background) -> **Liquid LFM (open weights, revenue-capped commercial use)** -> Apple AFM / Google Gemini Nano (closed, OS-bundled, API-only).
- Liquid is the only player in this list whose **primary business** is the small model itself; every big-tech competitor subsidizes its small models from another profit pool (cloud, ads, devices). That makes Liquid's license threshold the monetization mechanism and also a friction point versus Apache/MIT alternatives.
- Liquid's approach differs technically: architecture discovered by hardware-in-the-loop search (hybrid conv/attention blocks in LFM2 — unsourced background detail) versus distilled/shrunk transformers from the others; and family breadth (text, vision, audio, omni, "decision") from one ~1.7B base.
- Task-specialized tiny models (FunctionGemma 270M; LFM2.5-350M; Liquid "nano" task models — the last is unsourced background) show convergence on "fine-tune a sub-1B model per task" as the on-device pattern.

### Gaps
- No sourced data this session on Meta **MobileLLM / MobileLLM-Pro** (license, sizes) — searches returned nothing.
- Gemma 3n details (MatFormer, effective 2B/4B memory footprint) not sourced this session; Gemma 4 license conflict unresolved.
- Llama small-model license terms and Google Gemini Nano / AICore details not sourced.
- No independent head-to-head quality benchmark of LFM3 vs Gemma 4 / Qwen3.5 / Phi at equal size was retrieved; Liquid's speed/quality claims are vendor claims.

## 3. Runtime/SDK players: how do llama.cpp, ExecuTorch, MLX, LiteRT/AI Edge, ONNX Runtime, MLC, Apple frameworks, Nexa, RunAnywhere, Cactus, LEAP, Ollama compare?

### Takeaway
Open-source runtimes backed by big tech or community (llama.cpp, ExecuTorch, MLX, LiteRT-LM, ONNX Runtime, Core AI) own the kernel layer; startups (Cactus, Nexa, RunAnywhere) differentiate on cross-platform mobile SDK ergonomics, NPU access, hybrid cloud fallback, and fleet/control-plane features. Liquid's LEAP is a model-centric toolkit that **rides on** third-party runtimes rather than competing as an engine.

### Cited Findings
**Open-source / platform runtimes**
- llama.cpp: most portable option ("pick this when portability matters more than peak throughput"); most popular OSS local engine; GGUF most widely supported format. — [Edge LLM Runtime Stack 2026](https://edgeaistack-site.pages.dev/blog/edge-llm-runtime-stack-2026/); [Cactus comparison page](https://cactuscompute.com/compare/best-on-device-llm-framework)
- ExecuTorch: Meta's arXiv paper benchmarks it against llama.cpp, ONNX Runtime, LiteRT and CoreML using framework versions as of 31 Mar 2026 on Qwen3 0.6B, Llama 3.2 1B, Phi-4 Mini with quantization standardized where possible. — [arXiv 2605.08195](https://arxiv.org/pdf/2605.08195)
- LiteRT-LM (Google AI Edge): launched 7 Apr 2026 with Gemma 4 support, targeting phones, IoT, desktop; one review says NPU acceleration is still early access and it "needs a year of community baking". — [Edge LLM Runtime Stack 2026](https://edgeaistack-site.pages.dev/blog/edge-llm-runtime-stack-2026/); [shiporskip.io](https://shiporskip.io/compare/litert-lm-google-ai-edge-on-device-llm-inference-framework-vs-meta-llama-4-scout-maverick-quantized-on-device-inference)
- Community Apple-silicon benchmark (iPhone 17 Pro, Qwen3-0.6B): MLX Swift 167.2 tok/s on GPU vs LiteRT-LM 121.0; LiteRT-LM was "the only arm under a gigabyte" on footprint and resident memory; repo also covers llama.cpp, CoreML, ExecuTorch and Apple's Core AI; author revised earlier results. — [john-rocky/apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench)
- MLX: strongest on Apple silicon, narrower ecosystem. — [Edge LLM Runtime Stack 2026](https://edgeaistack-site.pages.dev/blog/edge-llm-runtime-stack-2026/)
- Apple Core AI (WWDC, June 2026): lower-level framework for bringing your own model on-device; Foundation Models framework is the higher-level API over Apple's own model (plus PCC and third-party models). One report says Apple plans to open-source "the framework" later in summer 2026 (unverified which framework/whether it happened). — [Callstack](https://callstack.com/blog/on-device-ai-after-wwdc-2026-whats-new); [implicator.ai](https://www.implicator.ai/apple-expands-xcode-and-foundation-models-for-wwdc-ai-developers/)
- General runtime primer covering ONNX Runtime, TFLite/LiteRT, Core ML, ExecuTorch etc. — [aman.ai ML runtimes primer](https://aman.ai/primers/ai/ml-runtimes/)

**Cactus (YC S25)**
- Launched on Hacker News as "Launch HN: Cactus (YC S25)" (Sept 2025 thread). — [Hacker News](https://news.ycombinator.com/item?id=45291024)
- Self-description: hybrid AI inference engine for mobile, desktop and edge hardware; covers LLMs, speech transcription, vision, embeddings; SDKs for Swift, Kotlin, Flutter, React Native, Python, C++, Rust. — [Cactus docs v1.9](https://docs.cactuscompute.com/v1.9/); [Cactus Engine](https://cactuscompute.com/engine)
- Cactus v1 (covered by InfoQ, Dec 2025): cross-platform LLM inference on mobile; v1 SDK in beta with bindings for React Native, Flutter, Kotlin Multiplatform. — [InfoQ](https://www.infoq.com/news/2025/12/cactus-on-device-inference/); [letsdatascience](https://letsdatascience.com/news/cactus-releases-v1-sdk-for-on-device-inference-ae037d65)
- Hybrid routing: requests go to a small on-device model or a cloud model; routing "measures model 'confidence'" in real time (vendor claim). — [Cactus comparison page](https://cactuscompute.com/compare/best-on-device-llm-framework)
- 2026 release timeline (vendor docs): Jan — Apple NPU/RAM optimisations; Feb — hybrid inference with GCP, INT4, lossless quantisation; May — GPU support for Macs and VR headsets; Jul — aggressive optimisations to run directly on wearables. — [Cactus docs v1.9](https://docs.cactuscompute.com/v1.9/)
- Traction (vendor claim, Oct 2025): production deployments collectively serving 500,000+ weekly inference tasks. — [Hugging Face blog by Cactus co-founder](https://huggingface.co/blog/rshemet/cactus-on-device-inference)
- Also ships its own small models, e.g. "Whistle", a 16.9 MB on-device speech-to-text model (third-party review). — [Kompozy review](https://kompozy.io/reviews/whistle)
- **License is contested/unclear**: an HN commenter said Cactus changed "from Apache 2.0 to non commercial" two weeks before launch; team replied it "is still free for personal & small pro[jects]" and the wording was clarified. Third-party listing: free for individuals, students, non-profits, and orgs with both <$2M total funding and <$2M gross annual revenue; larger orgs need a commercial license from Cactus Compute, Inc. But the PyPI page for cactus-compute 4.1.0 (released 7 Nov 2026) lists Apache-2.0, and one Cactus comparison page says the engine is "free under the MIT license". — [HN thread](https://news.ycombinator.com/item?id=45291024); [HN reply](https://news.ycombinator.com/item?id=45291865); [nextautomation.us](https://nextautomation.us/compare/cactus-alternative); [PyPI](https://pypi.org/project/cactus-compute/4.1.0/); [Cactus compare page](https://cactuscompute.com/compare/best-open-source-on-device-ai). **Conflict unresolved — check the GitHub LICENSE.**
- Pricing: team stated pricing is per monthly active device regardless of utilization; no public dollar figures found. — [HN thread](https://news.ycombinator.com/item?id=45291024)
- Funding: reported backers include Y Combinator, Oxford Seed Fund and Google for Startups; no amount found. — [extruct.ai](https://www.extruct.ai/hub/cactuscompute-com/) (aggregator; unverified)

**Liquid LEAP**
- LEAP (Liquid Edge AI Platform) v0 launched July 2025 alongside the Apollo iOS app; pitched for deploying on phones, laptops, cars without cloud, "10 lines of code". — [edgeir.com](https://reimagine.edgeir.com/liquid-ai-unveils-leap-and-apollo-to-slash-cloud-dependence-in-edge-ai-deployments-20250721); [VentureBeat](https://venturebeat.com/ai/finally-a-dev-kit-for-designing-on-device-mobile-ai-apps-is-here-liquid-ais-leap)
- Aug 2025: LEAP expanded to laptops with AMD Ryzen / Ryzen AI acceleration. — [Liquid press release](https://www.liquid.ai/press/liquids-edge-ai-platform-leap-expands-support-to-laptops-with-best-in-class-performance-on-amd-ryzen-tm-and-ryzen-ai-tm-processors)
- By 2026 LEAP is described as an SDK "for fine-tuning and deploying LFMs across llama.cpp, MLX, ONNX, CoreML, SGLang, and vLLM runtimes" — i.e., a toolkit over third-party engines; a July 2026 profile describes it as a platform/SDK to deploy foundation models into iOS and Android apps. — [hokai.io profile](https://hokai.io/hub/companies/liquid-ai); [latent.space wiki](https://latent.space/wiki/liquid-ai-7ebae2f8/)
- No evidence found of an ExecuTorch backend in LEAP's current runtime list. — [hokai.io profile](https://hokai.io/hub/companies/liquid-ai) (absence of evidence; early LEAP v0 did use ExecuTorch per unsourced background)

**Nexa AI, RunAnywhere, others**
- Nexa SDK: open-source SDK running multimodal and text models across CPU, GPU and NPUs. — [aicoolies listing](https://aicoolies.com/tools/nexa-sdk)
- RunAnywhere (YC W2026): unified SDK for running multimodal models locally on iOS/Android; business model is "open-source SDK (distribution) + paid control plane (monetization)"; a reviewer rates its technical moat "strong, decaying" and says "the fleet management layer is where the contract lives". — [startuphub.ai](https://www.startuphub.ai/ai-news/claude-s-corner/2026/claudes-corner-runanywhere-yc-w2026) (AI-written analysis column; treat as commentary)
- Funding figures for Nexa (~$16.5M est.), RunAnywhere ($10k / $130k / $12M across aggregators) are inconsistent and unverified. — [startup-seeker](https://startup-seeker.com/company/nexa~ai); [startuphub.ai](https://www.startuphub.ai/startups/runanywhere); [everydev.ai](https://www.everydev.ai/developers/runanywhere-inc)
- Cactus publishes head-to-head marketing pages vs Nexa AI and "best Nexa alternative", indicating it treats Nexa as a direct competitor. — [Cactus vs Nexa](https://cactuscompute.com/compare/cactus-vs-nexa-ai)

### Inferences
- Comparison grid (synthesis; cells without a citation above are background knowledge):

  | Player | Layer | Owner/backing | Platforms | Models | Openness | Monetization |
  |---|---|---|---|---|---|---|
  | llama.cpp | runtime + GGUF format | community (ggml) | everything, CPU-first | any GGUF | MIT | none |
  | ExecuTorch | runtime | Meta | mobile/embedded, many NPU delegates | PyTorch exports | BSD | none (ecosystem) |
  | MLX | runtime/framework | Apple | Apple silicon only | MLX weights | MIT | none (hardware pull) |
  | LiteRT / LiteRT-LM / MediaPipe | runtime + task APIs | Google | Android, iOS, web, IoT | Gemma-first | Apache | none (ecosystem) |
  | ONNX Runtime | runtime | Microsoft | cross-platform, Windows/Copilot+ | ONNX | MIT | none (ecosystem) |
  | Core ML -> Core AI; Foundation Models | runtime + system model API | Apple | Apple only | BYO (Core AI) / AFM (FM) | closed (possibly partly opening) | device sales |
  | Ollama / LM Studio | desktop model manager over llama.cpp/MLX | startups | desktop | GGUF/MLX | open core | cloud/enterprise |
  | Nexa SDK | runtime + SDK, NPU focus | startup | mobile/PC | multi | open SDK | enterprise |
  | RunAnywhere | SDK + control plane | YC W26 startup | iOS/Android | multi (wraps runtimes) | OSS SDK | paid control plane |
  | **Cactus** | own kernels + engine + SDKs + hybrid router | YC S25 startup | iOS, Android, Mac, wearables, Pi | LFM, Gemma, Qwen, Whisper-class etc. | source-available / thresholded (disputed) | per-MAD licensing, cloud fallback |
  | **Liquid LEAP** | model catalog + fine-tune + deploy toolkit over third-party runtimes | Liquid AI | iOS, Android, laptops | LFM-centric | SDK free; models thresholded | sells models/enterprise deals |

- Cactus's distinguishing choices vs peers: (a) own kernels rather than wrapping llama.cpp (early versions did wrap llama.cpp — unsourced background), (b) cross-platform app frameworks (React Native/Flutter/KMP) as first-class, (c) built-in confidence-based cloud fallback, (d) low-RAM/budget-device focus.
- The biggest structural threat to all independent runtime startups is OS vendors absorbing the layer (Apple Foundation Models + Core AI, Google AICore/LiteRT-LM) — the OS-bundled model costs the developer zero bytes of download and zero licensing.

### Gaps
- No independent benchmark including Cactus vs ExecuTorch/LiteRT-LM/llama.cpp/MLX on identical hardware was found; Cactus numbers are vendor-reported.
- Cactus GitHub stars, team size, founders' bios (names seen in snippets: Henry Ndubuaku as co-founder; HF blog author handle "rshemet") and funding amount were not verifiable (YC page fetch blocked).
- LEAP's current (late-2026) feature set — e.g., whether it includes a model-bundling/OTA service, any pricing — could not be read from Liquid's docs (fetch blocked).
- MLC LLM, Ollama, ONNX Runtime GenAI, MediaPipe: no sourced 2026 facts gathered.

## 4. Direct relationship between Liquid AI and Cactus

### Takeaway
They are publicly **partners and complements**: Liquid names Cactus as an inference/distribution partner and credits Cactus benchmarks in its launch posts; Cactus natively supports LFM2/2.5 and republishes LFM weights. There is mild overlap/competition at the developer-SDK layer (LEAP vs Cactus SDK) and Cactus runs SEO pages such as "best Liquid AI alternative" — but both sides' own words frame it as runtime (Cactus) + models (Liquid).

### Cited Findings
- Liquid's LFM2-24B-A2B launch post lists Cactus Compute among inference and distribution partners "to deploy across mobile, desktop, and terminal"; Cactus co-founder Henry Ndubuaku is quoted saying the Cactus engine fully supports the 24B MoE model, with 229 tok/s prefill and 27 tok/s decode on M4 Pro CPU at INT8 (1k prefill / 100 decode tokens; vendor-reported; figures surfaced from the Japanese version of the post). — [Liquid AI blog (EN)](https://www.liquid.ai/en/blog/lfm2-24b-a2b-from-cloud-to-ai-pc); [Liquid AI blog (JA)](https://www.liquid.ai/ja/blog/lfm2-24b-a2b-from-cloud-to-ai-pc)
- Cactus published its own posts on that launch. — [Cactus blog](https://cactuscompute.com/blog/lfm2-24b-a2b)
- Liquid's LFM2.5-350M materials note that Cactus Compute "showcased the LFM2.5 350M running exceptionally fast on sub-$300 devices like the iPhone 13 Mini, Google Pixel 6a, and Raspberry Pi 5". — [Liquid AI LFM page](https://www.liquid.ai/lfm); [Liquid AI blog](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind)
- Cactus's post title: "LFM-2.5-350m on Cactus: 140 tok/sec, Single Core, 355 MB"; INT8 model is 355 MB; runtime RAM 56-85 MB on Apple devices and 300-330 MB on Android; tested on seven devices from Vision Pro to Raspberry Pi 5 (vendor claims). — [Cactus blog](https://www.cactuscompute.com/blog/lfm2-5-350m)
- Cactus hosts converted LFM weights under its own Hugging Face org (e.g., Cactus-Compute/LFM2-700M, Cactus-Compute/LFM2-2.6B). — [HF: LFM2-700M](https://huggingface.co/Cactus-Compute/LFM2-700M); [HF: LFM2-2.6B](https://huggingface.co/Cactus-Compute/LFM2-2.6B)
- Cactus's comparison page is titled "Cactus vs Liquid AI: Inference Engine vs Efficient Model Provider" and states: "They serve complementary roles: Cactus provides the runtime, while Liquid AI provides the models"; "Cactus already supports Liquid AI's LFM2 models natively"; Cactus "can run Liquid AI's LFM models alongside other architectures like Gemma and Qwen through its unified API". — [Cactus compare page](https://cactuscompute.com/compare/cactus-vs-liquid-ai)
- A Cactus page characterizes Liquid as primarily offering macOS/Linux support through Python, "with mobile deployment requiring a third-party runtime" (vendor characterization of a partner/competitor; arguably understates LEAP's iOS/Android SDKs). — [Cactus: best Liquid AI alternative](https://cactuscompute.com/compare/best-liquid-ai-alternative)
- Search-surfaced guidance attributed to Liquid: for iOS/Android deployment of LFMs "you would use a mobile inference engine like Cactus" (source page not confirmed — surfaced in a search summary of Liquid/Cactus pages; treat as unverified attribution). — [Cactus compare page](https://cactuscompute.com/compare/cactus-vs-liquid-ai)
- Cactus also runs competitive-SEO pages: "Best Liquid AI Alternative in 2026: On-Device AI Inference Engines Compared" and "Liquid AI vs Nexa AI". — [Cactus](https://cactuscompute.com/compare/best-liquid-ai-alternative); [Cactus](https://cactuscompute.com/compare/liquid-ai-vs-nexa-ai)
- At the LFM3 launch (27 Oct 2026) the named partners were Ambarella, AMD, Arm, Intel/OpenVINO, NVIDIA, OpenRouter, Qualcomm, Vercel; **Cactus was not among partners surfaced** in coverage, and LFM3 ships with native llama.cpp, MLX and vLLM support. — [Liquid AI LFM3 blog](https://www.liquid.ai/blog/lfm3); [Unite.AI](https://www.unite.ai/liquid-ai-launches-lfm3-family-of-six-open-weight-2b-class-models/)

### Inferences
- **Complement (primary)**: Liquid needs fast mobile runtimes to make its small-model speed claims tangible on cheap phones; Cactus needs differentiated small models that make its engine look good. Each uses the other as proof: Liquid cites Cactus device benchmarks; Cactus markets LFM support.
- **Compete (secondary, at the SDK seam)**: a mobile developer choosing "how do I ship an LFM in my app" can pick LEAP's SDK (Liquid's own path, sitting on llama.cpp/MLX/CoreML/ONNX) or Cactus's SDK (own engine, multi-vendor models, cloud fallback). LEAP is LFM-centric; Cactus is model-agnostic — so Cactus also commoditizes Liquid by letting developers swap LFM for Gemma/Qwen behind one API.
- **Asymmetry**: Cactus depends on a supply of good open small models but not on Liquid specifically; Liquid depends on runtimes but not on Cactus specifically (LFM3's day-0 runtime list is llama.cpp/MLX/vLLM plus chip-vendor stacks). The relationship is a non-exclusive ecosystem partnership, not a contractual alliance (no evidence of equity, revenue share or exclusivity found).
- Possible signal to check: Cactus's absence from surfaced LFM3 launch-partner lists could mean (a) simply not surfaced in search snippets, (b) Cactus support lagging a new architecture, or (c) Liquid emphasizing silicon vendors and its own LEAP path. Unresolved.
- License stacking matters for joint customers: an app shipping LFM on Cactus faces two thresholded licenses (Liquid: $10M revenue; Cactus: reportedly $2M funding/revenue) — a combined cost consideration versus e.g. Gemma/Qwen on llama.cpp (free/free).

### Gaps
- No formal partnership press release, commercial agreement, or investment tie between the two was found.
- Could not confirm whether Cactus supports LFM3 as of Nov 2026 (no result; Cactus docs fetch blocked).
- Could not read the full Cactus-vs-Liquid page or Liquid's docs to verify exact wording/dates of quotes; the "use a mobile inference engine like Cactus" attribution to Liquid is unverified.
- No independent third-party (HN / r/LocalLLaMA) discussion comparing LEAP and Cactus was retrieved.

## 5. Market size/trends; drivers; obstacles; hybrid local+cloud routing

### Takeaway
Market-research estimates vary by >5x depending on definition (narrow "on-device AI" ~$11-18B in 2025 growing ~26-28% CAGR; broader "edge AI"/"on-device intelligence" $25-66B in 2026), so cite ranges not point figures. Practitioner consensus in 2026 is that **hybrid local+cloud is the default architecture**, with memory, thermals/battery, NPU/operator fragmentation and model-update logistics as the main obstacles.

### Cited Findings
**Market size (all from paid-report vendors; methodologies unverified)**
- Spherical Insights: on-device AI $11.01B in 2025 -> $128.57B by 2035, 27.86% CAGR (2026-2035). — [Spherical Insights](https://www.sphericalinsights.com/blogs/top-10-companies-in-global-on-device-ai-market-2026-2035-spherical-insights-analysis)
- SNS Insider: $17.61B in 2025 -> $115.74B by 2033, 26.57% CAGR. — [SNS Insider](https://www.snsinsider.com/reports/on-device-ai-market-8740)
- Another estimate: $10.76B in 2025 -> $75.5B by 2033. — [press release via Ironton Tribune](https://smb.irontontribune.com/article/On-Device-AI-Market-Set-to-Surpass-USD-755-Billion-by-2033-as-Demand-for-Real-Time-Intelligence-and-Privacy-First-Computing-Accelerates/6a32e629564d536c69a5223a)
- Ken Research: 26.23% CAGR 2026-2031 to $71.25B by 2031. — [Ken Research](https://www.kenresearch.com/industry-reports/global-on-device-ai-market.md)
- KBV Research: $54.79B by 2032 at 27.1% CAGR. — [KBV Research](https://www.kbvresearch.com/press-release/on-device-ai-market)
- Broader "on-device intelligence": $66.3B in 2026 -> $327.5B by 2034, 22.1% CAGR. — [Research and Markets](https://www.researchandmarkets.com/reports/6243836/on-device-intelligence-market-outlook-market)
- Roots Analysis "edge AI": $25.84B in 2026 -> $245.18B by 2040, 17.44% CAGR. — [Roots Analysis](https://www.rootsanalysis.com/edge-ai-market)
- Commentary that on-device LLMs are "challenging cloud-only AI economics". — [DataM Intelligence](https://www.datamintelligence.com/blogs/edge-ai-on-device-llms-cloud-economics)

**Drivers**
- Practitioner guides frame the decision as a latency-privacy-cost triangle ("golden triangle") for where to run inference. — [tianpan.co, 10 Apr 2026](https://tianpan.co/blog/2026-04-10-hybrid-cloud-edge-llm-inference-when-to-run-on-device)
- Cost: one article claims local models handle ~85% of traffic in production hybrid setups, cutting cloud spend 60-80%; another claims only ~3% of tokens need a large cloud model to reach equivalent quality (both unverified author claims). — [tianpan.co](https://tianpan.co/blog/2026-04-10-hybrid-cloud-edge-llm-inference-when-to-run-on-device)
- Vendor-side framing: LEAP/Apollo pitched to "slash cloud dependence"; Cactus v1 pitched as "zero latency and full privacy". — [edgeir.com](https://reimagine.edgeir.com/liquid-ai-unveils-leap-and-apollo-to-slash-cloud-dependence-in-edge-ai-deployments-20250721); [InfoQ](https://www.infoq.com/news/2025/12/cactus-on-device-inference/)
- Demand signals from enterprises: Shopify signed a multi-year, "multi-million dollar" LFM licensing deal (Nov 2025, per Liquid); Liquid lists an April 2026 Mercedes-Benz partnership. — [remio.ai](https://www.remio.ai/post/liquid-ai-bets-on-edge-models-and-customer-controlled-inference); [Liquid company news](https://www.liquid.ai/news/company-news)

**Obstacles**
- Memory: token generation is memory-bandwidth-bound; KV cache grows with context; mobile OS low-memory killers terminate apps using too much memory during LLM tasks. — [tianpan.co, 17 Apr 2026](https://tianpan.co/blog/2026-04-17-on-device-llm-inference-edge-ai-production)
- NPU/hardware fragmentation: "NPU compatibility varies by chipset. Each manufacturer implements a different set of operators"; older devices lack fast NPUs, forcing capability-based fallbacks. — [callsphere.ai](https://callsphere.ai/blog/edge-ai-on-device-llms-qualcomm-apple-google-2026); [petronellatech](https://petronellatech.com/blog/?p=24486)
- Battery/thermals: sustained inference causes thermal throttling; users on the same model version get inconsistent behavior due to throttling, memory pressure, NPU availability. — [tianpan.co](https://tianpan.co/blog/2026-04-19-on-device-llm-inference-model-update-propagation)
- Model update propagation/version fragmentation called "the central operational challenge of on-device AI". — [tianpan.co](https://tianpan.co/blog/2026-04-19-on-device-llm-inference-model-update-propagation)
- NPU access in open runtimes is still immature (LiteRT-LM NPU acceleration "early access" in 2026). — [shiporskip.io](https://shiporskip.io/compare/litert-lm-google-ai-edge-on-device-llm-inference-framework-vs-meta-llama-4-scout-maverick-quantized-on-device-inference)

**Hybrid local+cloud routing trend**
- Routing is the make-or-break component; routers in practice are small classifiers/heuristics, with escalation on low confidence or constrained compute (thermal, battery). — [tianpan.co](https://tianpan.co/blog/2026-04-10-hybrid-cloud-edge-llm-inference-when-to-run-on-device)
- Platform owners went hybrid in 2026: Apple's Foundation Models framework reportedly spans on-device, Private Cloud Compute and third-party cloud models in one Swift API ("Apple's Foundation Models Become a Hybrid AI Platform"). — [appbot.co](https://appbot.co/blog/apple-wwdc-2026-ai-foundation-model-update/)
- Cactus repositioned from pure on-device to "hybrid AI inference engine", adding hybrid inference with GCP in Feb 2026 and confidence-based routing. — [Cactus docs v1.9](https://docs.cactuscompute.com/v1.9/)
- Liquid's LFM2-24B-A2B "from cloud to AI PC" and LFM3's vLLM/OpenRouter availability show Liquid also serving cloud/customer-controlled inference, not only edge. — [Liquid AI blog](https://www.liquid.ai/blog/lfm2-24b-a2b-from-cloud-to-ai-pc); [remio.ai](https://www.remio.ai/post/liquid-ai-bets-on-edge-models-and-customer-controlled-inference)

### Inferences
- Because market-size numbers are definitional (many include NPU silicon and device hardware, not software), none should be read as the revenue pool for model licensing or runtime SDKs; that software slice is not separately sized in any source found.
- Hybrid routing is where Cactus is trying to move monetization (cloud fallback + per-device pricing), and it is the layer Apple just bundled into the OS — both validation and threat.
- Liquid's obstacle list differs: its models attack the memory/latency constraints directly (smaller, faster), which is why it co-designs with silicon vendors.

### Gaps
- No tier-1 analyst figures (Gartner, IDC, Counterpoint, Canalys) on gen-AI smartphone/AI PC shipments were retrieved this session; report writer should not cite such numbers from these notes.
- No sourced data on the size of the on-device *software/SDK* market specifically.
- The 85% / 60-80% / 3% hybrid-routing statistics are single-author blog claims without traceable primary sources.

## 6. Business models: model licensing vs. infrastructure/SDK monetization; which is more defensible?

### Takeaway
Commentators lean toward "the model is not the moat" and see bare runtimes as only temporarily defensible; the durable value is argued to sit in orchestration/fleet management, integration depth, OEM/hardware relationships and enterprise contracts. On that view, Liquid's defensibility rests on OEM/enterprise licensing and hardware co-design (not weights), and Cactus's on becoming the control plane/routing layer (not kernels). No rigorous analyst comparison of the two models exists in what was found.

### Cited Findings
**Liquid AI (model licensing)**
- Raised a $250M Series A in Dec 2024 at roughly $2B-$2.4B valuation (most-cited figure; AMD-led per unsourced background). — [aibusinessweekly](https://aibusinessweekly.net/p/liquid-ai-250-million-series-a-foundation-models); [multiples.vc](https://multiples.vc/private-comps/liquid-ai)
- PitchBook reportedly lists an undisclosed early-stage deal dated Aug 2026, unconfirmed by company or major outlets. — [revenuememo](https://www.revenuememo.com/p/liquid-ai-funding) (unverified)
- Revenue model: enterprise engagements, custom model work and OEM licensing for on-device deployments rather than metered API; strategy is open weights to attract developers, then monetize enterprise deployment/custom models/OEM licensing. — [remio.ai](https://www.remio.ai/post/liquid-ai-bets-on-edge-models-and-customer-controlled-inference)
- License threshold ($10M revenue) "gives individual developers and smaller startups wide access while keeping an enterprise licensing business". — [FourWeekMBA](https://fourweekmba.com/ai-liquid-ai-opens-lfm3-free-to-firms-under-10m-in-revenue/)
- Only concrete disclosed deal: Shopify multi-year, "multi-million dollar" license (Nov 2025). No audited revenue; a "$250M ARR 2026" figure on FourWeekMBA is speculative, and other aggregator revenue figures are implausible. — [remio.ai](https://www.remio.ai/post/liquid-ai-bets-on-edge-models-and-customer-controlled-inference); [FourWeekMBA](https://fourweekmba.com/liquid-ais-2-35b-business-model-mit-scientists-built-ai-that-thinks-like-a-worm-and-its-1000x-more-efficient) (flag: speculative)

**Cactus (infrastructure/SDK)**
- Source-available engine free below thresholds (reported <$2M funding and <$2M revenue), commercial license above; pricing per monthly active device; optional cloud services (hybrid fallback). — [nextautomation.us](https://nextautomation.us/compare/cactus-alternative); [HN thread](https://news.ycombinator.com/item?id=45291024)
- The license switch away from Apache 2.0 drew criticism at launch on HN. — [HN thread](https://news.ycombinator.com/item?id=45291024)

**Commentary on defensibility**
- "The model was never the moat. It was the entry fee." — [Towards AI](https://pub.towardsai.net/the-model-was-never-the-moat-a4818cb76097)
- Flybridge (VC) on edge AI: researchers they spoke with think compression methods are "close to commoditized". — [Flybridge](https://www.flybridge.com/ideas/the-bow/opportunities-in-edge-ai-infrastructure-and-applications)
- On an on-device SDK startup (RunAnywhere): technical moat "strong, decaying"; model is OSS SDK for distribution + paid control plane; "the fleet management layer is where the contract lives". — [startuphub.ai](https://www.startuphub.ai/ai-news/claude-s-corner/2026/claudes-corner-runanywhere-yc-w2026)
- Edge AI business-model map: primary moat types are "integration depth, certification cycles, and workflow embedment". — [newmarketpitch](https://newmarketpitch.com/blogs/news/edge-ai-business-model) (methodology unclear)
- Orchestration (local-vs-cloud routing, context) proposed as the stack's value layer; counterpoint that cheap local inference may prevent platform-layer consolidation. — [liveinthefuture.org](https://liveinthefuture.org/stories/ambient-ai-smart-devices-platform-layer)
- Open-source serving engines erode vendor lock-in ("vLLM and SGLang reduce vendor moat"). — [Sacra](https://sacra.com/chat/h/4146308f-f545-461d-af7b-b3ee8132b686/)
- Additional thesis pieces (not read in full): "AI's Next Battleground: The Edge"; "The Internet of Intelligence: A Thesis and Playbook for Edge AI". — [dslevy substack](https://dslevy.substack.com/p/ais-next-battleground-the-edge); [ondeviceguy substack via reader](https://hn.jakechampion.name/reader/https://ondeviceguy.substack.com/p/the-internet-of-intelligence-a-thesis)

### Inferences
- **Model licensing (Liquid)** — strengths: large enterprise/OEM contract sizes (Shopify, Mercedes-Benz), capital ($250M) to keep training, architecture + hardware-in-the-loop IP that is hard to replicate, silicon-vendor co-marketing. Weaknesses: competes with free Apache/MIT models from Google, Microsoft, Alibaba, HF subsidized by other businesses; open weights + threshold license is hard to enforce; must keep winning the efficiency frontier every release cycle; OS-bundled models (Apple AFM, Gemini Nano) cap the consumer-app opportunity.
- **Infra/SDK (Cactus)** — strengths: model-agnostic (benefits from every new small model, including Liquid's), sticky once embedded in shipped apps, recurring per-device pricing, hybrid routing creates a cloud revenue line. Weaknesses: free big-tech/community runtimes (llama.cpp, ExecuTorch, LiteRT-LM, MLX, Core AI); OS vendors absorbing runtime + hybrid routing; source-available licensing friction with developers; low thresholds ($2M) may push startups to free alternatives; several near-identical YC-backed competitors (RunAnywhere, Nexa).
- Both companies have converged on the same commercial pattern: **free below a size threshold, paid enterprise license above** — Liquid on weights ($10M revenue), Cactus on engine (~$2M funding/revenue). Both are "open-ish core" plays reliant on enterprise conversion.
- Scale asymmetry: Liquid is a ~$2B+ valued, $250M-funded model lab with named enterprise customers; Cactus is a seed-stage YC S25 company. Liquid can afford to treat runtimes as partners/commodities; Cactus's risk is that model vendors (LEAP) and OS vendors both squeeze the SDK layer.
- Commentator consensus (weak evidence, mostly blogs): neither weights nor kernels alone are durable; defensibility accrues to whoever owns the customer relationship + deployment control plane + hardware integration. Liquid is pursuing that via OEM/enterprise deals and LEAP; Cactus via hybrid routing and per-device contracts.

### Gaps
- No tier-1 analyst or named-VC piece directly comparing model-licensing vs SDK monetization in on-device AI was found; the defensibility section rests on blogs, a VC essay (Flybridge) and an AI-written startup column.
- No verified revenue for either company; no verified Cactus funding amount, customer names or price points.
- Liquid's above-threshold license pricing and whether LEAP itself is monetized are unknown.
