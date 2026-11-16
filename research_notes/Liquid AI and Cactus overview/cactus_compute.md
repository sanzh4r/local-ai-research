# Cactus (Cactus Compute, Inc.) — on-device AI startup, YC S25 — notes as of 16 Nov 2026

**Research-method caveat (read first).** This session's network sandbox blocked direct fetches of cactuscompute.com, docs.cactuscompute.com, ycombinator.com, news.ycombinator.com, huggingface.co, infoq.com, marktechpost.com, raspberrypi.com and most other hosts. Only github.com could be fetched directly (repo landing page + README, read 16 Nov 2026). Everything else below comes from web-search result summaries of the cited pages, not from reading the full pages. Items are marked **[direct]** (page read directly), **[search]** (search-engine summary of the cited page), **[company claim]**, **[independent]**, or **[unverified]**. The GitHub API was also unavailable, so release tags, the LICENSE file history and exact star history could not be checked.

**Disambiguation.** Not to be confused with (a) "Cactus" / oncactus.com, a separate YC company (AI copilot for home-service businesses) that raised a $7M seed led by Wellington Management in Nov 2025; (b) Cactus Communications; (c) Cactus AI home security; (d) CACTUS content-management / real-estate underwriting software. Funding databases frequently mix (a) with Cactus Compute — [pulse2](https://pulse2.com/cactus-7-million-seed-funding-raised-to-launch-ai-copilot-for-home-service-businesses), [YC oncactus page](https://www.ycombinator.com/companies/oncactus).

**Headline finding.** Cactus has pivoted substantially during 2026. In 2025 it was an inference-infrastructure company (cross-platform engine/SDK for running third-party open models on phones, first as a llama.cpp wrapper, then its own "Kernel/Graph/Engine" stack). From May 2026 onward it has shipped its own tiny foundation models (Needle, Whistle, Mirage) plus its own quantization format (CQ / "Cactus Quants"), and by Nov 2026 its YC tagline is "Automation Foundation Models For Tiny Devices" and its website title is "Sub-30MB Foundation Models for Tiny Devices". The general engine has been demoted to a secondary package (`cactus-engine`).

## 1. Company basics: founders, founding date, YC batch, funding, team, location

### Takeaway
Cactus Compute, Inc. was founded in 2025 by Roman Shemet (CEO) and Henry Ndubuaku (CTO), went through YC Summer 2025, is listed in San Francisco with ~8 employees, and has raised only a small disclosed amount (roughly $0.5M–$1M per directories; no primary-source round announcement found).

### Cited Findings
- YC profile lists Cactus Compute as YC Summer 2025, founded 2025, 8 employees, San Francisco; tagline "Automation Foundation Models For Tiny Devices"; description "On-device AI for phones, wearables, smart home and small robots". (Listing data reported as ~490 days old in one search summary, so headcount may be stale.) **[search]** — [YC company page](https://www.ycombinator.com/companies/cactus)
- An earlier version of the same YC page carried the tagline "Low-latency AI engine for mobile devices & wearables" — evidence of repositioning. **[search]** — [translated YC page cache](https://www-ycombinator-com.translate.goog/companies/cactus?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc)
- Roman Shemet is CEO; YC bio: former quant & economist with product and data-engineering background. Henry Ndubuaku is CTO; YC bio: "sidestepped an AI/ML role at Nvidia to work on Cactus". **[search]** — [YC company page](https://www.ycombinator.com/companies/cactus)
- The founders reportedly met through YC's co-founder matching programme in London. **[search, secondary profile, unverified]** — [Yespress](https://yespress.io/cactus-yc-s25)
- Funding: EveryDev.ai lists $1M total, including a $500K YC seed dated 2025-08-01; Extruct lists $500K–$1M; Caplight lists a seed round dated 2 Sep 2025 backed by YC with no amount. **[search, directories, unverified]** — [EveryDev](https://www.everydev.ai/developers/cactus-compute), [Extruct](https://www.extruct.ai/hub/cactuscompute-com/), [Caplight](https://www.caplight.com/company/cactuscompute)
- Cactus's own docs say it received funding "from YC & Oxford" and that in Sep 2025 it "Launched Cactus Kernel, Graph & Engine, raised more funding"; another directory names Oxford Seed Fund and Google for Startups as backers. **[search; company claim / directory]** — [Cactus docs v1.9](https://docs.cactuscompute.com/v1.9/), [Extruct](https://www.extruct.ai/hub/cactuscompute-com/)
- Conflicting/unreliable: Yespress mentions an "$8M seed in February 2025" and a headcount range of ~8–27; this conflicts with every other source and is likely contamination from another company. Reforgers lists team size 9; Crunchbase 1–10. **[unverified]** — [Yespress](https://yespress.io/cactus-yc-s25), [Reforgers](https://reforgers.com/startups/cactus), [Crunchbase](https://www.crunchbase.com/organization/cactus-4616)
- A "Cactus Seed Deck" page at seed.cactuscompute.com describes a different concept: a pay-per-runtime compute marketplace running on idle mobile devices at a 25–50% discount to cloud, acquiring compute suppliers via telecom partnerships. Undated. **[search]** — [seed.cactuscompute.com](https://seed.cactuscompute.com/)

### Inferences
- The seed-deck page plausibly reflects the original pre-YC idea (which would also explain the "Cactus Compute" name); the company has repositioned at least twice: distributed phone compute → mobile inference engine (2025) → tiny foundation models (2026). The first step is my inference, not something a source states.
- With ~8 people and ~$1M disclosed, Cactus is a very small company relative to Liquid AI; any larger round is undisclosed or not found.

### Gaps
- No primary-source announcement of a priced seed round, valuation or lead investor. No TechCrunch-type coverage found.
- Exact incorporation date, London vs. SF history, and current headcount not confirmed (YC page could not be opened directly).

## 2. Product in the 2025–early 2026 era: the Cactus inference engine and SDKs (infrastructure)

### Takeaway
Through early 2026 Cactus was infrastructure, not models: a cross-platform on-device inference SDK that began as a llama.cpp-based wrapper and was rewritten in Sep 2025 into its own ARM-focused stack (Cactus Kernels, Cactus Graph, Cactus Engine) with a proprietary model format, OpenAI-compatible APIs, cloud handoff and NPU support that was clearest on Apple.

### Cited Findings
- Launch HN (thread id 45291024, about 14 months before Nov 2026, i.e. mid-Sep 2025): "Cactus (YC S25) – AI inference on smartphones"; supports "any LLM or VLM available on HuggingFace"; ships React Native, Flutter and Kotlin bindings. **[search; company claim]** — [Launch HN](https://news.ycombinator.com/item?id=45291024), [YC launch post](https://www.ycombinator.com/launches/Nwp-cactus-deploy-ai-models-locally-on-smartphones)
- Architecture of the v1 rewrite: three layers. Cactus Kernels = hand-written ARM NEON SIMD kernels (matmul, attention, convolution, quantization, DSP, image processing); Cactus Graph = zero-copy computation graph, described as "PyTorch-like for mobile"; Cactus Engine = chat/speech/vision/RAG layer with OpenAI-compatible APIs. Docs give Sep 2025 as the v1 launch. **[search]** — [Cactus docs v1.12](https://docs.cactuscompute.com/v1.12/), [third-party explainer](https://han-oqo.github.io/study/cactus-on-device-engine.html)
- InfoQ (Dec 2025), "Cactus v1: Cross-Platform LLM Inference on Mobile with Zero Latency and Full Privacy": reports sub-50ms time-to-first-token, and that it is free for students, educators, non-profits and small businesses. **[search; trade press relaying company claims]** — [InfoQ](https://www.infoq.com/news/2025/12/cactus-on-device-inference/)
- Company site (engine page): runs on iOS, Android, macOS and wearables from a single SDK. **[search; company claim]** — [cactuscompute.com/engine](https://cactuscompute.com/engine)
- Cloud handoff / hybrid: the docs list cloud handoff as an engine feature and a third-party summary calls it a "hybrid edge-cloud inference engine"; Cactus's own comparison page says cloud fallback is billed separately when enabled. **[search]** — [Cactus docs v1.12](https://docs.cactuscompute.com/v1.12/), [Cactus vs Nexa AI](https://cactuscompute.com/compare/cactus-vs-nexa-ai)
- NPU: at Launch HN a founder said the engine was optimized for mobile CPU and NPU hybrid kernels were still in internal testing. Later docs list Apple NPU support for several Whisper and Parakeet variants. A Dev.to competitor comparison (written by a NobodyWho author) says Qualcomm, MediaTek and Exynos NPU support remained on the roadmap; a Mintlify-hosted doc mirror claims 5–11x speedups on Apple, Qualcomm and MediaTek NPUs. Sources conflict; non-Apple NPU support is **[unverified]**. — [HN comment](https://news.ycombinator.com/item?id=45300133), [Cactus docs v1.11](https://docs.cactuscompute.com/v1.11/), [Dev.to NobodyWho vs Cactus](https://dev.to/pielounw/nobodywho-vs-cactus-compared-on-engine-design-model-format-hardware-platforms-cloud-and-1518), [Mintlify mirror](https://mintlify.wiki/cactus-compute/cactus)
- GitHub repo description as indexed earlier in 2026: "Quantization, kernels, runtime and inference engine for mobiles, wearables, smart home and robots." **[search]** — [GitHub tree/main (indexed)](https://github.com/cactus-compute/cactus/tree/main)
- Docs versions v1.6 through v1.14 are indexed, indicating steady engine releases through 2026. **[search]** — [docs v1.14](https://docs.cactuscompute.com/v1.14/)
- As of 16 Nov 2026 the README states: "The earlier general inference SDK is now `cactus-engine`" (PyPI package). The main repo no longer documents the engine, SDK bindings or benchmarks. **[direct]** — [GitHub README](https://github.com/cactus-compute/cactus/blob/main/README.md)

### Inferences
- The original (pre-Sep 2025) product being a llama.cpp wrapper is consistent with the Hugging Face blog by Roman Shemet and the HN discussion, but I could not open either page; treat the specific "llama.cpp-based v0 → own engine v1" description as well-attested by the brief and secondary sources but not directly verified here.
- The engine is now a supporting asset (it is the C++ runtime the tiny models execute on) rather than the headline product.

### Gaps
- Could not confirm the current Swift / C++ / Kotlin / Flutter / React Native SDK status, where the `cactus-engine` source now lives (github.com/cactus-compute/cactus-engine returned 404 on 16 Nov 2026), or whether the mobile SDKs are still maintained.
- No detail found on the telemetry dashboard of the 2025 commercial product beyond the Launch HN statement that the commercial licence included cloud telemetry.

## 3. Product as of Nov 2026: Cactus's own tiny models (Needle, Whistle, Mirage) and Cactus Quants

### Takeaway
Yes — as of 2026 Cactus builds its own models. It ships three sub-30 MB task-specific foundation models (Needle for tool calling, Whistle for speech-to-text, Mirage for vision), all running on its own C++ engine and delivered through one Python package, plus an extreme-low-bit quantization format (CQ) that it has also applied to a 120B Nvidia Nemotron model for Macs.

### Cited Findings
- Repo landing page (16 Nov 2026): "Sub-50mb agentic, voice and vision intelligence for consumer devices"; one Python package (`pip install -U cactus-compute`, Python 3.9+) providing tool calling, structured extraction, transcription, image understanding and fine-tuning; inference runs locally; engine and weights download on first use. **[direct]** — [GitHub repo](https://github.com/cactus-compute/cactus)
- Website title (Nov 2026): "Cactus - Sub-30MB Foundation Models for Tiny Devices". **[search]** — [cactuscompute.com](https://cactuscompute.com/)
- **Needle v1 (mid-May 2026; one source dates it 18 May 2026):** 26M-parameter single-shot function-calling model distilled from Gemini (teacher reported as Gemini 3.1 Flash Lite by some sources); synthetic training data across 15 categories (timers, messaging, navigation, smart home, etc.); 14 MB at INT4; MIT licence; claimed 6,000 tok/s prefill and 1,200 tok/s decode on consumer devices. **[search; company claims via secondary blogs]** — [NYU Shanghai RITS](https://rits.shanghai.nyu.edu/ai/cactus-releases-needle-a-26m-distilled-model-for-on-device-tool-calling/), [andrew.ooo review](https://andrew.ooo/posts/needle-26m-function-calling-model-review/)
- **Needle 2 (Aug 2026; MarkTechPost article dated 13 Aug 2026):** 45M parameters, shipped as a single 14 MB binary with model and engine together and "CQ2" 2-bit weights baked in; a full session runs in 28 MB RAM; arXiv paper 2607.18363. Company-reported: 800+ tok/s overall and 500+ tok/s prefill on Raspberry Pi 5; 63.7% on "Mobile Actions"; "trades wins" with FunctionGemma 270M, LFM2.5 230M and Apple FM. Licence reported as Apache-2.0 by most sources, MIT by one. **[search; company claims]** — [Cactus blog](https://cactuscompute.com/blog/needle-2), [MarkTechPost](https://www.marktechpost.com/2026/08/13/cactus-compute-needle-2-45m-parameter-tool-calling-model/), [mer.vin](https://mer.vin/news/needle-2-cactus-14mb-on-device-tool-calling-model/)
- Needle 2 limitations noted by a third-party reviewer: no free-text fallback, hard 256-token context budget, not a general chatbot — it only maps instruction + tool schemas to a JSON function call. **[search; independent review]** — [Mi&Bee blog](https://blog.mickeyzzc.tech/en/posts/aihelper/needle2-on-device-tool-calling/)
- **Needle 3 (17 Sep 2026):** 8–29 MB depending on how many layers are deployed; does tool calling, structured extraction into typed schemas and text embeddings; the 121M-parameter variant scores 86.0% on Cactus's own Mobile Actions benchmark vs 88.4% for DeepSeek V4 Flash. Fine-tunable. **[search; company claims]** — [Better Stack guide](https://betterstack.com/community/guides/ai/needle3-edge/)
- **Whistle (2 Oct 2026):** 16.9 MB speech-to-text model, CPU-only, same C++ engine as Needle, up to 30-second clips, seven languages, word timestamps and speech embeddings. **[search + direct README for feature list]** — [RuntimeWire](https://runtimewire.com/article/cactus-whistle-16-9mb-local-speech-model), [GitHub README](https://github.com/cactus-compute/cactus/blob/main/README.md)
- **Mirage (5–6 Nov 2026):** 22 MB vision model covering object detection, OCR, segmentation, "decision" and embedding; company claims 74 ms on Samsung Galaxy S26 NPU or 107 ms on its CPU. README describes it as image description + structured extraction, hosted at Hugging Face `Cactus-Compute/mirage`. **[search; company claim] / [direct]** — [Cactus on X](https://x.com/cactuscompute), [AI Adoption Agency](https://aiadoptionagency.com/cactus-compute-mirage/), [GitHub README](https://github.com/cactus-compute/cactus/blob/main/README.md)
- **Cactus Quants (CQ) on large models (12 Nov 2026):** Cactus says it compressed Nvidia Nemotron-3.5-SuperVL-120B (120B-A12B) to 1.418 effective bits per weight (computed over the whole file); runs on Apple silicon via `cactus-mlx` (macOS 26+, Python 3.12+), with a local OpenAI-compatible server. The blog says it is a custom checkpoint format, not a stock MLX-LM preset, and that the Mac result does not show it fits on an iPhone. **[search; company claim] + [direct README confirms cactus-mlx and the HF checkpoint name]** — [Cactus blog: Nemotron CQ](https://cactuscompute.com/blog/nemotron-cq), [HF model](https://huggingface.co/Cactus-Compute/nemotron-3.5-120b-cq), [GitHub README](https://github.com/cactus-compute/cactus/blob/main/README.md)
- Cactus's own follow-up post concedes a large quality cost: the Nemotron CQ release solved 7 "DeepSWE" tasks versus 24 for the original model. **[search; company-reported]** — [Cactus blog: quantization and multi-step workflows](https://cactuscompute.com/blog/quantization-and-multi-step-workflows)
- The `cactus-mlx` adapter "currently supports Nemotron-H CQ checkpoints" only; greedy generation uses "verified MTP" by default. **[direct]** — [GitHub README](https://github.com/cactus-compute/cactus/blob/main/README.md)

### Inferences
- Cactus is now both a model builder and an infrastructure builder, with the weight shifted to models: small proprietary-architecture, openly licensed, task-specific models (action/voice/vision), each co-designed with its runtime and quantization. This is a different bet from Liquid AI's general-purpose small LMs — Cactus's models are far smaller (tens of MB vs hundreds of MB to GB) and narrower (no free-form chat in Needle).
- Naming LFM2.5 230M as a benchmark rival for Needle 2 signals that Cactus now positions itself partly as a competitor to Liquid AI in the function-calling niche, not only as a runtime for Liquid's models.

### Gaps
- Could not read the Needle 2 arXiv paper or model cards directly; architecture details (encoder-decoder vs decoder-only, training recipe for Needle 2/3) not captured.
- Mirage accuracy benchmarks and licence not found. Whistle licence not explicitly confirmed (repo-level licence is Apache 2.0).

## 4. Third-party models supported/distributed (engine era) and Hugging Face org

### Takeaway
The engine supported a curated set of open third-party models converted to Cactus's own weight format — Gemma, Qwen, Liquid AI's LFM2/LFM2-VL, plus Whisper, Parakeet and Moonshine for speech — distributed via the Hugging Face `Cactus-Compute` org; Cactus did not train these.

### Cited Findings
- Docs list LLM/VLM support for Gemma (3, 3n, 4), Qwen (3, 3.5) and LiquidAI LFM2 and LFM2-VL; speech support for Whisper, Parakeet and Moonshine, with Apple NPU support for some Whisper/Parakeet variants. **[search]** — [Cactus docs v1.11](https://docs.cactuscompute.com/v1.11/), [DeepWiki](https://deepwiki.com/cactus-compute/cactus)
- LFM2 variants named in Cactus docs: LFM2-350M, LFM2-700M, LFM2-8B-A1B, LFM2-2.6B, plus LFM2-VL; a `Cactus-Compute/LFM2-700M` repo exists on Hugging Face. **[search]** — [Cactus docs v1.8](https://docs.cactuscompute.com/v1.8/), [HF LFM2-700M](https://huggingface.co/Cactus-Compute/LFM2-700M)
- Runtime binaries and supported-platform list are now hosted at Hugging Face `Cactus-Compute/cactus`. **[direct]** — [GitHub README](https://github.com/cactus-compute/cactus/blob/main/README.md)
- A competitor-authored comparison notes Cactus uses its own model format (rather than GGUF). **[search; competitor source]** — [Dev.to NobodyWho vs Cactus](https://dev.to/pielounw/nobodywho-vs-cactus-compared-on-engine-design-model-format-hardware-platforms-cloud-and-1518)
- FunctionGemma: Cactus co-ran a "Google DeepMind x Cactus Compute Global Hackathon" (San Francisco session 21 Feb 2026; described as six cities, one day, with AI Tinkerers) where teams ran FunctionGemma-270M on Cactus with Gemini cloud fallback and competed on local/cloud routing (`generate_hybrid()` in the starter repo `cactus-compute/functiongemma-hackathon`). **[search]** — [AI Tinkerers SF](https://sf.aitinkerers.org/p/google-deepmind-x-cactus-compute-global-hackathon-san-francisco), [DeepWiki starter repo](https://deepwiki.com/cactus-compute/functiongemma-hackathon), [iTWire](https://itwire.com/business-it-news/security/global-hackathon-%E2%80%98unites-london,-boston-and-san-francisco-to-build-the-future-of-personal-ai-on-the-edge%E2%80%99.html)

### Inferences
- The Feb 2026 FunctionGemma hackathon (tool-calling routing) directly precedes Needle (May 2026, distilled from Gemini); the hackathon work likely informed the decision to build their own tool-calling model. This sequencing is my inference.

### Gaps
- Full Hugging Face org listing (model count, download numbers, whether SmolLM/embedding models are still hosted) could not be retrieved — huggingface.co blocked.
- No evidence found of Cactus fine-tuning FunctionGemma itself for release.

## 5. Licensing, pricing and how they make money

### Takeaway
Monetization has been unstable: Apache 2.0 at first, a switch to a non-commercial/source-available licence around early Sep 2025 with paid commercial licences priced per monthly active device (plus separately billed cloud fallback), and — as of Nov 2026 — the main repo (models + prebuilt runtime) is back under Apache 2.0 with no public pricing. The current revenue model is unclear.

### Cited Findings
- On the Launch HN thread (Sep 2025) a commenter complained the project had changed licence "two weeks" earlier from Apache 2.0 to non-commercial and called it a loss of trust; another argued a paid tier does not fix that. Founders replied it is free for hobbyists and personal projects with a paid licence for commercial use, and later said the licence wording had been clarified. **[search of HN mirrors]** — [Launch HN](https://news.ycombinator.com/item?id=45291024), [HN mirror](https://remix-clone-hacker-news.flameddd1.workers.dev/item/45291024)
- Launch HN: paid licence priced "per monthly active device, regardless of utilization"; commercial licence included cloud telemetry (and, per a secondary profile, hardware acceleration in enterprise tiers). No per-device rate published. **[search; company claim]** — [Launch HN](https://news.ycombinator.com/item?id=45291024), [Yespress](https://yespress.io/cactus-yc-s25)
- Later engine licence terms (source-available): free for individuals, students, educational institutions, registered non-profits, and organisations with under $2M total funding and under $2M annual gross revenue; larger organisations need a commercial licence from Cactus Compute, Inc. **[search; third-party description of licence]** — [ai-tldr tool page](https://ai-tldr.dev/tools/cactus/), [InfoQ](https://www.infoq.com/news/2025/12/cactus-on-device-inference/)
- Cloud fallback is an optional paid cloud API billed separately. **[search; company page]** — [Cactus vs Nexa AI](https://cactuscompute.com/compare/cactus-vs-nexa-ai)
- As of 16 Nov 2026 the main repo README states the licence is Apache 2.0 and that this also covers the prebuilt runtime binaries. Needle v1 was reported as MIT. **[direct] / [search]** — [GitHub README](https://github.com/cactus-compute/cactus/blob/main/README.md), [andrew.ooo](https://andrew.ooo/posts/needle-26m-function-calling-model-review/)
- Telemetry: the Python package collects anonymous usage counts by default; opt out with `CACTUS_TELEMETRY=0`. **[direct]** — [GitHub README](https://github.com/cactus-compute/cactus/blob/main/README.md)
- A brand-data site reports no public pricing tiers detected on the website. **[search]** — [webclaw](https://webclaw.io/data/startups/company/cactuscompute.com)

### Inferences
- Licence history: Apache 2.0 → non-commercial (c. late Aug/early Sep 2025) → source-available with $2M threshold (by Dec 2025) → Apache 2.0 for the 2026 model-centric repo. The date of the return to Apache is not established; it could apply only to the new repo contents while `cactus-engine` retains the threshold licence.
- With the core artefacts Apache-licensed and no public pricing, revenue (if any) likely comes from enterprise/OEM licensing, custom fine-tuning or support, and cloud fallback — this is speculation; no source confirms current revenue streams or revenue figures.

### Gaps
- Current licence of `cactus-engine` and of the mobile SDKs; the LICENSE file and its git history could not be inspected.
- No revenue, ARR, or paying-customer disclosures found.

## 6. Performance claims and independent verification

### Takeaway
Nearly all performance numbers are company-reported; the few independent checks (a Raspberry Pi write-up, two third-party Whistle analyses, one hobbyist Apple-silicon benchmark) broadly confirm that the models run fast on small hardware but show the marketing comparisons are selective.

### Cited Findings
- Engine, Launch HN (Sep 2025): 16–20 tok/s on Pixel 6a / Galaxy S21 / iPhone 11 Pro; 50–70 tok/s on Pixel 9 / Galaxy S25 / iPhone 16 (model/quantization not captured in the summary). **[search; company claim]** — [Launch HN](https://news.ycombinator.com/item?id=45291024)
- Later secondary summary: 70+ tok/s on iPhone 17 and Galaxy S25 Ultra with NPU optimization. **[search; company claim via secondary]** — [Yespress](https://yespress.io/cactus-yc-s25)
- Sub-50 ms time-to-first-token (v1 engine). **[search; company claim relayed by InfoQ]** — [InfoQ](https://www.infoq.com/news/2025/12/cactus-on-device-inference/)
- Third-party claim of 0.2 s per image for a VLM on iPhone 17 Pro. **[search; unverified]** — [BrightCoding](https://www.blog.brightcoding.dev/2026/04/02/cactus-ai-engine-the-revolutionary-mobile-inference-framework)
- Needle 2 independent-ish check: Raspberry Pi's official news blog ran it on a Pi 5 CPU alone; one test took 78 ms to turn a typed command into an action. **[search; independent hands-on]** — [Raspberry Pi news](https://www.raspberrypi.com/news/turn-text-input-into-actions-with-needle-a-14mb-function-calling-llm/); see also [peppe8o benchmark guide](https://peppe8o.com/needle-2-raspberry-pi-benchmark-guide/)
- Whistle model card (self-reported WER): LibriSpeech clean 4.31, LibriSpeech other 10.49, TED-LIUM 7.61, FLEURS avg 21.4, AMI 26.07; 16.9 MB vs 145.3 MB for Whisper base; Whisper/Moonshine figures were taken from published papers rather than re-run. **[search; company claim]** — [HF Whistle README](https://huggingface.co/Cactus-Compute/whistle/blob/557a18bb60b49497a0a3a0125d3bb65dda6200bf/README.md)
- Independent critique: Whistle beats Whisper base on LibriSpeech clean/other (4.31 vs 4.9; 10.49 vs 11.0) but loses on TED-LIUM (7.61 vs 5.0) and AMI (26.07 vs 21.5) — "wins two of six comparable benchmarks, not most"; the FLEURS reference bar used for Whisper (24.5) is not supported by the Whisper paper (21.0 mean); speed comparison uses greedy fp32 PyTorch Whisper while the WER panel quotes 5-beam Whisper. **[search; independent]** — [ai.satyajit.si](https://ai.satyajit.si/articles/whistle-stt)
- Independent hands-on: 10.27% WER for Whistle vs 12.46% greedy Whisper base and 10.83% beam on a paired subset; 3.75% WER with silence-based splitting vs 6.74% with fixed windows. **[search; independent]** — [HAIQ](https://haiq.agency/en/cactus-whistle-vs-whisper-base-16-9-mb-cpu-wer-33-ms-streaming)
- Hobbyist Apple-silicon benchmark repo (iPhone 17 Pro, M4 Max): places Cactus second in one phone-class speed ranking, behind MLX-OptiQ and MLX-PTQ and ahead of Apple Core AI. Run by an individual, not a lab. **[search; independent]** — [john-rocky/apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench)
- Ultrametric "arena" (agent-readiness user stories, not raw speed): llama.cpp beat Cactus 26–19 with 41 draws. **[search; independent, methodology weak]** — [Ultrametric](https://ultrametric.ai/arena/local-llm-runtimes/battle/llama-cpp-vs-cactus-compute)

### Inferences
- No vendor-neutral benchmark was found comparing Cactus against llama.cpp, ExecuTorch, LiteRT, MLC, Nexa and RunAnywhere on the same model, quantization and device. Cactus's use of its own model format and its own "Mobile Actions" benchmark makes like-for-like comparison harder.

### Gaps
- RAM and energy figures for the engine era (company comparison tables) could not be retrieved; no independent energy measurements found.
- "Mobile Actions" benchmark provenance (whether it is Google's dataset or Cactus's variant) not confirmed.

## 7. Customers, adoption, partnerships (including Liquid AI)

### Takeaway
Adoption is developer-community-led (≈14.8k GitHub stars; big Hacker News launches) with few named production customers; the Liquid AI relationship is a real but informal ecosystem tie — Cactus is a named local-deployment route in Liquid launch posts and natively supports LFM2 — with no formal partnership announcement found, and no evidence of deals with Nothing or Oppo.

### Cited Findings
- GitHub (16 Nov 2026): 14.8k stars, 1.0k forks, 82 watchers, 5 open issues, 386 commits on main; Releases section empty. **[direct]** — [GitHub repo](https://github.com/cactus-compute/cactus)
- Launch HN (Sep 2025): AnythingLLM and KinAI named as production apps, together running 500k+ weekly inference tasks. **[search; company claim]** — [Launch HN](https://news.ycombinator.com/item?id=45291024)
- YC page: customer use cases include private RAG pipelines, offline fallback for big remote models, phone agents for gallery/calendar tasks, and medical/privacy-sensitive apps. **[search; company claim]** — [YC company page](https://www.ycombinator.com/companies/cactus)
- Third-party profile claims 3,000+ developers and 100+ integrated products. **[unverified]** — [Reforgers](https://reforgers.com/startups/cactus)
- Needle v1's HN post reportedly reached 557–583 points and ~157 comments (#3 on launch day) in May 2026 (sources disagree on the score). **[search; secondary]** — [the-agent-report](https://the-agent-report.com/2026/05/needle-gemini-tool-calling-distilled-26m-parameter-model/), [devpik](https://www.devpik.com/blog/needle-26m-tool-calling-model-complete-guide)
- **Liquid AI:** Liquid's LFM2-24B-A2B launch post ("From Cloud to AI PC … Alongside Our Growing Partner Ecosystem") lists Cactus as a local deployment route "across mobile, desktop, and terminal via Cactus, Ollama, LM Studio, and Nexa AI" and links a Cactus example, "How to build on-device coding agents with Cactus & LFM2-24B-A2B". **[search; Liquid primary source]** — [Liquid AI blog](https://www.liquid.ai/en/blog/lfm2-24b-a2b-from-cloud-to-ai-pc)
- Cactus published its own write-up on running LFM2-24B-A2B locally on a Mac (24B total parameters, ~2B active per token), calling it a step up from LFM2-8B-A1B. **[search; company]** — [Cactus blog](https://cactuscompute.com/blog/lfm2-24b-a2b)
- Cactus's comparison page "Cactus vs Liquid AI" says Cactus natively supports LFM2 and frames Liquid as relying on third-party runtimes like Cactus for on-device mobile deployment. **[search; company marketing framing — note Liquid has its own LEAP SDK, so this framing is contestable]** — [Cactus vs Liquid AI](https://cactuscompute.com/compare/cactus-vs-liquid-ai)
- Google DeepMind: co-branded global hackathon, Feb 2026 (see section 4). **[search]** — [AI Tinkerers showcase](https://sf.aitinkerers.org/hackathons/h_DRGnrtIWaG8/showcase)
- Raspberry Pi featured Needle on its official news blog. **[search]** — [Raspberry Pi news](https://www.raspberrypi.com/news/turn-text-input-into-actions-with-needle-a-14mb-function-calling-llm/)
- Nothing / Oppo: a targeted search found no evidence of any Cactus partnership with either. **[search; negative result]** — [search surfaced only unrelated Oppo profile](https://akta.pro/company-profiles/0000p4j/oppo)

### Inferences
- The Liquid relationship is best described as "launch-ecosystem partner / supported runtime", now complicated by Cactus benchmarking its own Needle against LFM2.5 230M — partner at the runtime layer, competitor at the tiny-tool-calling-model layer.
- Named production customers (AnythingLLM, KinAI) date from Sep 2025; no newer named commercial customers or OEM design wins were found.

### Gaps
- No download counts (PyPI, npm, pub.dev, HF) retrieved. No verified list of apps shipping Cactus in 2026.
- Any OEM relationships (Nothing, Oppo, Samsung, etc.) are unconfirmed; if the brief's premise comes from founder LinkedIn/X posts, those could not be accessed.

## 8. Positioning, competitors, and latest news through Nov 2026

### Takeaway
Cactus now positions itself as the maker of ultra-small "automation" foundation models for phones, wearables, smart-home devices and small robots, competing with both runtimes (llama.cpp, ExecuTorch, LiteRT, MLX, Nexa, RunAnywhere, NobodyWho) and tiny-model vendors (Google FunctionGemma, Liquid LFM2.5, Apple on-device FMs, Whisper/Moonshine).

### Cited Findings
- Timeline (all 2026 unless noted): Sep 2025 v1 Kernel/Graph/Engine + Launch HN → Dec 2025 InfoQ coverage of v1 → 21 Feb Google DeepMind x Cactus hackathon → Apr: engine docs/coverage continue (v1.x) → mid-May Needle v1 (26M) → Aug Needle 2 (45M, 14 MB, arXiv 2607.18363) → 17 Sep Needle 3 (8–29 MB) → 2 Oct Whistle (16.9 MB STT) → 5–6 Nov Mirage (22 MB vision) → 12 Nov Nemotron-3.5-SuperVL-120B at 1.418 bits/weight via Cactus Quants. **[search]** — [Launch HN](https://news.ycombinator.com/item?id=45291024), [InfoQ](https://www.infoq.com/news/2025/12/cactus-on-device-inference/), [AI Tinkerers](https://sf.aitinkerers.org/p/google-deepmind-x-cactus-compute-global-hackathon-san-francisco), [MarkTechPost](https://www.marktechpost.com/2026/08/13/cactus-compute-needle-2-45m-parameter-tool-calling-model/), [Better Stack](https://betterstack.com/community/guides/ai/needle3-edge/), [RuntimeWire Whistle](https://runtimewire.com/article/cactus-whistle-16-9mb-local-speech-model), [Cactus blog Nemotron CQ](https://cactuscompute.com/blog/nemotron-cq)
- At Launch HN the founders told desktop users to use Hugging Face, llama.cpp, Ollama, vLLM or MLX instead — Cactus targeted phones. By Nov 2026 it also ships a Mac MLX path. **[search] / [direct]** — [Launch HN](https://news.ycombinator.com/item?id=45291024), [GitHub README](https://github.com/cactus-compute/cactus/blob/main/README.md)
- Cactus maintains "Cactus vs X" comparison pages (Nexa AI, Liquid AI). **[search]** — [vs Nexa AI](https://cactuscompute.com/compare/cactus-vs-nexa-ai), [vs Liquid AI](https://cactuscompute.com/compare/cactus-vs-liquid-ai)
- Needle 2 is benchmarked by Cactus against FunctionGemma 270M, LFM2.5 230M and Apple FM; Needle 3 against DeepSeek V4 Flash; Whistle against Whisper base and Moonshine. **[search; company claims]** — [RuntimeWire Needle 2](https://runtimewire.com/article/cactus-needle-2-14mb-agent-model-tiny-devices), [Better Stack](https://betterstack.com/community/guides/ai/needle3-edge/), [HF Whistle README](https://huggingface.co/Cactus-Compute/whistle/blob/557a18bb60b49497a0a3a0125d3bb65dda6200bf/README.md)
- Target hardware for the new models is "tiny devices": Raspberry Pi-class boards, wearables, smart home, robots, budget phones. **[search]** — [Cactus blog Needle 2](https://cactuscompute.com/blog/needle-2), [YC company page](https://www.ycombinator.com/companies/cactus)

### Inferences
- Strategic read: a tiny team could not win a general mobile-runtime race against Google (LiteRT), Meta (ExecuTorch), Apple (MLX/Core AI) and llama.cpp, nor monetize a restrictive licence after community backlash; the pivot to permissively licensed, extremely small, co-designed model+runtime bundles gives it a distinctive niche (always-on tool calling / voice / vision in tens of MB). This is my interpretation.
- The Nemotron 120B CQ release suggests quantization technology ("Cactus Quants") may become a third product line beyond tiny models, though the company's own data shows heavy quality loss at 1.4 bits.

### Gaps
- No mainstream tech-press (TechCrunch etc.) coverage found; most 2026 coverage is small blogs/aggregators restating company posts.
- No information on funding, hiring or customer news in H2 2026.
