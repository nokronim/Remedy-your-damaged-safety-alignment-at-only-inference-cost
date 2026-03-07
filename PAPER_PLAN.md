# ACL Paper Plan: Inference-Time Safety Recovery for Fine-Tuned LLMs

## Working Title
**"Safety Alignment Is Model-Dependent: Inference-Time Recovery for Fine-Tuned Language Models"**

Alternative titles (shorter, punchier):
- "Hypernetwork-Generated Safety Heads for Fine-Tuned Language Models"
- "Zero-Shot Safety Recovery at Inference Time via Activation-Conditioned Hypernetworks"

---

## Core Narrative (The Story Arc)

**Problem:** Fine-tuning LLMs on benign downstream tasks degrades safety alignment. The open ecosystem produces thousands of fine-tuned checkpoints, and re-aligning each one is prohibitively expensive.

**Key Insight:** Safety is model-dependent — the same prompt may be safe on one fine-tuned model but unsafe on another. A fine-tuned model's intermediate activations encode a behavioral fingerprint that captures how its safety properties differ from the base model.

**Solution:** Train a hypernetwork that reads a fine-tuned model's activations and generates a lightweight side network ("safety head") that acts as a prompt-level safety classifier. At deployment, given any new fine-tuned checkpoint, we extract activations from a small calibration set and instantiate a model-specific safety head — zero-shot, no retraining needed.

**Result:** Across 4 holdout splits × 2 model families (Qwen2-7B, LLaMA-3-8B), generated safety heads recover >95% harmful detection while preserving domain task performance, generalizing to unseen fine-tuning domains.

---

## Section-by-Section Plan

### Abstract (~200 words)
- Problem: safety fragility under fine-tuning
- Gap: existing methods require per-model retraining
- Our approach: activation-conditioned hypernetwork → side network weights → inference-time safety classifier
- Key design: layer-wise factorized generation, direction-magnitude split, template-based corrections
- Results: across N unseen domains and 2 model architectures, recovers X% harmful detection while maintaining Y% task accuracy
- Significance: scalable, training-free safety assessment for the fine-tuned model ecosystem

### 1. Introduction (~1.5 pages)

**Para 1 — Context:** LLMs are safety-aligned via RLHF/DPO, but fine-tuning breaks this. Cite Qi et al. (2024), Fraser et al. (2025). Even benign task adaptation degrades refusal behavior.

**Para 2 — Existing Solutions & Limitations:**
- Post-hoc re-alignment (LISA, etc.): requires safety data + training per checkpoint
- External classifiers (Llama Guard, prompt shields): model-agnostic, miss model-dependent degradation
- Activation-based detection (HiddenDetect): per-model probes, don't transfer

**Para 3 — Our Insight:** Safety is model-dependent. Introduce the concept: same prompt, different safety status depending on the checkpoint. This motivates learning a *model-conditioned* safety mechanism.

**Para 4 — Our Approach:** Activation-conditioned hypernetwork generates a lightweight side network. Layer-wise factorized generation. Three-stage pipeline: (1) fine-tune, (2) train per-domain side networks, (3) train hypernetwork across domains. At test time: extract activations → generate safety head → classify prompts.

**Para 5 — Contributions:**
1. Model-conditioned safety formulation: safety classifier weights are a function of the fine-tuned model's activations
2. Layer-wise factorized hypernetwork with direction-magnitude activation processing and template-based correction generation
3. Multi-holdout evaluation across 2 architectures (Qwen2-7B, LLaMA-3-8B) and 4 dataset splits, demonstrating zero-shot generalization to unseen fine-tuning domains

**FIGURE 1 (page 1, top):** Overview figure showing the full pipeline. See [Figure Descriptions](#figures-and-diagrams) below.

### 2. Related Work (~1 page)

**2.1 Safety Alignment Fragility**
- Fine-tuning degrades safety: Qi et al. (2024), Qi et al. (2025) "tokens deep", Fraser et al. (2025)
- Harmful fine-tuning attacks: intentional poisoning vs. benign degradation
- LISA: lazy safety alignment via additional training (Huang et al., 2024)
- Key gap: all require per-model intervention

**2.2 Safety Classifiers and Guardrails**
- Prompt-level classifiers: Llama Guard, Aegis, WildGuard
- Adversarial prompt shields: Kim et al. (2024)
- Activation-based detection: HiddenDetect (Jiang et al., 2025), representation engineering
- Key gap: model-agnostic (ignore fine-tuning effects) or per-model (don't transfer)

**2.3 Hypernetworks for Weight Generation**
- Hypernetwork survey: Chauhan et al. (2024)
- Context-conditioned weight generation: HyperNetworks (Ha et al., 2017)
- Task-conditioned adapters: hypernetwork generates LoRA/adapter weights conditioned on task embeddings
- Key difference: we condition on model activations (not task descriptions), generating safety-specific weights

**2.4 Side Networks and Ladder Tuning**
- Ladder Side-Tuning (Sung et al., 2022): lightweight side network with ladder connections to frozen backbone
- Parameter-efficient fine-tuning: LoRA, adapters
- Our contribution: LST architecture repurposed for safety classification, with weights generated by hypernetwork

### 3. Methodology (~2.5 pages)

**3.1 Problem Formulation**
- Given: base model M₀ (safety-aligned), fine-tuned model M_d (domain d)
- Goal: for unseen M_d*, produce safety classifier f_θ(x | M_d*) that outputs safety score per prompt x
- Approach: learn generator G that maps model activations A(M_d*) to classifier weights θ

**3.2 Side Network Architecture (Safety Head)**
- Ladder Side-Tuning architecture:
  - Side transformer with K=12 layers, reduced hidden dim (TRF=4)
  - Downsample projection: backbone_hidden → side_hidden
  - Ladder connections: gated fusion μ·h_ladder + (1-μ)·h_side
  - Lambda classifier head: mean-pool side hidden states → MLP → scalar safety logit (λ)
- Lambda interpretation: λ≈0 for safe/domain prompts, λ≈1 for harmful prompts
- Training objective: selective SFT loss on harmful prompts + lambda classification loss on all prompts

**FIGURE 2:** Side network architecture diagram. See below.

**3.3 Activation Extraction**
- For each fine-tuned model M_d, extract activations from N_cal calibration prompts
- Activations: hidden states from each transformer layer (layer i → activation a_i)
- Calibration set: domain-specific prompts (not harmful prompts)
- Key: activations serve as a compact behavioral fingerprint of the fine-tuned model

**3.4 Hypernetwork Architecture (V15)**

**3.4.1 Layer-wise Factorized Generation**
- Naive approach: map all activations → all side network weights (high-dimensional, ill-conditioned)
- Our approach: predict each side network layer's weights independently from corresponding backbone layer activations
- For layer i: θ_i = H_i(a_i), where H_i is the hypernetwork sub-module for layer i
- Shared hypernetwork weights across layers with layer-specific adapters

**3.4.2 Direction-Magnitude Split**
- Activation processing: split activation deltas into unit direction d̂ and log-magnitude m
- Direction encoder: project to local_proj_dim, apply attention
- Magnitude encoder: MLP on log-magnitudes
- Fusion: concatenate direction and magnitude features → generate weight corrections
- Motivation: direction captures *what changed* during fine-tuning, magnitude captures *how much*

**3.4.3 Template-Based Correction Generation**
- Learn a shared weight template T (initialized as mean of ground-truth side networks across training domains)
- Hypernetwork generates low-rank corrections ΔW, not full weights
- Final weights: W = T + s·ΔW (s = learnable scale, initialized small)
- Benefit: reduces output complexity from generating ~237M params to generating small corrections

**3.4.4 Lambda Parameter Generation**
- Lambda classifier weights generated separately via LoRA-style delta prediction
- Freeze lambda normalization layers (use template norms)
- Predict only the functional components that affect lambda classification

**FIGURE 3:** Hypernetwork architecture diagram showing direction-magnitude split and template corrections. See below.

**3.5 Training**
- Training data: K domains (fine-tuned models) with ground-truth side networks
- Loss: L = L_recon + α·L_lambda + β·L_functional
  - L_recon: weight reconstruction loss (MSE between generated and ground-truth side network weights)
  - L_lambda: lambda classification accuracy on calibration prompts
  - L_functional: behavioral loss — generated side network should produce correct lambda on harmful probes
- Holdout evaluation: M domains held out during training, evaluated zero-shot
- Hyperparameters: 300-400 epochs, lr=5e-5, hidden_dim=512, rank=4

**3.6 Inference (Zero-Shot Safety Head Generation)**
- Given new fine-tuned model M_d*:
  1. Extract activations from N_cal=50 calibration prompts
  2. Feed activations through hypernetwork → generate side network weights
  3. Load generated side network alongside M_d*
  4. For each input prompt: compute lambda score; λ>0.5 → harmful (refuse), λ<0.5 → safe (proceed)
- No training, no safety data needed at deployment time

### 4. Experiments (~2 pages)

**4.1 Setup**

**Models:**
- Qwen2-7B-Instruct (3584 hidden, 28 layers)
- LLaMA-3-8B-Instruct (4096 hidden, 32 layers)

**Domains (fine-tuning tasks):** 23 total across categories:
| Category | Domains |
|----------|---------|
| Math | gsm8k, competition_math |
| Reasoning | arc, commonsenseqa, hellaswag, piqa, socialqa |
| Knowledge | mmlu, nqopen, boolq, squad |
| Medical | pubmedqa |
| Code | codefeedback |
| Summarization | dialogsum |
| Classification | agnews, sst2 |
| Instruction-following | dolly, openorca, wizardlm, alpaca, lima, openhermes, norobots |

**Fine-tuning:** LoRA (rank 16, alpha 32) on each domain → merge into base model

**Safety Dataset:** BeaverTails-Evaluation (700 curated harmful prompts from PKU-Alignment)

**Metrics:**
- Domain Task Accuracy: task-specific accuracy on the fine-tuning domain's test set
- Domain Lambda Accuracy (dom_λ): % of safe/domain prompts correctly classified as safe (λ<0.5)
- Harmful Lambda Accuracy (harm_λ): % of harmful prompts correctly classified as harmful (λ>0.5)
- Lambda Gap: harm_λ - dom_λ (higher = better discrimination)

**4.2 Evaluation Protocol: Multi-Holdout Splits**

Key experimental design: hold out M domains from hypernetwork training, evaluate zero-shot generation on held-out domains.

**Unseen Datasets (4 splits):**
| Split | Held-out Domains |
|-------|-----------------|
| A | competition_math, arc, boolq, sst2, dolly |
| B | gsm8k, hellaswag, mmlu, codefeedback, wizardlm |
| C | gsm8k, hellaswag, dolly, codefeedback |
| D | competition_math, boolq, sst2, wizardlm |

Each split holds out 1 domain per category, so training always sees at least one domain from each category.

**Unseen Domains (category-level holdout):**
Hold out entire categories: Medical (pubmedqa) + Code (codefeedback) + Math (gsm8k, competition_math)
Tests generalization to domain *types* never seen during training.

**4.3 Baselines**
1. **LoRA-finetuned model (no safety):** raw fine-tuned model, no safety mechanism. Shows safety degradation.
2. **Base instruct model:** original safety-aligned model before fine-tuning. Shows upper bound on safety + lower bound on domain performance.
3. **Oracle side network:** per-domain trained side network (not generated). Shows ceiling performance.
4. **External classifier (Llama Guard / WildGuard):** model-agnostic safety classifier. Shows model-independent baseline.

**TABLE 1 (Main Results):** Lambda accuracy on unseen dataset splits, averaged across splits A-D.
Columns: Domain | Task Acc | Dom λ-Acc | Harm λ-Acc | (for each of: LoRA-only, LST-generated, Oracle)

**TABLE 2:** Unseen domain (category-level) results.

**TABLE 3:** Comparison with external classifiers (Llama Guard, WildGuard) on the same held-out domains.

**4.4 Main Results**
- Generated safety heads recover >95% harmful detection on most unseen domains
- Task accuracy preserved (within 1-3% of LoRA-finetuned model)
- Lambda gap consistently positive (+0.6 to +0.9) across domains
- Works across both model architectures (Qwen2, LLaMA3)

**4.5 Ablation Studies**

**TABLE 4:** Ablation table showing contribution of each component:
| Variant | harm_λ (avg) |
|---------|-------------|
| Full V15 (direction-magnitude + templates + LoRA lambda) | X% |
| − template corrections (generate full weights) | -Y% |
| − direction-magnitude split (raw activations) | -Z% |
| − lambda LoRA delta | -W% |
| − behavioral probes (weight recon only) | -V% |

**4.6 Analysis**

**Calibration sensitivity:** How does N_cal (number of calibration prompts) affect performance? Plot harm_λ vs N_cal (10, 25, 50, 100).

**FIGURE 4:** Calibration prompt sensitivity plot.

**Cross-architecture transfer:** Train on Qwen2 domains, test on LLaMA3 (or vice versa). Does the method transfer across architectures? (Likely not — activations are architecture-specific. But worth discussing as a limitation.)

**Lambda distribution visualization:** Show lambda score distributions for safe vs. harmful prompts on an unseen domain. Should show clear separation.

**FIGURE 5:** Lambda distribution histograms (safe vs harmful) for a representative unseen domain.

### 5. Conclusions (~0.5 page)
- We introduced model-conditioned safety: treating safety as dependent on the fine-tuned checkpoint
- Hypernetwork generates lightweight safety heads from activations, enabling zero-shot safety recovery
- Validated across 2 architectures, 23 domains, 4 holdout splits
- Practical implications: safety-as-a-service for the fine-tuned model ecosystem
- Future work: cross-architecture transfer, scaling to larger models, extending beyond prompt-level classification to generation-level safety

### Limitations (~0.5 page)
- Requires per-domain side network training for hypernetwork training set (one-time cost, but labor-intensive)
- Lambda classifier operates at prompt level — does not prevent harmful continuations of safe-looking prompts
- Tested on 7B-8B models; scaling behavior to 70B+ unknown
- Side network adds ~3% parameters and slight inference latency
- BeaverTails as sole harmful benchmark — broader safety taxonomies needed
- No multilingual evaluation

### Appendix
- A: Full domain list with dataset statistics
- B: Side network architecture details (layer counts, dims, parameter counts)
- C: Hypernetwork training hyperparameters
- D: Per-domain results for all 4 splits (full tables)
- E: Additional ablation details
- F: Compute costs (training time, GPU hours)

---

## Figures and Diagrams

### FIGURE 1: Pipeline Overview (Page 1, full-width)
**Description:** Three-panel horizontal diagram showing the full pipeline.
- **Panel A (Training — Offline):** Shows K fine-tuned models (M_d1, M_d2, ..., M_dK), each with activations extracted. Arrows from activations into "Hypernetwork" box. Ground-truth side networks shown as supervision signal. Caption: "Train hypernetwork across K domains with ground-truth side networks."
- **Panel B (Deployment — Zero-Shot):** A new unseen fine-tuned model M_d*, activations extracted from small calibration set (50 prompts), fed into trained hypernetwork, which outputs generated side network weights. Caption: "Given unseen model, extract activations and generate safety head."
- **Panel C (Inference):** Input prompt flows through M_d* backbone AND generated side network in parallel. Side network outputs lambda score. Decision: λ<0.5 → safe (green check), λ>0.5 → harmful (red X). Caption: "Classify prompts at inference time."
- **Style:** Clean, minimal, use arrows and boxes. Color-code: blue for backbone, orange for side network, red for harmful, green for safe.

### FIGURE 2: Side Network Architecture (Section 3.2)
**Description:** Vertical diagram showing backbone transformer layers (left, frozen) and side network layers (right, smaller).
- Show backbone layers 1-28 with every K-th layer connected via "Downsample" arrow to side network layers 1-12
- Ladder connections: gated arrows from backbone hidden states to side network
- At the bottom: side network hidden states → mean pool → MLP → λ (lambda score)
- Label key components: Downsample (Linear), Ladder Gate (μ), Side Attention, Side MLP, Lambda Head
- Show gating formula: μ·h_ladder + (1-μ)·h_side
- **Style:** Ladder-like visual, emphasizing the skip connections that give LST its name

### FIGURE 3: Hypernetwork Architecture (Section 3.4)
**Description:** Detailed diagram of the V15 hypernetwork for one layer.
- **Input:** Raw activation vector from layer i
- **Branch 1 (Direction):** Normalize → unit direction d̂ → Linear(proj_dim=256) → Attention → direction features
- **Branch 2 (Magnitude):** Log(||a||) → MLP(128) → magnitude features
- **Fusion:** Concatenate direction + magnitude → MLP → low-rank correction ΔW
- **Template:** Show template T (learned, shared across domains) + scale·ΔW → final weights W_i
- **Output:** Generated weights for side network layer i
- Show this is repeated per-layer (layer-wise factorization)
- **Style:** Flow diagram, left-to-right. Color: blue for direction path, green for magnitude path, orange for fusion.

### FIGURE 4: Calibration Sensitivity (Section 4.6)
**Description:** Line plot.
- X-axis: Number of calibration prompts (10, 25, 50, 100)
- Y-axis: Harmful lambda accuracy (%)
- Lines: one per held-out domain (or averaged with confidence band)
- Show that 50 prompts is sufficient (diminishing returns after)
- **Style:** Clean matplotlib plot, with clear legend

### FIGURE 5: Lambda Score Distributions (Section 4.6)
**Description:** Overlapping histograms or violin plots.
- Two distributions per subplot: safe/domain prompts (blue) and harmful prompts (red)
- Show for 2-3 representative unseen domains (e.g., gsm8k, arc, dolly)
- Clear separation between distributions demonstrates lambda classifier effectiveness
- Mark threshold at λ=0.5
- **Style:** Side-by-side subplots, semi-transparent overlapping histograms

### FIGURE 6 (Optional): Safety Degradation Motivation
**Description:** Bar chart showing harmful compliance rates of fine-tuned vs. base models.
- X-axis: domain (gsm8k, arc, dolly, etc.)
- Y-axis: % of harmful prompts answered (lower is safer)
- Two bars per domain: Base model (low, safe) vs. LoRA-finetuned (high, unsafe)
- Motivates the problem: fine-tuning drastically increases harmful compliance
- **Style:** Grouped bar chart, red bars for fine-tuned, green for base

---

## Tables

### TABLE 1: Main Results — Unseen Dataset Generalization
Rows: held-out domains (averaged across splits or shown per-split)
Columns: Method | Task Acc (%) | Dom λ-Acc (%) | Harm λ-Acc (%)
Methods: LoRA-only (no safety), Generated Safety Head (ours), Oracle Side Net, Base Model (no FT)

### TABLE 2: Unseen Domain (Category-Level) Generalization
Same format as Table 1, but for category-level holdout (medical + code + math)

### TABLE 3: Comparison with External Safety Classifiers
Rows: held-out domains
Columns: Method | Harmful Detection Rate (%) | False Positive Rate (%)
Methods: Ours, Llama Guard, WildGuard, Keyword-based

### TABLE 4: Ablation Study
Rows: ablation variants
Columns: harm_λ (%), dom_λ (%), Δ from full

### TABLE 5 (Appendix): Full Per-Domain Results
Complete results for all 4 splits × all domains × both architectures

---

## Key References to Add

**Safety fragility:**
- Qi et al. (2024) — Fine-tuning compromises safety [ALREADY IN BIB]
- Qi et al. (2025) — Safety alignment tokens deep [ALREADY IN BIB]
- Fraser et al. (2025) — Fine-tuning lowers safety [ALREADY IN BIB]
- Yang et al. (2024) — Shadow Alignment: fine-tuning risks
- Zhan et al. (2024) — Removing RLHF protections

**Safety classifiers / guardrails:**
- Inan et al. (2023) — Llama Guard
- Han et al. (2024) — WildGuard
- Kim et al. (2024) — Adversarial Prompt Shield [ALREADY IN BIB]
- Jiang et al. (2025) — HiddenDetect [ALREADY IN BIB]

**Post-hoc safety restoration:**
- Huang et al. (2024) — LISA [ALREADY IN BIB]
- Hsu et al. (2024) — Safe LoRA
- Yi et al. (2024) — Vaccine: perturbation-aware alignment
- Rosati et al. (2024) — Representation noising

**Hypernetworks:**
- Ha et al. (2017) — HyperNetworks (original)
- Chauhan et al. (2024) — Hypernetwork survey [ALREADY IN BIB]
- von Oswald et al. (2020) — Continual learning with hypernetworks
- Navon et al. (2023) — Equivariant architectures for hypernetworks

**Side networks / parameter-efficient methods:**
- Sung et al. (2022) — Ladder Side-Tuning (LST) [MUST ADD]
- Hu et al. (2022) — LoRA
- Houlsby et al. (2019) — Parameter-efficient transfer with adapters

**Safety datasets:**
- Ji et al. (2024) — BeaverTails
- Mazeika et al. (2024) — HarmBench

---

## Novelty Arguments (for reviewers)

1. **New formulation:** Model-conditioned safety (safety depends on the checkpoint, not just the prompt). This is distinct from model-agnostic classifiers.

2. **Amortized safety:** One-time hypernetwork training enables zero-shot safety head generation for any new fine-tuned model. Cost is O(1) per new model at deployment, vs. O(training) for re-alignment methods.

3. **Technical contributions:** Direction-magnitude activation processing + template-based correction generation are novel hypernetwork design choices, motivated by the specific challenges of activation-to-weight mapping.

4. **Practical significance:** Addresses a real and growing problem — the proliferation of fine-tuned models with degraded safety. Our method is the first to offer scalable, training-free safety restoration.

---

## Potential Reviewer Concerns & Responses

**Q: Why not just use Llama Guard / external classifier?**
A: External classifiers are model-agnostic — they don't account for how fine-tuning changes safety boundaries. Table 3 shows our model-conditioned approach outperforms external classifiers on fine-tuned models.

**Q: The hypernetwork still requires training ground-truth side networks for many domains. How scalable is this?**
A: This is a one-time cost. Once trained, the hypernetwork generalizes to unseen domains zero-shot. We show generalization across 4 holdout splits and even to unseen domain categories.

**Q: Side network adds parameters and latency.**
A: The side network is ~3% of backbone parameters (237M for Qwen2-7B). Lambda inference requires only a forward pass through 12 small transformer layers — negligible compared to the 28-layer backbone.

**Q: Only tested on 7B-8B models.**
A: Fair limitation. The layer-wise factorized design should scale, but empirical validation on larger models is future work. Discuss in Limitations.

**Q: Lambda operates at prompt level — what about harmful continuations?**
A: Yes, this is a prompt-level classifier. It catches the majority of harmful prompts upfront. Generation-level safety (monitoring during decoding) is complementary and orthogonal to our approach.

---

## Writing Timeline Suggestion

1. **Methods section first** — this is the core technical contribution, write it carefully
2. **Experiments setup** — define all baselines, metrics, splits clearly
3. **Results tables** — fill in once experiments complete
4. **Introduction** — refine the narrative after methods/experiments are solid
5. **Related work** — comprehensive but concise
6. **Abstract** — write last, distill the whole paper into 200 words
7. **Figures** — can be delegated in parallel

---

## Page Budget (8 pages + refs)

| Section | Pages |
|---------|-------|
| Abstract | 0.3 |
| Introduction + Figure 1 | 1.5 |
| Related Work | 1.0 |
| Methodology + Figures 2-3 | 2.5 |
| Experiments + Tables 1-4 + Figures 4-5 | 2.5 |
| Conclusions + Limitations | 0.5 |
| **Total** | **8.3** → trim to 8 |
