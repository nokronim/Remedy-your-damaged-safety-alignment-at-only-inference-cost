# Hypernetwork Architecture — Full Technical Details

## Overview

The hypernetwork takes an **activation fingerprint** from a fine-tuned LLM and outputs the complete weights of a **safety head** (~230M parameters). The safety head is a small side transformer that classifies prompts as safe or harmful.

---

## 1. Input: Activation Fingerprint

For a fine-tuned model M_d, we run N_cal=50 domain-specific calibration prompts through it and collect the hidden state at each transformer layer after the final token position:

```
a_l = (1/N_cal) * sum_i h_l(x_i)    for l = 1, ..., L
```

- L = 32 layers (for LLaMA-3-8B)
- Each a_l is a vector in R^4096 (backbone hidden dim)
- Total input: 32 vectors of 4096 dimensions = 131K numbers
- These vectors encode how fine-tuning shifted the model's representations

---

## 2. Direction-Magnitude Decomposition

Each activation vector is split into two components:

```
direction:  d_l = a_l / ||a_l||       (unit vector, 4096-dim)
magnitude:  m_l = log ||a_l||          (scalar)
```

**Why decompose?**
- Direction captures WHAT changed (which direction in representation space fine-tuning pushed the model)
- Magnitude captures HOW MUCH it changed (a model trained 10 epochs shifts more than 1 epoch)
- Empirically: most domains share similar directions in early/middle layers (cosine >0.95) but diverge sharply in later layers (cosine 0.2-0.5). Magnitudes vary 2-3x across domains at all layers.
- Separating them lets each encoder specialize

**Direction encoder:**
```
z_dir = CrossLayerAttention(W_proj * d_l)
```
- W_proj ∈ R^(256 × 4096): projects 4096-dim direction to 256-dim
- CrossLayerAttention: 4 queries, 4 heads, processes all L=32 layers jointly
- Why attention? Because the relationship between directions across layers matters (e.g., layer 5 direction interacting with layer 20 direction)
- Output: 512-dim embedding

**Magnitude encoder:**
```
z_mag = MLP(m_l)
```
- Input: 32 scalars (one per layer)
- 2-layer MLP with hidden dim 128
- Output: 128-dim embedding
- Why MLP? Magnitudes are just scalars — no cross-layer interaction needed

**Fusion:**
```
z_fused = [z_dir; z_mag]    (concatenation, 640-dim)
```

---

## 3. Weight Generation

### 3.1 Layer-wise Factorized Generation

Instead of predicting all ~230M parameters at once, we predict each side-network layer's weights independently. Each layer's generation is conditioned on **both local and global signals**:

```
theta_k = H_k(local_k, global_emb)    for k = 1, ..., K=12 side layers
```

Where l(k) is the backbone layer connected to side layer k (every other layer: 0, 2, 4, ..., 22).

**Two conditioning signals per layer:**

1. **Local signal** (layer-specific, 257-dim):
   - Per-layer direction projection: `local_dir = W_local_proj * d_{l(k)}` → 256-dim
   - Per-layer log-magnitude: `local_mag = log ||a_{l(k)}||` → 1 scalar
   - Concatenated: `local_k = [local_dir; local_mag]` → 257-dim
   - This captures what happened at THIS specific layer

2. **Global signal** (shared across all layers, 512-dim):
   - The fused direction-magnitude embedding from step 2 (z_fused → domain_fusion → global_emb)
   - This captures the overall behavioral shift of the model across ALL layers
   - Provides context: "what kind of model is this?" while local says "what happened at this layer?"

**Layer generation network:**
- Input: cat(local_k, global_emb) = 257 + 512 = 769-dim
- Architecture: Linear(769, 512) → GELU → Linear(512, 512) → GELU → Linear(512, output_dim)
- Most parameters are shared across all 12 layers
- Each layer has a small **layer-specific adapter** (additional linear projection) that specializes the output

**Why both?** Local-only would miss cross-layer patterns (e.g., a model fine-tuned on math shifts differently at layers 5 vs 25, but the global pattern tells you "this is a math model"). Global-only would generate the same weights for every layer, missing layer-specific structure.

**What weights are generated per layer:**
- Self-attention: Q, K, V, O projection matrices (each h_s × h_s = 1024 × 1024)
- MLP: gate_proj, up_proj, down_proj (1024 × 3584 each)
- LayerNorm: weight and bias (1024 each)
- Downsample projection: W_down ∈ R^(1024 × 4096) — projects backbone hidden to side hidden
- Gate scalar: mu_k

Per side layer: ~15.2M parameters
Total across 12 layers: ~182.5M parameters

### 3.2 Lambda Classifier Generation (Separate Branch)

The safety classifier (the MLP that produces the final safe/harmful score) is generated via a dedicated branch:

1. **Cross-layer attention** aggregates information across ALL 32 layers into a global embedding
2. This global embedding goes through an MLP
3. Output uses **low-rank factorization**: W_cls = B · A
   - A ∈ R^(16 × 1024), B ∈ R^(1 × 16)
   - Rank r = 16
   - Reduces generated parameters while preserving expressiveness

Lambda classifier total: ~1.1M parameters

---

## 4. Training

### 4.1 Ground-Truth Pairs

Before hypernetwork training, we collect (model, safety head) pairs:
- 22 domains × 2 models (LLaMA-3, Qwen2) = up to 44 pairs
- For each domain: freeze the fine-tuned model, train a side network on safe+harmful data
- All side networks share the same random initialization → weight spaces are aligned (no permutation ambiguity)

### 4.2 Training Losses

Three complementary losses:

```
L = L_recon + α * L_cls + β * L_func
```

**L_recon** (weight reconstruction, α_weight=1):
- MSE between generated weights and ground-truth weights
- Provides dense supervision across all ~230M dimensions
- Without this, the prediction is severely underconstrained

**L_cls** (classifier accuracy, α=10):
- BCE on safety score predictions
- Uses the SAME calibration prompts used for activation extraction
- Directly supervises the lambda classifier on calibration data
- Higher weight (10x) because lambda classifier is small but critical

**L_func** (behavioral probes, β=5):
- BCE on safety score predictions using a SEPARATE batch of safe and harmful probe prompts
- 50 harmful probes (from BeaverTails) + 50 safe probes (from domain data)
- NOT used for activation extraction — tests generalization
- Critical finding: removing safe probes causes classifier to mark everything as harmful (accuracy collapses)

### 4.3 Training Setup

- Optimizer: Adam, lr = 5e-5
- Epochs: 100
- Hidden dim: 512
- Batch: iterate through all K training domains per epoch
- Each epoch: for each domain, extract activations → generate weights → compute loss → backprop
- Training time: ~20 minutes on single A100

---

## 5. Inference (Deployment)

Given a new fine-tuned model M_d* (never seen during training):

1. **Extract activations**: Run 50 calibration prompts through M_d*, collect hidden states at each layer, average them. (~15 seconds)

2. **Hypernetwork forward pass**: Feed the 32 activation vectors through the hypernetwork. Out come ~230M weight values. (~1 second)

3. **Load safety head**: Create the side network, load the generated weights. Attach to the frozen M_d*. (~1 second)

4. **Per-prompt inference**: For each input prompt:
   - Prompt flows through both backbone AND side network in parallel
   - Side network produces safety score λ
   - If λ ≤ 0.5: base model output (unchanged, no accuracy impact)
   - If λ > 0.5: return refusal template

Total deployment: **under 30 seconds**. No training, no gradients, no safety data needed.

---

## 6. Architecture Summary

```
Input:  32 activation vectors × 4096 dim  (from fine-tuned model)
                    ↓
        Direction-Magnitude Decomposition
        ├── Directions (32 × 4096) → CrossLayerAttention → 512-dim
        └── Magnitudes (32 scalars) → MLP → 128-dim
                    ↓
              Fused: 640-dim
                    ↓
        Layer-wise Factorized Generation
        ├── Side Layer 1 weights (~15.2M params)
        ├── Side Layer 2 weights (~15.2M params)
        ├── ...
        ├── Side Layer 12 weights (~15.2M params)
        ├── Downsample projections (12 × 4096 × 1024 = ~50.3M params)
        └── Lambda classifier (low-rank, ~1.1M params)
                    ↓
Output: Complete safety head weights (~230M params total)
```

## 7. Key Design Decisions

| Decision | Why |
|----------|-----|
| Direction-magnitude split | Disentangles what changed from how much; different encoders can specialize |
| Cross-layer attention for directions | Directions across layers interact (layer 5 relates to layer 20) |
| Simple MLP for magnitudes | Magnitudes are just 32 scalars, no cross-layer structure needed |
| Layer-wise factorization | Reduces output dimensionality from 230M to ~15M per prediction |
| Low-rank classifier generation | Lambda classifier is small but critical; low-rank preserves expressiveness efficiently |
| Three losses | L_recon regularizes weight space; L_cls supervises on calibration data; L_func ensures generalization to unseen prompts |
| Shared initialization for GT side networks | Eliminates permutation ambiguity — weight-space MSE is well-defined |

## 8. Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone hidden dim (h) | 4096 (LLaMA-3) / 3584 (Qwen2) |
| Side hidden dim (h_s) | 1024 / 896 |
| Side layers (K) | 12 |
| Reduction factor (r) | 4 |
| Hypernetwork hidden dim | 512 |
| Direction projection dim | 256 |
| Magnitude hidden dim | 128 |
| Classifier LoRA rank | 16 |
| N_cal (calibration prompts) | 50 |
| Training epochs | 100 |
| Learning rate | 5e-5 |
| L_cls weight (α) | 10 |
| L_func weight (β) | 5 |
