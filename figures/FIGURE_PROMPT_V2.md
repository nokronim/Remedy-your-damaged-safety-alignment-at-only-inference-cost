# Figure Prompts V2 — Rich Context, Creative Freedom

For each prompt: paste into Claude web, ask for SVG artifact. Iterate freely.

---

## Figure 1: Framework Overview

```
I'm writing a paper about SafeHead, a system for restoring safety in fine-tuned language models. I need an SVG figure for the paper. Here's everything about how the system works — design a clear, elegant diagram that communicates the key ideas. Style it like a top ML conference paper (NeurIPS, ICML, ECCV). Use serif font. 700x500 viewBox.

BACKGROUND:
When you fine-tune an LLM (like LLaMA or Qwen) on a task dataset, the model gets better at that task but loses its safety alignment — it starts complying with harmful requests it would have refused before. This happens even with benign training data and is a major problem in the open-source model ecosystem where thousands of fine-tuned models circulate.

OUR SOLUTION has three stages:

STAGE 1 — COLLECTING TRAINING DATA (done once):
We take an aligned base model and create many fine-tuned variants using LoRA on different domains (BoolQ for QA, GSM8K for math, Alpaca for instructions, etc. — 22 domains total). For each fine-tuned model, we train a small "safety head" — a classifier that can detect harmful prompts for that specific model. So we end up with 22 pairs: each pair is a (fine-tuned model, its safety head). These pairs are the training data for the next stage.

STAGE 2 — TRAINING THE HYPERNETWORK (done once):
We want to learn a function that can produce a safety head for ANY fine-tuned model, even ones we've never seen. We do this by training a hypernetwork — a neural network that generates the weights of another neural network.

The input to the hypernetwork is an "activation fingerprint" of a fine-tuned model. We run a small set of calibration prompts through the model and record the hidden states at each transformer layer. These hidden states encode how fine-tuning shifted the model's representations — essentially a behavioral fingerprint of what the model learned.

The hypernetwork takes this fingerprint and outputs the complete set of weights for a safety head. It's trained using the 22 pairs from Stage 1: for each pair, extract the fingerprint, generate weights, and compare against the ground-truth safety head using MSE loss plus behavioral losses.

STAGE 3 — DEPLOYMENT (per new model, takes seconds):
When someone gives us a new fine-tuned model we've never seen before:
1. Run a handful of calibration prompts through it to get the activation fingerprint
2. Feed the fingerprint through the trained hypernetwork in a single forward pass
3. Out comes a complete safety head, tailored to this specific model
4. Attach the safety head — it classifies each incoming prompt as safe or harmful
5. Safe prompts go through normally. Harmful prompts get refused.

No training needed. No safety data needed. No modification to the model. Takes seconds.

KEY EMPHASIS in the figure:
- The contrast between "one-time training" (stages 1-2) and "instant deployment" (stage 3)
- The hypernetwork is the central innovation
- The output is a model-SPECIFIC safety classifier (not one-size-fits-all)
- The speed of deployment
```

---

## Figure 2: Hypernetwork Architecture

```
I need an SVG figure showing the internal architecture of a hypernetwork that generates safety classifier weights. This is for an academic ML paper. Style like NeurIPS/ICML. Serif font. 680x500 viewBox.

WHAT THE HYPERNETWORK DOES:
It takes a behavioral fingerprint of a fine-tuned LLM and outputs the complete weights of a safety classifier (about 230M parameters total). The challenge is that we're predicting millions of weights from just a few activation vectors — a severely ill-conditioned problem. Two key design choices make this tractable.

THE INPUT — ACTIVATION FINGERPRINT:
We run N=50 calibration prompts through the fine-tuned model and collect the hidden state at each of the L=32 transformer layers (averaging across prompts). This gives us 32 vectors, each of dimension 4096. These vectors encode how fine-tuning shifted the model's internal representations.

DESIGN CHOICE 1 — DIRECTION-MAGNITUDE DECOMPOSITION:
Each activation vector is decomposed into:
- A unit DIRECTION vector (a/||a||) — this captures WHAT changed in representation space. Different fine-tuning tasks push representations in different geometric directions.
- A scalar MAGNITUDE (log ||a||) — this captures HOW MUCH it changed. A model trained for 10 epochs shifts more than one trained for 1 epoch, but in a similar direction.

These two components carry fundamentally different information and are processed by different encoders:
- Directions go through a cross-layer ATTENTION module (because the relationship between directions across layers matters — the direction at layer 5 interacts with the direction at layer 20)
- Magnitudes go through a simple MLP (they're just 32 scalars — no cross-layer interaction needed)
The outputs are fused (concatenated) into a joint embedding.

DESIGN CHOICE 2 — LAYER-WISE FACTORIZED GENERATION:
Rather than predicting all 230M parameters at once, we predict each safety-head layer's weights independently from the corresponding backbone activation. A shared weight-generation MLP (with layer-specific adapters) takes the fused embedding and outputs the weights for one layer at a time. This reduces the effective output dimensionality dramatically.

SEPARATE CLASSIFIER BRANCH:
The safety classifier (the small MLP that produces the final safe/harmful score) is generated via a dedicated branch. It first aggregates information across ALL layers using cross-layer attention to produce a global embedding, then maps it through an MLP with low-rank output factorization (W = B·A, rank 16) for efficiency.

OUTPUT:
The complete set of safety head weights — ready to load alongside any fine-tuned model.

KEY EMPHASIS:
- The direction/magnitude split is the most visually interesting part
- Show that different layers are processed independently (factorization)
- The input is compact (32 vectors) but the output is high-dimensional (230M weights)
- The attention module for directions vs MLP for magnitudes shows the architectural asymmetry
```

---

## Figure 3: Safety Head (Side Network)

```
I need an SVG figure showing how a safety head works alongside a frozen LLM backbone during inference. Academic ML paper style. Serif font. 680x420 viewBox.

CONTEXT:
After the hypernetwork generates the safety head weights, we load them into a small neural network that runs alongside the main model. The safety head's job: for each input prompt, decide whether it's safe (let the model respond normally) or harmful (refuse to answer).

THE BACKBONE (main model):
A standard transformer with L layers (e.g., 32 for LLaMA-3-8B). It's completely frozen — we never modify its weights. The input prompt flows through it normally. The backbone has hidden dimension h=4096.

THE SIDE NETWORK (safety head):
A much smaller transformer with K=12 layers, hidden dimension h_s=1024 (only 1/4 of the backbone). It runs in parallel with the backbone but is connected to it through "ladder connections." The side network contains only about 3% of the backbone's parameters.

LADDER CONNECTIONS:
At selected backbone layers, the hidden states are projected (downsampled) from 4096 dimensions to 1024 and fed into the corresponding side network layer. The fusion uses a learnable gate:

h_side = gate * downsample(h_backbone) + (1-gate) * h_side_prev

The gate controls how much backbone information flows into the side network at each layer. This is called "Ladder Side-Tuning" (LST), originally designed for parameter-efficient task adaptation — we repurpose it for safety classification.

THE SAFETY SCORE:
After the final side network layer, hidden states are mean-pooled across the sequence and passed through a small MLP to produce a single scalar — the safety score λ ∈ [0,1].

SELECTIVE ROUTING (the key mechanism):
This is what preserves task accuracy:
- If the safety score says "safe": the side network is COMPLETELY BYPASSED. The backbone's output goes through as-is. The model responds normally. Zero accuracy impact.
- If the safety score says "harmful": a pre-written refusal response is returned instead of the model's generation.

This means safe prompts NEVER interact with the safety head at all — the only overhead is computing the safety score, and if it's safe, there's no modification to the output. This is why task accuracy is preserved within <1%.

KEY EMPHASIS:
- The backbone is large and frozen (not modified)
- The side network is tiny (~3% of backbone)
- The ladder connections show information flowing from backbone to side network
- The routing decision at the bottom is the critical mechanism
- Safe prompts bypass everything — that's why accuracy is preserved
```

---

## Tips:
- "make it more professional and clean"
- "less saturated colors"
- "can you try a vertical layout instead?"
- "make the [X] the visual focal point"
- "add more whitespace"
- "the text is too small"
