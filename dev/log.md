# Experiment Log

A running summary documenting some experiments and findings. Started ~Jan 7 2026.

---

## 2026-01-08: exp_grad_clip - Gradient Clipping

**Hypothesis:** Gradient clipping may be unnecessary overhead. Tested L2 norm clipping at various thresholds (0.25, 0.5, 1.0, 2.0) and elementwise clipping.

**Results:**

- No benefit at any scale tested (d12, d20)

- All variants within noise (~0.9827 val_bpb)

- Grad norm never exceeds 1.0 naturally, so clipping is always inactive

- Clipping adds ~2% time overhead from the all-reduce

**Bug Found:** Original implementation clipped local gradients before sync. Since this codebase doesn't use DDP (gradient sync is in the optimizers), each rank was clipping based on its own local norm. Fixed on the branch with proper distributed all-reduce.

**Observation:** modded-nanogpt does not appear to clip either right now.

**Recommendation:** Disable by default (`--grad_clip=0.0`). The code naturally produces well-behaved gradients.

---
