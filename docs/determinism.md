# Why the atomic converter is order-unstable

Not a bug report. The converter this replaces is correct, and understanding
*why* it is both correct and unstable is the whole point.

## The mechanism, in one paragraph

The converter processes a selection row in tiles of `BLOCK_N` elements. Each tile
scans its own elements, counts how many survive ownership filtering and paged
translation, and then reserves a contiguous output segment for those survivors
with a single `atomic_add` on a per-row counter:

```
base = atomic_add(&counter[row], tile_count)     # <- the only interesting line
out[row][base + local_position] = physical
```

Every tile ends up somewhere. No element is lost, none is duplicated, and the
final counter is the exact valid count. The *set* is a function of the input
alone.

The *order* is not. `base` depends on how many elements the tiles that reached
the atomic *first* happened to contribute — and which tiles reach it first is a
function of scheduling, occupancy, clock, memory pressure, what else is on the
device, and nothing you control. At a top-k of 2048 with `BLOCK_N=128` there are
16 tiles per row racing for 16 segments, which is why the shuffling is coarse and
blocky rather than a fine permutation.

You can watch it:

```bash
python -m bench.order_instability
```

Measured on two H100s, one selection row, 20 replays of byte-identical input:
**17 to 20 distinct consumed orders**. Both `silkern` implementations return
exactly one.

## Why a different order is a different answer

Sparse attention gathers the selected KV rows and reduces over them. In finite
precision, addition is not associative:

```
(a + b) + c  ≠  a + (b + c)
```

The differences are tiny — last-place bits — and they are also *amplified*: they
pass through a softmax, then a projection, then 47 more layers, and at the end
they choose between two logits that were nearly tied. Most steps, the argmax is
unchanged. Some steps, it isn't. Then the sequence forks and every subsequent
token is different.

This is why "the outputs are numerically close" is not a defense. Nobody claimed
they were far apart. The claim is that they are *not the same*, and greedy
decoding is a discontinuous function of them.

## What this costs you, concretely

**RL rollouts.** You generate a trajectory, compute rewards, and update against
the log-probs you recorded. If a replay of the same prefix produces different
log-probs, your importance ratios are computed against a policy the model did not
actually execute. The bias is small per token and does not average out, because
it is correlated with exactly the near-tie states where the gradient is largest.

**Evaluation.** Two runs of the same checkpoint on the same benchmark produce
different scores. You cannot separate a real 0.3-point regression from converter
noise without repeating everything many times — which is expensive, and which
people therefore do not do.

**Debugging.** A failure reproduces on Tuesday and not Wednesday. Bisection is
meaningless when the oracle is itself nondeterministic. This is the cost people
underestimate most, because it shows up as engineer-weeks rather than as a
number on a dashboard.

**Prefix caching.** A cached prefix and a freshly computed one take different
tile paths, so they disagree in the last bits — which surfaces as "caching
changes my outputs," a bug report that is very hard to close.

## Why the obvious fixes don't work

| Fix | Why not |
|---|---|
| Sort the output prefix afterward | Allocates a workspace, and the *selector's* order is not sorted order — you would be imposing a different arbitrary order, not restoring the intended one |
| One `atomic_add` per element instead of per tile | Strictly worse: finer-grained racing, more atomic traffic, still unspecified |
| A grid-wide barrier before reserving | Not available inside a single kernel launch without cooperative groups, which constrains occupancy and complicates capture |
| Serialize the tiles | Throws away the parallelism the tiling existed to provide |
| `torch.use_deterministic_algorithms(True)` | Governs PyTorch's own kernels. This converter is a custom Triton kernel; the flag does not reach it |

What actually works is not racing in the first place. A row-wide `cumsum` gives
each element its destination as a pure function of the input
(`localize_rowwise`), or bounded tile scans compose through a deterministic
tile-prefix pass in caller-owned workspace (`localize_hierarchical`). Neither
uses an atomic. Both are cheaper than or comparable to the baseline — see
[`dispatch.md`](dispatch.md).

## What determinism here does *not* buy you

Being precise about the boundary, because overclaiming here is easy:

- **This is not end-to-end bitwise reproducibility.** Other nondeterminism
  remains: reduction order inside attention and GEMM kernels, cuBLAS algorithm
  selection, NCCL reduction order, batch-composition effects under continuous
  batching. `silkern` removes one specific source; it does not certify the stack.
- **It is not a correctness fix.** The atomic converter is not wrong. If your
  pipeline genuinely does not care about order, you lose nothing by keeping it —
  though you also gain nothing by keeping it, since the deterministic converter
  is cheaper.
- **It says nothing about which order is *better*.** Preserving the selector's
  order is a determinism property, not a quality one. No claim is made that
  selector order improves model output.
