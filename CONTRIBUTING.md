# Contributing

## The one rule

**The oracle is the specification.**
[`silkern.localize_reference`](silkern/contract.py) defines what correct means. A
kernel change that does not reproduce it elementwise is wrong, no matter how fast
it is. If you believe the oracle is wrong, change the oracle first, in its own
commit, with the reasoning — do not change a kernel to match a new belief about
the contract.

## Setup

```bash
pip install -e ".[dev]"
python -m pytest -q -m "not gpu"      # runs anywhere, no GPU needed
python -m ruff check .
```

With a CUDA device and Triton:

```bash
pip install -e ".[dev,gpu]"
python -m pytest -q                   # adds the CUDA suites
python -m silkern                       # the conformance matrix
```

## Scope

`silkern` is a converter, not a framework. In scope:

- the localization contract and its implementations
- correctness and determinism verification
- integrations that install the converter at a specific upstream call site
- evidence for claims made in this repository

Out of scope: attention kernels, selectors, KV-cache management, scheduling,
anything that would make this a serving stack.

## Changing a kernel

1. Add the failing case to `tests/test_contract.py` first, against the oracle.
2. Make the change.
3. `python -m pytest -q` — the ported GPU suites must pass.
4. `python -m silkern` — the full conformance matrix, including replay and guard
   checks.
5. If you touched anything performance-relevant, run
   `python -m bench.bench_converter --with-atomic` before and after and put both
   numbers in the PR. A converter that wins in isolation can lose end to end;
   [`docs/dispatch.md`](docs/dispatch.md) has a worked example of exactly that.

The Triton kernel bodies are carried verbatim from the implementation the
recorded evidence was collected on. Reformatting them invalidates that link, so
`ruff` is configured to leave them alone. Behavior changes are welcome; cosmetic
churn in `silkern/kernels.py` is not.

## Adding a geometry to the vLLM adapter

The `QUALIFIED_*` constants in
[`silkern/integrations/vllm.py`](silkern/integrations/vllm.py) are a claim that the
geometry has been checked, not a wishlist. To widen one:

1. Add the geometry to a `silkern.conformance()` matrix and run it on real hardware.
2. Paste the report in the PR.
3. Then edit the constant.

In that order. The adapter fails closed for a reason.

## Adding evidence

Anything in [`evidence/`](evidence/) must be raw analyzer output, not a
hand-written summary, and must be listed in `SHA256SUMS`. If a measurement did
not clear its own criterion, it still goes in —
[`evidence/03`](evidence/03-subpath-timing/) is an abstention and is kept
deliberately. A repository that publishes only its wins is advertising, not
evidence.

## Reporting a conformance failure

Include:

- the geometry line from `report.summary()` (it names the failing check)
- `torch.cuda.get_device_name(0)`, and your torch and triton versions
- whether the pure-Python oracle path also disagrees (`localize_reference` alone)

A `conformance()` failure is a real bug and takes priority over everything else
here.
