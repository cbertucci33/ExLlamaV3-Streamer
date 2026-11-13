# ExLlamaV3 Streamer

Experimental disk-backed calibration and streamed output compilation for
[ExLlamaV3](https://github.com/turboderp-org/exllamav3). This is a complete
source derivative, including the original runtime and conversion CLI.

## What streams

`convert.py --calibration_state_storage disk` stores calibration hidden states,
preserved cross-module context, reference states, and parked Hessians in disk
packs. Rows are reconstructed for the active call. Hessian leases load only
the groups being quantized and retire completed spills. Output compilation
copies file-backed tensor payloads in chunks instead of collecting a complete
output shard as resident tensors. Memory storage remains the default.

This does not make every weight out-of-core or impose a hard RAM/VRAM cap.
The converter still loads active modules and linear weights. Capture, row
execution, Hessian factorization, quantization buffers, transformed tensors,
and GPU kernels require memory. A large module or Hessian can still fail to
fit. Disk capacity and throughput also matter; there is no automatic disk
quota. Use a fast local disk with room for states, Hessians, checkpoints,
quantized intermediates, and final output.

### Sensitivity measurement is a separate path

The upstream `sc_measure.py --streaming` (also `--load-mode streaming`) loads
one top-level module at a time. Its cached states remain in system RAM.
`--max-sys` budgets pending sensitivity states by splitting passes; it is not
a total-process memory limit. This path does not use the converter's disk
state backend. The largest module and Hessian/noise workspace must still fit
on the device. `sc_optimize.py` converts measurements into a mixed-rate recipe.

## Install from this checkout

Use Python 3.10.11 or newer, a supported GPU, matching PyTorch/CUDA tooling,
and a C++ build toolchain. Native Windows also requires the MSVC build tools
and `triton-windows`, declared in `pyproject.toml`. See the retained
[upstream installation guide](UPSTREAM_README.md) for platform requirements.
Its upstream clone commands and release wheels install upstream, not this
derivative. Run these commands from this checkout instead:

```sh
python -m venv .venv
# Activate .venv using your platform's activation script.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install .
```

Select the PyTorch CUDA index that matches your driver and toolkit; `cu128`
is an example. The isolated package build installs extension sources; the
first import compiles the extension with the local toolchain. With PyTorch
and build requirements already installed, `pip install --no-build-isolation .`
can precompile instead. No custom wheel is provided. Keep the source checkout
for the CLI scripts.

## Convert and resume

From the checkout, with an unquantized model in `models/source`:

```sh
python convert.py -i models/source -o models/converted -w work/conversion   -b 4.0 -cr 250 -cc 2048 --calibration_state_storage disk
python convert.py -w work/conversion -r
```

These options are checked against the actual parser in the CPU tests. The
example itself requires a compatible model and GPU and is not run by those
tests. Use new, separate input, work, and output directories: conversion can
overwrite output files. Do not place unrelated files in the work directory.
Disk checkpoints contain absolute paths. Resume at the same work path with
the same source files, calibration data, recipe, and computation settings.
Do not move or edit spill packs. Resume rejects conflicting computation
options, storage-backend changes, and `--override_anyway`. Some scheduler
and diagnostic options may be changed. Completed module boundaries restore
row state, preserved context, random state, and committed output inventory;
unfinished work is repeated. Source safetensor identity uses size and mtime,
not a full content digest; do not treat it as tamper-proof verification.

For mixed rates, pass `--recipe recipes/mixed.yaml` instead of relying on a
uniform `-b` allocation. Recipes must cover the target model's tensors.
See [conversion options](doc/convert.md) and [recipe workflow](doc/optimize.md).
Those retained upstream documents describe memory-mode behavior unless
this README says otherwise. Decoder target bits per weight are not the
whole artifact's effective size.

## Validation and limits

Earlier GPU acceptance completed two full conversions of the same roughly
1.7-billion-parameter hybrid checkpoint, at mixed decoder targets of 4.0 and
5.0 bits per weight, with 250 x 2048 calibration tokens on native Windows
and one RTX 5080. Representative disk/memory pilots matched 86 tensors
exactly across convolution and attention blocks. Scaling pilots, three
interruption/recovery cases, twelve CLI rejection cases, and 27 committed
module hashes were checked; the completed storage audit found no stale packs.
These are prior scoped acceptance results, not GPU tests rerun for this
release. Private calibration data, raw reports, and model artifacts are not
distributed.

The repeatable release checks are CPU-only:

```sh
python -m unittest discover -s streamer_tests -v
```

They cover storage round trips and aliases, NumPy scalars, tensor layouts,
Hessian leases and retirement, chunked compilation, checkpoint recovery,
and parser/resume guards. They load storage modules without initializing
the GPU runtime. They do not validate CUDA kernels or model quality.

Multi-GPU execution, side-model conversion, full-depth stock-memory parity,
every architecture and MoE configuration, and all crash/publication windows
remain unqualified. Power-loss durability is not established. This is working
experimental software, not a universal memory guarantee or a production
readiness claim.

## Credits and license

ExLlamaV3 Streamer derives from **turboderp-org/exllamav3 v1.6.6**, revision
`bd1f519172406a9560bd5c90e61000691bb69a07`. The original runtime, kernels,
quantizer, sensitivity streaming, documentation, and tests are credited to
turboderp and the upstream contributors. Disk-backed conversion storage,
checkpoint integration, and streamed compilation are the derivative changes.

This release uses a clean upstream snapshot with reviewed feature files,
not private development history. It is a source derivative, not a claim of
a GitHub fork-network relationship or upstream endorsement. The original
[MIT license](LICENSE), public attribution, and [upstream README](UPSTREAM_README.md)
are retained. See [provenance](PROVENANCE.md).
