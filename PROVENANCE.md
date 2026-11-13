# Source provenance

- Upstream: https://github.com/turboderp-org/exllamav3
- Baseline: v1.6.6, `bd1f519172406a9560bd5c90e61000691bb69a07`
- License: MIT; original copyright preserved in LICENSE.
- Release strategy: clean source snapshot plus reviewed conversion changes.
- Original upstream README: UPSTREAM_README.md.

No private Git history is included. Upstream CI and funding configuration are
omitted because their release jobs and sponsorship settings belong to upstream.
The Python package and imports remain named `exllamav3` for compatibility.
The package version is the upstream base version, not a separate wheel release.
Do not install upstream wheels expecting this checkout's conversion changes.

The seven added conversion modules implement row packs, disk input slices,
checkpoint recovery, Hessian ownership, affinity samples, external diagonal
loading, and chunked output. Integration changes are in conversion/compile.py,
conversion/convert_model.py, conversion/moe_affinity.py, loader/safetensors_alt.py,
modules/block_sparse_mlp.py, modules/dsv4.py, modules/hyperconnections.py,
modules/module.py, and sc_measure.py. The latter only defers the optional
datasets import; sensitivity module streaming is inherited from upstream.
