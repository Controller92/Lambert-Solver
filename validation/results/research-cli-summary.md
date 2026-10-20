# Research CLI validation summary

13 workflow tests passed with actual CUDA and real SPICE kernels enabled; 64 Python mathematical tests and 23 native integration tests remained passing.

Workflow tests cover every native candidate versus the Python reference on analytical states, fixed-axis versus geometric orientation, M>0 reduction, chunk tails, failure/no-fake-minimum behavior, C3 overflow, fractional/legacy configs, invalid settings, menu/help/batch dry runs, real SPICE cache integrity/coverage/lifetime, and saved-plot hash checks.

The final paper-preset batch took 3.486 seconds. Raw per-run timings and conditions are in research-cli-example-runs.json; stdout is in research-cli-paper-runs.txt.

Independent Gooding checks passed for 204 sampled paper-grid cells and all nine multi-revolution smoke cells. The real multi-revolution smoke selected an M=1 right-branch solution as its global minimum.

Explicit kernel downloads were tested from an empty directory and all pinned sizes/hashes matched. Previously loaded SPICE kernels are preserved and rejected rather than cleared or silently changing provenance.

Native numerical source and ABI were not modified by this increment. A stale Python docstring about native M>0 support was corrected. Full numerical grids/figures remain local outputs; only compact validation summaries and logs are tracked in Git.
