# CIRCUIT-Memory

Workspace for the **CIRCUIT** episodic-memory research project — adaptive confidence-weighted
multi-axis intersection retrieval, benchmarked on **EPBench** (the Tulving Episodic Memory Benchmark)
and compared against ARTEM/STEM and other memory systems.

## Contents
- **`EDA_EPBench_chi_tiet.md`** — detailed EDA of the EPBench dataset (4 universes, 27 books,
  event model `(t,s,e,c)`, task taxonomy, score landscape).
- **`episodic-memory-benchmark/`** — vendored copy of [ahstat/episodic-memory-benchmark](https://github.com/ahstat/episodic-memory-benchmark) (EPBench, ICLR 2025). See its `LICENSE`.
- **`A-mem-sys/`** — vendored copy of [WujiangXu/A-mem-sys](https://github.com/WujiangXu/A-mem-sys) (A-MEM, NeurIPS 2025). See its `LICENSE`.

> Third-party folders retain their original `LICENSE` files; they are included here as the
> systems CIRCUIT builds on / benchmarks against.
