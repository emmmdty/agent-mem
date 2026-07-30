<div align="center">

# agent-mem

**Learn Agent Memory from scratch — as an engineering system you can build, benchmark, and attack**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/content-CC%20BY--NC--SA%204.0-lightgrey.svg)](./LICENSE.txt)
[![Code License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](./LICENSE-CODE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

[Read online](https://emmmdty.github.io/agent-mem/) · [Roadmap](./ROADMAP.md) · [Contributing](./CONTRIBUTING.md) · [简体中文](./README.md)

</div>

> **Note**: the tutorial body is written in Simplified Chinese. This page is a summary for English readers.

## What this is

`agent-mem` is an engineering-first tutorial on **LLM agent memory**. Instead of cataloguing frameworks, it asks you to answer three questions by the end:

1. Which **representations** and which **operations** make up a memory system — and why is that decomposition more useful for writing code than the working/episodic/semantic/procedural split borrowed from cognitive psychology?
2. For the design you picked: how many tokens, milliseconds, and dollars does one write and one retrieval cost?
3. If someone poisons your memory store, what happens — and how would your tests catch it?

The book is built around **MiniMem**, a memory system written from scratch across the chapters: starting from a naive conversation buffer and growing into vector retrieval, an entity graph, bi-temporal facts, self-organizing notes, layered paging, a skill library, and an evaluation harness. Every chapter adds one module behind one stable interface.

## Framework: representation × operation × substrate

```
Representation              Operation                 Substrate
├─ Parametric               ├─ Encoding               ├─ Memory / files
│  (in weights, ch.9)       ├─ Consolidation          ├─ Vector store
├─ Contextual-unstructured  ├─ Indexing               ├─ Graph store
│  (raw text / summaries)   ├─ Retrieval              ├─ Relational store
└─ Contextual-structured    ├─ Updating               └─ Model weights
   (graphs / skills)        ├─ Forgetting
                            ├─ Compression
                            └─ Reflection
```

## Chapters

| # | Topic | MiniMem module | Compute |
| :-- | :--- | :--- | :--- |
| 1 | Why agents need memory | `MemoryStore`, `BufferMemory` | CPU |
| 2 | Context & long-context memory | `WindowMemory` | CPU / 1 GPU |
| 3 | Retrieval-augmented memory | `VectorMemory` | CPU |
| 4 | Structured & graph memory | `GraphMemory` | CPU |
| 5 | Temporal awareness & knowledge evolution | `TemporalGraphMemory` | CPU |
| 6 | Agentic & self-organizing memory | `AgenticMemory` | CPU + API |
| 7 | OS-style layered memory & scheduling | `LayeredMemory` | CPU + API |
| 8 | Experience & skill memory | `SkillMemory` | CPU + API |
| 9 | Parametric memory & continual learning | (comparison only) | **GPU** |
| 10 | Evaluation, security, and deployment | `EvalHarness` | CPU + API |

Everything except chapter 9 runs on CPU with a single LLM API key.

## On the numbers you'll read elsewhere

Vendor-reported scores on LoCoMo / LongMemEval contradict each other and are hard to reproduce; independent audits have found answer-key errors and lenient LLM judges in those benchmarks. This tutorial tags every third-party number with its provenance (📄 paper / 🏢 vendor claim / ⚠️ disputed / 🕐 time-sensitive) and never uses a disputed number as a conclusion. See chapter 10.

## Quick start

```bash
git clone https://github.com/emmmdty/agent-mem.git
cd agent-mem
python -m venv .venv && source .venv/bin/activate
pip install -e .
python docs/chapter1/code/01_goldfish_agent.py
```

## License

Content: [CC BY-NC-SA 4.0](./LICENSE.txt). Code: [MIT](./LICENSE-CODE).
