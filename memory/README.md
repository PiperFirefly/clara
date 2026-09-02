# Agent's Memory Store (Stage 1)

CoALA split: `facts` (structured, exact lookup) + `memories` (episodic/semantic, vector-indexed).

- **Storage:** SQLite at `memory.db`
- **Embeddings:** `fastembed` → `BAAI/bge-small-en-v1.5` (384-dim), ONNX runtime, no torch
- **Retrieval:** brute-force cosine (numpy) — fine up to tens of thousands of memories; add an ANN index (sqlite-vec / lancedb) when scale demands.

## Usage

```bash
source ~/venvs/memory/bin/activate
python memstore.py remember "text" --kind episodic [--importance 0.5]
python memstore.py recall "query" --k 5
python memstore.py associate "query" --k 3
python memstore.py facts set <key> <value>
python memstore.py facts get <key>
python memstore.py stats
python memstore.py seed    # (re)seed identity
```

## Roadmap
- **Stage 2:** pi extension — `remember`/`recall` tools + `resources_discover` injection + `session_before_compact` consolidation.
- **Stage 3:** HippoRAG-style knowledge graph + Personalized PageRank for associative recall.
- **Stage 4:** background consolidation micro-agent (dedup/evaluate/verify/enhance + forgetting-curve).
- **Stage 5:** the hive — multiple micro-LLMs, one per memory function.
