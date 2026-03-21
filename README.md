# Live Knowledge Graphs

Runtime knowledge graph infrastructure for AI agent systems. Unlike static knowledge bases that are frozen at training time, this system maintains a live, mutable graph that agents query and update during inference.

The core engine is **cc-graph-kernel**: a deterministic context slicer written in Rust that answers one question:

> Given a target turn in a conversation DAG, which other turns are *allowed to influence meaning*?

## Results

Evaluated on 39 real agent conversations (5,000 turns), with a knowledge graph containing **71,130 triples** across 11 domains:

| Metric | Value | What it measures |
|--------|-------|------------------|
| Pairwise path accuracy | **81.0%** | Valid KG paths scored higher than hard negatives |
| Cohen's d | **2.23** | Effect size (large = clear separation) |
| Valid path mean score | 6.44 | Average reward for structurally valid paths |
| Hard negative mean | 1.63 | Average reward for shuffled/corrupted paths |
| Recovery margin | 0.87 | How well slices recover from degraded context |

Full evaluation data in [`results/evaluation-v2.json`](results/evaluation-v2.json).

## Architecture

```
                    +------------------+
                    |  Anchor Turn ID  |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   SlicePolicyV1   |
                    |  (max_nodes=256,  |
                    |   max_radius=10,  |
                    |   phase_weights)  |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v----------+       +----------v---------+
    |   Context Slicer   |       |   Knowledge DB     |
    |   (BFS expansion,  |       |   (triple store,   |
    |    priority queue,  |       |    multi-hop BFS,  |
    |    phase scoring)   |       |    entity aliases) |
    +---------+----------+       +----------+---------+
              |                             |
    +---------v----------+       +----------v---------+
    |   Graph Store      |       |   Query Cache      |
    |   (Postgres/SQLite/ |       |   (Moka, LRU,     |
    |    In-Memory)       |       |    TTL eviction)   |
    +---------+----------+       +--------------------+
              |
    +---------v----------+
    |   SliceExport      |
    |  +fingerprint      |
    |  +admissibility    |
    |   token (HMAC)     |
    +--------------------+
```

### Core Components

**Context Slicer** (`slicer.rs`, 395 lines). BFS expansion from an anchor turn through the conversation DAG. Uses a priority queue weighted by phase importance, salience scores, and distance decay. Produces a deterministic slice: same anchor + same policy + same graph state = identical output, byte-for-byte.

**Admissibility Tokens** (`types/admissible.rs`, `types/slice.rs`). HMAC-SHA256 tokens that cryptographically bind a slice to its construction parameters. Downstream services verify that a context window was produced by an authorized slicer with a specific policy, without needing access to the signing secret.

**TurnSnapshot DAG** (`types/turn.rs`, `store/`). The conversation graph where nodes are turns (user/assistant messages with phase labels, salience scores, content hashes) and edges are parent-child relationships. Backed by PostgreSQL, SQLite, or in-memory stores.

**Knowledge DB** (`store/knowledge_db.rs`, `store/knowledge_db_pg.rs`, `store/knowledge_db_sqlite.rs`). Triple store (subject-predicate-object) with confidence scores, multi-hop BFS traversal, entity alias resolution, and D3/Mermaid/DOT visualization export.

**Atlas** (`atlas/`). Batch operations over the graph: parallel slicing across multiple anchors, overlap analysis (Jaccard similarity between slices), influence scoring (which turns appear in the most slices), and deterministic bundle export with schema versioning.

**Policy Engine** (`policy/`). Phase-weighted priority scoring with quantized float hashing for cross-platform determinism. Policies are registered at runtime, versioned, and their parameter hashes are included in admissibility tokens.

### Module Map

```
crates/graph-kernel/src/
  lib.rs                    # Public API surface, re-exports
  canonical.rs              # Deterministic serialization (xxHash64)
  canonical_content.rs      # Content normalization + hashing
  slicer.rs                 # Core BFS context slicer
  atlas/
    mod.rs                  # Atlas orchestration
    batch_slicer.rs         # Parallel multi-anchor slicing
    bundler.rs              # Deterministic bundle export
    influence.rs            # Turn influence scoring
    overlap.rs              # Slice overlap analysis (Jaccard)
    snapshot.rs             # Graph state snapshots
  policy/
    mod.rs                  # Policy registry
    v1.rs                   # SlicePolicyV1 with phase weights
    scoring.rs              # Priority score computation
  service/
    mod.rs                  # Axum service setup
    routes.rs               # HTTP route handlers
    state.rs                # Shared service state
    cache.rs                # Moka query cache
    middleware.rs            # CORS, tracing, timeouts
    normalize.rs            # Entity name normalization
    knowledge_handlers.rs   # Knowledge CRUD + traversal
    visualization.rs        # D3, Mermaid, DOT export
  store/
    mod.rs                  # Store trait definitions
    memory.rs               # In-memory graph store
    postgres.rs             # PostgreSQL graph store
    sqlite.rs               # SQLite graph store
    knowledge_db.rs         # Knowledge triple types
    knowledge_db_dyn.rs     # Dynamic dispatch wrapper
    knowledge_db_pg.rs      # PostgreSQL knowledge backend
    knowledge_db_sqlite.rs  # SQLite knowledge backend
  types/
    mod.rs                  # Core type definitions
    turn.rs                 # TurnSnapshot, TurnId, Role, Phase
    edge.rs                 # Edge, EdgeType
    slice.rs                # SliceExport, Fingerprint, HMAC tokens
    admissible.rs           # Evidence bundles, verification
    verification.rs         # Token verifier with LRU cache
    boundary.rs             # Slice boundary guards
    sufficiency.rs          # Diversity metrics, salience stats
    provenance.rs           # Replay provenance chain
    incident.rs             # Quarantine, incident tracking
    pool.rs                 # Turn pool management
    task_ticket.rs          # Task routing tickets
```

## API Endpoints

The service exposes a REST API on port 8001 (configurable via `PORT`):

### Context Slicing

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/slice` | Construct a context slice around an anchor turn |
| POST | `/api/slice/batch` | Batch slice multiple anchors |
| POST | `/api/verify_token` | Verify an admissibility token |

### Knowledge Graph

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/knowledge` | Query triples with filters |
| POST | `/api/knowledge` | Add a single triple |
| POST | `/api/knowledge/batch` | Batch insert triples |
| DELETE | `/api/knowledge` | Delete triples by filter |
| GET | `/api/knowledge/stats` | Graph statistics |
| GET | `/api/knowledge/aliases` | Entity alias lookup |
| POST | `/api/knowledge/traverse` | Multi-hop BFS traversal |

### Visualization

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/knowledge/graph` | D3-compatible JSON |
| GET | `/api/knowledge/graph.mermaid` | Mermaid diagram |
| GET | `/api/knowledge/graph.dot` | Graphviz DOT |

### Policy Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/policies` | List registered policies |
| POST | `/api/policies` | Register a new policy |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Full health with DB status |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe |
| GET | `/health/startup` | Startup probe |

## Quick Start

### Build the library

```bash
# Default features (no database, no HTTP service)
cargo build

# With PostgreSQL backend + HTTP service
cargo build --features service

# With SQLite backend + HTTP service
cargo build --features service-sqlite
```

### Run the service

```bash
export DATABASE_URL="postgres://user:pass@localhost:5432/graphkernel"
export KERNEL_HMAC_SECRET="your-secret-key"
export PORT=8001

cargo run --features service
```

### Run tests

```bash
cargo test
# 158 tests: 131 unit + 14 integration + 13 golden + 2 doc-tests
```

### Python client

```python
from python.kg_client import KGClient

kg = KGClient("http://localhost:8001")

# Add knowledge
kg.add_triple("spore", "built_with", "swiftui", confidence=0.95)

# Query
results = kg.query(subject="spore")

# Multi-hop traversal
paths = kg.traverse("spore", max_hops=3)
for path in paths:
    print(" -> ".join(path.entities))
```

### Path reward scoring

```python
from python.kg_reward import KGPath, PathEdge, compute_path_reward

path = KGPath(edges=[
    PathEdge("spore", "built_with", "swiftui", confidence=0.95),
    PathEdge("swiftui", "is_a", "framework", confidence=0.99),
])

reward = compute_path_reward(path)  # Higher = better trajectory
```

## Project Structure

```
live-knowledge-graphs/
  Cargo.toml                    # Workspace root
  LICENSE                       # MIT
  README.md                     # This file
  paper/
    paper.md                    # Research paper (765 lines)
  results/
    evaluation-v2.json          # Evaluation on 39 conversations
  crates/
    graph-kernel/               # Full Rust implementation
      Cargo.toml                # Crate manifest with feature flags
      src/                      # 43 Rust source files, ~14K lines
      benches/                  # Criterion benchmarks
      tests/                    # Integration + golden tests
  python/
    kg_client.py                # Zero-dependency Python client
    kg_reward.py                # KG path reward function
  examples/
    traverse.py                 # BFS traversal + scoring example
```

**55 files. 14,932 lines of Rust. 763 lines of Python. 158 tests.**

## Feature Flags

| Flag | Dependencies Added | What It Enables |
|------|--------------------|-----------------|
| `postgres` | sqlx, tokio | PostgreSQL graph store |
| `sqlite` | sqlx/sqlite, tokio | SQLite graph store |
| `service` | axum, tower, tower-http, moka + postgres | Full HTTP service |
| `service-sqlite` | axum, tower, tower-http, moka + sqlite | HTTP service with SQLite |

## Determinism Guarantees

Every operation in the graph kernel is deterministic:

1. **Canonical serialization**: All types serialize to canonical JSON (sorted keys via serde). Hashed with xxHash64 for fingerprints.
2. **Quantized floats**: Policy parameters use integer quantization (multiply by 1e6, round to i64) before hashing, ensuring identical hashes across Rust and Python.
3. **Ordered traversal**: BFS uses a deterministic priority queue. Turn ordering is canonical (by TurnId). Edge ordering is canonical (parent, child).
4. **HMAC binding**: Admissibility tokens bind slice_id + anchor_id + policy_id + policy_params_hash + graph_snapshot_hash + schema_version. Any change invalidates the token.

## License

MIT
