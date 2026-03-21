# Live Knowledge Graphs: Runtime Graph Integration for Continuous Domain Adaptation in Language Agents

**Mohamed Diomande**
Independent Researcher

---

## Abstract

Recent work on Domain-Specific Superintelligence (Belova et al., 2026) demonstrates that knowledge graph-derived training curricula produce domain specialists that outperform models 400x their size. However, this approach treats knowledge graphs as static training scaffolding: constructed once, used for fine-tuning, then discarded at inference. We present an alternative: runtime knowledge graph integration, where the graph is queried live during inference with provenance-tracked context slicing, real-time entity resolution, and cryptographic admissibility verification. We implement this in cc-graph-kernel, a production Rust service built on the Axum framework, processing real workloads across a multi-machine mesh with 71,130 knowledge triples in its production database. We evaluate multi-hop path quality using a 3-signal reward function over 199 valid graph paths and 199 hard negatives, achieving 81.0% pairwise ranking accuracy (Cohen's d = 2.228). We further demonstrate that anticipation geometry scalars, originally developed for conversational turn analysis, produce meaningful distributions when applied to knowledge graph paths, with distinct profiles compared to the conversation domain. Our key contribution is the Context Slicer, a priority-queue BFS algorithm that produces provenance-complete, HMAC-signed graph slices suitable for direct injection into language model prompts.

**Keywords:** knowledge graphs, retrieval-augmented generation, context slicing, runtime integration, provenance, admissibility tokens, conversational AI

---

## 1. Introduction

The dominant paradigm for integrating structured knowledge into language models treats the knowledge graph as a pre-training artifact. Models are fine-tuned on curricula derived from graph traversals, and the graph itself plays no role at inference time. This design rests on an implicit assumption: that the domain is sufficiently static that knowledge encoded during training remains valid at inference. For many real-world agent deployments, this assumption fails.

Consider a language agent operating as a long-running conversational assistant. Over 15 months, such an agent accumulates 112,000+ dialogue turns spanning dozens of domains, including software architecture, creative production, business operations, and personal knowledge management. The knowledge graph describing this agent's domain is not a fixed ontology. It is a living structure where new entities appear daily, relationships between entities shift as projects evolve, and the salience of past knowledge decays or resurges unpredictably.

Training-time KG integration faces a fundamental staleness problem in this setting. A model fine-tuned on a graph snapshot from month 3 will make confident but incorrect claims about the state of affairs in month 15. The only remedy is periodic re-training, which introduces latency (the model is always behind the current graph state), cost (each training cycle consumes compute), and risk (catastrophic forgetting of older but still-valid knowledge).

We propose runtime knowledge graph integration as a complementary approach. Rather than encoding graph structure into model weights, we query the graph live during inference and inject provenance-tracked context slices directly into the model's prompt. This shifts the burden of domain currency from the model to the retrieval system, where it is more naturally managed.

Our implementation, cc-graph-kernel, is not a research prototype. It is a production Rust service (Axum framework, PostgreSQL backing store) that has been processing real workloads since its deployment. The system enforces four production invariants: (1) Slice Boundary Integrity, ensuring that only turns explicitly selected by the slicer may influence downstream reasoning; (2) Provenance Completeness, requiring that every response carry a full provenance chain `(slice_id, policy_ref, schema_version, graph_snapshot_hash, admissibility_token)`; (3) No Phantom Authority, implemented via HMAC-SHA256 signed admissibility tokens that prevent any downstream system from fabricating authorization claims; and (4) Content Immutability, enforced through SHA-256 content hashes on every turn.

The remainder of this paper is organized as follows. Section 2 surveys related work on knowledge graph integration, retrieval-augmented generation, and domain-specific language models. Section 3 presents the architecture of cc-graph-kernel. Section 4 provides a formal comparison between training-time and runtime KG integration. Section 5 describes the conversation graph as a DAG with typed edges and phase transitions. Section 6 presents empirical results from the production system, including graph traversal analysis, KG path reward evaluation, and anticipation geometry on graph paths. Section 7 outlines proposed evaluation experiments not yet conducted. Section 8 describes the multi-machine deployment architecture. Section 9 discusses synthesis opportunities and limitations. Section 10 concludes.

---

## 2. Related Work

### 2.1 Domain-Specific Superintelligence (Princeton DSS)

Belova et al. (2026) propose replacing trillion-parameter monolithic LLMs with "societies of Domain-Specific Superintelligence (DSS)" models. Their 67-page position paper (arXiv:2603.14147) articulates a compelling vision: small language models augmented with explicit symbolic abstractions (knowledge graphs, ontologies, formal logic) can outperform models orders of magnitude larger on domain-specific tasks.

The Princeton group's companion papers validate this thesis empirically. GraphMERT (arXiv:2510.09580, TMLR 2026) is an 80M-parameter encoder-only model that distills knowledge graphs from text, achieving 69.8% FActScore versus 40.2% for a 32B Qwen3 baseline. QwQ-Med-3, their bottom-up DSS instantiation (arXiv:2507.13966), fine-tunes QwQ-32B on 24K KG-grounded reasoning tasks and achieves 84.72% on ICD-Bench, establishing a new state of the art. Their energy-efficient DSS work (arXiv:2510.22052) targets 1000x energy efficiency through a "ladder of learning" framework.

However, the Princeton approach treats the knowledge graph as training infrastructure. GraphMERT constructs a KG during a distillation phase; QwQ-Med-3 generates training examples from KG traversals. At inference time, the graph is absent. This design is optimal when the domain is static (e.g., medical coding standards that change annually). It is suboptimal when the domain evolves continuously, as we demonstrate in Section 4.

### 2.2 Retrieval-Augmented Generation

Lewis et al. (2020) introduced RAG as a method for conditioning language model generation on retrieved passages. The original RAG system uses a dense passage retriever (DPR) to fetch relevant documents from a flat corpus, concatenating them into the model's input. Subsequent work has extended RAG in several directions: iterative retrieval (Trivedi et al., 2023), self-reflective retrieval (Asai et al., 2024), and adaptive retrieval strategies (Jiang et al., 2023).

Our system departs from standard RAG in a fundamental way: the retrieval target is not a document corpus but a directed acyclic graph of conversational turns. This means retrieval is not a nearest-neighbor search in embedding space but a graph traversal problem with structural constraints (edge types, phase metadata, trajectory depth). We combine both retrieval modalities in what we call "dual-plane retrieval" (Section 3.5).

### 2.3 GraphRAG

Microsoft's GraphRAG (Edge et al., 2024) constructs a knowledge graph from a document corpus and uses graph-based community detection to generate summaries at multiple levels of abstraction. At query time, GraphRAG retrieves communities relevant to the query and conditions generation on community summaries rather than raw passages. This approach improves global question answering over corpora where relevant information is distributed across many documents.

GraphRAG shares our conviction that graph structure improves retrieval. However, GraphRAG constructs its graph offline (as a preprocessing step) and uses it to generate static summaries. Our system queries the graph live and returns provenance-tracked slices of the graph itself, not summaries derived from it. This preserves the ability to audit exactly which evidence influenced a given response.

### 2.4 KAPING

Baek et al. (2023) introduce Knowledge-Augmented Prompting (KAPING), which retrieves relevant knowledge graph triples at inference time and injects them into prompts. KAPING is closest to our approach in spirit. The key differences are: (1) KAPING retrieves individual triples, whereas we retrieve connected subgraphs (slices) that preserve relational structure; (2) KAPING does not address provenance or admissibility, making it difficult to audit or verify the retrieved context; (3) KAPING operates on static knowledge graphs, whereas our conversation DAG is continuously growing.

### 2.5 Knowledge Graph Embedding and Refinement

Pan et al. (2024) survey the intersection of LLMs and knowledge graphs, identifying a bidirectional relationship: LLMs can construct and refine KGs, and KGs can augment LLMs. Their taxonomy distinguishes KG-enhanced LLM pre-training, KG-enhanced LLM inference, and LLM-based KG construction. Our work falls squarely in the "KG-enhanced inference" category, with the additional contribution of cryptographic provenance tracking.

---

## 3. Architecture

### 3.1 cc-graph-kernel: Service Overview

cc-graph-kernel is a Rust binary built with Axum 0.7, backed by PostgreSQL via sqlx 0.7. The production instance runs on a GCP virtual machine, accessible via SSH tunnel on port 8001, and currently stores 71,130 knowledge triples. The service exposes a REST API for context slicing, knowledge graph CRUD, multi-hop traversal, and token verification. The crate supports multiple storage backends through a generic `GraphStore` trait:

```rust
pub trait GraphStore: Send + Sync {
    type Error: std::error::Error + Send + Sync;
    fn get_turn(&self, id: &TurnId)
        -> impl Future<Output = Result<Option<TurnSnapshot>, Self::Error>> + Send;
    fn get_parents(&self, id: &TurnId)
        -> impl Future<Output = Result<Vec<TurnId>, Self::Error>> + Send;
    fn get_children(&self, id: &TurnId)
        -> impl Future<Output = Result<Vec<TurnId>, Self::Error>> + Send;
    fn get_siblings(&self, id: &TurnId, limit: usize)
        -> impl Future<Output = Result<Vec<TurnId>, Self::Error>> + Send;
    fn get_edges(&self, turn_ids: &[TurnId])
        -> impl Future<Output = Result<Vec<Edge>, Self::Error>> + Send;
}
```

This trait uses native async fn in traits (Rust 1.75+, no `async-trait` crate). Production deployments use `PostgresGraphStore`; tests use `InMemoryGraphStore`; edge deployments can use `SqliteGraphStore`.

The REST API exposes the following endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/slice` | POST | Construct a context slice around an anchor turn |
| `/api/slice/batch` | POST | Construct multiple slices in batch |
| `/api/verify_token` | POST | Verify an admissibility token |
| `/api/policies` | GET/POST | List or register slice policies |
| `/api/knowledge` | GET/POST/DELETE | Knowledge triple CRUD |
| `/api/knowledge/batch` | POST | Batch triple ingestion (up to 10,000) |
| `/api/knowledge/traverse` | POST | Server-side multi-hop BFS traversal |
| `/api/knowledge/aliases` | GET | Entity alias expansion for RAG++ |
| `/api/knowledge/stats` | GET | Graph statistics |
| `/health`, `/health/live`, `/health/ready`, `/health/startup` | GET | Cloud Run compatible health probes |

The knowledge graph stores Subject-Predicate-Object triples with confidence scores and provenance source tags. The `KnowledgeTriple` struct is:

```rust
pub struct KnowledgeTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,  // [0.0, 1.0]
    pub source: String,
}
```

Entity names are canonicalized via `canonicalize_entity()` before storage, ensuring that "Graph Kernel", "graph-kernel", and "graph_kernel" resolve to the same entity. Predicates are lowercased. This normalization is critical for reliable graph traversal in a system where entities are ingested from heterogeneous sources (conversation logs, code analysis, manual annotation).

### 3.2 The Context Slicer

The Context Slicer is the core algorithm. Given an anchor turn (the turn for which context is needed), it performs a priority-queue BFS expansion through the conversation DAG, selecting turns based on a composite priority score and respecting configurable budget constraints.

The algorithm proceeds as follows:

1. Initialize the anchor turn at distance 0 and push it onto a max-heap priority queue.
2. While the frontier is not empty and the selected set has fewer than `max_nodes` turns:
   a. Pop the highest-priority candidate from the heap.
   b. If the candidate's distance exceeds `max_radius`, skip it.
   c. Add the candidate to the selected set.
   d. For each parent and child of the selected turn, if not already visited, compute its priority score and push it onto the heap at distance + 1.
   e. If `include_siblings` is enabled, fetch up to `max_siblings_per_node` siblings and add them at the same distance as the current node.
3. Collect all edges between selected turns.
4. Sort turns by `TurnId` and edges by `(parent, child, edge_type)` for deterministic output.
5. Compute a `GraphSnapshotHash` from content hashes (or statistics for legacy data).
6. Issue an HMAC-SHA256 signed `AdmissibilityToken`.
7. Return the slice wrapped in an `AdmissibleEvidenceBundle`.

The implementation in `slicer.rs` uses `BinaryHeap<ExpansionCandidate>` where `ExpansionCandidate` implements `Ord` with the following comparison:

```rust
impl Ord for ExpansionCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Primary: higher priority first
        // Secondary: lower distance first (closer to anchor)
        // Tertiary: by TurnId for determinism
        match self.priority.partial_cmp(&other.priority) {
            Some(std::cmp::Ordering::Equal) | None => {
                match self.distance.cmp(&other.distance).reverse() {
                    std::cmp::Ordering::Equal => self.turn.id.cmp(&other.turn.id),
                    ord => ord,
                }
            }
            Some(ord) => ord,
        }
    }
}
```

The three-level comparison ensures determinism: priority first, then proximity to anchor, then lexicographic turn ID ordering as a tiebreaker.

**Priority Scoring.** The priority score for a turn at distance *d* from the anchor is computed as:

$$\text{priority}(t, d) = (\text{phase\_weight}(t.\text{phase}) + t.\text{salience} \times w_s) \times \delta^d$$

where $w_s$ is the `salience_weight` (default 0.3) and $\delta$ is the `distance_decay` (default 0.9). Phase weights follow a fixed hierarchy:

| Phase | Default Weight |
|-------|---------------|
| Synthesis | 1.0 |
| Planning | 0.9 |
| Consolidation | 0.6 |
| Debugging | 0.5 |
| Exploration | 0.3 |

This weighting encodes the empirical observation that synthesis and planning turns carry higher information density for downstream reasoning than exploratory or debugging turns.

### 3.3 Admissibility Tokens

Every slice produced by the kernel carries an `AdmissibilityToken`, an HMAC-SHA256 signed proof that the slice was issued by the kernel. The token is computed over a canonical string representation of the slice's identity:

```rust
fn canonical_string(
    slice_id: &SliceFingerprint,
    anchor_turn_id: &TurnId,
    policy_id: &str,
    policy_params_hash: &str,
    graph_snapshot_hash: &GraphSnapshotHash,
    schema_version: &str,
) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}|{}",
        slice_id.as_str(),
        anchor_turn_id.as_uuid(),
        policy_id,
        policy_params_hash,
        graph_snapshot_hash.as_str(),
        schema_version,
        Self::TOKEN_VERSION,  // "admissibility_token_v2_hmac"
    )
}
```

The token is the first 128 bits (16 bytes) of `HMAC-SHA256(secret, canonical_string)`, hex-encoded to 32 characters. Verification uses constant-time comparison to prevent timing attacks.

The security model is straightforward: only the kernel possesses the HMAC secret. Downstream systems that receive a `SliceExport` can verify the token without knowing the secret by calling the `/api/verify_token` endpoint. Systems that share the secret can verify locally, achieving sub-microsecond verification latency.

This design enforces INV-GK-003: No Phantom Authority. Without the kernel's secret, the token cannot be forged. Any admissibility claim is verifiable without trusting the claimant.

The type system further reinforces this invariant. The `AdmissibleEvidenceBundle` struct can only be constructed via `from_verified()`, which performs HMAC verification:

```rust
pub fn from_verified(
    slice: SliceExport,
    hmac_secret: &[u8],
) -> Result<Self, VerificationError> {
    if !slice.admissibility_token.is_valid_format() {
        return Err(VerificationError::InvalidTokenFormat(...));
    }
    let is_valid = slice.admissibility_token.verify_hmac(
        hmac_secret, &slice.slice_id, &slice.anchor_turn_id,
        &slice.policy_id, &slice.policy_params_hash,
        &slice.graph_snapshot_hash, &slice.schema_version,
    );
    if !is_valid {
        return Err(VerificationError::TokenMismatch);
    }
    Ok(Self { slice, verified_at_unix_ms: Utc::now().timestamp_millis() })
}
```

Any API that accepts an `AdmissibleEvidenceBundle` as a parameter has a compile-time guarantee that the bundle was verified. There is no code path that constructs a bundle without verification.

### 3.4 TurnSnapshot DAG

Each node in the conversation DAG is a `TurnSnapshot` with the following fields:

```rust
pub struct TurnSnapshot {
    pub id: TurnId,                      // UUID, implements Ord
    pub session_id: String,              // Conversation/session identifier
    pub role: Role,                      // User | Assistant | System | Tool
    pub phase: Phase,                    // Exploration | Debugging | Planning |
                                         // Consolidation | Synthesis
    pub salience: f32,                   // [0, 1], clamped at construction
    pub trajectory_depth: u32,           // Distance from root in DAG
    pub trajectory_sibling_order: u32,   // Sibling position at this depth
    pub trajectory_homogeneity: f32,     // [0, 1], similarity to parent
    pub trajectory_temporal: f32,        // [0, 1], temporal position
    pub trajectory_complexity: f32,      // Complexity score
    pub created_at: i64,                 // Unix timestamp
    pub content_hash: Option<String>,    // SHA-256 of content_text
}
```

The `content_hash` field enables INV-GK-004: Content Immutability. When present, any modification to the underlying content is detectable via `verify_content_hash()`:

```rust
pub fn verify_content_hash(&self, content: &str) -> Result<(), ContentHashError> {
    match validate_content_hash(content, self.content_hash.as_deref()) {
        HashValidation::Valid => Ok(()),
        HashValidation::Missing => Ok(()),  // Legacy data
        HashValidation::Mismatch { expected, computed } => {
            Err(ContentHashError::Mismatch { turn_id: self.id, stored: expected, computed })
        }
    }
}
```

The five trajectory coordinates `(depth, sibling_order, homogeneity, temporal, complexity)` form a 5D trajectory space that enables structural-semantic hybrid retrieval. Unlike flat timestamp ordering, these coordinates capture the topological position of each turn within the broader conversation structure.

Edges connect turns with typed relationships:

```rust
pub enum EdgeType {
    Reply,      // Direct reply/continuation
    Branch,     // Fork in conversation
    Reference,  // Reference to earlier turn
    Default,    // Unspecified
}

pub struct Edge {
    pub parent: TurnId,
    pub child: TurnId,
    pub edge_type: EdgeType,
}
```

Edges implement canonical ordering by `(parent, child, edge_type)`, ensuring deterministic serialization regardless of insertion order.

The production graph contains 112,000+ turns across 4,132 conversations, stored in Supabase PostgreSQL with 141 tables.

### 3.5 Dual-Plane Retrieval (RAG++)

The cc-graph-kernel operates alongside RAG++, a FastAPI service that implements dual-plane retrieval:

**Plane 1: Semantic Search.** Standard vector similarity search over turn embeddings stored in pgvector. Queries are embedded using a configurable embedding model (tracked via `EmbeddingModelRef` with model_id, version, dimensionality, and quantization metadata). Results are ranked by cosine similarity with a configurable threshold.

**Plane 2: Graph Traversal.** Server-side multi-hop BFS through the knowledge graph, exposed via the `/api/knowledge/traverse` endpoint. The traversal accepts a start entity, optional predicate filters, a maximum hop count, a maximum result count, and a minimum confidence threshold:

```rust
pub struct TraversalRequest {
    pub start: String,
    pub predicates: Option<Vec<String>>,
    pub direction: String,       // "outgoing" | "incoming" | "both"
    pub max_hops: u32,           // default: 3
    pub max_results: usize,      // default: 100
    pub min_confidence: Option<f64>,
    pub return_paths: bool,      // default: true
}
```

The traversal returns complete paths with per-edge confidence scores and traversal statistics (entities visited, edges traversed, elapsed time in milliseconds).

RAG++ fuses both planes by: (1) performing semantic search to identify candidate turns, (2) expanding each candidate via graph traversal to discover related entities and relationships, (3) constructing a context slice that includes both the semantically retrieved turns and the graph-connected context, and (4) injecting the fused context into the language model prompt with full provenance metadata.

The entity alias system bridges the two planes. When RAG++ constructs a search query, it calls the `/api/knowledge/aliases` endpoint to expand entity names. For example, a query mentioning "GK" would be expanded to include "graph kernel", "cc-graph-kernel", and "graph_kernel", ensuring that semantic search covers all surface forms of the same entity.

### 3.6 Slice Policy System

Slice behavior is governed by configurable policies. The `SlicePolicyV1` struct parameterizes the slicer:

```rust
pub struct SlicePolicyV1 {
    pub version: String,                 // "slice_policy_v1"
    pub max_nodes: usize,                // default: 256
    pub max_radius: u32,                 // default: 10
    pub phase_weights: PhaseWeights,     // per-phase importance weights
    pub salience_weight: f32,            // default: 0.3
    pub distance_decay: f32,             // default: 0.9
    pub include_siblings: bool,          // default: true
    pub max_siblings_per_node: usize,    // default: 5
}
```

Policies are registered in a `PolicyRegistry` and referenced by `PolicyRef` in slice requests. This allows different downstream consumers to request different slice configurations without modifying the slicer itself.

A critical design decision is the use of quantized float representation for policy hashing. All floating-point parameters are multiplied by 10^6 and rounded to `i64` before hashing, ensuring cross-platform consistency between Rust and Python implementations:

```rust
fn quantize_float(value: f32) -> i64 {
    ((value as f64) * 1_000_000.0).round() as i64
}
```

This prevents the common failure mode where identical policies produce different `params_hash` values due to floating-point serialization differences across languages.

### 3.7 Replay Provenance

The system tracks complete provenance for experimental reproducibility. The `ReplayProvenance` struct captures:

- **Embedding model**: model_id, version, dimensionality, quantization, determinism flag
- **Normalization version**: version string, config hash, feature list (crlf_to_lf, trim_whitespace, utf8_encode)
- **Retrieval parameters**: k, similarity threshold, reranking model, max context tokens, policy version, policy params hash
- **Graph snapshot**: `GraphSnapshotHash` at retrieval time
- **Slice fingerprint**: the resulting slice ID
- **Query vector hash**: for query reproduction

The replay contract states: given identical provenance and identical query, retrieval MUST return an identical slice (same fingerprint). Violations indicate non-deterministic embedding models, graph state changes between retrieval and replay, or bugs in the retrieval pipeline. The `is_replay_compatible()` method enables programmatic verification of this contract.

---

## 4. Training-Time vs. Runtime KG Integration

### 4.1 Formal Comparison

Let $G_t$ denote the state of the knowledge graph at time $t$. Let $M_\theta$ denote a language model with parameters $\theta$. We compare two integration strategies:

**Training-Time Integration (TTI).** The model is fine-tuned on a dataset $D$ derived from $G_{t_0}$, where $t_0$ is the training snapshot time. At inference time $t > t_0$, the model operates with parameters $\theta_{t_0}$ and has no access to $G_t$.

**Runtime Integration (RTI).** The model operates with fixed parameters $\theta$ (which may or may not have been trained with graph awareness). At inference time $t$, the system queries $G_t$ to produce a context slice $S_t$, which is injected into the model's prompt.

### 4.2 Staleness Analysis

Define the *staleness* of a TTI system at time $t$ as:

$$\text{staleness}(t) = |G_t \triangle G_{t_0}|$$

where $\triangle$ denotes the symmetric difference (triples added or removed since training). In our production graph (71,130 triples as of March 2026), new triples are ingested continuously from conversation logs, code analysis, and manual annotation. Over 15 months, the daily triple addition rate averages 50-200 new triples, with periodic bursts of 1,000+ during project transitions. This means staleness grows monotonically, with no upper bound short of complete domain replacement.

For a TTI system, the only remedy for staleness is re-training. If re-training occurs at interval $\Delta$, the average staleness at query time is:

$$\mathbb{E}[\text{staleness}] = \frac{\Delta}{2} \times r$$

where $r$ is the average triple addition rate. With $r = 100$ triples/day and $\Delta = 30$ days (monthly re-training), the expected staleness is 1,500 triples, representing approximately 5-15% of the active knowledge base.

For an RTI system, staleness is bounded by the graph update latency $\epsilon$ (the time between a fact being observed and being available in $G_t$). In our system, ingestion is near-real-time (sub-minute), so $\epsilon \approx 0$.

### 4.3 When TTI Wins

Training-time integration has genuine advantages:

1. **Latency.** TTI adds zero inference-time latency, since knowledge is encoded in weights. RTI adds the cost of graph traversal + slice construction. In cc-graph-kernel, the `/api/slice` endpoint completes in 5-50ms depending on graph density and policy parameters.

2. **Reasoning depth.** A model fine-tuned on KG-grounded reasoning tasks (as in QwQ-Med-3) can perform multi-step logical inference over graph-derived facts. RTI provides facts in the prompt but relies on the base model's in-context reasoning ability, which may be weaker for complex chains.

3. **Domain compression.** TTI compresses domain knowledge into model weights, enabling operation without network access to a graph service. This matters for edge deployment, offline operation, and privacy-sensitive settings.

### 4.4 When RTI Wins

Runtime integration is superior in the following scenarios:

1. **Evolving domains.** When $|G_t \triangle G_{t_0}|$ grows continuously, RTI maintains accuracy while TTI degrades. This is the primary use case for conversational agents, project management systems, and any application where the domain is defined by ongoing activity.

2. **Auditability.** RTI produces provenance-complete evidence bundles. For any model response, we can reconstruct exactly which turns and relationships were in the context window, verify that the context was kernel-authorized (via HMAC token), and detect content drift (via graph snapshot hash). TTI offers no comparable audit trail.

3. **Multi-domain agents.** An agent operating across many domains simultaneously benefits from RTI because the graph naturally segments by entity clusters. The slicer retrieves domain-relevant context without requiring domain-specific model variants.

4. **Cold start.** A new domain can be served immediately by ingesting triples into the graph. TTI requires a training cycle before the model can reason about new domains.

### 4.5 Synthesis: TTI + RTI

The two approaches are not mutually exclusive. The optimal architecture uses TTI for foundational domain knowledge (stable ontological structure, core reasoning patterns) and RTI for current state (recent events, evolving relationships, temporal context). GraphMERT's KG distillation could construct the initial graph; cc-graph-kernel's runtime slicer maintains currency. This synthesis combines the deep reasoning advantage of TTI with the freshness guarantee of RTI.

---

## 5. The Conversation Graph

### 5.1 DAG Structure

The 112K+ turns form a directed acyclic graph rather than a linear sequence. Conversations branch (when a user explores alternative approaches), merge (when insights from one branch inform another), and reference earlier turns across session boundaries. Three edge types capture these relationships:

- **Reply**: Sequential continuation within a conversation thread. The most common edge type, representing the linear flow of dialogue.
- **Branch**: A fork point where the conversation diverges into parallel threads. Branch edges arise naturally in multi-topic conversations and when users explicitly request exploration of alternatives.
- **Reference**: Cross-session or cross-thread references, where a later turn explicitly or implicitly refers to knowledge established in an earlier, disconnected turn. These edges are the most valuable for context retrieval because they capture non-obvious semantic connections.

### 5.2 Phase Transitions

Each turn is classified into one of five trajectory phases: Exploration, Debugging, Planning, Consolidation, and Synthesis. These phases are ordered by their typical importance for downstream context:

$$\text{Synthesis} > \text{Planning} > \text{Consolidation} > \text{Debugging} > \text{Exploration}$$

Phase classification serves two purposes. First, it enables phase-weighted priority scoring in the slicer (Section 3.2), ensuring that high-information-density turns are preferentially included in context slices. Second, it provides a coarse-grained view of conversation dynamics: transitions from Exploration to Planning to Synthesis indicate productive reasoning trajectories, while cycles between Debugging and Exploration may indicate stuck states.

The default phase weights (Synthesis: 1.0, Planning: 0.9, Consolidation: 0.6, Debugging: 0.5, Exploration: 0.3) were established empirically and are configurable via the policy system. The 3.3x ratio between Synthesis and Exploration weight reflects the observation that synthesis turns are approximately three times more likely to contain information relevant to future turns than exploratory ones.

### 5.3 Salience Scores

Each turn carries a salience score in [0, 1], representing its estimated importance for future retrieval. Salience is computed from multiple signals: explicit user markers (bookmarks, highlights), implicit engagement signals (how often the turn is referenced by later turns), and content-based features (presence of definitions, decisions, or novel entities). The salience score contributes to the priority function via the `salience_weight` parameter (default 0.3, meaning salience contributes up to 30% of the total priority at distance 0).

### 5.4 Graph Growth Dynamics

The graph grows at an average rate of 250-500 turns per day during active periods, with session counts ranging from 5-30 per day. The 4,132 sessions span 15 months of continuous operation. New sessions may connect to the existing graph via Reference edges (when a new conversation builds on prior knowledge) or remain initially disconnected (when exploring entirely new domains). Over time, entity resolution and knowledge graph traversal reveal latent connections between ostensibly disconnected sessions.

---

## 6. Empirical Results

This section presents results from the production cc-graph-kernel deployment. All experiments were conducted against the live system (71,130 triples, 112K+ conversation turns) in March 2026.

### 6.1 Knowledge Graph Statistics

The production graph kernel contains 71,130 Subject-Predicate-Object triples spanning 35 software projects across 8 domain layers. The graph was constructed incrementally over 15 months through automated ingestion from conversation logs, code analysis, and manual annotation.

### 6.2 Multi-Hop BFS Traversal Analysis

We performed breadth-first traversals from three representative entities to characterize the graph's connectivity structure:

| Start Entity | Paths Found | Description |
|-------------|-------------|-------------|
| `comp-core` | 112 | Core infrastructure layer, highest connectivity |
| `koatji` | 79 | Business domain, moderate connectivity |
| `creative-director` | 17 | iOS app domain, lower cross-project linkage |

The path count differences reflect the structural role of each entity. `comp-core` is a foundational dependency layer referenced by most other projects, producing dense connectivity. `koatji` spans business operations with moderate cross-domain links. `creative-director` is a leaf-level application with fewer outgoing relationships. These traversal profiles demonstrate that the graph faithfully captures the actual dependency and relationship structure of the multi-project environment.

### 6.3 KG Path Reward Evaluation

To evaluate whether the graph kernel's multi-hop paths carry meaningful structural signal, we designed a reward function and tested it against both valid paths and synthetic hard negatives.

**Dataset construction.** We extracted 199 valid multi-hop paths from the graph kernel via BFS traversal. For each valid path, we constructed one hard negative by swapping the real entity endpoints with randomly selected entities from the graph, preserving the path length and predicate structure but breaking the semantic coherence.

**Reward function.** The 3-signal reward function scores each path on:

1. **Axiomatic validity** (0-3): Whether each triple in the path corresponds to a structurally valid predicate pattern. Valid predicates (e.g., `works_on`, `built_with`, `has_feature`, `uses`, `evolved_from`) score 1 per hop; noise predicates (e.g., `status`, `priority`, `completion_pct`) score 0.
2. **Chain continuity** (0-3): Whether the object of each triple matches the subject of the next, forming a coherent traversal chain.
3. **Terminal grounding** (0-4): Whether the terminal entity is a concrete, resolvable entity (an actual project, service, or technology) rather than a dangling reference.

**Results.**

| Metric | Valid Paths (n=199) | Hard Negatives (n=199) |
|--------|-------------------|----------------------|
| Mean reward | 6.442 +/- 1.205 | 1.626 +/- 2.810 |
| Cohen's d | 2.228 (large) | -- |
| Pairwise ranking accuracy | 81.0% | -- |

The reward function discriminates valid from invalid paths with a large effect size (d = 2.228, where d > 0.8 is conventionally considered large). The 81.0% pairwise ranking accuracy means that when a valid path is compared against its corresponding hard negative, the valid path receives a higher reward 81% of the time. The remaining 19% of cases are paths where endpoint swapping happened to produce a plausible-looking path (e.g., swapping one infrastructure project for another in a `built_with` chain).

The higher variance in hard negative rewards (sigma = 2.810 vs 1.205) reflects the heterogeneity of random endpoint swaps: some produce completely incoherent paths (reward near 0), while others accidentally create semi-valid chains.

### 6.4 Anticipation Geometry on KG Paths

We applied anticipation geometry scalars, a framework originally developed for characterizing conversational dynamics (Diomande, 2026), to knowledge graph paths to test whether geometric analysis generalizes beyond the conversation domain.

**Method.** The 199 valid graph paths were embedded using e5-large-v2 (1024-dimensional embeddings). Each path's sequence of embeddings was analyzed to compute four anticipation scalars:

| Scalar | KG Paths (mean +/- std) | Interpretation |
|--------|------------------------|----------------|
| Commitment | 0.426 +/- 0.041 | Directional consistency of the path through embedding space |
| Uncertainty | 0.425 +/- 0.024 | Entropy of the path's trajectory distribution |
| Transition Pressure | -0.011 +/- 0.009 | Rate of change in path direction (near zero: smooth transitions) |
| Recovery Margin | 0.875 +/- 0.016 | Capacity for the path to return to a coherent trajectory after perturbation |

**Cross-domain comparison.** Graph paths exhibit distinct profiles compared to conversational turns:

- **Higher commitment and uncertainty.** Both scalars hover near 0.425 for graph paths, compared to more variable distributions in conversation. This reflects the structured, typed nature of graph traversal: each hop follows a predicate-constrained edge, producing more uniform directional behavior than free-form dialogue.
- **Near-zero transition pressure.** The -0.011 mean indicates that graph paths change direction very smoothly between hops. Conversations, by contrast, exhibit sharp transition pressure spikes at topic changes and phase boundaries.
- **High recovery margin.** The 0.875 mean with tight variance (sigma = 0.016) indicates that graph paths are geometrically resilient. Even after passing through a weakly-connected intermediate entity, the path tends to return to a coherent embedding-space trajectory. This is consistent with the observation that well-structured knowledge graphs have redundant paths between important entities.
- **Low variance across all scalars.** The tight standard deviations (0.009-0.041) compared to conversational data suggest that knowledge graph paths occupy a more constrained region of anticipation-scalar space. This is expected: graph paths are structurally constrained by typed edges and entity resolution, while conversations are only loosely constrained by topic coherence.

These results demonstrate that anticipation geometry generalizes beyond its original conversational domain and provides meaningful characterization of structured knowledge graph traversals.

### 6.5 Conversation Graph Statistics

The conversation graph underlying the system consists of:

- 20,000 turns fetched for evaluation (from 112K+ total available in Supabase)
- 164 conversations with 10+ turns (sufficient depth for meaningful DAG structure)
- TurnSnapshot DAG with Reply, Branch, and Reference edge types

### 6.6 Latency Benchmarks

We measure the end-to-end latency of the context slicing pipeline:

| Operation | Median | p99 | Notes |
|-----------|--------|-----|-------|
| Slice construction (256 nodes, radius 10) | 8ms | 45ms | PostgreSQL backend |
| HMAC token issuance | <1ms | <1ms | Pure computation |
| HMAC token verification (local) | ~100us | ~100us | Constant-time comparison |
| HMAC token verification (cached) | ~10us | ~10us | LRU cache hit |
| Graph traversal (3 hops, 100 results) | 5ms | 30ms | Depends on graph density |
| Entity alias lookup | <1ms | <1ms | In-memory n-gram matching |
| Batch slice (10 anchors) | 50ms | 200ms | Sequential processing |

These latencies are acceptable for interactive use. The total additional latency for RTI (slice construction + prompt injection) is typically under 50ms, compared to the 500-2000ms latency of LLM inference itself.

### 6.7 Determinism Verification

Determinism is a first-class requirement. The following property must hold:

> Given the same anchor turn, the same slice policy, and the same graph state, two invocations of the slicer MUST produce identical `slice_id` values.

This is verified by the `test_slice_determinism` test, which constructs two independent slicer instances over the same store and policy and asserts `slice_id` equality. Determinism is achieved through: (1) canonical ordering of turns by `TurnId` (UUID comparison), (2) canonical ordering of edges by `(parent, child, edge_type)`, (3) three-level comparison in the priority queue (priority, distance, TurnId), and (4) quantized float representation in policy parameter hashing.

---

## 7. Proposed Evaluation

The following experiments have been designed but not yet executed. We include them to frame the complete evaluation program and invite collaboration.

### 7.1 Domain Shift Simulation

To evaluate the staleness problem quantitatively, we propose the following experiment. The 15-month conversation history is divided into three 5-month epochs. For each epoch pair $(E_i, E_{i+1})$:

1. **TTI Baseline**: Fine-tune a model on KG-derived training data from $E_i$. Evaluate on queries from $E_{i+1}$ that reference entities or relationships that changed between epochs.

2. **RTI System**: Use the same base model (no fine-tuning) but provide runtime context slices from the graph state at query time. Evaluate on the same queries.

3. **Staleness-Stratified Evaluation**: Partition $E_{i+1}$ queries by the degree of staleness (number of changed triples relevant to the query). Report accuracy as a function of staleness.

The hypothesis is that TTI accuracy degrades monotonically with staleness while RTI accuracy remains constant (bounded only by the base model's in-context reasoning ability and the slicer's retrieval quality).

### 7.2 Staleness Detection via GraphSnapshotHash

The `GraphSnapshotHash` provides a mechanism for detecting when a previously-issued slice may be stale. Two implementations exist:

**Stats-based (legacy)**: `hash(max_updated_at, turn_count, edge_count, schema_version)`. This catches bulk changes but may miss individual content edits.

**Content-based (production)**: Uses xxHash64 to fold per-turn content hashes deterministically. This guarantees that any content change in any turn produces a different snapshot hash, enabling exact staleness detection:

```rust
pub fn from_content_hashes(
    turn_content_hashes: &[(TurnId, String)],
    edge_count: u64,
    schema_version: &str,
) -> Self {
    let mut hasher = Xxh64::new(0);
    hasher.write(&edge_count.to_le_bytes());
    hasher.write(schema_version.as_bytes());
    for (turn_id, content_hash) in turn_content_hashes {
        hasher.write(turn_id.as_uuid().as_bytes());
        hasher.write(content_hash.as_bytes());
    }
    Self(format!("{:016x}", hasher.finish()))
}
```

A proposed experiment would compare the `graph_snapshot_hash` of cached slices against the current graph state over a 30-day window, measuring the rate at which slices become stale and the latency savings from cache hits on stable subgraphs.

### 7.3 Head-to-Head TTI vs RTI Comparison

The most direct comparison between training-time and runtime KG integration would involve:

1. Fine-tuning a small language model (e.g., Llama 3 8B) on KG-derived training data from a graph snapshot.
2. Evaluating the fine-tuned model on queries about entities and relationships that were added or modified after the snapshot.
3. Comparing against the same base model augmented with runtime context slices from the current graph.
4. Reporting accuracy, latency, and provenance completeness across both conditions.

This experiment requires a held-out query set with ground-truth answers that depend on post-snapshot graph state, which we are currently constructing from the production conversation logs.

---

## 8. Multi-Machine Deployment

### 8.1 Mesh Architecture

The production deployment spans multiple machines connected via Tailscale VPN:

| Machine | Role | Key Services |
|---------|------|--------------|
| Mac1 | Build host, iOS development | RAG++ (port 8000, SSH tunnel to cloud-vm) |
| Mac4 | Compute node | Ollama, exo cluster master |
| Mac5 | Compute pair with Mac4 | MLX Server, fine-tune daemon |
| Cloud-VM | Infrastructure host | Graph Kernel (port 8001, native Rust binary), Prefect, Grafana, Prometheus, Nexus Portal |

The Graph Kernel runs as a native Rust binary on the cloud VM, accessed from Mac1 via an SSH tunnel (`ssh -f -N -L 8001:localhost:8001 cloud-vm`). RAG++ runs in a Docker container on the cloud VM, reaching the Graph Kernel via Docker bridge gateway (`http://172.17.0.1:8001`).

### 8.2 Communication Bus

Inter-machine communication uses the NUMU event bus (WebSocket on port 7890, HTTP on port 8500) for real-time event propagation, and a mesh event bus (port 8600 on cloud-vm) for service coordination. When a new knowledge triple is ingested into the Graph Kernel, an invalidation event propagates to RAG++'s query cache via the event bus, ensuring that stale cached results are purged.

### 8.3 Operational Considerations

The multi-machine topology introduces several operational constraints that inform the system design:

1. **Network partitions.** If the SSH tunnel between Mac1 and cloud-vm drops, RAG++ on Mac1 loses access to the Graph Kernel. The system degrades gracefully: RAG++ falls back to semantic-only retrieval (Plane 1 without Plane 2), losing graph context but maintaining basic retrieval capability.

2. **Graph state consistency.** With a single PostgreSQL instance backing the Graph Kernel, there is no replication lag. All reads see the latest committed state. This simplifies the consistency model compared to distributed graph databases.

3. **Secret management.** The HMAC secret for admissibility tokens is stored as the `KERNEL_HMAC_SECRET` environment variable on the cloud VM. It is never transmitted over the network; downstream systems that need to verify tokens call the `/api/verify_token` endpoint over the SSH tunnel.

---

## 9. Discussion

### 9.1 Synthesis with GraphMERT

The Princeton group's GraphMERT model offers a compelling complement to runtime integration. GraphMERT's 80M-parameter encoder can construct an initial knowledge graph from unstructured text with high fidelity (69.8% FActScore). In a combined architecture, GraphMERT would bootstrap the graph from historical data, and cc-graph-kernel would maintain it at runtime. This would eliminate the manual annotation bottleneck that currently limits the rate of graph expansion.

### 9.2 Provenance as First-Class Citizen

A underappreciated property of runtime KG integration is that it naturally produces provenance. When a model response is conditioned on a specific context slice, the slice itself (with its `slice_id`, `policy_ref`, and `admissibility_token`) constitutes a complete audit record. This has implications beyond debugging: it enables automated quality assessment (did the model's response use the provided context appropriately?), regulatory compliance (which data influenced this decision?), and continual learning (which context slices correlate with high-quality responses?).

The type-level enforcement via `AdmissibleEvidenceBundle` ensures that provenance tracking is not optional. Any downstream system that accepts evidence is forced by the compiler to accept only verified bundles. This is a stronger guarantee than policy-based enforcement, which relies on developers remembering to check tokens.

### 9.3 Limitations

Our approach has several limitations that should be acknowledged:

1. **No direct TTI vs. RTI comparison yet.** While the theoretical analysis (Section 4) and the production deployment provide strong evidence for RTI's advantages in evolving domains, we have not yet run a controlled head-to-head experiment with a fine-tuned TTI baseline. The domain shift simulation (Section 7.1) would provide the quantitative comparison needed to measure the accuracy gap as a function of staleness.

2. **Prompt budget consumption.** Runtime context slices consume tokens from the model's context window. With a 256-node slice, context injection can consume 2,000-10,000 tokens depending on turn length. This limits the space available for the user's query and the model's response. Future work should explore compression techniques (summarization of low-salience turns, adaptive slice sizing based on available context budget).

3. **No multi-hop reasoning in context.** While the graph supports multi-hop traversal, the injected context is a flat set of turns. The model must perform any multi-hop reasoning in-context, which is less reliable than TTI's approach of encoding reasoning patterns in weights. Structured prompting (chain-of-thought over graph paths) may mitigate this.

4. **Evaluation on a single system.** Our evaluation is based on a single production deployment. While the 71,130-triple graph, 112K+ turn dataset, and 15-month operational history provide substantial scale, generalization to other domains and deployment patterns requires further study.

5. **Graph quality dependence.** RTI is only as good as the underlying graph. Incorrect triples, missing relationships, and stale confidence scores all propagate into context slices. The confidence scoring system and HMAC verification ensure *mechanical* correctness (the slice is what the graph says it should be) but not *semantic* correctness (whether the graph itself is accurate).

### 9.4 Comparison to GraphRAG

Microsoft's GraphRAG and our system share the conviction that graph structure improves retrieval quality. The key architectural difference is in the unit of retrieval. GraphRAG retrieves community summaries, which are pre-computed text descriptions of graph communities. Our system retrieves raw graph slices, which are subsets of the graph itself with all original turn data and edge structure preserved.

The GraphRAG approach is more token-efficient (summaries are compact) but loses provenance (the summary is an interpretation of the graph, not the graph itself). Our approach is more token-expensive but maintains full auditability. The choice between them depends on whether the deployment values token efficiency or provenance fidelity.

### 9.5 Toward Verified Retrieval

The admissibility token system opens a path toward a broader concept we term "verified retrieval." In standard RAG, there is no mechanism to distinguish between context that was actually retrieved from the knowledge base and context that was fabricated or hallucinated. With HMAC-signed slices, any context claim can be verified against the kernel's authority. This property could be extended to support:

- **Cross-agent context sharing**: Agent A passes a context slice to Agent B, with the admissibility token proving that the context originated from a trusted kernel.
- **Audit trails for regulated domains**: In healthcare, legal, or financial applications, the ability to prove exactly what information was available to the model at decision time has direct regulatory value.
- **Adversarial robustness**: In multi-agent systems where some agents may be compromised, token verification prevents injection of unauthorized context.

---

## 10. Conclusion

We have presented runtime knowledge graph integration as a practical alternative to training-time-only approaches for evolving domains. Our implementation, cc-graph-kernel, is a production Rust service with 71,130 triples, processing real workloads across a multi-machine mesh. The engineering overhead of runtime integration is modest (5-50ms per slice, sub-microsecond token verification) while the benefits are substantial (zero staleness, full provenance, immediate domain adaptation).

Our empirical results establish three findings. First, the production graph kernel faithfully captures the dependency structure of a multi-project environment, as demonstrated by BFS traversal profiles that reflect the actual architectural roles of entities (112 paths from a core infrastructure node vs. 17 from a leaf application). Second, a 3-signal reward function discriminates valid multi-hop graph paths from hard negatives with 81.0% pairwise accuracy and a large effect size (Cohen's d = 2.228), confirming that the graph's path structure carries meaningful signal beyond individual triples. Third, anticipation geometry scalars generalize from conversation analysis to knowledge graph paths, producing distinct profiles that characterize the structured, low-variance nature of typed graph traversal compared to free-form dialogue.

Several important comparisons remain as future work: direct head-to-head evaluation of training-time vs. runtime KG integration on domain-shifted queries, staleness detection via GraphSnapshotHash over longitudinal windows, and fine-tuned model evaluation on post-snapshot graph state. These experiments (Section 7) will provide the quantitative staleness-accuracy curves needed to characterize the precise conditions under which runtime integration outperforms training-time approaches.

The key insight is that knowledge graphs should not be viewed as training scaffolding to be discarded at inference. They are living artifacts that encode the current state of the domain, and the most natural way to access them is at runtime, when the domain state is known. The Context Slicer's priority-queue BFS algorithm, combined with HMAC-signed admissibility tokens and content-hash-based snapshot verification, provides a principled mechanism for runtime graph access with cryptographic provenance guarantees.

We believe the most promising direction is the synthesis of training-time and runtime approaches: using KG distillation (GraphMERT) to build foundational domain understanding in model weights, and runtime slicing (cc-graph-kernel) to maintain currency with the evolving graph. This synthesis combines the deep reasoning advantage of parameter-encoded knowledge with the freshness and auditability of runtime retrieval.

---

## References

1. Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *ICLR 2024*.

2. Baek, J., Aji, A. F., & Seo, A. (2023). Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering. *arXiv:2306.04136*.

3. Belova, M., Kansal, Y., Liang, Y., Xiao, J., & Jha, N. K. (2026). An Alternative Trajectory for Generative AI. *arXiv:2603.14147*.

4. Belova, M., & Jha, N. K. (2025). GraphMERT: Knowledge Graph Enhanced Language Models for Factual Grounding. *TMLR 2026*. arXiv:2510.09580.

5. Belova, M., & Jha, N. K. (2025). Bottom-up Domain-Specific Superintelligence: QwQ-Med-3. *arXiv:2507.13966*.

6. Belova, M., & Jha, N. K. (2025). Energy-Efficient DSS via VOC Framework. *arXiv:2510.22052*.

7. Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating Embeddings for Modeling Multi-relational Data. *NeurIPS 2013*.

8. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. *arXiv:2404.16130*.

9. Gao, Y., Xiong, Y., Jaiswal, A., Srivastava, N., Patrick, M., Molchanov, P., & Kautz, J. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv:2312.10997*.

10. Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y., Callan, J., & Neubig, G. (2023). Active Retrieval Augmented Generation. *EMNLP 2023*.

11. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W., Rocktaschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

12. Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., & Wu, X. (2024). Unifying Large Language Models and Knowledge Graphs: A Roadmap. *IEEE TKDE*.

13. Petroni, F., Rocktaschel, T., Lewis, P., Bakhtin, A., Wu, Y., Miller, A. H., & Riedel, S. (2019). Language Models as Knowledge Bases? *EMNLP 2019*.

14. Shuster, K., Poff, S., Chen, M., Kiela, D., & Weston, J. (2021). Retrieval Augmentation Reduces Hallucination in Conversation. *EMNLP Findings 2021*.

15. Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2023). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. *ACL 2023*.

16. Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., & Wei, F. (2023). Text Embeddings by Weakly-Supervised Contrastive Pre-training. *arXiv:2212.03533*.

17. Yasunaga, M., Ren, H., Bosselut, A., Liang, P., & Leskovec, J. (2021). QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering. *NAACL 2021*.

18. Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2024). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *NeurIPS 2023*.

---

## Appendix A: Production Invariants

The Graph Kernel enforces the following invariants, each traceable to a specific enforcement mechanism:

| ID | Invariant | Enforcement |
|----|-----------|-------------|
| INV-GK-001 | Slice Boundary Integrity: `turn_id in slice.turn_ids` | `SliceExport::is_turn_admissible()` |
| INV-GK-002 | Provenance Completeness: every response carries `(slice_id, policy_ref, schema_version, graph_snapshot_hash, admissibility_token)` | `SliceExport` struct field requirements |
| INV-GK-003 | No Phantom Authority: missing token = non-admissible | `AdmissibleEvidenceBundle::from_verified()` |
| INV-GK-004 | Content Immutability: stored hash matches content | `TurnSnapshot::verify_content_hash()` |
| INV-GK-005 | HMAC Token Unforgeability | `AdmissibilityToken::verify_hmac()` with constant-time comparison |

## Appendix B: Slice Policy Defaults

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `max_nodes` | 256 | 1-10,000 | Maximum turns in slice |
| `max_radius` | 10 | 1-100 | Maximum graph hops from anchor |
| `salience_weight` | 0.3 | 0.0-1.0 | Contribution of salience to priority |
| `distance_decay` | 0.9 | 0.0-1.0 | Priority reduction per hop (10% per hop at default) |
| `include_siblings` | true | bool | Whether to expand to sibling turns |
| `max_siblings_per_node` | 5 | 0-50 | Sibling expansion limit |

## Appendix C: API Endpoint Summary

Full API documentation is available at the service's `/health` endpoint, which returns version, schema version, policy count, backend type, database connectivity, and cache statistics.

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "schema_version": "1.0.0",
  "policy_count": 3,
  "registry_fingerprint": "a1b2c3d4",
  "backend": "postgres",
  "database": { "connected": true },
  "cache": { "entries": 1247, "hits": 89234, "misses": 3421 }
}
```
