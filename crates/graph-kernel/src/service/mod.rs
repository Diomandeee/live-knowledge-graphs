//! Graph Kernel REST Service
//!
//! Exposes the Graph Kernel as a REST API for slice-conditioned retrieval.
//!
//! ## Endpoints
//!
//! - `POST /api/slice` - Construct a context slice around an anchor
//! - `POST /api/slice/batch` - Batch slice construction
//! - `POST /api/verify_token` - Verify an admissibility token
//! - `GET /api/policies` - List registered policies
//! - `POST /api/policies` - Register a new policy
//! - `GET /api/knowledge` - Query knowledge triples
//! - `POST /api/knowledge` - Add a single triple
//! - `DELETE /api/knowledge` - Delete triples
//! - `POST /api/knowledge/batch` - Batch ingest triples (high-performance)
//! - `GET /api/knowledge/stats` - Knowledge graph statistics
//! - `POST /api/knowledge/traverse` - Multi-hop graph traversal
//! - `GET /api/knowledge/graph` - D3-compatible graph visualization
//! - `GET /api/knowledge/graph.mermaid` - Mermaid diagram
//! - `GET /api/knowledge/graph.dot` - Graphviz DOT format
//! - `GET /health` - Detailed service health (includes cache stats)
//! - `GET /health/live` - Liveness probe
//! - `GET /health/ready` - Readiness probe
//! - `GET /health/startup` - Startup probe

pub mod cache;
pub mod knowledge_handlers;
pub mod middleware;
pub mod normalize;
pub mod routes;
pub mod state;
pub mod visualization;

pub use cache::QueryCache;
pub use middleware::{metrics_middleware, record_slice_metrics, record_token_verification};
pub use routes::{create_router, AppState};
pub use state::{ServiceState, PolicyRegistry, PolicyRef};

