//! # cc-graph-kernel
//!
//! Deterministic context slicing for conversation DAGs.
//!
//! The Graph Kernel answers one question:
//!
//! > Given a target turn, which other turns are **allowed to influence meaning**?
//!
//! ## Core Contract
//!
//! 1. Given a target turn (node), deterministically select a context neighborhood (slice)
//! 2. Produce a **slice fingerprint hash** for downstream provenance
//! 3. Export the slice as a stable, ordered bundle of turn IDs + edges + metadata
//!
//! ## Architecture
//!
//! ```text
//! Target Turn → SlicePolicy → Expansion → SliceExport → SliceFingerprint
//!                    ↓
//!              GraphStore (Postgres or Memory)
//! ```
//!
//! ## Determinism Guarantees
//!
//! - Same anchor + same policy + same graph state → identical slice_id
//! - Edge ordering is canonical (parent, child)
//! - Turn ordering is canonical (by TurnId)

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod types;
pub mod policy;
pub mod store;
pub mod slicer;
pub mod canonical;
pub mod canonical_content;
pub mod atlas;

#[cfg(feature = "service")]
pub mod service;

// Re-exports
pub use types::{TurnId, TurnSnapshot, Edge, EdgeType, Role, Phase};
pub use types::slice::{SliceExport, SliceFingerprint, GraphSnapshotHash, AdmissibilityToken};
pub use types::admissible::{AdmissibleEvidenceBundle, VerificationError};
pub use types::verification::{TokenVerifier, VerificationMode, VerificationResult, CacheConfig, CacheStats};
pub use types::sufficiency::{
    DiversityMetrics, SalienceStats, SufficiencyPolicy, SufficiencyCheck,
    SufficiencyViolation, EvidenceBundle, EvidenceBundleError,
};
pub use types::boundary::{
    SliceBoundaryGuard, BoundedQueryBuilder, BoundaryViolation, BoundaryCheck,
};
pub use types::provenance::{
    ReplayProvenance, EmbeddingModelRef, RetrievalParams, NormalizationVersion,
    ProvenanceBuilder, ProvenanceError,
};
pub use types::incident::{
    Severity, IncidentType, Incident, QuarantinedToken,
    IncidentMetrics, NoOpMetrics, TestMetrics,
    QUARANTINE_TABLE_SCHEMA, INCIDENT_TABLE_SCHEMA,
};
pub use canonical_content::CANONICAL_CONTENT_VERSION;
pub use policy::{SlicePolicyV1, PhaseWeights};
pub use store::GraphStore;
#[cfg(feature = "postgres")]
pub use store::PostgresGraphStore;
#[cfg(feature = "sqlite")]
pub use store::SqliteGraphStore;
pub use slicer::ContextSlicer;
pub use canonical::{to_canonical_bytes, canonical_hash, canonical_hash_hex};
pub use canonical_content::{
    normalize_text, canonical_content, compute_content_hash,
    verify_content_hash, validate_content_hash, HashValidation,
};

// Atlas re-exports
pub use atlas::{
    GraphSnapshot, SnapshotInput, SnapshotStore,
    BatchSlicer, BatchSliceResult, SliceRegistry, SliceRegistryEntry, AnchorSet,
    OverlapAnalyzer, OverlapGraph, OverlapEdge,
    TurnInfluence, InfluenceScores, PhaseCounts, BridgeTurn, PhaseTopologyStats,
    compute_influence, extract_bridges, compute_phase_topology,
    AtlasBundler, AtlasManifest, AtlasArtifactPaths, PhaseTopology, AtlasStats,
    ATLAS_SCHEMA_VERSION,
};

// Service re-exports (when service feature is enabled)
#[cfg(feature = "service")]
pub use service::{create_router, ServiceState, PolicyRegistry, PolicyRef};

/// Schema version for all graph kernel types.
/// Increment on breaking changes to any schema type.
pub const GRAPH_KERNEL_SCHEMA_VERSION: &str = "1.0.0";

/// Default policy version identifier.
pub const DEFAULT_POLICY_VERSION: &str = "slice_policy_v1";

