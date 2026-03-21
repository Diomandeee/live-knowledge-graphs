//! Core types for the graph kernel.

pub mod turn;
pub mod edge;
pub mod slice;
pub mod admissible;
pub mod verification;
pub mod sufficiency;
pub mod boundary;
pub mod provenance;
pub mod incident;

pub use turn::{TurnId, TurnSnapshot, Role, Phase, ContentHashError};
pub use edge::{Edge, EdgeType};
pub use slice::{SliceExport, SliceFingerprint, GraphSnapshotHash, AdmissibilityToken};
pub use admissible::{AdmissibleEvidenceBundle, VerificationError};
pub use verification::{TokenVerifier, VerificationMode, VerificationResult, CacheConfig, CacheStats};
pub use sufficiency::{
    DiversityMetrics, SalienceStats, SufficiencyPolicy, SufficiencyCheck,
    SufficiencyViolation, EvidenceBundle, EvidenceBundleError,
};
pub use boundary::{
    SliceBoundaryGuard, BoundedQueryBuilder, BoundaryViolation, BoundaryCheck,
};
pub use provenance::{
    ReplayProvenance, EmbeddingModelRef, RetrievalParams, NormalizationVersion,
    ProvenanceBuilder, ProvenanceError,
};
pub use incident::{
    Severity, IncidentType, Incident, QuarantinedToken,
    IncidentMetrics, NoOpMetrics, TestMetrics,
    QUARANTINE_TABLE_SCHEMA, INCIDENT_TABLE_SCHEMA,
};

