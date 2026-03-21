//! Atlas: Reproducible graph structure computation.
//!
//! The Atlas module provides tools for computing the global structure of
//! a conversation DAG through bounded, versioned passes:
//!
//! 1. **Snapshot**: Capture a deterministic fingerprint of graph state
//! 2. **Anchors**: Select representative turns for slicing
//! 3. **Tiling**: Generate slices for each anchor
//! 4. **Overlap**: Compute structural relationships between slices
//! 5. **Bundle**: Package all artifacts with a manifest
//!
//! ## Core Contract
//!
//! Given the same `snapshot_id` and `policy_id`, an Atlas Run produces
//! byte-identical artifacts.
//!
//! ## Architecture
//!
//! ```text
//! GraphStore → Snapshot → Anchors → Slices → Overlap → Manifest
//!                ↓           ↓         ↓         ↓         ↓
//!          snapshot_id  anchors.jsonl  slices/  overlap.json  atlas_manifest.json
//! ```

pub mod snapshot;
pub mod batch_slicer;
pub mod overlap;
pub mod influence;
pub mod bundler;

// Re-exports
pub use snapshot::{GraphSnapshot, SnapshotInput, SnapshotStore};
pub use batch_slicer::{BatchSlicer, BatchSliceResult, SliceRegistry, SliceRegistryEntry, AnchorSet};
pub use overlap::{OverlapAnalyzer, OverlapGraph, OverlapEdge};
pub use influence::{TurnInfluence, InfluenceScores, PhaseCounts, BridgeTurn, PhaseTopologyStats, compute_influence, extract_bridges, compute_phase_topology};
pub use bundler::{AtlasBundler, AtlasManifest, AtlasArtifactPaths, PhaseTopology, AtlasStats};

/// Atlas schema version. Increment on breaking changes.
pub const ATLAS_SCHEMA_VERSION: &str = "atlas_v1";

