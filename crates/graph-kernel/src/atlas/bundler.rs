//! Atlas bundler for packaging all artifacts with a manifest.
//!
//! The bundler produces a complete, hashable Atlas package that
//! can be verified and replayed.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::canonical::canonical_hash_hex;
use super::{
    GraphSnapshot,
    BatchSliceResult,
    OverlapGraph,
    InfluenceScores,
    ATLAS_SCHEMA_VERSION,
};

/// Paths to Atlas artifacts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasArtifactPaths {
    /// Path to graph snapshot file.
    pub snapshot: String,
    /// Path to anchors file.
    pub anchors: String,
    /// Directory containing slice files.
    pub slices_dir: String,
    /// Path to slice registry file.
    pub slice_registry: String,
    /// Path to overlap graph file.
    pub overlap_graph: String,
    /// Path to turn influence scores file.
    pub turn_influence: String,
    /// Path to phase topology file.
    pub phase_topology: String,
}

impl Default for AtlasArtifactPaths {
    fn default() -> Self {
        Self {
            snapshot: "graph_snapshot_v1.json".to_string(),
            anchors: "anchors_v1.jsonl".to_string(),
            slices_dir: "slices_v1/".to_string(),
            slice_registry: "slice_registry_v1.jsonl".to_string(),
            overlap_graph: "overlap_graph_v1.json".to_string(),
            turn_influence: "turn_influence_v1.jsonl".to_string(),
            phase_topology: "phase_topology_v1.json".to_string(),
        }
    }
}

/// Phase topology summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTopology {
    /// Average overlap between phase pairs.
    pub phase_pair_overlaps: BTreeMap<String, f32>,
    /// Representative slice IDs for each phase.
    pub phase_centroids: BTreeMap<String, Vec<String>>,
    /// Number of cross-phase bridge turns.
    pub bridge_turn_count: usize,
    /// Content hash.
    pub topology_hash: String,
}

impl PhaseTopology {
    /// Create a new phase topology.
    pub fn new(
        phase_pair_overlaps: BTreeMap<String, f32>,
        phase_centroids: BTreeMap<String, Vec<String>>,
        bridge_turn_count: usize,
    ) -> Self {
        let hash_input = (&phase_pair_overlaps, &phase_centroids, bridge_turn_count);
        let topology_hash = canonical_hash_hex(&hash_input);

        Self {
            phase_pair_overlaps,
            phase_centroids,
            bridge_turn_count,
            topology_hash,
        }
    }
}

/// The complete Atlas manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasManifest {
    /// Unique identifier for this Atlas run.
    pub atlas_id: String,
    /// Schema version.
    pub version: String,
    /// Source graph snapshot ID.
    pub snapshot_id: String,
    /// Anchor set hash.
    pub anchor_set_hash: String,
    /// Slice registry hash.
    pub slice_registry_hash: String,
    /// Overlap graph hash.
    pub overlap_graph_hash: String,
    /// Turn influence scores hash.
    pub turn_influence_hash: String,
    /// Phase topology hash.
    pub phase_topology_hash: String,
    /// Unix timestamp when computed.
    pub computed_at: i64,
    /// Paths to all artifacts.
    pub artifact_paths: AtlasArtifactPaths,
    /// Summary statistics.
    pub stats: AtlasStats,
}

/// Summary statistics for an Atlas run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasStats {
    /// Number of turns in source graph.
    pub turn_count: u64,
    /// Number of edges in source graph.
    pub edge_count: u64,
    /// Number of anchors selected.
    pub anchor_count: usize,
    /// Number of slices generated.
    pub slice_count: usize,
    /// Number of overlap edges.
    pub overlap_edge_count: usize,
    /// Number of cross-phase bridge turns.
    pub bridge_turn_count: usize,
}

/// Builder for Atlas manifests.
pub struct AtlasBundler {
    snapshot: Option<GraphSnapshot>,
    batch_result: Option<BatchSliceResult>,
    overlap_graph: Option<OverlapGraph>,
    influence_scores: Option<InfluenceScores>,
    phase_topology: Option<PhaseTopology>,
    artifact_paths: AtlasArtifactPaths,
}

impl AtlasBundler {
    /// Create a new bundler.
    pub fn new() -> Self {
        Self {
            snapshot: None,
            batch_result: None,
            overlap_graph: None,
            influence_scores: None,
            phase_topology: None,
            artifact_paths: AtlasArtifactPaths::default(),
        }
    }

    /// Set custom artifact paths.
    pub fn with_paths(mut self, paths: AtlasArtifactPaths) -> Self {
        self.artifact_paths = paths;
        self
    }

    /// Set the graph snapshot.
    pub fn snapshot(mut self, snapshot: GraphSnapshot) -> Self {
        self.snapshot = Some(snapshot);
        self
    }

    /// Set the batch slice result.
    pub fn batch_result(mut self, result: BatchSliceResult) -> Self {
        self.batch_result = Some(result);
        self
    }

    /// Set the overlap graph.
    pub fn overlap_graph(mut self, graph: OverlapGraph) -> Self {
        self.overlap_graph = Some(graph);
        self
    }

    /// Set the influence scores.
    pub fn influence_scores(mut self, scores: InfluenceScores) -> Self {
        self.influence_scores = Some(scores);
        self
    }

    /// Set the phase topology.
    pub fn phase_topology(mut self, topology: PhaseTopology) -> Self {
        self.phase_topology = Some(topology);
        self
    }

    /// Build the Atlas manifest.
    ///
    /// Panics if required components are missing.
    pub fn build(self) -> AtlasManifest {
        let snapshot = self.snapshot.expect("snapshot is required");
        let batch_result = self.batch_result.expect("batch_result is required");
        let overlap_graph = self.overlap_graph.expect("overlap_graph is required");
        let influence_scores = self.influence_scores.expect("influence_scores is required");
        let phase_topology = self.phase_topology.expect("phase_topology is required");

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        // Compute atlas_id from all component hashes
        let atlas_id_input = AtlasIdInput {
            snapshot_id: snapshot.snapshot_id.clone(),
            anchor_set_hash: batch_result.anchor_set_hash.clone(),
            slice_registry_hash: batch_result.registry.registry_hash.clone(),
            overlap_graph_hash: overlap_graph.graph_hash.clone(),
            turn_influence_hash: influence_scores.scores_hash.clone(),
            phase_topology_hash: phase_topology.topology_hash.clone(),
        };
        let atlas_id = canonical_hash_hex(&atlas_id_input);

        let stats = AtlasStats {
            turn_count: snapshot.turn_count,
            edge_count: snapshot.edge_count,
            anchor_count: batch_result.registry.entries.len(),
            slice_count: batch_result.slices.len(),
            overlap_edge_count: overlap_graph.edges.len(),
            bridge_turn_count: phase_topology.bridge_turn_count,
        };

        AtlasManifest {
            atlas_id,
            version: ATLAS_SCHEMA_VERSION.to_string(),
            snapshot_id: snapshot.snapshot_id,
            anchor_set_hash: batch_result.anchor_set_hash,
            slice_registry_hash: batch_result.registry.registry_hash,
            overlap_graph_hash: overlap_graph.graph_hash,
            turn_influence_hash: influence_scores.scores_hash,
            phase_topology_hash: phase_topology.topology_hash,
            computed_at: now,
            artifact_paths: self.artifact_paths,
            stats,
        }
    }

    /// Try to build, returning None if components are missing.
    pub fn try_build(self) -> Option<AtlasManifest> {
        if self.snapshot.is_none()
            || self.batch_result.is_none()
            || self.overlap_graph.is_none()
            || self.influence_scores.is_none()
            || self.phase_topology.is_none()
        {
            return None;
        }
        Some(self.build())
    }
}

impl Default for AtlasBundler {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal struct for computing atlas_id.
#[derive(Serialize)]
struct AtlasIdInput {
    snapshot_id: String,
    anchor_set_hash: String,
    slice_registry_hash: String,
    overlap_graph_hash: String,
    turn_influence_hash: String,
    phase_topology_hash: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::{SnapshotInput, BatchSliceResult, SliceRegistry, SliceRegistryEntry};
    use crate::types::{TurnId, Edge, EdgeType};
    use uuid::Uuid;

    fn make_test_snapshot() -> GraphSnapshot {
        let turn1 = TurnId::new(Uuid::new_v4());
        let turn2 = TurnId::new(Uuid::new_v4());

        let input = SnapshotInput {
            turn_ids: vec![turn1.clone(), turn2.clone()],
            edges: vec![Edge::new(turn1, turn2, EdgeType::Reply)],
            timestamps: vec![1000, 2000],
        };

        GraphSnapshot::compute(&input)
    }

    fn make_test_batch_result() -> BatchSliceResult {
        BatchSliceResult {
            snapshot_id: "test_snapshot".to_string(),
            anchor_set_hash: "anchor_hash".to_string(),
            policy_id: "policy_v1".to_string(),
            policy_params_hash: "params_hash".to_string(),
            slices: vec![],
            registry: SliceRegistry::new(vec![
                SliceRegistryEntry {
                    anchor_turn_id: "turn1".to_string(),
                    slice_id: "slice1".to_string(),
                    turn_count: 5,
                    edge_count: 4,
                    policy_params_hash: "params_hash".to_string(),
                },
            ]),
        }
    }

    fn make_test_overlap_graph() -> OverlapGraph {
        OverlapGraph::new(vec![], 1, 0.0)
    }

    fn make_test_influence_scores() -> InfluenceScores {
        InfluenceScores::new(vec![], 0)
    }

    fn make_test_phase_topology() -> PhaseTopology {
        PhaseTopology::new(BTreeMap::new(), BTreeMap::new(), 0)
    }

    #[test]
    fn test_bundler_build() {
        let manifest = AtlasBundler::new()
            .snapshot(make_test_snapshot())
            .batch_result(make_test_batch_result())
            .overlap_graph(make_test_overlap_graph())
            .influence_scores(make_test_influence_scores())
            .phase_topology(make_test_phase_topology())
            .build();

        assert!(!manifest.atlas_id.is_empty());
        assert_eq!(manifest.version, ATLAS_SCHEMA_VERSION);
        assert_eq!(manifest.stats.anchor_count, 1);
    }

    #[test]
    fn test_bundler_try_build_incomplete() {
        let result = AtlasBundler::new()
            .snapshot(make_test_snapshot())
            .try_build();

        assert!(result.is_none());
    }

    #[test]
    fn test_manifest_determinism() {
        let snapshot = make_test_snapshot();
        let batch_result = make_test_batch_result();
        let overlap_graph = make_test_overlap_graph();
        let influence_scores = make_test_influence_scores();
        let phase_topology = make_test_phase_topology();

        let manifest1 = AtlasBundler::new()
            .snapshot(snapshot.clone())
            .batch_result(batch_result.clone())
            .overlap_graph(overlap_graph.clone())
            .influence_scores(influence_scores.clone())
            .phase_topology(phase_topology.clone())
            .build();

        let manifest2 = AtlasBundler::new()
            .snapshot(snapshot)
            .batch_result(batch_result)
            .overlap_graph(overlap_graph)
            .influence_scores(influence_scores)
            .phase_topology(phase_topology)
            .build();

        assert_eq!(manifest1.atlas_id, manifest2.atlas_id);
    }
}

