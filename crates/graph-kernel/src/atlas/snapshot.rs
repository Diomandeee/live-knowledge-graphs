//! Graph snapshot identity for deterministic dataset versioning.
//!
//! A `GraphSnapshot` captures a fingerprint of the entire graph state
//! before any computation begins, ensuring reproducibility.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::canonical::{canonical_hash_hex, to_canonical_bytes};
use crate::types::{TurnId, Edge};
use crate::GRAPH_KERNEL_SCHEMA_VERSION;

/// A deterministic fingerprint of the graph state.
///
/// This becomes the "dataset version" for provenance tracking.
/// Any downstream artifact should reference the `snapshot_id` to
/// prove it was computed against a specific graph state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    /// Unique identifier for this snapshot (xxh64 of all components).
    pub snapshot_id: String,
    /// Total number of turns in the graph.
    pub turn_count: u64,
    /// Total number of edges in the graph.
    pub edge_count: u64,
    /// Maximum timestamp across all turns.
    pub max_timestamp: i64,
    /// Schema version used for types.
    pub schema_version: String,
    /// Hash of sorted turn UUIDs.
    pub turn_id_hash: String,
    /// Hash of sorted (parent, child) edge pairs.
    pub edge_pair_hash: String,
    /// Unix timestamp when this snapshot was computed.
    pub computed_at: i64,
}

/// Input data for computing a graph snapshot.
#[derive(Debug, Clone)]
pub struct SnapshotInput {
    /// All turn IDs in the graph.
    pub turn_ids: Vec<TurnId>,
    /// All edges in the graph.
    pub edges: Vec<Edge>,
    /// Timestamps for each turn (aligned with turn_ids).
    pub timestamps: Vec<i64>,
}

impl GraphSnapshot {
    /// Compute a new graph snapshot from input data.
    ///
    /// The snapshot ID is computed deterministically from:
    /// - Turn count and edge count
    /// - Maximum timestamp
    /// - Schema version
    /// - Hash of sorted turn IDs
    /// - Hash of sorted edge pairs
    pub fn compute(input: &SnapshotInput) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let turn_count = input.turn_ids.len() as u64;
        let edge_count = input.edges.len() as u64;
        let max_timestamp = input.timestamps.iter().copied().max().unwrap_or(0);

        // Compute turn_id_hash from sorted turn IDs
        let sorted_turn_ids: BTreeSet<_> = input.turn_ids.iter().collect();
        let turn_id_strings: Vec<String> = sorted_turn_ids
            .iter()
            .map(|t| t.as_uuid().to_string())
            .collect();
        let turn_id_hash = canonical_hash_hex(&turn_id_strings);

        // Compute edge_pair_hash from sorted (parent, child) pairs
        let mut edge_pairs: Vec<(String, String)> = input
            .edges
            .iter()
            .map(|e| (e.parent.as_uuid().to_string(), e.child.as_uuid().to_string()))
            .collect();
        edge_pairs.sort();
        let edge_pair_hash = canonical_hash_hex(&edge_pairs);

        // Compute snapshot_id from all components
        let id_input = SnapshotIdInput {
            turn_count,
            edge_count,
            max_timestamp,
            schema_version: GRAPH_KERNEL_SCHEMA_VERSION.to_string(),
            turn_id_hash: turn_id_hash.clone(),
            edge_pair_hash: edge_pair_hash.clone(),
        };
        let snapshot_id = canonical_hash_hex(&id_input);

        Self {
            snapshot_id,
            turn_count,
            edge_count,
            max_timestamp,
            schema_version: GRAPH_KERNEL_SCHEMA_VERSION.to_string(),
            turn_id_hash,
            edge_pair_hash,
            computed_at: now,
        }
    }

    /// Serialize to canonical JSON bytes.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        to_canonical_bytes(self)
    }

    /// Verify that this snapshot matches the given input.
    pub fn verify(&self, input: &SnapshotInput) -> bool {
        let recomputed = Self::compute(input);
        self.snapshot_id == recomputed.snapshot_id
    }
}

/// Internal struct for computing snapshot_id hash.
#[derive(Serialize)]
struct SnapshotIdInput {
    turn_count: u64,
    edge_count: u64,
    max_timestamp: i64,
    schema_version: String,
    turn_id_hash: String,
    edge_pair_hash: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Extended GraphStore trait for snapshot computation
// ─────────────────────────────────────────────────────────────────────────────

/// Extension trait for graph stores that can compute snapshots.
pub trait SnapshotStore {
    /// Error type for store operations.
    type Error: std::error::Error;

    /// Get all turn IDs in the graph.
    fn get_all_turn_ids(&self) -> Result<Vec<TurnId>, Self::Error>;

    /// Get all edges in the graph.
    fn get_all_edges(&self) -> Result<Vec<Edge>, Self::Error>;

    /// Get timestamps for a list of turn IDs.
    fn get_timestamps(&self, turn_ids: &[TurnId]) -> Result<Vec<i64>, Self::Error>;

    /// Compute a graph snapshot.
    fn compute_snapshot(&self) -> Result<GraphSnapshot, Self::Error> {
        let turn_ids = self.get_all_turn_ids()?;
        let edges = self.get_all_edges()?;
        let timestamps = self.get_timestamps(&turn_ids)?;

        let input = SnapshotInput {
            turn_ids,
            edges,
            timestamps,
        };

        Ok(GraphSnapshot::compute(&input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EdgeType;
    use uuid::Uuid;

    fn make_turn_id() -> TurnId {
        TurnId::new(Uuid::new_v4())
    }

    #[test]
    fn test_snapshot_determinism() {
        let turn1 = make_turn_id();
        let turn2 = make_turn_id();
        let turn3 = make_turn_id();

        let edges = vec![
            Edge::new(turn1.clone(), turn2.clone(), EdgeType::Reply),
            Edge::new(turn2.clone(), turn3.clone(), EdgeType::Reply),
        ];

        let input = SnapshotInput {
            turn_ids: vec![turn1.clone(), turn2.clone(), turn3.clone()],
            edges: edges.clone(),
            timestamps: vec![1000, 2000, 3000],
        };

        let snapshot1 = GraphSnapshot::compute(&input);
        let snapshot2 = GraphSnapshot::compute(&input);

        assert_eq!(snapshot1.snapshot_id, snapshot2.snapshot_id);
        assert_eq!(snapshot1.turn_count, 3);
        assert_eq!(snapshot1.edge_count, 2);
        assert_eq!(snapshot1.max_timestamp, 3000);
    }

    #[test]
    fn test_snapshot_differs_on_change() {
        let turn1 = make_turn_id();
        let turn2 = make_turn_id();

        let input1 = SnapshotInput {
            turn_ids: vec![turn1.clone(), turn2.clone()],
            edges: vec![Edge::new(turn1.clone(), turn2.clone(), EdgeType::Reply)],
            timestamps: vec![1000, 2000],
        };

        let turn3 = make_turn_id();
        let input2 = SnapshotInput {
            turn_ids: vec![turn1.clone(), turn2.clone(), turn3.clone()],
            edges: vec![Edge::new(turn1.clone(), turn2.clone(), EdgeType::Reply)],
            timestamps: vec![1000, 2000, 3000],
        };

        let snapshot1 = GraphSnapshot::compute(&input1);
        let snapshot2 = GraphSnapshot::compute(&input2);

        assert_ne!(snapshot1.snapshot_id, snapshot2.snapshot_id);
    }

    #[test]
    fn test_snapshot_order_independence() {
        let turn1 = make_turn_id();
        let turn2 = make_turn_id();
        let turn3 = make_turn_id();

        // Input with turns in one order
        let input1 = SnapshotInput {
            turn_ids: vec![turn1.clone(), turn2.clone(), turn3.clone()],
            edges: vec![
                Edge::new(turn1.clone(), turn2.clone(), EdgeType::Reply),
                Edge::new(turn2.clone(), turn3.clone(), EdgeType::Reply),
            ],
            timestamps: vec![1000, 2000, 3000],
        };

        // Input with turns in different order (but same timestamps aligned)
        let input2 = SnapshotInput {
            turn_ids: vec![turn3.clone(), turn1.clone(), turn2.clone()],
            edges: vec![
                Edge::new(turn2.clone(), turn3.clone(), EdgeType::Reply),
                Edge::new(turn1.clone(), turn2.clone(), EdgeType::Reply),
            ],
            timestamps: vec![3000, 1000, 2000],
        };

        let snapshot1 = GraphSnapshot::compute(&input1);
        let snapshot2 = GraphSnapshot::compute(&input2);

        // Snapshots should be identical because we sort before hashing
        assert_eq!(snapshot1.snapshot_id, snapshot2.snapshot_id);
    }

    #[test]
    fn test_snapshot_verify() {
        let turn1 = make_turn_id();
        let turn2 = make_turn_id();

        let input = SnapshotInput {
            turn_ids: vec![turn1.clone(), turn2.clone()],
            edges: vec![Edge::new(turn1.clone(), turn2.clone(), EdgeType::Reply)],
            timestamps: vec![1000, 2000],
        };

        let snapshot = GraphSnapshot::compute(&input);
        assert!(snapshot.verify(&input));

        // Modify input - should fail verification
        let modified_input = SnapshotInput {
            turn_ids: vec![turn1.clone()],
            edges: vec![],
            timestamps: vec![1000],
        };
        assert!(!snapshot.verify(&modified_input));
    }
}

