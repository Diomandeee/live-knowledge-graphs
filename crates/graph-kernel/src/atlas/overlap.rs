//! Slice overlap computation for Atlas.
//!
//! Computes structural relationships between slices based on shared turns.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::canonical::canonical_hash_hex;
use crate::types::SliceExport;

/// An edge in the slice overlap graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OverlapEdge {
    /// First slice ID.
    pub slice_a: String,
    /// Second slice ID.
    pub slice_b: String,
    /// Number of shared turns.
    pub shared_turns: usize,
    /// Jaccard similarity: |A ∩ B| / |A ∪ B|.
    pub jaccard: f32,
}

impl OverlapEdge {
    /// Create a new overlap edge.
    pub fn new(slice_a: String, slice_b: String, shared_turns: usize, jaccard: f32) -> Self {
        // Ensure canonical ordering
        let (slice_a, slice_b) = if slice_a < slice_b {
            (slice_a, slice_b)
        } else {
            (slice_b, slice_a)
        };

        Self {
            slice_a,
            slice_b,
            shared_turns,
            jaccard,
        }
    }
}

/// The complete slice overlap graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapGraph {
    /// All overlap edges (slice pairs with non-zero overlap).
    pub edges: Vec<OverlapEdge>,
    /// Total number of slices in the graph.
    pub slice_count: usize,
    /// Content hash for integrity verification.
    pub graph_hash: String,
    /// Minimum Jaccard threshold used (0.0 means all overlaps included).
    pub min_jaccard: f32,
}

impl OverlapGraph {
    /// Create a new overlap graph from edges.
    pub fn new(edges: Vec<OverlapEdge>, slice_count: usize, min_jaccard: f32) -> Self {
        let graph_hash = canonical_hash_hex(&edges);
        Self {
            edges,
            slice_count,
            graph_hash,
            min_jaccard,
        }
    }

    /// Get all edges for a given slice.
    pub fn edges_for_slice(&self, slice_id: &str) -> Vec<&OverlapEdge> {
        self.edges
            .iter()
            .filter(|e| e.slice_a == slice_id || e.slice_b == slice_id)
            .collect()
    }

    /// Get the neighbor slice IDs for a given slice.
    pub fn neighbors(&self, slice_id: &str) -> Vec<&str> {
        self.edges_for_slice(slice_id)
            .iter()
            .map(|e| {
                if e.slice_a == slice_id {
                    e.slice_b.as_str()
                } else {
                    e.slice_a.as_str()
                }
            })
            .collect()
    }

    /// Find highly connected slices (hubs).
    pub fn hub_slices(&self, min_degree: usize) -> Vec<(String, usize)> {
        let mut degrees: BTreeMap<&str, usize> = BTreeMap::new();

        for edge in &self.edges {
            *degrees.entry(&edge.slice_a).or_default() += 1;
            *degrees.entry(&edge.slice_b).or_default() += 1;
        }

        let mut hubs: Vec<_> = degrees
            .into_iter()
            .filter(|(_, d)| *d >= min_degree)
            .map(|(id, d)| (id.to_string(), d))
            .collect();

        hubs.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by degree descending
        hubs
    }
}

/// Analyzer for computing slice overlaps.
pub struct OverlapAnalyzer {
    /// Minimum Jaccard similarity to include an edge.
    pub min_jaccard: f32,
}

impl OverlapAnalyzer {
    /// Create a new analyzer with default settings.
    pub fn new() -> Self {
        Self { min_jaccard: 0.0 }
    }

    /// Create an analyzer with a minimum Jaccard threshold.
    pub fn with_min_jaccard(min_jaccard: f32) -> Self {
        Self { min_jaccard }
    }

    /// Compute the overlap graph from a set of slices.
    pub fn compute(&self, slices: &[SliceExport]) -> OverlapGraph {
        // Build turn sets for each slice
        let slice_turns: Vec<(String, BTreeSet<String>)> = slices
            .iter()
            .map(|s| {
                let turns: BTreeSet<String> = s
                    .turns
                    .iter()
                    .map(|t| t.id.as_uuid().to_string())
                    .collect();
                (s.slice_id.to_string(), turns)
            })
            .collect();

        let mut edges = Vec::new();

        // Compare all pairs
        for i in 0..slice_turns.len() {
            for j in (i + 1)..slice_turns.len() {
                let (id_a, turns_a) = &slice_turns[i];
                let (id_b, turns_b) = &slice_turns[j];

                let intersection: BTreeSet<_> = turns_a.intersection(turns_b).collect();
                let shared = intersection.len();

                if shared > 0 {
                    let union_size = turns_a.len() + turns_b.len() - shared;
                    let jaccard = shared as f32 / union_size as f32;

                    if jaccard >= self.min_jaccard {
                        edges.push(OverlapEdge::new(
                            id_a.clone(),
                            id_b.clone(),
                            shared,
                            jaccard,
                        ));
                    }
                }
            }
        }

        // Sort edges for determinism
        edges.sort_by(|a, b| {
            (&a.slice_a, &a.slice_b).cmp(&(&b.slice_a, &b.slice_b))
        });

        OverlapGraph::new(edges, slices.len(), self.min_jaccard)
    }
}

impl Default for OverlapAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TurnId, TurnSnapshot, Phase, Role};
    use uuid::Uuid;

    fn make_turn(id: &str) -> TurnSnapshot {
        TurnSnapshot::new(
            TurnId::new(Uuid::parse_str(id).unwrap()),
            "session".to_string(),
            Role::User,
            Phase::Exploration,
            0.5,
            0, 0, 0.5, 0.5, 1.0,
            1000,
        )
    }

    fn make_slice(_id: &str, turn_ids: &[&str]) -> SliceExport {
        let turns: Vec<TurnSnapshot> = turn_ids.iter().map(|t| make_turn(t)).collect();
        let anchor = turns[0].id.clone();

        // Use SliceExport::new_for_test for unit tests
        SliceExport::new_for_test(
            anchor,
            turns,
            vec![],
            "test".to_string(),
            "hash".to_string(),
        )
    }

    #[test]
    fn test_overlap_computation() {
        let uuid1 = "00000000-0000-0000-0000-000000000001";
        let uuid2 = "00000000-0000-0000-0000-000000000002";
        let uuid3 = "00000000-0000-0000-0000-000000000003";
        let uuid4 = "00000000-0000-0000-0000-000000000004";

        // Slice A: {1, 2, 3}
        // Slice B: {2, 3, 4}
        // Overlap: {2, 3} -> Jaccard = 2/4 = 0.5
        let slice_a = make_slice("slice_a", &[uuid1, uuid2, uuid3]);
        let slice_b = make_slice("slice_b", &[uuid2, uuid3, uuid4]);

        let analyzer = OverlapAnalyzer::new();
        let graph = analyzer.compute(&[slice_a, slice_b]);

        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].shared_turns, 2);
        assert!((graph.edges[0].jaccard - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_no_overlap() {
        let uuid1 = "00000000-0000-0000-0000-000000000001";
        let uuid2 = "00000000-0000-0000-0000-000000000002";
        let uuid3 = "00000000-0000-0000-0000-000000000003";
        let uuid4 = "00000000-0000-0000-0000-000000000004";

        // Disjoint slices
        let slice_a = make_slice("slice_a", &[uuid1, uuid2]);
        let slice_b = make_slice("slice_b", &[uuid3, uuid4]);

        let analyzer = OverlapAnalyzer::new();
        let graph = analyzer.compute(&[slice_a, slice_b]);

        assert_eq!(graph.edges.len(), 0);
    }

    #[test]
    fn test_min_jaccard_filter() {
        let uuid1 = "00000000-0000-0000-0000-000000000001";
        let uuid2 = "00000000-0000-0000-0000-000000000002";
        let uuid3 = "00000000-0000-0000-0000-000000000003";
        let uuid4 = "00000000-0000-0000-0000-000000000004";
        let uuid5 = "00000000-0000-0000-0000-000000000005";

        // Slice A: {1, 2, 3, 4, 5} (5 turns)
        // Slice B: {1} (1 turn)
        // Overlap: {1} -> Jaccard = 1/5 = 0.2
        let slice_a = make_slice("slice_a", &[uuid1, uuid2, uuid3, uuid4, uuid5]);
        let slice_b = make_slice("slice_b", &[uuid1]);

        // Without filter
        let analyzer = OverlapAnalyzer::new();
        let graph = analyzer.compute(&[slice_a.clone(), slice_b.clone()]);
        assert_eq!(graph.edges.len(), 1);

        // With filter > 0.2
        let analyzer = OverlapAnalyzer::with_min_jaccard(0.3);
        let graph = analyzer.compute(&[slice_a, slice_b]);
        assert_eq!(graph.edges.len(), 0);
    }

    #[test]
    fn test_hub_detection() {
        let uuid1 = "00000000-0000-0000-0000-000000000001";
        let uuid2 = "00000000-0000-0000-0000-000000000002";
        let uuid3 = "00000000-0000-0000-0000-000000000003";

        // Slice A shares with B and C
        // Slice B shares with A
        // Slice C shares with A
        // A is a hub with degree 2
        let slice_a = make_slice("slice_a", &[uuid1, uuid2]);
        let slice_b = make_slice("slice_b", &[uuid1, uuid3]);
        let slice_c = make_slice("slice_c", &[uuid2, uuid3]);

        let analyzer = OverlapAnalyzer::new();
        let graph = analyzer.compute(&[slice_a, slice_b, slice_c]);

        // All slices should have degree 2 (each overlaps with the other 2)
        let hubs = graph.hub_slices(2);
        assert_eq!(hubs.len(), 3);
    }

    #[test]
    fn test_determinism() {
        let uuid1 = "00000000-0000-0000-0000-000000000001";
        let uuid2 = "00000000-0000-0000-0000-000000000002";
        let uuid3 = "00000000-0000-0000-0000-000000000003";

        let slice_a = make_slice("slice_a", &[uuid1, uuid2]);
        let slice_b = make_slice("slice_b", &[uuid2, uuid3]);

        let analyzer = OverlapAnalyzer::new();

        // Compute twice with different input order
        let graph1 = analyzer.compute(&[slice_a.clone(), slice_b.clone()]);
        let graph2 = analyzer.compute(&[slice_b, slice_a]);

        assert_eq!(graph1.graph_hash, graph2.graph_hash);
    }
}

