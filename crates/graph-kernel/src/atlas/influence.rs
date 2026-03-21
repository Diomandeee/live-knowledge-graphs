//! Turn influence scoring for Atlas.
//!
//! Computes approximate global centrality by measuring how often
//! turns appear across different context slices.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::canonical::canonical_hash_hex;
use crate::types::{Phase, SliceExport};

/// Phase distribution counts.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PhaseCounts {
    /// Count of Exploration phase slices.
    pub exploration: u32,
    /// Count of Debugging phase slices.
    pub debugging: u32,
    /// Count of Planning phase slices.
    pub planning: u32,
    /// Count of Consolidation phase slices.
    pub consolidation: u32,
    /// Count of Synthesis phase slices.
    pub synthesis: u32,
}

impl PhaseCounts {
    /// Increment count for a phase.
    pub fn increment(&mut self, phase: Phase) {
        match phase {
            Phase::Exploration => self.exploration += 1,
            Phase::Debugging => self.debugging += 1,
            Phase::Planning => self.planning += 1,
            Phase::Consolidation => self.consolidation += 1,
            Phase::Synthesis => self.synthesis += 1,
        }
    }

    /// Total count across all phases.
    pub fn total(&self) -> u32 {
        self.exploration + self.debugging + self.planning + self.consolidation + self.synthesis
    }

    /// Check if this turn appears in multiple phases.
    pub fn is_cross_phase(&self) -> bool {
        let non_zero = [
            self.exploration > 0,
            self.debugging > 0,
            self.planning > 0,
            self.consolidation > 0,
            self.synthesis > 0,
        ];
        non_zero.iter().filter(|&&x| x).count() > 1
    }

    /// Get the dominant phase (most occurrences).
    pub fn dominant_phase(&self) -> Option<Phase> {
        let counts = [
            (Phase::Exploration, self.exploration),
            (Phase::Debugging, self.debugging),
            (Phase::Planning, self.planning),
            (Phase::Consolidation, self.consolidation),
            (Phase::Synthesis, self.synthesis),
        ];

        counts
            .into_iter()
            .filter(|(_, c)| *c > 0)
            .max_by_key(|(_, c)| *c)
            .map(|(p, _)| p)
    }
}

/// Influence score for a single turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TurnInfluence {
    /// Turn ID.
    pub turn_id: String,
    /// Number of slices containing this turn.
    pub slice_count: u32,
    /// Fraction of total slices containing this turn.
    pub slice_fraction: f32,
    /// Phase distribution of slices containing this turn.
    pub phase_distribution: PhaseCounts,
    /// Whether this turn bridges multiple phases.
    pub is_bridge: bool,
}

/// Collection of influence scores for all turns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfluenceScores {
    /// Individual turn influence scores (sorted by turn_id).
    pub scores: Vec<TurnInfluence>,
    /// Total number of slices analyzed.
    pub total_slices: usize,
    /// Content hash for integrity verification.
    pub scores_hash: String,
}

impl InfluenceScores {
    /// Create new influence scores from a list.
    pub fn new(mut scores: Vec<TurnInfluence>, total_slices: usize) -> Self {
        // Sort by turn_id for determinism
        scores.sort_by(|a, b| a.turn_id.cmp(&b.turn_id));

        let scores_hash = canonical_hash_hex(&scores);

        Self {
            scores,
            total_slices,
            scores_hash,
        }
    }

    /// Get influence for a specific turn.
    pub fn get(&self, turn_id: &str) -> Option<&TurnInfluence> {
        self.scores.iter().find(|s| s.turn_id == turn_id)
    }

    /// Get the most influential turns (highest slice_count).
    pub fn top_influential(&self, n: usize) -> Vec<&TurnInfluence> {
        let mut sorted: Vec<_> = self.scores.iter().collect();
        sorted.sort_by(|a, b| b.slice_count.cmp(&a.slice_count));
        sorted.into_iter().take(n).collect()
    }

    /// Get all bridge turns (appear in multiple phases).
    pub fn bridge_turns(&self) -> Vec<&TurnInfluence> {
        self.scores.iter().filter(|s| s.is_bridge).collect()
    }

    /// Get turns that appear in at least N slices.
    pub fn with_min_slices(&self, min: u32) -> Vec<&TurnInfluence> {
        self.scores.iter().filter(|s| s.slice_count >= min).collect()
    }
}

/// Compute influence scores from slices.
///
/// For each turn, counts how many slices it appears in and
/// what phases those slices represent.
pub fn compute_influence(slices: &[SliceExport]) -> InfluenceScores {
    // Map: turn_id -> (slice_count, phase_counts)
    let mut turn_data: BTreeMap<String, (u32, PhaseCounts)> = BTreeMap::new();

    let total_slices = slices.len();

    for slice in slices {
        // Determine the anchor's phase for this slice
        let anchor_phase = slice
            .turns
            .iter()
            .find(|t| t.id == slice.anchor_turn_id)
            .map(|t| t.phase)
            .unwrap_or(Phase::Exploration);

        for turn in &slice.turns {
            let turn_id = turn.id.as_uuid().to_string();
            let entry = turn_data.entry(turn_id).or_default();
            entry.0 += 1;
            entry.1.increment(anchor_phase);
        }
    }

    let scores: Vec<TurnInfluence> = turn_data
        .into_iter()
        .map(|(turn_id, (slice_count, phase_distribution))| {
            let slice_fraction = slice_count as f32 / total_slices as f32;
            let is_bridge = phase_distribution.is_cross_phase();

            TurnInfluence {
                turn_id,
                slice_count,
                slice_fraction,
                phase_distribution,
                is_bridge,
            }
        })
        .collect();

    InfluenceScores::new(scores, total_slices)
}

/// Bridge turn information for phase topology.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BridgeTurn {
    /// Turn ID.
    pub turn_id: String,
    /// Phases this turn bridges.
    pub bridged_phases: Vec<Phase>,
    /// Total slice appearances.
    pub total_appearances: u32,
}

/// Extract bridge turns from influence scores.
pub fn extract_bridges(scores: &InfluenceScores) -> Vec<BridgeTurn> {
    scores
        .bridge_turns()
        .iter()
        .map(|t| {
            let mut phases = Vec::new();
            let pd = &t.phase_distribution;
            if pd.exploration > 0 {
                phases.push(Phase::Exploration);
            }
            if pd.debugging > 0 {
                phases.push(Phase::Debugging);
            }
            if pd.planning > 0 {
                phases.push(Phase::Planning);
            }
            if pd.consolidation > 0 {
                phases.push(Phase::Consolidation);
            }
            if pd.synthesis > 0 {
                phases.push(Phase::Synthesis);
            }

            BridgeTurn {
                turn_id: t.turn_id.clone(),
                bridged_phases: phases,
                total_appearances: t.slice_count,
            }
        })
        .collect()
}

/// Phase topology analysis: how phases relate in the overlap space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTopologyStats {
    /// Average Jaccard overlap between slice pairs where anchors have different phases.
    /// Key format: "phase1_phase2" (alphabetically sorted).
    pub phase_pair_overlaps: BTreeMap<String, f32>,
    /// Representative slice IDs per phase (top N by connectivity).
    pub phase_centroids: BTreeMap<String, Vec<String>>,
    /// Turns that appear in slices of multiple phases.
    pub cross_phase_bridges: Vec<BridgeTurn>,
    /// Content hash.
    pub stats_hash: String,
}

impl PhaseTopologyStats {
    /// Create new phase topology stats.
    pub fn new(
        phase_pair_overlaps: BTreeMap<String, f32>,
        phase_centroids: BTreeMap<String, Vec<String>>,
        cross_phase_bridges: Vec<BridgeTurn>,
    ) -> Self {
        use crate::canonical::canonical_hash_hex;
        let hash_input = (&phase_pair_overlaps, &phase_centroids, &cross_phase_bridges);
        let stats_hash = canonical_hash_hex(&hash_input);

        Self {
            phase_pair_overlaps,
            phase_centroids,
            cross_phase_bridges,
            stats_hash,
        }
    }
}

/// Compute phase topology from slices and overlap data.
pub fn compute_phase_topology(
    slices: &[SliceExport],
    overlap_edges: &[super::OverlapEdge],
    max_centroids_per_phase: usize,
) -> PhaseTopologyStats {
    use std::collections::HashMap;
    
    // Build slice_id -> phase mapping
    let slice_phases: HashMap<String, Phase> = slices
        .iter()
        .map(|s| {
            let anchor_phase = s
                .turns
                .iter()
                .find(|t| t.id == s.anchor_turn_id)
                .map(|t| t.phase)
                .unwrap_or(Phase::Exploration);
            (s.slice_id.to_string(), anchor_phase)
        })
        .collect();

    // Compute phase pair overlaps
    let mut pair_sums: HashMap<String, (f32, usize)> = HashMap::new();
    for edge in overlap_edges {
        if let (Some(phase_a), Some(phase_b)) = (
            slice_phases.get(&edge.slice_a),
            slice_phases.get(&edge.slice_b),
        ) {
            if phase_a != phase_b {
                let key = make_phase_pair_key(*phase_a, *phase_b);
                let entry = pair_sums.entry(key).or_insert((0.0, 0));
                entry.0 += edge.jaccard;
                entry.1 += 1;
            }
        }
    }

    let phase_pair_overlaps: BTreeMap<String, f32> = pair_sums
        .into_iter()
        .map(|(k, (sum, count))| (k, sum / count as f32))
        .collect();

    // Compute phase centroids (most connected slices per phase)
    let mut phase_connectivity: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for edge in overlap_edges {
        if let Some(phase) = slice_phases.get(&edge.slice_a) {
            *phase_connectivity
                .entry(phase_to_string(*phase))
                .or_default()
                .entry(edge.slice_a.clone())
                .or_default() += 1;
        }
        if let Some(phase) = slice_phases.get(&edge.slice_b) {
            *phase_connectivity
                .entry(phase_to_string(*phase))
                .or_default()
                .entry(edge.slice_b.clone())
                .or_default() += 1;
        }
    }

    let phase_centroids: BTreeMap<String, Vec<String>> = phase_connectivity
        .into_iter()
        .map(|(phase, slices)| {
            let mut sorted: Vec<_> = slices.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let top: Vec<String> = sorted
                .into_iter()
                .take(max_centroids_per_phase)
                .map(|(id, _)| id)
                .collect();
            (phase, top)
        })
        .collect();

    // Extract bridge turns
    let influence = compute_influence(slices);
    let cross_phase_bridges = extract_bridges(&influence);

    PhaseTopologyStats::new(phase_pair_overlaps, phase_centroids, cross_phase_bridges)
}

fn make_phase_pair_key(a: Phase, b: Phase) -> String {
    let a_str = phase_to_string(a);
    let b_str = phase_to_string(b);
    if a_str < b_str {
        format!("{}_{}", a_str, b_str)
    } else {
        format!("{}_{}", b_str, a_str)
    }
}

fn phase_to_string(phase: Phase) -> String {
    match phase {
        Phase::Exploration => "exploration".to_string(),
        Phase::Debugging => "debugging".to_string(),
        Phase::Planning => "planning".to_string(),
        Phase::Consolidation => "consolidation".to_string(),
        Phase::Synthesis => "synthesis".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TurnId, TurnSnapshot, Role};
    use uuid::Uuid;

    fn make_turn(id: &str, phase: Phase) -> TurnSnapshot {
        TurnSnapshot::new(
            TurnId::new(Uuid::parse_str(id).unwrap()),
            "session".to_string(),
            Role::User,
            phase,
            0.5,
            0, 0, 0.5, 0.5, 1.0,
            1000,
        )
    }

    fn make_slice(_id: &str, turns: Vec<TurnSnapshot>) -> SliceExport {
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
    fn test_influence_computation() {
        let uuid1 = "00000000-0000-0000-0000-000000000001";
        let uuid2 = "00000000-0000-0000-0000-000000000002";
        let uuid3 = "00000000-0000-0000-0000-000000000003";

        // Turn 1 appears in both slices (2 times)
        // Turn 2 appears in slice A only (1 time)
        // Turn 3 appears in slice B only (1 time)
        let slice_a = make_slice("a", vec![
            make_turn(uuid1, Phase::Exploration),
            make_turn(uuid2, Phase::Exploration),
        ]);
        let slice_b = make_slice("b", vec![
            make_turn(uuid1, Phase::Synthesis),
            make_turn(uuid3, Phase::Synthesis),
        ]);

        let scores = compute_influence(&[slice_a, slice_b]);

        assert_eq!(scores.total_slices, 2);

        let turn1 = scores.get(uuid1).unwrap();
        assert_eq!(turn1.slice_count, 2);
        assert!((turn1.slice_fraction - 1.0).abs() < 0.01);
        assert!(turn1.is_bridge); // Appears in Exploration and Synthesis

        let turn2 = scores.get(uuid2).unwrap();
        assert_eq!(turn2.slice_count, 1);
        assert!(!turn2.is_bridge);

        let turn3 = scores.get(uuid3).unwrap();
        assert_eq!(turn3.slice_count, 1);
        assert!(!turn3.is_bridge);
    }

    #[test]
    fn test_top_influential() {
        let uuid1 = "00000000-0000-0000-0000-000000000001";
        let uuid2 = "00000000-0000-0000-0000-000000000002";

        // Turn 1 appears in 3 slices, Turn 2 in 1
        let slice_a = make_slice("a", vec![make_turn(uuid1, Phase::Exploration)]);
        let slice_b = make_slice("b", vec![make_turn(uuid1, Phase::Exploration)]);
        let slice_c = make_slice("c", vec![make_turn(uuid1, Phase::Exploration), make_turn(uuid2, Phase::Exploration)]);

        let scores = compute_influence(&[slice_a, slice_b, slice_c]);

        let top = scores.top_influential(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].turn_id, uuid1);
        assert_eq!(top[0].slice_count, 3);
    }

    #[test]
    fn test_bridge_extraction() {
        let uuid1 = "00000000-0000-0000-0000-000000000001";

        let slice_a = make_slice("a", vec![make_turn(uuid1, Phase::Exploration)]);
        let slice_b = make_slice("b", vec![make_turn(uuid1, Phase::Synthesis)]);

        let scores = compute_influence(&[slice_a, slice_b]);
        let bridges = extract_bridges(&scores);

        assert_eq!(bridges.len(), 1);
        assert_eq!(bridges[0].turn_id, uuid1);
        assert!(bridges[0].bridged_phases.contains(&Phase::Exploration));
        assert!(bridges[0].bridged_phases.contains(&Phase::Synthesis));
    }

    #[test]
    fn test_determinism() {
        let uuid1 = "00000000-0000-0000-0000-000000000001";
        let uuid2 = "00000000-0000-0000-0000-000000000002";

        let slice_a = make_slice("a", vec![make_turn(uuid1, Phase::Exploration)]);
        let slice_b = make_slice("b", vec![make_turn(uuid2, Phase::Synthesis)]);

        // Compute twice with different order
        let scores1 = compute_influence(&[slice_a.clone(), slice_b.clone()]);
        let scores2 = compute_influence(&[slice_b, slice_a]);

        assert_eq!(scores1.scores_hash, scores2.scores_hash);
    }
}

