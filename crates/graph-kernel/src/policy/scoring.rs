//! Priority scoring for slice expansion.

use crate::types::TurnSnapshot;
use super::v1::SlicePolicyV1;

/// Compute priority score for a turn at a given distance from anchor.
///
/// Higher score = higher priority for inclusion in slice.
///
/// Formula:
/// ```text
/// priority = (phase_weight + salience * salience_weight) * distance_decay^distance
/// ```
///
/// ## Parameters
///
/// - `turn`: The turn being scored
/// - `distance`: Graph distance from the anchor turn (0 for anchor itself)
/// - `policy`: The slice policy with weights and decay
pub fn priority_score(turn: &TurnSnapshot, distance: u32, policy: &SlicePolicyV1) -> f32 {
    let phase_score = policy.phase_weights.get(turn.phase);
    let salience_score = turn.salience * policy.salience_weight;
    let distance_penalty = policy.distance_decay.powi(distance as i32);

    (phase_score + salience_score) * distance_penalty
}

/// Candidate turn for expansion with its priority and distance.
#[derive(Debug, Clone)]
pub struct ExpansionCandidate {
    /// Turn snapshot.
    pub turn: TurnSnapshot,
    /// Graph distance from anchor.
    pub distance: u32,
    /// Computed priority score.
    pub priority: f32,
}

impl ExpansionCandidate {
    /// Create a new expansion candidate.
    pub fn new(turn: TurnSnapshot, distance: u32, policy: &SlicePolicyV1) -> Self {
        let priority = priority_score(&turn, distance, policy);
        Self {
            turn,
            distance,
            priority,
        }
    }
}

// Implement ordering for priority queue (max-heap by priority)
impl PartialEq for ExpansionCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.turn.id == other.turn.id
    }
}

impl Eq for ExpansionCandidate {}

impl PartialOrd for ExpansionCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TurnId, Role, Phase};
    use uuid::Uuid;

    fn make_turn(id: u128, salience: f32, phase: Phase) -> TurnSnapshot {
        TurnSnapshot::new(
            TurnId::new(Uuid::from_u128(id)),
            "session_1".to_string(),
            Role::User,
            phase,
            salience,
            1,
            0,
            0.5,
            0.5,
            1.0,
            1000,
        )
    }

    #[test]
    fn test_priority_score_phase_matters() {
        let policy = SlicePolicyV1::default();

        let synthesis_turn = make_turn(1, 0.5, Phase::Synthesis);
        let exploration_turn = make_turn(2, 0.5, Phase::Exploration);

        let score1 = priority_score(&synthesis_turn, 0, &policy);
        let score2 = priority_score(&exploration_turn, 0, &policy);

        assert!(score1 > score2, "Synthesis should score higher than Exploration");
    }

    #[test]
    fn test_priority_score_salience_matters() {
        let policy = SlicePolicyV1::default();

        let high_salience = make_turn(1, 1.0, Phase::Consolidation);
        let low_salience = make_turn(2, 0.0, Phase::Consolidation);

        let score1 = priority_score(&high_salience, 0, &policy);
        let score2 = priority_score(&low_salience, 0, &policy);

        assert!(score1 > score2, "High salience should score higher");
    }

    #[test]
    fn test_priority_score_distance_decay() {
        let policy = SlicePolicyV1::default();
        let turn = make_turn(1, 0.5, Phase::Planning);

        let score0 = priority_score(&turn, 0, &policy);
        let score1 = priority_score(&turn, 1, &policy);
        let score5 = priority_score(&turn, 5, &policy);

        assert!(score0 > score1, "Score should decrease with distance");
        assert!(score1 > score5, "Score should decrease with distance");
    }

    #[test]
    fn test_candidate_ordering() {
        let policy = SlicePolicyV1::default();

        let c1 = ExpansionCandidate::new(make_turn(1, 1.0, Phase::Synthesis), 0, &policy);
        let c2 = ExpansionCandidate::new(make_turn(2, 0.5, Phase::Exploration), 0, &policy);
        let c3 = ExpansionCandidate::new(make_turn(3, 1.0, Phase::Synthesis), 5, &policy);

        // c1 > c2 (synthesis > exploration)
        assert!(c1 > c2);
        // c1 > c3 (same phase but closer distance)
        assert!(c1 > c3);
    }
}

