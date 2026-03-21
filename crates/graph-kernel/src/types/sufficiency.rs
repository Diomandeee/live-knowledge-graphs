//! Sufficiency framework for evidence quality enforcement.
//!
//! ## Purpose
//!
//! This module prevents **gaming** of the promotion pipeline by requiring
//! evidence to meet minimum diversity criteria. A slice of 10 identical
//! low-quality turns should NOT be sufficient for promotion.
//!
//! ## Sufficiency Dimensions
//!
//! Evidence sufficiency is measured across multiple dimensions:
//!
//! | Dimension | What It Measures | Why It Matters |
//! |-----------|-----------------|----------------|
//! | **Role Diversity** | Mix of user/assistant turns | Single-party evidence is weak |
//! | **Phase Coverage** | Coverage of conversation phases | Single-phase evidence misses context |
//! | **Salience Spread** | Distribution of salience scores | All low-salience is suspicious |
//! | **Turn Count** | Minimum number of turns | Too few turns = insufficient context |
//! | **Unique Sessions** | Distinct session IDs | Cross-session evidence is stronger |
//!
//! ## Security Model
//!
//! `EvidenceBundle` wraps `AdmissibleEvidenceBundle` (verified slice) with
//! sufficiency metrics. This creates a two-layer enforcement:
//!
//! 1. **Admissibility**: Token verification (kernel-issued)
//! 2. **Sufficiency**: Diversity metrics (policy-enforced)
//!
//! Both layers must pass for evidence to be used in promotion.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::admissible::AdmissibleEvidenceBundle;
use super::turn::{TurnId, Role, Phase};

/// Diversity metrics computed from a slice's turns.
///
/// These metrics quantify how diverse the evidence is across multiple
/// dimensions. Higher diversity generally indicates stronger evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityMetrics {
    /// Total number of turns in the slice.
    pub turn_count: usize,

    /// Number of unique roles represented (user, assistant, system).
    pub unique_roles: usize,

    /// Breakdown of turns by role.
    pub role_distribution: HashMap<Role, usize>,

    /// Number of unique phases represented.
    pub unique_phases: usize,

    /// Breakdown of turns by phase.
    pub phase_distribution: HashMap<Phase, usize>,

    /// Number of unique session IDs.
    pub unique_sessions: usize,

    /// Salience statistics.
    pub salience_stats: SalienceStats,

    /// Whether there's meaningful conversation exchange (user + assistant).
    pub has_exchange: bool,
}

/// Statistical summary of salience scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalienceStats {
    /// Minimum salience in the slice.
    pub min: f32,
    /// Maximum salience in the slice.
    pub max: f32,
    /// Mean salience across all turns.
    pub mean: f32,
    /// Standard deviation of salience scores.
    pub std_dev: f32,
    /// Count of high-salience turns (>= 0.7).
    pub high_salience_count: usize,
}

impl DiversityMetrics {
    /// Compute diversity metrics from an admissible evidence bundle.
    pub fn from_bundle(bundle: &AdmissibleEvidenceBundle) -> Self {
        let slice = bundle.slice();
        let turns = &slice.turns;

        // Count roles
        let mut role_distribution: HashMap<Role, usize> = HashMap::new();
        for turn in turns {
            *role_distribution.entry(turn.role).or_insert(0) += 1;
        }

        // Count phases
        let mut phase_distribution: HashMap<Phase, usize> = HashMap::new();
        for turn in turns {
            *phase_distribution.entry(turn.phase).or_insert(0) += 1;
        }

        // Count unique sessions
        let unique_sessions: HashSet<_> = turns.iter().map(|t| &t.session_id).collect();

        // Compute salience stats
        let saliences: Vec<f32> = turns.iter().map(|t| t.salience).collect();
        let salience_stats = Self::compute_salience_stats(&saliences);

        // Check for meaningful exchange
        let has_user = role_distribution.contains_key(&Role::User);
        let has_assistant = role_distribution.contains_key(&Role::Assistant);
        let has_exchange = has_user && has_assistant;

        Self {
            turn_count: turns.len(),
            unique_roles: role_distribution.len(),
            role_distribution,
            unique_phases: phase_distribution.len(),
            phase_distribution,
            unique_sessions: unique_sessions.len(),
            salience_stats,
            has_exchange,
        }
    }

    fn compute_salience_stats(saliences: &[f32]) -> SalienceStats {
        if saliences.is_empty() {
            return SalienceStats {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                high_salience_count: 0,
            };
        }

        let min = saliences.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = saliences.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = saliences.iter().sum();
        let mean = sum / saliences.len() as f32;

        let variance: f32 = saliences.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f32>() / saliences.len() as f32;
        let std_dev = variance.sqrt();

        let high_salience_count = saliences.iter().filter(|&&s| s >= 0.7).count();

        SalienceStats {
            min,
            max,
            mean,
            std_dev,
            high_salience_count,
        }
    }
}

/// Policy defining minimum sufficiency requirements.
///
/// Evidence must meet ALL requirements to be considered sufficient.
/// This prevents gaming with homogeneous low-quality turns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SufficiencyPolicy {
    /// Minimum number of turns required.
    pub min_turns: usize,

    /// Minimum number of unique roles required (1 = any, 2 = must have exchange).
    pub min_roles: usize,

    /// Minimum number of unique phases required.
    pub min_phases: usize,

    /// Minimum number of high-salience turns required.
    pub min_high_salience: usize,

    /// Require meaningful exchange (user + assistant).
    pub require_exchange: bool,

    /// Minimum mean salience score.
    pub min_mean_salience: f32,
}

impl Default for SufficiencyPolicy {
    /// Default policy for production use.
    ///
    /// These defaults are intentionally strict to prevent gaming.
    fn default() -> Self {
        Self {
            min_turns: 3,           // At least 3 turns
            min_roles: 2,           // Must have user + assistant
            min_phases: 1,          // At least one phase
            min_high_salience: 1,   // At least one high-salience turn
            require_exchange: true, // Must be a conversation
            min_mean_salience: 0.3, // Average salience above threshold
        }
    }
}

impl SufficiencyPolicy {
    /// Create a lenient policy for testing.
    pub fn lenient() -> Self {
        Self {
            min_turns: 1,
            min_roles: 1,
            min_phases: 1,
            min_high_salience: 0,
            require_exchange: false,
            min_mean_salience: 0.0,
        }
    }

    /// Create a strict policy for high-stakes promotions.
    pub fn strict() -> Self {
        Self {
            min_turns: 5,
            min_roles: 2,
            min_phases: 2,
            min_high_salience: 2,
            require_exchange: true,
            min_mean_salience: 0.5,
        }
    }

    /// Check if metrics satisfy this policy.
    pub fn is_satisfied(&self, metrics: &DiversityMetrics) -> bool {
        metrics.turn_count >= self.min_turns
            && metrics.unique_roles >= self.min_roles
            && metrics.unique_phases >= self.min_phases
            && metrics.salience_stats.high_salience_count >= self.min_high_salience
            && (!self.require_exchange || metrics.has_exchange)
            && metrics.salience_stats.mean >= self.min_mean_salience
    }

    /// Get detailed violation report.
    pub fn check(&self, metrics: &DiversityMetrics) -> SufficiencyCheck {
        let mut violations = Vec::new();

        if metrics.turn_count < self.min_turns {
            violations.push(SufficiencyViolation::InsufficientTurns {
                required: self.min_turns,
                actual: metrics.turn_count,
            });
        }

        if metrics.unique_roles < self.min_roles {
            violations.push(SufficiencyViolation::InsufficientRoles {
                required: self.min_roles,
                actual: metrics.unique_roles,
            });
        }

        if metrics.unique_phases < self.min_phases {
            violations.push(SufficiencyViolation::InsufficientPhases {
                required: self.min_phases,
                actual: metrics.unique_phases,
            });
        }

        if metrics.salience_stats.high_salience_count < self.min_high_salience {
            violations.push(SufficiencyViolation::InsufficientHighSalience {
                required: self.min_high_salience,
                actual: metrics.salience_stats.high_salience_count,
            });
        }

        if self.require_exchange && !metrics.has_exchange {
            violations.push(SufficiencyViolation::NoExchange);
        }

        if metrics.salience_stats.mean < self.min_mean_salience {
            violations.push(SufficiencyViolation::LowMeanSalience {
                required: self.min_mean_salience,
                actual: metrics.salience_stats.mean,
            });
        }

        SufficiencyCheck {
            is_sufficient: violations.is_empty(),
            violations,
            metrics: metrics.clone(),
        }
    }
}

/// Result of a sufficiency check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SufficiencyCheck {
    /// Whether the evidence is sufficient.
    pub is_sufficient: bool,
    /// List of policy violations (empty if sufficient).
    pub violations: Vec<SufficiencyViolation>,
    /// The computed metrics.
    pub metrics: DiversityMetrics,
}

/// Specific sufficiency violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SufficiencyViolation {
    /// Not enough turns in the slice.
    InsufficientTurns {
        /// Minimum required.
        required: usize,
        /// Actual count.
        actual: usize,
    },
    /// Not enough distinct roles.
    InsufficientRoles {
        /// Minimum required.
        required: usize,
        /// Actual count.
        actual: usize,
    },
    /// Not enough distinct phases.
    InsufficientPhases {
        /// Minimum required.
        required: usize,
        /// Actual count.
        actual: usize,
    },
    /// Not enough high-salience turns.
    InsufficientHighSalience {
        /// Minimum required.
        required: usize,
        /// Actual count.
        actual: usize,
    },
    /// No meaningful conversation exchange.
    NoExchange,
    /// Mean salience too low.
    LowMeanSalience {
        /// Minimum required.
        required: f32,
        /// Actual value.
        actual: f32,
    },
}

impl std::fmt::Display for SufficiencyViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientTurns { required, actual } => {
                write!(f, "Insufficient turns: {} required, {} found", required, actual)
            }
            Self::InsufficientRoles { required, actual } => {
                write!(f, "Insufficient roles: {} required, {} found", required, actual)
            }
            Self::InsufficientPhases { required, actual } => {
                write!(f, "Insufficient phases: {} required, {} found", required, actual)
            }
            Self::InsufficientHighSalience { required, actual } => {
                write!(f, "Insufficient high-salience turns: {} required, {} found", required, actual)
            }
            Self::NoExchange => {
                write!(f, "No meaningful exchange: requires both user and assistant turns")
            }
            Self::LowMeanSalience { required, actual } => {
                write!(f, "Low mean salience: {:.2} required, {:.2} found", required, actual)
            }
        }
    }
}

/// Evidence bundle combining admissibility and sufficiency.
///
/// This is the highest-level evidence type, ensuring both:
/// 1. **Admissibility**: HMAC token verification (kernel authorization)
/// 2. **Sufficiency**: Diversity metrics (policy enforcement)
///
/// Use this type in promotion APIs to enforce both guarantees.
///
/// # Example
///
/// ```rust,ignore
/// // In promotion API handler:
/// fn promote_turn(
///     turn_id: TurnId,
///     evidence: &EvidenceBundle,  // Type enforces both checks
/// ) -> Result<(), PromotionError> {
///     // evidence is guaranteed to be:
///     // 1. Kernel-authorized (AdmissibleEvidenceBundle)
///     // 2. Sufficiently diverse (SufficiencyCheck passed)
///
///     // Safe to proceed with promotion
///     ...
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceBundle {
    /// The verified, admissible slice.
    bundle: AdmissibleEvidenceBundle,
    /// Computed diversity metrics.
    metrics: DiversityMetrics,
    /// Policy that was satisfied.
    policy_id: String,
}

/// Error when creating an evidence bundle.
#[derive(Debug, thiserror::Error)]
pub enum EvidenceBundleError {
    /// Sufficiency policy not satisfied.
    #[error("Evidence does not satisfy sufficiency policy: {0}")]
    InsufficientEvidence(String),
}

impl EvidenceBundle {
    /// Create an evidence bundle from an admissible bundle.
    ///
    /// Fails if the evidence does not satisfy the sufficiency policy.
    ///
    /// # Arguments
    /// * `bundle` - Verified admissible evidence
    /// * `policy` - Sufficiency policy to enforce
    /// * `policy_id` - Identifier for the policy (for audit trail)
    ///
    /// # Returns
    /// - `Ok(EvidenceBundle)` if evidence is sufficient
    /// - `Err(EvidenceBundleError)` if policy violated
    pub fn from_admissible(
        bundle: AdmissibleEvidenceBundle,
        policy: &SufficiencyPolicy,
        policy_id: impl Into<String>,
    ) -> Result<Self, EvidenceBundleError> {
        let metrics = DiversityMetrics::from_bundle(&bundle);
        let check = policy.check(&metrics);

        if !check.is_sufficient {
            let violations: Vec<String> = check.violations.iter()
                .map(|v| v.to_string())
                .collect();
            return Err(EvidenceBundleError::InsufficientEvidence(
                violations.join("; ")
            ));
        }

        Ok(Self {
            bundle,
            metrics,
            policy_id: policy_id.into(),
        })
    }

    /// Create an evidence bundle with lenient policy (for testing).
    #[cfg(test)]
    pub fn from_admissible_lenient(bundle: AdmissibleEvidenceBundle) -> Self {
        let metrics = DiversityMetrics::from_bundle(&bundle);
        Self {
            bundle,
            metrics,
            policy_id: "lenient_test".to_string(),
        }
    }

    /// Get the underlying admissible bundle.
    pub fn admissible_bundle(&self) -> &AdmissibleEvidenceBundle {
        &self.bundle
    }

    /// Get the diversity metrics.
    pub fn metrics(&self) -> &DiversityMetrics {
        &self.metrics
    }

    /// Get the anchor turn ID.
    pub fn anchor_turn_id(&self) -> TurnId {
        self.bundle.anchor_turn_id()
    }

    /// Get the policy ID that was satisfied.
    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    /// Get the number of turns.
    pub fn num_turns(&self) -> usize {
        self.metrics.turn_count
    }

    /// Check if a turn is admissible in this bundle.
    pub fn is_turn_admissible(&self, turn_id: &TurnId) -> bool {
        self.bundle.is_turn_admissible(turn_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TurnSnapshot, SliceExport, GraphSnapshotHash};
    use uuid::Uuid;

    fn make_turn(id: u128, role: Role, phase: Phase, salience: f32, session: &str) -> TurnSnapshot {
        TurnSnapshot::new(
            TurnId::new(Uuid::from_u128(id)),
            session.to_string(),
            role,
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

    fn make_admissible_bundle(turns: Vec<TurnSnapshot>) -> AdmissibleEvidenceBundle {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let anchor = turns[0].id;
        let snapshot = GraphSnapshotHash::new("test_snapshot".to_string());

        let slice = SliceExport::new_with_secret(
            secret,
            anchor,
            turns,
            vec![],
            "test_policy".to_string(),
            "params_hash".to_string(),
            snapshot,
        );

        AdmissibleEvidenceBundle::from_verified(slice, secret).unwrap()
    }

    #[test]
    fn test_diversity_metrics_single_turn() {
        let turns = vec![
            make_turn(1, Role::User, Phase::Exploration, 0.5, "session1"),
        ];
        let bundle = make_admissible_bundle(turns);
        let metrics = DiversityMetrics::from_bundle(&bundle);

        assert_eq!(metrics.turn_count, 1);
        assert_eq!(metrics.unique_roles, 1);
        assert_eq!(metrics.unique_phases, 1);
        assert_eq!(metrics.unique_sessions, 1);
        assert!(!metrics.has_exchange);
    }

    #[test]
    fn test_diversity_metrics_conversation() {
        let turns = vec![
            make_turn(1, Role::User, Phase::Exploration, 0.8, "session1"),
            make_turn(2, Role::Assistant, Phase::Planning, 0.6, "session1"),
            make_turn(3, Role::User, Phase::Synthesis, 0.9, "session1"),
        ];
        let bundle = make_admissible_bundle(turns);
        let metrics = DiversityMetrics::from_bundle(&bundle);

        assert_eq!(metrics.turn_count, 3);
        assert_eq!(metrics.unique_roles, 2);
        assert_eq!(metrics.unique_phases, 3);
        assert_eq!(metrics.unique_sessions, 1);
        assert!(metrics.has_exchange);
        assert_eq!(metrics.salience_stats.high_salience_count, 2); // 0.8 and 0.9
    }

    #[test]
    fn test_salience_stats() {
        let turns = vec![
            make_turn(1, Role::User, Phase::Exploration, 0.2, "s1"),
            make_turn(2, Role::Assistant, Phase::Exploration, 0.4, "s1"),
            make_turn(3, Role::User, Phase::Exploration, 0.6, "s1"),
            make_turn(4, Role::Assistant, Phase::Exploration, 0.8, "s1"),
        ];
        let bundle = make_admissible_bundle(turns);
        let metrics = DiversityMetrics::from_bundle(&bundle);

        assert!((metrics.salience_stats.min - 0.2).abs() < 0.001);
        assert!((metrics.salience_stats.max - 0.8).abs() < 0.001);
        assert!((metrics.salience_stats.mean - 0.5).abs() < 0.001);
        assert_eq!(metrics.salience_stats.high_salience_count, 1); // Only 0.8
    }

    #[test]
    fn test_sufficiency_policy_default_satisfied() {
        let turns = vec![
            make_turn(1, Role::User, Phase::Exploration, 0.8, "s1"),
            make_turn(2, Role::Assistant, Phase::Planning, 0.6, "s1"),
            make_turn(3, Role::User, Phase::Synthesis, 0.5, "s1"),
        ];
        let bundle = make_admissible_bundle(turns);
        let metrics = DiversityMetrics::from_bundle(&bundle);

        let policy = SufficiencyPolicy::default();
        assert!(policy.is_satisfied(&metrics));
    }

    #[test]
    fn test_sufficiency_policy_insufficient_turns() {
        let turns = vec![
            make_turn(1, Role::User, Phase::Exploration, 0.8, "s1"),
        ];
        let bundle = make_admissible_bundle(turns);
        let metrics = DiversityMetrics::from_bundle(&bundle);

        let policy = SufficiencyPolicy::default(); // min_turns = 3
        let check = policy.check(&metrics);

        assert!(!check.is_sufficient);
        assert!(check.violations.iter().any(|v| matches!(v, SufficiencyViolation::InsufficientTurns { .. })));
    }

    #[test]
    fn test_sufficiency_policy_no_exchange() {
        let turns = vec![
            make_turn(1, Role::User, Phase::Exploration, 0.8, "s1"),
            make_turn(2, Role::User, Phase::Planning, 0.7, "s1"),
            make_turn(3, Role::User, Phase::Synthesis, 0.9, "s1"),
        ];
        let bundle = make_admissible_bundle(turns);
        let metrics = DiversityMetrics::from_bundle(&bundle);

        let policy = SufficiencyPolicy::default(); // require_exchange = true
        let check = policy.check(&metrics);

        assert!(!check.is_sufficient);
        assert!(check.violations.iter().any(|v| matches!(v, SufficiencyViolation::NoExchange)));
    }

    #[test]
    fn test_sufficiency_policy_low_salience() {
        let turns = vec![
            make_turn(1, Role::User, Phase::Exploration, 0.1, "s1"),
            make_turn(2, Role::Assistant, Phase::Planning, 0.1, "s1"),
            make_turn(3, Role::User, Phase::Synthesis, 0.1, "s1"),
        ];
        let bundle = make_admissible_bundle(turns);
        let metrics = DiversityMetrics::from_bundle(&bundle);

        let policy = SufficiencyPolicy::default(); // min_high_salience = 1, min_mean_salience = 0.3
        let check = policy.check(&metrics);

        assert!(!check.is_sufficient);
        assert!(check.violations.iter().any(|v| matches!(v, SufficiencyViolation::InsufficientHighSalience { .. })));
        assert!(check.violations.iter().any(|v| matches!(v, SufficiencyViolation::LowMeanSalience { .. })));
    }

    #[test]
    fn test_evidence_bundle_creation_success() {
        let turns = vec![
            make_turn(1, Role::User, Phase::Exploration, 0.8, "s1"),
            make_turn(2, Role::Assistant, Phase::Planning, 0.6, "s1"),
            make_turn(3, Role::User, Phase::Synthesis, 0.5, "s1"),
        ];
        let admissible = make_admissible_bundle(turns);

        let policy = SufficiencyPolicy::default();
        let result = EvidenceBundle::from_admissible(admissible, &policy, "default_v1");

        assert!(result.is_ok());
        let bundle = result.unwrap();
        assert_eq!(bundle.num_turns(), 3);
        assert_eq!(bundle.policy_id(), "default_v1");
    }

    #[test]
    fn test_evidence_bundle_creation_failure() {
        let turns = vec![
            make_turn(1, Role::User, Phase::Exploration, 0.1, "s1"),
        ];
        let admissible = make_admissible_bundle(turns);

        let policy = SufficiencyPolicy::default();
        let result = EvidenceBundle::from_admissible(admissible, &policy, "default_v1");

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Insufficient turns"));
    }

    #[test]
    fn test_lenient_policy() {
        let turns = vec![
            make_turn(1, Role::User, Phase::Exploration, 0.1, "s1"),
        ];
        let bundle = make_admissible_bundle(turns);
        let metrics = DiversityMetrics::from_bundle(&bundle);

        let policy = SufficiencyPolicy::lenient();
        assert!(policy.is_satisfied(&metrics));
    }

    #[test]
    fn test_strict_policy() {
        let turns = vec![
            make_turn(1, Role::User, Phase::Exploration, 0.8, "s1"),
            make_turn(2, Role::Assistant, Phase::Planning, 0.7, "s1"),
            make_turn(3, Role::User, Phase::Synthesis, 0.6, "s1"),
        ];
        let bundle = make_admissible_bundle(turns);
        let metrics = DiversityMetrics::from_bundle(&bundle);

        let policy = SufficiencyPolicy::strict(); // min_turns = 5, min_phases = 2, min_high_salience = 2
        let check = policy.check(&metrics);

        // Should fail strict policy (only 3 turns, needs 5)
        assert!(!check.is_sufficient);
    }

    #[test]
    fn test_multi_session_diversity() {
        let turns = vec![
            make_turn(1, Role::User, Phase::Exploration, 0.8, "session_a"),
            make_turn(2, Role::Assistant, Phase::Planning, 0.7, "session_b"),
            make_turn(3, Role::User, Phase::Synthesis, 0.6, "session_c"),
        ];
        let bundle = make_admissible_bundle(turns);
        let metrics = DiversityMetrics::from_bundle(&bundle);

        assert_eq!(metrics.unique_sessions, 3);
    }

    #[test]
    fn test_violation_display() {
        let v = SufficiencyViolation::InsufficientTurns { required: 5, actual: 2 };
        assert_eq!(v.to_string(), "Insufficient turns: 5 required, 2 found");

        let v = SufficiencyViolation::NoExchange;
        assert!(v.to_string().contains("user and assistant"));
    }
}
