//! SlicePolicy v1: Phase-weighted priority frontier with budgets.
//!
//! ## Float Normalization for Deterministic Hashing
//!
//! Floats are quantized to integers before hashing to avoid cross-platform
//! and cross-language serialization differences. The quantization factor
//! is 1e6 (multiply by 1,000,000 and round to i64).
//!
//! This ensures both Rust and Python produce identical `params_hash` values.

use serde::{Deserialize, Serialize};
use crate::canonical::canonical_hash_hex;
use crate::types::Phase;
use crate::DEFAULT_POLICY_VERSION;

/// Quantization factor for float normalization.
/// Floats are multiplied by this value and rounded to i64.
const FLOAT_QUANTIZATION_FACTOR: f64 = 1_000_000.0;

/// Phase weights for priority scoring.
///
/// Higher weight = higher priority in slice selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseWeights {
    /// Weight for Synthesis phase (highest importance).
    pub synthesis: f32,
    /// Weight for Planning phase.
    pub planning: f32,
    /// Weight for Consolidation phase.
    pub consolidation: f32,
    /// Weight for Debugging phase.
    pub debugging: f32,
    /// Weight for Exploration phase (lowest importance).
    pub exploration: f32,
}

impl PhaseWeights {
    /// Create new phase weights.
    pub fn new(
        synthesis: f32,
        planning: f32,
        consolidation: f32,
        debugging: f32,
        exploration: f32,
    ) -> Self {
        Self {
            synthesis,
            planning,
            consolidation,
            debugging,
            exploration,
        }
    }

    /// Get weight for a phase.
    pub fn get(&self, phase: Phase) -> f32 {
        match phase {
            Phase::Synthesis => self.synthesis,
            Phase::Planning => self.planning,
            Phase::Consolidation => self.consolidation,
            Phase::Debugging => self.debugging,
            Phase::Exploration => self.exploration,
        }
    }
}

impl Default for PhaseWeights {
    fn default() -> Self {
        Self {
            synthesis: 1.0,
            planning: 0.9,
            consolidation: 0.6,
            debugging: 0.5,
            exploration: 0.3,
        }
    }
}

impl PhaseWeights {
    /// Convert to quantized representation for deterministic hashing.
    fn to_quantized(&self) -> QuantizedPhaseWeights {
        QuantizedPhaseWeights {
            synthesis: quantize_float(self.synthesis),
            planning: quantize_float(self.planning),
            consolidation: quantize_float(self.consolidation),
            debugging: quantize_float(self.debugging),
            exploration: quantize_float(self.exploration),
        }
    }
}

/// Quantize a float to an i64 for deterministic hashing.
fn quantize_float(value: f32) -> i64 {
    ((value as f64) * FLOAT_QUANTIZATION_FACTOR).round() as i64
}

/// Quantized phase weights for deterministic hashing.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantizedPhaseWeights {
    synthesis: i64,
    planning: i64,
    consolidation: i64,
    debugging: i64,
    exploration: i64,
}

/// Quantized policy parameters for deterministic hashing.
///
/// All floats are quantized to i64 to ensure cross-platform consistency.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantizedPolicyParams {
    version: String,
    max_nodes: usize,
    max_radius: u32,
    phase_weights: QuantizedPhaseWeights,
    salience_weight: i64,
    distance_decay: i64,
    include_siblings: bool,
    max_siblings_per_node: usize,
}

/// Slice policy version 1.
///
/// Controls how context slices are expanded around an anchor turn.
///
/// ## Parameters
///
/// - `max_nodes`: Maximum turns in the slice (budget cap)
/// - `max_radius`: Maximum graph distance from anchor (hop limit)
/// - `phase_weights`: Importance weights per phase
/// - `salience_weight`: How much salience contributes to priority
/// - `distance_decay`: Priority decay per hop (0.9 = 10% loss per hop)
/// - `include_siblings`: Whether to include sibling turns
/// - `max_siblings_per_node`: Limit on siblings per parent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlicePolicyV1 {
    /// Policy version identifier.
    pub version: String,
    /// Maximum number of turns in the slice.
    pub max_nodes: usize,
    /// Maximum graph distance from anchor.
    pub max_radius: u32,
    /// Phase importance weights.
    pub phase_weights: PhaseWeights,
    /// Weight for salience in priority scoring.
    pub salience_weight: f32,
    /// Priority decay per hop (0.0-1.0).
    pub distance_decay: f32,
    /// Whether to include sibling turns.
    pub include_siblings: bool,
    /// Maximum siblings to include per parent.
    pub max_siblings_per_node: usize,
}

impl SlicePolicyV1 {
    /// Create a new policy with custom parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_nodes: usize,
        max_radius: u32,
        phase_weights: PhaseWeights,
        salience_weight: f32,
        distance_decay: f32,
        include_siblings: bool,
        max_siblings_per_node: usize,
    ) -> Self {
        Self {
            version: DEFAULT_POLICY_VERSION.to_string(),
            max_nodes,
            max_radius,
            phase_weights,
            salience_weight: salience_weight.clamp(0.0, 1.0),
            distance_decay: distance_decay.clamp(0.0, 1.0),
            include_siblings,
            max_siblings_per_node,
        }
    }

    /// Get the policy ID.
    pub fn policy_id(&self) -> &str {
        &self.version
    }

    /// Compute a hash of the policy parameters.
    ///
    /// Uses quantized float representation to ensure cross-platform consistency.
    /// Floats are multiplied by 1e6 and rounded to i64 before hashing.
    ///
    /// This ensures identical `params_hash` values across:
    /// - Rust and Python
    /// - Different float serialization settings
    /// - Different serde_json versions
    pub fn params_hash(&self) -> String {
        let quantized = self.to_quantized();
        canonical_hash_hex(&quantized)
    }

    /// Convert to quantized representation for deterministic hashing.
    fn to_quantized(&self) -> QuantizedPolicyParams {
        QuantizedPolicyParams {
            version: self.version.clone(),
            max_nodes: self.max_nodes,
            max_radius: self.max_radius,
            phase_weights: self.phase_weights.to_quantized(),
            salience_weight: quantize_float(self.salience_weight),
            distance_decay: quantize_float(self.distance_decay),
            include_siblings: self.include_siblings,
            max_siblings_per_node: self.max_siblings_per_node,
        }
    }

    /// Create a minimal policy for testing.
    #[cfg(test)]
    pub fn minimal() -> Self {
        Self {
            version: DEFAULT_POLICY_VERSION.to_string(),
            max_nodes: 10,
            max_radius: 3,
            phase_weights: PhaseWeights::default(),
            salience_weight: 0.3,
            distance_decay: 0.9,
            include_siblings: false,
            max_siblings_per_node: 0,
        }
    }
}

impl Default for SlicePolicyV1 {
    fn default() -> Self {
        Self {
            version: DEFAULT_POLICY_VERSION.to_string(),
            max_nodes: 256,
            max_radius: 10,
            phase_weights: PhaseWeights::default(),
            salience_weight: 0.3,
            distance_decay: 0.9,
            include_siblings: true,
            max_siblings_per_node: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_weights_get() {
        let weights = PhaseWeights::default();
        assert_eq!(weights.get(Phase::Synthesis), 1.0);
        assert_eq!(weights.get(Phase::Exploration), 0.3);
    }

    #[test]
    fn test_policy_params_hash_determinism() {
        let policy1 = SlicePolicyV1::default();
        let policy2 = SlicePolicyV1::default();

        assert_eq!(policy1.params_hash(), policy2.params_hash());
    }

    #[test]
    fn test_policy_params_hash_changes() {
        let policy1 = SlicePolicyV1::default();
        let mut policy2 = SlicePolicyV1::default();
        policy2.max_nodes = 128; // Change a parameter

        assert_ne!(policy1.params_hash(), policy2.params_hash());
    }
}

