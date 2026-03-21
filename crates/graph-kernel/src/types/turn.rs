//! Turn types for the graph kernel.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::fmt;

/// Unique identifier for a turn in the conversation DAG.
///
/// Wraps a UUID and implements `Ord` for deterministic ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TurnId(Uuid);

impl TurnId {
    /// Create a new TurnId from a UUID.
    pub fn new(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Create a new TurnId from a UUID string.
    pub fn from_str(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }

    /// Get the inner UUID.
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }

    /// Generate a new random TurnId (for testing).
    #[cfg(test)]
    pub fn random() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for TurnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for TurnId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

/// Role of the turn author.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// User message.
    User,
    /// Assistant/AI response.
    Assistant,
    /// System message.
    System,
    /// Tool/function call result.
    Tool,
}

impl Role {
    /// Parse role from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "user" => Some(Self::User),
            "assistant" => Some(Self::Assistant),
            "system" => Some(Self::System),
            "tool" => Some(Self::Tool),
            _ => None,
        }
    }
}

impl Default for Role {
    fn default() -> Self {
        Self::User
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::System => write!(f, "system"),
            Self::Tool => write!(f, "tool"),
        }
    }
}

/// Trajectory phase of the turn.
///
/// Phases are ordered by their typical importance for context:
/// Synthesis > Planning > Consolidation > Debugging > Exploration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    /// Exploratory thinking, brainstorming.
    Exploration,
    /// Debugging, troubleshooting.
    Debugging,
    /// Planning, strategizing.
    Planning,
    /// Consolidating, summarizing.
    Consolidation,
    /// Synthesizing, creating new understanding.
    Synthesis,
}

impl Phase {
    /// Parse phase from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "exploration" => Some(Self::Exploration),
            "debugging" => Some(Self::Debugging),
            "planning" => Some(Self::Planning),
            "consolidation" => Some(Self::Consolidation),
            "synthesis" => Some(Self::Synthesis),
            _ => None,
        }
    }

    /// Get the default weight for this phase.
    pub fn default_weight(&self) -> f32 {
        match self {
            Self::Synthesis => 1.0,
            Self::Planning => 0.9,
            Self::Consolidation => 0.6,
            Self::Debugging => 0.5,
            Self::Exploration => 0.3,
        }
    }
}

impl Default for Phase {
    fn default() -> Self {
        Self::Exploration
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exploration => write!(f, "exploration"),
            Self::Debugging => write!(f, "debugging"),
            Self::Planning => write!(f, "planning"),
            Self::Consolidation => write!(f, "consolidation"),
            Self::Synthesis => write!(f, "synthesis"),
        }
    }
}

/// Snapshot of a turn for slicing.
///
/// Contains minimal fields needed for context selection.
/// Ordered by TurnId for deterministic serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSnapshot {
    /// Unique turn identifier.
    pub id: TurnId,
    /// Session/conversation identifier.
    pub session_id: String,
    /// Role of the author.
    pub role: Role,
    /// Trajectory phase.
    pub phase: Phase,
    /// Salience score [0, 1].
    pub salience: f32,
    /// Trajectory depth (distance from root).
    pub trajectory_depth: u32,
    /// Sibling order at this depth.
    pub trajectory_sibling_order: u32,
    /// Homogeneity with parent [0, 1].
    pub trajectory_homogeneity: f32,
    /// Temporal position [0, 1].
    pub trajectory_temporal: f32,
    /// Complexity score.
    pub trajectory_complexity: f32,
    /// Unix timestamp of creation.
    pub created_at: i64,
    /// SHA-256 hash of content_text for immutable graph snapshots.
    pub content_hash: Option<String>,
}

impl TurnSnapshot {
    /// Create a new turn snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: TurnId,
        session_id: String,
        role: Role,
        phase: Phase,
        salience: f32,
        trajectory_depth: u32,
        trajectory_sibling_order: u32,
        trajectory_homogeneity: f32,
        trajectory_temporal: f32,
        trajectory_complexity: f32,
        created_at: i64,
    ) -> Self {
        Self {
            id,
            session_id,
            role,
            phase,
            salience: salience.clamp(0.0, 1.0),
            trajectory_depth,
            trajectory_sibling_order,
            trajectory_homogeneity: trajectory_homogeneity.clamp(0.0, 1.0),
            trajectory_temporal: trajectory_temporal.clamp(0.0, 1.0),
            trajectory_complexity,
            created_at,
            content_hash: None,
        }
    }

    /// Create a new turn snapshot with content hash.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_content_hash(
        id: TurnId,
        session_id: String,
        role: Role,
        phase: Phase,
        salience: f32,
        trajectory_depth: u32,
        trajectory_sibling_order: u32,
        trajectory_homogeneity: f32,
        trajectory_temporal: f32,
        trajectory_complexity: f32,
        created_at: i64,
        content_hash: Option<String>,
    ) -> Self {
        Self {
            id,
            session_id,
            role,
            phase,
            salience: salience.clamp(0.0, 1.0),
            trajectory_depth,
            trajectory_sibling_order,
            trajectory_homogeneity: trajectory_homogeneity.clamp(0.0, 1.0),
            trajectory_temporal: trajectory_temporal.clamp(0.0, 1.0),
            trajectory_complexity,
            created_at,
            content_hash,
        }
    }

    /// Set the content hash on an existing TurnSnapshot.
    pub fn with_content_hash(mut self, content_hash: Option<String>) -> Self {
        self.content_hash = content_hash;
        self
    }

    /// Verify content hash matches actual content.
    ///
    /// # Arguments
    /// * `content` - The actual content text to verify against the stored hash
    ///
    /// # Returns
    /// * `Ok(())` if hash matches or no hash is stored (legacy data)
    /// * `Err(ContentHashError::Mismatch)` if hash doesn't match (tampering/corruption)
    ///
    /// # Security
    /// This enforces **INV-GK-004: Content Immutability**.
    /// Returns an error if a stored hash doesn't match the content.
    pub fn verify_content_hash(&self, content: &str) -> Result<(), ContentHashError> {
        use crate::canonical_content::validate_content_hash;
        use crate::canonical_content::HashValidation;

        match validate_content_hash(content, self.content_hash.as_deref()) {
            HashValidation::Valid => Ok(()),
            HashValidation::Missing => Ok(()), // Legacy data: no hash stored
            HashValidation::Mismatch { expected, computed } => {
                Err(ContentHashError::Mismatch {
                    turn_id: self.id,
                    stored: expected,
                    computed,
                })
            }
        }
    }

    /// Check if this turn has a content hash.
    pub fn has_content_hash(&self) -> bool {
        self.content_hash.is_some()
    }
}

/// Error when content hash verification fails.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ContentHashError {
    /// Stored hash doesn't match computed hash (tampering or corruption).
    #[error("Content hash mismatch for turn {turn_id}: stored={stored}, computed={computed}")]
    Mismatch {
        /// The turn ID where the mismatch was detected.
        turn_id: TurnId,
        /// The hash that was stored in the database.
        stored: String,
        /// The hash computed from the actual content.
        computed: String,
    },
}

// Implement Ord for TurnSnapshot based on TurnId for deterministic ordering
impl PartialEq for TurnSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for TurnSnapshot {}

impl PartialOrd for TurnSnapshot {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TurnSnapshot {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turn_id_ordering() {
        let id1 = TurnId::from_str("00000000-0000-0000-0000-000000000001").unwrap();
        let id2 = TurnId::from_str("00000000-0000-0000-0000-000000000002").unwrap();
        assert!(id1 < id2);
    }

    #[test]
    fn test_phase_weights() {
        assert!(Phase::Synthesis.default_weight() > Phase::Planning.default_weight());
        assert!(Phase::Planning.default_weight() > Phase::Consolidation.default_weight());
        assert!(Phase::Consolidation.default_weight() > Phase::Debugging.default_weight());
        assert!(Phase::Debugging.default_weight() > Phase::Exploration.default_weight());
    }

    #[test]
    fn test_role_parsing() {
        assert_eq!(Role::from_str("user"), Some(Role::User));
        assert_eq!(Role::from_str("ASSISTANT"), Some(Role::Assistant));
        assert_eq!(Role::from_str("invalid"), None);
    }

    #[test]
    fn test_content_hash_verification_valid() {
        use crate::canonical_content::compute_content_hash;

        let content = "Hello World";
        let hash = compute_content_hash(content);

        let turn = TurnSnapshot::new_with_content_hash(
            TurnId::from_str("00000000-0000-0000-0000-000000000001").unwrap(),
            "session_1".to_string(),
            Role::User,
            Phase::Exploration,
            0.5,
            0, 0, 0.5, 0.5, 1.0,
            1000,
            Some(hash),
        );

        assert!(turn.verify_content_hash(content).is_ok());
    }

    #[test]
    fn test_content_hash_verification_mismatch() {
        use crate::canonical_content::compute_content_hash;

        let original = "Hello World";
        let hash = compute_content_hash(original);

        let turn = TurnSnapshot::new_with_content_hash(
            TurnId::from_str("00000000-0000-0000-0000-000000000001").unwrap(),
            "session_1".to_string(),
            Role::User,
            Phase::Exploration,
            0.5,
            0, 0, 0.5, 0.5, 1.0,
            1000,
            Some(hash),
        );

        // Tampered content should fail
        let result = turn.verify_content_hash("Hello Tampered World");
        assert!(result.is_err());
        match result.unwrap_err() {
            ContentHashError::Mismatch { turn_id, .. } => {
                assert_eq!(turn_id, turn.id);
            }
        }
    }

    #[test]
    fn test_content_hash_verification_missing() {
        // Legacy turn without content hash
        let turn = TurnSnapshot::new(
            TurnId::from_str("00000000-0000-0000-0000-000000000001").unwrap(),
            "session_1".to_string(),
            Role::User,
            Phase::Exploration,
            0.5,
            0, 0, 0.5, 0.5, 1.0,
            1000,
        );

        // Should succeed (legacy data has no hash to verify)
        assert!(turn.verify_content_hash("Any content").is_ok());
        assert!(!turn.has_content_hash());
    }

    #[test]
    fn test_has_content_hash() {
        use crate::canonical_content::compute_content_hash;

        let turn_no_hash = TurnSnapshot::new(
            TurnId::from_str("00000000-0000-0000-0000-000000000001").unwrap(),
            "session_1".to_string(),
            Role::User,
            Phase::Exploration,
            0.5,
            0, 0, 0.5, 0.5, 1.0,
            1000,
        );

        let turn_with_hash = TurnSnapshot::new_with_content_hash(
            TurnId::from_str("00000000-0000-0000-0000-000000000002").unwrap(),
            "session_1".to_string(),
            Role::User,
            Phase::Exploration,
            0.5,
            0, 0, 0.5, 0.5, 1.0,
            1000,
            Some(compute_content_hash("test")),
        );

        assert!(!turn_no_hash.has_content_hash());
        assert!(turn_with_hash.has_content_hash());
    }
}

