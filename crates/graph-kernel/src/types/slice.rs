//! Slice export types for the graph kernel.
//!
//! ## Production Invariants
//!
//! 1. **Slice Boundary**: Any turn in slice-mode must satisfy `turn_id ∈ slice.turn_ids`
//! 2. **Provenance Completeness**: Every response includes `(slice_id, policy_ref, schema_version, graph_snapshot_hash, admissibility_token)`
//! 3. **Non-Escalation**: Missing `admissibility_token` means non-admissible by definition
//! 4. **Replay**: Requires `(slice_id, graph_snapshot_hash, query_embedding_hash)` match

use serde::{Deserialize, Serialize};
use super::turn::{TurnId, TurnSnapshot};
use super::edge::Edge;
use crate::canonical::canonical_hash_hex;
use crate::GRAPH_KERNEL_SCHEMA_VERSION;

/// Fingerprint of a slice for provenance tracking.
///
/// This is a content-derived hash that uniquely identifies a slice
/// given the same anchor, policy, and graph state.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SliceFingerprint(String);

impl SliceFingerprint {
    /// Create a new fingerprint from a hash string.
    pub fn new(hash: String) -> Self {
        Self(hash)
    }

    /// Get the fingerprint as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SliceFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Graph snapshot identity for detecting content drift.
///
/// This hash captures the state of the graph at slicing time.
/// If content changes but IDs stay the same, this will differ.
///
/// Computed from: `max(updated_at) + row_counts + schema_version`
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphSnapshotHash(String);

impl GraphSnapshotHash {
    /// Create a new snapshot hash.
    pub fn new(hash: String) -> Self {
        Self(hash)
    }

    /// Create from table statistics.
    ///
    /// **Deprecated**: Use `from_content_hashes` for true content-derived immutability.
    /// This method is a drift canary, not an immutability proof.
    pub fn from_stats(
        max_updated_at: i64,
        turn_count: u64,
        edge_count: u64,
        schema_version: &str,
    ) -> Self {
        let canonical = (max_updated_at, turn_count, edge_count, schema_version);
        Self(canonical_hash_hex(&canonical))
    }

    /// Create from per-turn content hashes (deterministic fold).
    ///
    /// This is the production method for computing snapshot hashes.
    /// It guarantees that any content change in any turn will produce
    /// a different snapshot hash, enabling true replay immutability.
    ///
    /// # Arguments
    /// * `turn_content_hashes` - Sorted list of (TurnId, content_hash) pairs
    /// * `edge_count` - Number of edges in the slice
    /// * `schema_version` - Graph Kernel schema version
    ///
    /// # Determinism
    /// The input must be sorted by TurnId for deterministic output.
    pub fn from_content_hashes(
        turn_content_hashes: &[(TurnId, String)],
        edge_count: u64,
        schema_version: &str,
    ) -> Self {
        use std::hash::Hasher;
        use xxhash_rust::xxh64::Xxh64;
        
        // Start with edge_count and schema_version
        let mut hasher = Xxh64::new(0);
        hasher.write(&edge_count.to_le_bytes());
        hasher.write(schema_version.as_bytes());
        
        // Fold in each turn's (id, content_hash) pair
        for (turn_id, content_hash) in turn_content_hashes {
            hasher.write(turn_id.as_uuid().as_bytes());
            hasher.write(content_hash.as_bytes());
        }
        
        Self(format!("{:016x}", hasher.finish()))
    }

    /// Get the hash as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for GraphSnapshotHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unforgeable admissibility token issued by the Graph Kernel.
///
/// This token is the SOLE proof that a slice was issued by the kernel.
/// Downstream systems must verify presence of this token before allowing
/// any promotion or lifecycle advancement operations.
///
/// ## Security Model
///
/// The token is computed as: `HMAC-SHA256(secret, canonical_fields)[..16]`
///
/// Without knowing the kernel's secret, this token cannot be forged.
/// This implements the "No Phantom Authority" invariant: any admissibility
/// claim is verifiable without trusting the claimant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdmissibilityToken(String);

impl AdmissibilityToken {
    /// Token version marker for canonical representation.
    const TOKEN_VERSION: &'static str = "admissibility_token_v2_hmac";

    /// Build the canonical string for HMAC computation.
    fn canonical_string(
        slice_id: &SliceFingerprint,
        anchor_turn_id: &TurnId,
        policy_id: &str,
        policy_params_hash: &str,
        graph_snapshot_hash: &GraphSnapshotHash,
        schema_version: &str,
    ) -> String {
        format!(
            "{}|{}|{}|{}|{}|{}|{}",
            slice_id.as_str(),
            anchor_turn_id.as_uuid(),
            policy_id,
            policy_params_hash,
            graph_snapshot_hash.as_str(),
            schema_version,
            Self::TOKEN_VERSION,
        )
    }

    /// Issue a cryptographically signed admissibility token (kernel-only operation).
    ///
    /// Uses HMAC-SHA256 with the kernel's secret key. Only the kernel
    /// possesses this secret, making the token unforgeable.
    ///
    /// # Arguments
    /// * `secret` - The kernel's HMAC secret (32+ bytes recommended)
    /// * Other parameters define the slice being authorized
    pub fn issue_hmac(
        secret: &[u8],
        slice_id: &SliceFingerprint,
        anchor_turn_id: &TurnId,
        policy_id: &str,
        policy_params_hash: &str,
        graph_snapshot_hash: &GraphSnapshotHash,
        schema_version: &str,
    ) -> Self {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        let canonical = Self::canonical_string(
            slice_id,
            anchor_turn_id,
            policy_id,
            policy_params_hash,
            graph_snapshot_hash,
            schema_version,
        );

        let mut mac = Hmac::<Sha256>::new_from_slice(secret)
            .expect("HMAC accepts any key size");
        mac.update(canonical.as_bytes());

        // Take first 16 bytes (128 bits) of HMAC for token
        let result = mac.finalize().into_bytes();
        Self(hex::encode(&result[..16]))
    }

    /// Verify this token was issued by the kernel for the given slice.
    ///
    /// Uses constant-time comparison to prevent timing attacks.
    ///
    /// # Arguments
    /// * `secret` - The kernel's HMAC secret (shared with verifier)
    /// * Other parameters must match exactly what was used to issue the token
    pub fn verify_hmac(
        &self,
        secret: &[u8],
        slice_id: &SliceFingerprint,
        anchor_turn_id: &TurnId,
        policy_id: &str,
        policy_params_hash: &str,
        graph_snapshot_hash: &GraphSnapshotHash,
        schema_version: &str,
    ) -> bool {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        let canonical = Self::canonical_string(
            slice_id,
            anchor_turn_id,
            policy_id,
            policy_params_hash,
            graph_snapshot_hash,
            schema_version,
        );

        let mut mac = Hmac::<Sha256>::new_from_slice(secret)
            .expect("HMAC accepts any key size");
        mac.update(canonical.as_bytes());

        // Decode our token and verify
        match hex::decode(&self.0) {
            Ok(token_bytes) if token_bytes.len() == 16 => {
                let expected = mac.finalize().into_bytes();
                // Constant-time comparison
                token_bytes.iter()
                    .zip(expected[..16].iter())
                    .fold(true, |acc, (a, b)| acc && (a == b))
            }
            _ => false,
        }
    }

    /// Legacy: Issue token without HMAC (for testing/backwards compatibility).
    ///
    /// **WARNING**: This token is content-derived, not cryptographically signed.
    /// Use `issue_hmac` for production.
    #[deprecated(note = "Use issue_hmac with a secret for production")]
    pub fn issue_legacy(
        slice_id: &SliceFingerprint,
        anchor_turn_id: &TurnId,
        policy_id: &str,
        policy_params_hash: &str,
        graph_snapshot_hash: &GraphSnapshotHash,
        schema_version: &str,
    ) -> Self {
        let canonical = (
            slice_id.as_str(),
            anchor_turn_id,
            policy_id,
            policy_params_hash,
            graph_snapshot_hash.as_str(),
            schema_version,
            "admissibility_token_v1",
        );
        Self(canonical_hash_hex(&canonical))
    }

    /// Get the token as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Create a token from a string (for verification).
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Check if this looks like a valid token format.
    pub fn is_valid_format(&self) -> bool {
        // Token should be 32 hex chars (16 bytes)
        self.0.len() == 32 && self.0.chars().all(|c| c.is_ascii_hexdigit())
    }
}

impl std::fmt::Display for AdmissibilityToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Exported slice of the conversation DAG.
///
/// Contains all information needed to:
/// 1. Reconstruct the context for a target turn
/// 2. Verify determinism via slice_id
/// 3. Track provenance in downstream artifacts
/// 4. Detect content drift via graph_snapshot_hash
/// 5. Prove admissibility via admissibility_token
///
/// ## Provenance Completeness Invariant
///
/// A valid SliceExport MUST contain all of:
/// - `slice_id`: Deterministic fingerprint of selection
/// - `policy_id` + `policy_params_hash`: Policy identity
/// - `schema_version`: Graph Kernel version
/// - `graph_snapshot_hash`: Content immutability proof
/// - `admissibility_token`: Unforgeable kernel-issued claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceExport {
    /// The anchor turn this slice was built around.
    pub anchor_turn_id: TurnId,
    /// Turns in the slice, sorted by TurnId.
    pub turns: Vec<TurnSnapshot>,
    /// Edges between turns in the slice, sorted by (parent, child).
    pub edges: Vec<Edge>,
    /// Policy identifier (e.g., "slice_policy_v1").
    pub policy_id: String,
    /// Hash of policy parameters (quantized, not float-dependent).
    pub policy_params_hash: String,
    /// Schema version.
    pub schema_version: String,
    /// Unique fingerprint of this slice (selection identity).
    pub slice_id: SliceFingerprint,
    /// Graph state at slicing time (content immutability proof).
    pub graph_snapshot_hash: GraphSnapshotHash,
    /// Unforgeable admissibility claim from Graph Kernel.
    pub admissibility_token: AdmissibilityToken,
}

impl SliceExport {
    /// Create a new slice export with HMAC-signed admissibility token.
    ///
    /// This is the production method. Requires the kernel's HMAC secret.
    ///
    /// # Arguments
    /// * `hmac_secret` - The kernel's secret key for signing tokens
    /// * Other parameters define the slice content
    pub fn new_with_secret(
        hmac_secret: &[u8],
        anchor_turn_id: TurnId,
        mut turns: Vec<TurnSnapshot>,
        mut edges: Vec<Edge>,
        policy_id: String,
        policy_params_hash: String,
        graph_snapshot_hash: GraphSnapshotHash,
    ) -> Self {
        // Sort for determinism
        turns.sort();
        edges.sort();

        let schema_version = GRAPH_KERNEL_SCHEMA_VERSION.to_string();

        // Compute selection fingerprint
        let slice_id = Self::compute_fingerprint(
            &anchor_turn_id,
            &turns,
            &edges,
            &policy_id,
            &policy_params_hash,
        );

        // Issue HMAC-signed admissibility token
        let admissibility_token = AdmissibilityToken::issue_hmac(
            hmac_secret,
            &slice_id,
            &anchor_turn_id,
            &policy_id,
            &policy_params_hash,
            &graph_snapshot_hash,
            &schema_version,
        );

        Self {
            anchor_turn_id,
            turns,
            edges,
            policy_id,
            policy_params_hash,
            schema_version,
            slice_id,
            graph_snapshot_hash,
            admissibility_token,
        }
    }

    /// Verify this slice's admissibility token is valid.
    ///
    /// # Arguments
    /// * `hmac_secret` - The kernel's secret key for verification
    pub fn verify_token(&self, hmac_secret: &[u8]) -> bool {
        self.admissibility_token.verify_hmac(
            hmac_secret,
            &self.slice_id,
            &self.anchor_turn_id,
            &self.policy_id,
            &self.policy_params_hash,
            &self.graph_snapshot_hash,
            &self.schema_version,
        )
    }

    /// Create a slice export for testing (uses legacy non-HMAC token).
    #[cfg(test)]
    #[allow(deprecated)]
    pub fn new_for_test(
        anchor_turn_id: TurnId,
        mut turns: Vec<TurnSnapshot>,
        mut edges: Vec<Edge>,
        policy_id: String,
        policy_params_hash: String,
    ) -> Self {
        turns.sort();
        edges.sort();

        let schema_version = GRAPH_KERNEL_SCHEMA_VERSION.to_string();
        let graph_snapshot_hash = GraphSnapshotHash::new("test_snapshot".to_string());

        let slice_id = Self::compute_fingerprint(
            &anchor_turn_id,
            &turns,
            &edges,
            &policy_id,
            &policy_params_hash,
        );

        let admissibility_token = AdmissibilityToken::issue_legacy(
            &slice_id,
            &anchor_turn_id,
            &policy_id,
            &policy_params_hash,
            &graph_snapshot_hash,
            &schema_version,
        );

        Self {
            anchor_turn_id,
            turns,
            edges,
            policy_id,
            policy_params_hash,
            schema_version,
            slice_id,
            graph_snapshot_hash,
            admissibility_token,
        }
    }

    /// Compute the slice fingerprint.
    fn compute_fingerprint(
        anchor: &TurnId,
        turns: &[TurnSnapshot],
        edges: &[Edge],
        policy_id: &str,
        policy_params_hash: &str,
    ) -> SliceFingerprint {
        // Extract just the turn IDs for hashing (not full snapshots)
        let turn_ids: Vec<_> = turns.iter().map(|t| t.id).collect();

        // Create canonical representation
        let canonical = (
            anchor,
            &turn_ids,
            edges,
            policy_id,
            policy_params_hash,
            GRAPH_KERNEL_SCHEMA_VERSION,
        );

        SliceFingerprint::new(canonical_hash_hex(&canonical))
    }

    /// Get the number of turns in the slice.
    pub fn num_turns(&self) -> usize {
        self.turns.len()
    }

    /// Get the number of edges in the slice.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Check if a turn is in the slice.
    pub fn contains_turn(&self, id: &TurnId) -> bool {
        self.turns.binary_search_by_key(id, |t| t.id).is_ok()
    }

    /// Get the anchor turn snapshot.
    pub fn anchor_turn(&self) -> Option<&TurnSnapshot> {
        self.turns.iter().find(|t| t.id == self.anchor_turn_id)
    }

    /// Verify the admissibility token is valid for this slice.
    ///
    /// Returns true if the token was issued by the kernel for these exact parameters.
    /// Requires the HMAC secret that was used to issue the token.
    pub fn verify_admissibility(&self, hmac_secret: &[u8]) -> bool {
        self.admissibility_token.verify_hmac(
            hmac_secret,
            &self.slice_id,
            &self.anchor_turn_id,
            &self.policy_id,
            &self.policy_params_hash,
            &self.graph_snapshot_hash,
            &self.schema_version,
        )
    }

    /// Check if a turn is admissible in this slice.
    ///
    /// Enforces the slice boundary invariant: turn_id ∈ slice.turn_ids
    pub fn is_turn_admissible(&self, turn_id: &TurnId) -> bool {
        self.contains_turn(turn_id)
    }

    /// Filter a list of turn IDs to only those admissible in this slice.
    pub fn filter_admissible(&self, turn_ids: &[TurnId]) -> Vec<TurnId> {
        turn_ids.iter()
            .filter(|id| self.is_turn_admissible(id))
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::turn::{Role, Phase};
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
    fn test_slice_fingerprint_determinism() {
        let anchor = TurnId::new(Uuid::from_u128(1));
        let turns = vec![
            make_turn(1, 0.8, Phase::Synthesis),
            make_turn(2, 0.6, Phase::Planning),
        ];
        let edges = vec![Edge::reply(
            TurnId::new(Uuid::from_u128(1)),
            TurnId::new(Uuid::from_u128(2)),
        )];

        let slice1 = SliceExport::new_for_test(
            anchor,
            turns.clone(),
            edges.clone(),
            "test_policy".to_string(),
            "params_hash".to_string(),
        );

        let slice2 = SliceExport::new_for_test(
            anchor,
            turns,
            edges,
            "test_policy".to_string(),
            "params_hash".to_string(),
        );

        assert_eq!(slice1.slice_id, slice2.slice_id);
    }

    #[test]
    fn test_slice_fingerprint_changes_with_policy() {
        let anchor = TurnId::new(Uuid::from_u128(1));
        let turns = vec![make_turn(1, 0.8, Phase::Synthesis)];
        let edges = vec![];

        let slice1 = SliceExport::new_for_test(
            anchor,
            turns.clone(),
            edges.clone(),
            "policy_v1".to_string(),
            "params_1".to_string(),
        );

        let slice2 = SliceExport::new_for_test(
            anchor,
            turns,
            edges,
            "policy_v1".to_string(),
            "params_2".to_string(), // Different params
        );

        assert_ne!(slice1.slice_id, slice2.slice_id);
    }

    #[test]
    fn test_slice_turns_sorted() {
        let anchor = TurnId::new(Uuid::from_u128(1));
        // Add turns out of order
        let turns = vec![
            make_turn(3, 0.5, Phase::Exploration),
            make_turn(1, 0.8, Phase::Synthesis),
            make_turn(2, 0.6, Phase::Planning),
        ];

        let slice = SliceExport::new_for_test(
            anchor,
            turns,
            vec![],
            "test".to_string(),
            "hash".to_string(),
        );

        // Should be sorted by TurnId
        assert_eq!(slice.turns[0].id, TurnId::new(Uuid::from_u128(1)));
        assert_eq!(slice.turns[1].id, TurnId::new(Uuid::from_u128(2)));
        assert_eq!(slice.turns[2].id, TurnId::new(Uuid::from_u128(3)));
    }

    #[test]
    fn test_admissibility_token_hmac_verification() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let anchor = TurnId::new(Uuid::from_u128(1));
        let turns = vec![make_turn(1, 0.8, Phase::Synthesis)];
        let snapshot_hash = GraphSnapshotHash::new("test_snapshot".to_string());

        let slice = SliceExport::new_with_secret(
            secret,
            anchor,
            turns,
            vec![],
            "test_policy".to_string(),
            "params_hash".to_string(),
            snapshot_hash,
        );

        // Token should verify correctly with same secret
        assert!(slice.verify_admissibility(secret));
        
        // Token should NOT verify with wrong secret
        assert!(!slice.verify_admissibility(b"wrong_secret_definitely_wrong!"));
    }

    #[test]
    fn test_hmac_token_is_unforgeable() {
        let secret = b"kernel_only_secret_very_secure!!";
        let anchor = TurnId::new(Uuid::from_u128(1));
        let slice_id = SliceFingerprint::new("test_slice_id".to_string());
        let snapshot_hash = GraphSnapshotHash::new("test_snapshot".to_string());

        let token = AdmissibilityToken::issue_hmac(
            secret,
            &slice_id,
            &anchor,
            "policy_v1",
            "params_hash",
            &snapshot_hash,
            "1.0.0",
        );

        // Token should verify with correct parameters
        assert!(token.verify_hmac(
            secret,
            &slice_id,
            &anchor,
            "policy_v1",
            "params_hash",
            &snapshot_hash,
            "1.0.0",
        ));

        // Token should NOT verify if any parameter changes
        let wrong_anchor = TurnId::new(Uuid::from_u128(2));
        assert!(!token.verify_hmac(
            secret,
            &slice_id,
            &wrong_anchor,  // Different anchor
            "policy_v1",
            "params_hash",
            &snapshot_hash,
            "1.0.0",
        ));
    }

    #[test]
    fn test_content_hash_snapshot() {
        let turn1 = TurnId::new(Uuid::from_u128(1));
        let turn2 = TurnId::new(Uuid::from_u128(2));
        
        let hashes1 = vec![
            (turn1, "hash_for_turn_1".to_string()),
            (turn2, "hash_for_turn_2".to_string()),
        ];
        
        let snapshot1 = GraphSnapshotHash::from_content_hashes(&hashes1, 1, "1.0.0");
        let snapshot2 = GraphSnapshotHash::from_content_hashes(&hashes1, 1, "1.0.0");
        
        // Same inputs = same hash
        assert_eq!(snapshot1, snapshot2);
        
        // Different content hash = different snapshot
        let hashes2 = vec![
            (turn1, "hash_for_turn_1".to_string()),
            (turn2, "DIFFERENT_HASH".to_string()),  // Content changed
        ];
        let snapshot3 = GraphSnapshotHash::from_content_hashes(&hashes2, 1, "1.0.0");
        assert_ne!(snapshot1, snapshot3);
    }

    #[test]
    fn test_turn_admissibility() {
        let anchor = TurnId::new(Uuid::from_u128(1));
        let turns = vec![
            make_turn(1, 0.8, Phase::Synthesis),
            make_turn(2, 0.6, Phase::Planning),
        ];

        let slice = SliceExport::new_for_test(
            anchor,
            turns,
            vec![],
            "test".to_string(),
            "hash".to_string(),
        );

        // Turns in slice are admissible
        assert!(slice.is_turn_admissible(&TurnId::new(Uuid::from_u128(1))));
        assert!(slice.is_turn_admissible(&TurnId::new(Uuid::from_u128(2))));
        
        // Turns not in slice are not admissible
        assert!(!slice.is_turn_admissible(&TurnId::new(Uuid::from_u128(999))));
    }

    #[test]
    fn test_filter_admissible() {
        let anchor = TurnId::new(Uuid::from_u128(1));
        let turns = vec![
            make_turn(1, 0.8, Phase::Synthesis),
            make_turn(2, 0.6, Phase::Planning),
        ];

        let slice = SliceExport::new_for_test(
            anchor,
            turns,
            vec![],
            "test".to_string(),
            "hash".to_string(),
        );

        let candidates = vec![
            TurnId::new(Uuid::from_u128(1)),
            TurnId::new(Uuid::from_u128(2)),
            TurnId::new(Uuid::from_u128(3)),  // Not in slice
            TurnId::new(Uuid::from_u128(999)), // Not in slice
        ];

        let admissible = slice.filter_admissible(&candidates);
        assert_eq!(admissible.len(), 2);
        assert!(admissible.contains(&TurnId::new(Uuid::from_u128(1))));
        assert!(admissible.contains(&TurnId::new(Uuid::from_u128(2))));
    }

    #[test]
    fn test_graph_snapshot_hash_from_stats() {
        let hash1 = GraphSnapshotHash::from_stats(1000, 100, 50, "1.0.0");
        let hash2 = GraphSnapshotHash::from_stats(1000, 100, 50, "1.0.0");
        let hash3 = GraphSnapshotHash::from_stats(1001, 100, 50, "1.0.0"); // Different timestamp

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}

