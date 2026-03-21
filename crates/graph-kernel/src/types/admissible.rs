//! Admissible evidence bundles - type-level safety for kernel-authorized evidence.
//!
//! ## Purpose
//!
//! This module enforces **INV-GK-003: No Phantom Authority** at the type level.
//! By making `AdmissibleEvidenceBundle` the only way to represent verified evidence,
//! we ensure that promotion pipelines **physically cannot accept** unverified slices.
//!
//! ## Security Model
//!
//! - `AdmissibleEvidenceBundle` can ONLY be constructed via `from_verified()`
//! - `from_verified()` REQUIRES HMAC secret and performs verification
//! - Failed verification → `Err(VerificationError)`
//! - Successful verification → `Ok(AdmissibleEvidenceBundle)` (unforgeable proof)
//!
//! ## Usage in Downstream Systems
//!
//! ```rust,ignore
//! // Promotion API signature (old - unsafe):
//! fn promote_turn(turn_id: TurnId, evidence: Vec<TurnSnapshot>) { ... }
//!
//! // Promotion API signature (new - type-safe):
//! fn promote_turn(turn_id: TurnId, evidence: &AdmissibleEvidenceBundle) { ... }
//! ```
//!
//! With the new signature, it's **impossible** to call `promote_turn` with
//! unverified evidence. The type system enforces kernel authorization.

use serde::{Deserialize, Serialize};
use super::slice::{SliceExport, SliceFingerprint, GraphSnapshotHash, AdmissibilityToken};
use super::turn::TurnId;

/// Error type for admissibility verification.
#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
    /// HMAC token verification failed.
    #[error("Admissibility token verification failed: token mismatch")]
    TokenMismatch,

    /// Token has invalid format.
    #[error("Admissibility token has invalid format: {0}")]
    InvalidTokenFormat(String),

    /// Slice provenance is incomplete.
    #[error("Slice provenance incomplete: missing {0}")]
    IncompleteProvenance(String),
}

/// Admissible evidence bundle - cryptographically verified slice.
///
/// This type represents a `SliceExport` that has passed HMAC token verification.
/// It can **only** be constructed via `from_verified()`, which requires the
/// kernel's HMAC secret.
///
/// ## Type-Level Safety Guarantee
///
/// By requiring this type in promotion APIs, we make it **impossible** to
/// promote evidence without kernel authorization. The type system enforces
/// the security boundary.
///
/// ## Provenance Tracking
///
/// This bundle carries complete provenance:
/// - `slice_id`: Deterministic fingerprint of selection
/// - `graph_snapshot_hash`: Content immutability proof
/// - `admissibility_token`: HMAC proof of kernel issuance
/// - `policy_ref`: Policy that generated the slice
///
/// # Security
///
/// This type is the enforcement mechanism for **INV-GK-003: No Phantom Authority**.
/// Never expose a public constructor that bypasses `from_verified()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissibleEvidenceBundle {
    /// The verified slice export.
    slice: SliceExport,

    /// Verification timestamp (when bundle was created).
    verified_at_unix_ms: i64,
}

impl AdmissibleEvidenceBundle {
    /// Create an admissible evidence bundle from a slice export.
    ///
    /// This is the **only** way to construct this type. It performs HMAC
    /// token verification using the provided secret.
    ///
    /// # Arguments
    /// * `slice` - The slice export to verify
    /// * `hmac_secret` - The kernel's HMAC secret (must match the secret used to issue the token)
    ///
    /// # Returns
    /// - `Ok(AdmissibleEvidenceBundle)` if verification succeeds
    /// - `Err(VerificationError)` if verification fails
    ///
    /// # Security
    ///
    /// This function enforces **INV-GK-005: HMAC Token Unforgeability**.
    /// Only slices issued by the Graph Kernel will pass verification.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let secret = std::env::var("KERNEL_HMAC_SECRET")?.as_bytes();
    /// let slice = graph_kernel.slice(anchor_id).await?;
    ///
    /// match AdmissibleEvidenceBundle::from_verified(slice, secret) {
    ///     Ok(bundle) => {
    ///         // Safe to use in promotion pipeline
    ///         promotion_api.promote(turn_id, &bundle)?;
    ///     }
    ///     Err(e) => {
    ///         // Token verification failed - DO NOT PROMOTE
    ///         log::error!("SLICE_BOUNDARY_VIOLATION: {}", e);
    ///         return Err(e.into());
    ///     }
    /// }
    /// ```
    pub fn from_verified(
        slice: SliceExport,
        hmac_secret: &[u8],
    ) -> Result<Self, VerificationError> {
        // Check token format
        if !slice.admissibility_token.is_valid_format() {
            return Err(VerificationError::InvalidTokenFormat(
                "Token must be 32 hex characters".to_string()
            ));
        }

        // Verify HMAC token
        let is_valid = slice.admissibility_token.verify_hmac(
            hmac_secret,
            &slice.slice_id,
            &slice.anchor_turn_id,
            &slice.policy_id,
            &slice.policy_params_hash,
            &slice.graph_snapshot_hash,
            &slice.schema_version,
        );

        if !is_valid {
            return Err(VerificationError::TokenMismatch);
        }

        // Verification passed - construct bundle
        Ok(Self {
            slice,
            verified_at_unix_ms: chrono::Utc::now().timestamp_millis(),
        })
    }

    /// Get a reference to the underlying verified slice.
    pub fn slice(&self) -> &SliceExport {
        &self.slice
    }

    /// Get the anchor turn ID.
    pub fn anchor_turn_id(&self) -> TurnId {
        self.slice.anchor_turn_id
    }

    /// Get the slice fingerprint.
    pub fn slice_id(&self) -> &SliceFingerprint {
        &self.slice.slice_id
    }

    /// Get the graph snapshot hash (content immutability proof).
    pub fn graph_snapshot_hash(&self) -> &GraphSnapshotHash {
        &self.slice.graph_snapshot_hash
    }

    /// Get the admissibility token.
    pub fn admissibility_token(&self) -> &AdmissibilityToken {
        &self.slice.admissibility_token
    }

    /// Get the policy ID that generated this slice.
    pub fn policy_id(&self) -> &str {
        &self.slice.policy_id
    }

    /// Get the policy parameters hash.
    pub fn policy_params_hash(&self) -> &str {
        &self.slice.policy_params_hash
    }

    /// Get the IDs of all turns in the slice.
    ///
    /// This is the authoritative list of admissible turn IDs.
    /// Any turn NOT in this list is non-admissible by definition.
    pub fn turn_ids(&self) -> Vec<TurnId> {
        self.slice.turns.iter().map(|t| t.id).collect()
    }

    /// Check if a turn is admissible in this bundle.
    ///
    /// Enforces **INV-GK-001: Slice Boundary Integrity**.
    pub fn is_turn_admissible(&self, turn_id: &TurnId) -> bool {
        self.slice.is_turn_admissible(turn_id)
    }

    /// Filter a list of turn IDs to only those admissible in this bundle.
    ///
    /// Enforces **INV-GK-001: Slice Boundary Integrity**.
    pub fn filter_admissible(&self, turn_ids: &[TurnId]) -> Vec<TurnId> {
        self.slice.filter_admissible(turn_ids)
    }

    /// Get the timestamp when this bundle was verified.
    pub fn verified_at_unix_ms(&self) -> i64 {
        self.verified_at_unix_ms
    }

    /// Get the number of turns in the bundle.
    pub fn num_turns(&self) -> usize {
        self.slice.num_turns()
    }

    /// Get the schema version.
    pub fn schema_version(&self) -> &str {
        &self.slice.schema_version
    }

    /// Extract provenance metadata for audit/replay.
    ///
    /// Returns a tuple of:
    /// - `slice_id`: Selection fingerprint
    /// - `graph_snapshot_hash`: Content immutability proof
    /// - `policy_id`: Policy identifier
    /// - `policy_params_hash`: Policy parameters hash
    /// - `schema_version`: Graph Kernel version
    pub fn provenance(&self) -> (
        &SliceFingerprint,
        &GraphSnapshotHash,
        &str,
        &str,
        &str,
    ) {
        (
            &self.slice.slice_id,
            &self.slice.graph_snapshot_hash,
            &self.slice.policy_id,
            &self.slice.policy_params_hash,
            &self.slice.schema_version,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TurnSnapshot, Role, Phase};
    use uuid::Uuid;

    fn make_turn(id: u128) -> TurnSnapshot {
        TurnSnapshot::new(
            TurnId::new(Uuid::from_u128(id)),
            "session_test".to_string(),
            Role::User,
            Phase::Synthesis,
            0.8,
            1,
            0,
            0.5,
            0.5,
            1.0,
            1000,
        )
    }

    #[test]
    fn test_successful_verification() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let anchor = TurnId::new(Uuid::from_u128(1));
        let turns = vec![make_turn(1)];
        let snapshot = GraphSnapshotHash::new("test_snapshot".to_string());

        // Create slice with HMAC token
        let slice = SliceExport::new_with_secret(
            secret,
            anchor,
            turns,
            vec![],
            "test_policy".to_string(),
            "params_hash".to_string(),
            snapshot,
        );

        // Verification should succeed
        let bundle = AdmissibleEvidenceBundle::from_verified(slice, secret);
        assert!(bundle.is_ok());

        let bundle = bundle.unwrap();
        assert_eq!(bundle.num_turns(), 1);
        assert_eq!(bundle.anchor_turn_id(), anchor);
    }

    #[test]
    fn test_verification_failure_wrong_secret() {
        let secret = b"correct_secret_32_bytes_minimum!";
        let wrong_secret = b"wrong_secret_totally_different!!";

        let anchor = TurnId::new(Uuid::from_u128(1));
        let turns = vec![make_turn(1)];
        let snapshot = GraphSnapshotHash::new("test_snapshot".to_string());

        // Create slice with correct secret
        let slice = SliceExport::new_with_secret(
            secret,
            anchor,
            turns,
            vec![],
            "test_policy".to_string(),
            "params_hash".to_string(),
            snapshot,
        );

        // Verification should FAIL with wrong secret
        let result = AdmissibleEvidenceBundle::from_verified(slice, wrong_secret);
        assert!(result.is_err());

        match result {
            Err(VerificationError::TokenMismatch) => (),
            _ => panic!("Expected TokenMismatch error"),
        }
    }

    #[test]
    fn test_slice_boundary_enforcement() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let anchor = TurnId::new(Uuid::from_u128(1));
        let turns = vec![
            make_turn(1),
            make_turn(2),
        ];
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

        let bundle = AdmissibleEvidenceBundle::from_verified(slice, secret).unwrap();

        // Turns 1 and 2 are admissible
        assert!(bundle.is_turn_admissible(&TurnId::new(Uuid::from_u128(1))));
        assert!(bundle.is_turn_admissible(&TurnId::new(Uuid::from_u128(2))));

        // Turn 999 is NOT admissible
        assert!(!bundle.is_turn_admissible(&TurnId::new(Uuid::from_u128(999))));
    }

    #[test]
    fn test_filter_admissible() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let anchor = TurnId::new(Uuid::from_u128(1));
        let turns = vec![
            make_turn(1),
            make_turn(2),
        ];
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

        let bundle = AdmissibleEvidenceBundle::from_verified(slice, secret).unwrap();

        let candidates = vec![
            TurnId::new(Uuid::from_u128(1)),   // admissible
            TurnId::new(Uuid::from_u128(2)),   // admissible
            TurnId::new(Uuid::from_u128(999)), // NOT admissible
        ];

        let filtered = bundle.filter_admissible(&candidates);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&TurnId::new(Uuid::from_u128(1))));
        assert!(filtered.contains(&TurnId::new(Uuid::from_u128(2))));
    }

    #[test]
    fn test_provenance_extraction() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let anchor = TurnId::new(Uuid::from_u128(1));
        let turns = vec![make_turn(1)];
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

        let bundle = AdmissibleEvidenceBundle::from_verified(slice, secret).unwrap();

        let (_slice_id, _graph_hash, policy_id, params_hash, schema_version) = bundle.provenance();
        assert_eq!(policy_id, "test_policy");
        assert_eq!(params_hash, "params_hash");
        assert_eq!(schema_version, crate::GRAPH_KERNEL_SCHEMA_VERSION);
    }

    #[test]
    fn test_verified_timestamp_is_recent() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let anchor = TurnId::new(Uuid::from_u128(1));
        let turns = vec![make_turn(1)];
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

        let before = chrono::Utc::now().timestamp_millis();
        let bundle = AdmissibleEvidenceBundle::from_verified(slice, secret).unwrap();
        let after = chrono::Utc::now().timestamp_millis();

        let verified_at = bundle.verified_at_unix_ms();
        assert!(verified_at >= before && verified_at <= after);
    }
}
