//! Deterministic context slicer.
//!
//! The slicer expands around an anchor turn using a priority queue,
//! respecting budget caps and producing a deterministic slice.

use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

use crate::policy::{SlicePolicyV1, scoring::ExpansionCandidate};
use crate::store::GraphStore;
use crate::types::{TurnId, TurnSnapshot, SliceExport, GraphSnapshotHash, AdmissibleEvidenceBundle, VerificationError};

/// Error type for slicer operations.
#[derive(Debug, thiserror::Error)]
pub enum SlicerError {
    /// Anchor turn not found.
    #[error("Anchor turn not found: {0}")]
    AnchorNotFound(TurnId),
    /// Store error.
    #[error("Store error: {0}")]
    StoreError(String),
    /// Verification error (should never happen - internal consistency violation).
    #[error("Internal verification error: {0}")]
    VerificationError(#[from] VerificationError),
}

impl SlicerError {
    /// Create a store error from any error type.
    pub fn from_store<E: std::error::Error>(e: E) -> Self {
        Self::StoreError(e.to_string())
    }
}

/// Deterministic context slicer.
///
/// Expands around an anchor turn to produce a context slice.
///
/// ## Algorithm
///
/// 1. Start with anchor turn at distance 0
/// 2. Add to frontier priority queue (max-heap by priority score)
/// 3. While frontier not empty and nodes < max_nodes:
///    - Pop highest priority candidate
///    - Add to slice
///    - Add unvisited parents/children to frontier (distance + 1)
///    - If include_siblings: add siblings up to limit
/// 4. Return slice (sorted for determinism)
///
/// ## Security
///
/// The slicer holds the HMAC secret for issuing admissibility tokens.
/// Only kernel-internal code should have access to the secret.
pub struct ContextSlicer<S: GraphStore> {
    store: Arc<S>,
    policy: SlicePolicyV1,
    /// HMAC secret for signing admissibility tokens.
    hmac_secret: Vec<u8>,
}

impl<S: GraphStore + Send + Sync + 'static> ContextSlicer<S> {
    /// Create a new context slicer with HMAC secret.
    ///
    /// # Arguments
    /// * `store` - The graph store backend
    /// * `policy` - Slice policy configuration
    /// * `hmac_secret` - Secret key for signing admissibility tokens (32+ bytes recommended)
    pub fn new(store: Arc<S>, policy: SlicePolicyV1, hmac_secret: Vec<u8>) -> Self {
        Self { store, policy, hmac_secret }
    }

    /// Create a slicer for testing (uses empty secret, tokens not cryptographically valid).
    #[cfg(test)]
    pub fn new_for_test(store: Arc<S>, policy: SlicePolicyV1) -> Self {
        Self::new(store, policy, b"test_secret_for_unit_tests".to_vec())
    }

    /// Create a context slice around an anchor turn.
    ///
    /// Returns an `AdmissibleEvidenceBundle` - a cryptographically verified slice
    /// that proves kernel authorization. This is the **only** way to obtain a
    /// verified slice from the kernel.
    ///
    /// ## Security
    ///
    /// By returning `AdmissibleEvidenceBundle` instead of raw `SliceExport`,
    /// we enforce **INV-GK-003: No Phantom Authority** at the API boundary.
    /// Downstream systems cannot accidentally operate on unverified slices.
    pub async fn slice(&self, anchor_id: TurnId) -> Result<AdmissibleEvidenceBundle, SlicerError> {
        // Get anchor turn
        let anchor = self.store.get_turn(&anchor_id).await
            .map_err(|e| SlicerError::StoreError(e.to_string()))?
            .ok_or_else(|| SlicerError::AnchorNotFound(anchor_id))?;

        // Initialize state
        let mut selected: Vec<TurnSnapshot> = Vec::new();
        let mut visited: HashSet<TurnId> = HashSet::new();
        let mut frontier: BinaryHeap<ExpansionCandidate> = BinaryHeap::new();

        // Start with anchor
        let anchor_candidate = ExpansionCandidate::new(anchor, 0, &self.policy);
        frontier.push(anchor_candidate);
        visited.insert(anchor_id);

        // Expand
        while let Some(candidate) = frontier.pop() {
            // Check budget
            if selected.len() >= self.policy.max_nodes {
                break;
            }

            // Check radius
            if candidate.distance > self.policy.max_radius {
                continue;
            }

            let turn_id = candidate.turn.id;
            let next_distance = candidate.distance + 1;
            let current_distance = candidate.distance;

            // Add to selected
            selected.push(candidate.turn);

            // Skip expansion if at max radius
            if next_distance > self.policy.max_radius {
                continue;
            }

            // Expand to parents
            let parents = self.store.get_parents(&turn_id).await
                .map_err(|e| SlicerError::StoreError(e.to_string()))?;
            
            for parent_id in parents {
                if !visited.contains(&parent_id) {
                    visited.insert(parent_id);
                    if let Some(parent) = self.store.get_turn(&parent_id).await
                        .map_err(|e| SlicerError::StoreError(e.to_string()))? 
                    {
                        let candidate = ExpansionCandidate::new(parent, next_distance, &self.policy);
                        frontier.push(candidate);
                    }
                }
            }

            // Expand to children
            let children = self.store.get_children(&turn_id).await
                .map_err(|e| SlicerError::StoreError(e.to_string()))?;
            
            for child_id in children {
                if !visited.contains(&child_id) {
                    visited.insert(child_id);
                    if let Some(child) = self.store.get_turn(&child_id).await
                        .map_err(|e| SlicerError::StoreError(e.to_string()))? 
                    {
                        let candidate = ExpansionCandidate::new(child, next_distance, &self.policy);
                        frontier.push(candidate);
                    }
                }
            }

            // Expand to siblings if enabled
            if self.policy.include_siblings && self.policy.max_siblings_per_node > 0 {
                let siblings = self.store.get_siblings(&turn_id, self.policy.max_siblings_per_node).await
                    .map_err(|e| SlicerError::StoreError(e.to_string()))?;
                
                for sibling_id in siblings {
                    if !visited.contains(&sibling_id) {
                        visited.insert(sibling_id);
                        if let Some(sibling) = self.store.get_turn(&sibling_id).await
                            .map_err(|e| SlicerError::StoreError(e.to_string()))? 
                        {
                            // Siblings are at the same distance as the current node
                            let candidate = ExpansionCandidate::new(sibling, current_distance, &self.policy);
                            frontier.push(candidate);
                        }
                    }
                }
            }
        }

        // Collect edges between selected turns
        let selected_ids: Vec<TurnId> = selected.iter().map(|t| t.id).collect();
        let edges = self.store.get_edges(&selected_ids).await
            .map_err(|e| SlicerError::StoreError(e.to_string()))?;

        // Compute graph snapshot hash from selected turns
        // Prefer content hashes for true immutability, fall back to stats
        #[allow(deprecated)]
        let graph_snapshot_hash = {
            // Check if all turns have content hashes
            let all_have_hashes = selected.iter().all(|t| t.content_hash.is_some());
            
            if all_have_hashes {
                // Use content-derived hash (production mode)
                let mut turn_hashes: Vec<(TurnId, String)> = selected
                    .iter()
                    .map(|t| (t.id, t.content_hash.clone().unwrap()))
                    .collect();
                // Sort by TurnId for determinism
                turn_hashes.sort_by(|a, b| a.0.cmp(&b.0));
                
                GraphSnapshotHash::from_content_hashes(
                    &turn_hashes,
                    edges.len() as u64,
                    crate::GRAPH_KERNEL_SCHEMA_VERSION,
                )
            } else {
                // Fall back to stats-based hash (backwards compatibility)
                let max_created_at = selected.iter()
                    .map(|t| t.created_at)
                    .max()
                    .unwrap_or(0);
                GraphSnapshotHash::from_stats(
                    max_created_at,
                    selected.len() as u64,
                    edges.len() as u64,
                    crate::GRAPH_KERNEL_SCHEMA_VERSION,
                )
            }
        };

        // Create slice export with HMAC-signed token
        let slice = SliceExport::new_with_secret(
            &self.hmac_secret,
            anchor_id,
            selected,
            edges,
            self.policy.policy_id().to_string(),
            self.policy.params_hash(),
            graph_snapshot_hash,
        );

        // Wrap in AdmissibleEvidenceBundle (verification always passes since we just issued the token)
        // This enforces INV-GK-003: No Phantom Authority at the API boundary
        let bundle = AdmissibleEvidenceBundle::from_verified(slice, &self.hmac_secret)?;
        Ok(bundle)
    }

    /// Get the policy.
    pub fn policy(&self) -> &SlicePolicyV1 {
        &self.policy
    }

    /// Get a reference to the store.
    pub fn store(&self) -> &S {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::InMemoryGraphStore;
    use crate::types::{Edge, Role, Phase, EdgeType};
    use uuid::Uuid;

    fn make_turn(id: u128, salience: f32, phase: Phase, depth: u32) -> TurnSnapshot {
        TurnSnapshot::new(
            TurnId::new(Uuid::from_u128(id)),
            "session_1".to_string(),
            Role::User,
            phase,
            salience,
            depth,
            0,
            0.5,
            0.5,
            1.0,
            1000,
        )
    }

    fn build_linear_graph(n: usize) -> Arc<InMemoryGraphStore> {
        let mut store = InMemoryGraphStore::new();
        
        for i in 1..=n {
            store.add_turn(make_turn(i as u128, 0.5, Phase::Consolidation, i as u32));
            if i > 1 {
                store.add_edge(Edge::new(
                    TurnId::new(Uuid::from_u128((i - 1) as u128)),
                    TurnId::new(Uuid::from_u128(i as u128)),
                    EdgeType::Reply,
                ));
            }
        }
        
        Arc::new(store)
    }

    #[tokio::test]
    async fn test_slice_includes_anchor() {
        let store = build_linear_graph(5);
        let policy = SlicePolicyV1::minimal();
        let slicer = ContextSlicer::new_for_test(store, policy);

        let anchor_id = TurnId::new(Uuid::from_u128(3));
        let bundle = slicer.slice(anchor_id).await.unwrap();

        // AdmissibleEvidenceBundle wraps the slice
        assert!(bundle.slice().contains_turn(&anchor_id));
        assert!(bundle.is_turn_admissible(&anchor_id));
    }

    #[tokio::test]
    async fn test_slice_respects_max_nodes() {
        let store = build_linear_graph(100);
        let mut policy = SlicePolicyV1::minimal();
        policy.max_nodes = 5;
        policy.max_radius = 100; // High radius to test node limit

        let slicer = ContextSlicer::new_for_test(store, policy);

        let anchor_id = TurnId::new(Uuid::from_u128(50));
        let bundle = slicer.slice(anchor_id).await.unwrap();

        assert!(bundle.num_turns() <= 5);
    }

    #[tokio::test]
    async fn test_slice_respects_max_radius() {
        let store = build_linear_graph(100);
        let mut policy = SlicePolicyV1::minimal();
        policy.max_nodes = 100; // High node limit to test radius
        policy.max_radius = 2;

        let slicer = ContextSlicer::new_for_test(store, policy);

        let anchor_id = TurnId::new(Uuid::from_u128(50));
        let bundle = slicer.slice(anchor_id).await.unwrap();

        // With radius 2, we can reach: 50, 49, 48, 51, 52 = 5 nodes
        assert!(bundle.num_turns() <= 5);
    }

    #[tokio::test]
    async fn test_slice_determinism() {
        let store = build_linear_graph(20);
        let policy = SlicePolicyV1::minimal();

        let slicer1 = ContextSlicer::new_for_test(Arc::clone(&store), policy.clone());
        let slicer2 = ContextSlicer::new_for_test(store, policy);

        let anchor_id = TurnId::new(Uuid::from_u128(10));
        let bundle1 = slicer1.slice(anchor_id).await.unwrap();
        let bundle2 = slicer2.slice(anchor_id).await.unwrap();

        // Slice IDs should be identical (determinism)
        assert_eq!(bundle1.slice_id().as_str(), bundle2.slice_id().as_str());
    }

    #[tokio::test]
    async fn test_hmac_token_verification() {
        let store = build_linear_graph(5);
        let policy = SlicePolicyV1::minimal();
        let secret = b"production_secret_key_32bytes!!".to_vec();
        let slicer = ContextSlicer::new(store, policy, secret.clone());

        let anchor_id = TurnId::new(Uuid::from_u128(3));
        let bundle = slicer.slice(anchor_id).await.unwrap();

        // The bundle exists, proving verification passed
        // AdmissibleEvidenceBundle can only be constructed via from_verified()
        assert!(bundle.num_turns() > 0);

        // Token on underlying slice should still verify with correct secret
        assert!(bundle.slice().verify_token(&secret));

        // Token should NOT verify with wrong secret
        assert!(!bundle.slice().verify_token(b"wrong_secret"));
    }

    #[tokio::test]
    async fn test_bundle_is_verified_proof() {
        // This test demonstrates that receiving an AdmissibleEvidenceBundle
        // is cryptographic proof of kernel authorization
        let store = build_linear_graph(5);
        let policy = SlicePolicyV1::minimal();
        let secret = b"production_secret_key_32bytes!!".to_vec();
        let slicer = ContextSlicer::new(store, policy, secret);

        let anchor_id = TurnId::new(Uuid::from_u128(3));

        // The fact that slice() returns Ok proves the bundle is verified
        // There is no way to get a bundle without verification
        let bundle = slicer.slice(anchor_id).await.expect(
            "Bundle creation should succeed - verification is internal"
        );

        // Bundle carries verified provenance
        let (slice_id, graph_hash, policy_id, _params, _schema) = bundle.provenance();
        assert!(!slice_id.as_str().is_empty());
        assert!(!graph_hash.as_str().is_empty());
        assert!(!policy_id.is_empty());
    }
}

