//! Batch slice generation for Atlas runs.
//!
//! Generates deterministic slices for a set of anchor turns,
//! producing a registry of all slices with their fingerprints.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::canonical::canonical_hash_hex;
use crate::policy::SlicePolicyV1;
use crate::slicer::{ContextSlicer, SlicerError};
use crate::store::GraphStore;
use crate::types::{TurnId, SliceExport};

/// Result of a batch slice operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSliceResult {
    /// Snapshot ID this batch was computed against.
    pub snapshot_id: String,
    /// Hash of the anchor set used.
    pub anchor_set_hash: String,
    /// Policy ID used for slicing.
    pub policy_id: String,
    /// Policy parameters hash.
    pub policy_params_hash: String,
    /// All generated slices.
    pub slices: Vec<SliceExport>,
    /// Registry of slice metadata.
    pub registry: SliceRegistry,
}

/// Registry of all slices in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceRegistry {
    /// Individual slice entries.
    pub entries: Vec<SliceRegistryEntry>,
    /// Hash of the registry for integrity verification.
    pub registry_hash: String,
}

/// Metadata for a single slice in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceRegistryEntry {
    /// The anchor turn this slice was built around.
    pub anchor_turn_id: String,
    /// Unique fingerprint of the slice.
    pub slice_id: String,
    /// Number of turns in the slice.
    pub turn_count: usize,
    /// Number of edges in the slice.
    pub edge_count: usize,
    /// Hash of policy parameters.
    pub policy_params_hash: String,
}

impl SliceRegistry {
    /// Create a new registry from entries.
    pub fn new(entries: Vec<SliceRegistryEntry>) -> Self {
        let registry_hash = canonical_hash_hex(&entries);
        Self {
            entries,
            registry_hash,
        }
    }

    /// Get entry by anchor turn ID.
    pub fn get_by_anchor(&self, anchor_id: &str) -> Option<&SliceRegistryEntry> {
        self.entries.iter().find(|e| e.anchor_turn_id == anchor_id)
    }

    /// Get entry by slice ID.
    pub fn get_by_slice_id(&self, slice_id: &str) -> Option<&SliceRegistryEntry> {
        self.entries.iter().find(|e| e.slice_id == slice_id)
    }
}

/// Batch slicer for generating slices across many anchors.
pub struct BatchSlicer<S: GraphStore + Send + Sync + 'static> {
    slicer: ContextSlicer<S>,
    policy: SlicePolicyV1,
}

impl<S: GraphStore + Send + Sync + 'static> BatchSlicer<S> {
    /// Create a new batch slicer with HMAC secret.
    pub fn new(store: Arc<S>, policy: SlicePolicyV1, hmac_secret: Vec<u8>) -> Self {
        let slicer = ContextSlicer::new(store, policy.clone(), hmac_secret);
        Self { slicer, policy }
    }

    /// Create for testing (uses test secret).
    #[cfg(test)]
    pub fn new_for_test(store: Arc<S>, policy: SlicePolicyV1) -> Self {
        Self::new(store, policy, b"test_secret_for_batch_slicer".to_vec())
    }

    /// Generate slices for all anchors.
    ///
    /// Returns slices in anchor order for determinism.
    pub async fn slice_all(
        &self,
        anchors: &[TurnId],
        snapshot_id: &str,
        anchor_set_hash: &str,
    ) -> Result<BatchSliceResult, SlicerError> {
        let policy_params_hash = canonical_hash_hex(&self.policy);

        let mut slices = Vec::with_capacity(anchors.len());
        let mut entries = Vec::with_capacity(anchors.len());

        for anchor in anchors {
            // slice() now returns AdmissibleEvidenceBundle, proving verification
            let bundle = self.slicer.slice(anchor.clone()).await?;
            let slice = bundle.slice();

            entries.push(SliceRegistryEntry {
                anchor_turn_id: anchor.as_uuid().to_string(),
                slice_id: slice.slice_id.to_string(),
                turn_count: slice.turns.len(),
                edge_count: slice.edges.len(),
                policy_params_hash: policy_params_hash.clone(),
            });

            slices.push(slice.clone());
        }

        let registry = SliceRegistry::new(entries);

        Ok(BatchSliceResult {
            snapshot_id: snapshot_id.to_string(),
            anchor_set_hash: anchor_set_hash.to_string(),
            policy_id: self.policy.version.clone(),
            policy_params_hash,
            slices,
            registry,
        })
    }

    /// Get the policy being used.
    pub fn policy(&self) -> &SlicePolicyV1 {
        &self.policy
    }
}

/// Anchor set with deterministic hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorSet {
    /// Anchor turn IDs (sorted for determinism).
    pub anchors: Vec<TurnId>,
    /// Selection policy used.
    pub selection_policy: String,
    /// Hash of the anchor set.
    pub anchor_set_hash: String,
}

impl AnchorSet {
    /// Create a new anchor set from turn IDs.
    pub fn new(mut anchors: Vec<TurnId>, selection_policy: &str) -> Self {
        // Sort for determinism
        anchors.sort();
        anchors.dedup();

        let anchor_strings: Vec<String> = anchors.iter().map(|a| a.as_uuid().to_string()).collect();
        let hash_input = (anchor_strings, selection_policy);
        let anchor_set_hash = canonical_hash_hex(&hash_input);

        Self {
            anchors,
            selection_policy: selection_policy.to_string(),
            anchor_set_hash,
        }
    }

    /// Number of anchors in the set.
    pub fn len(&self) -> usize {
        self.anchors.len()
    }

    /// Check if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }
}

/// Build a turn-to-slices index from a batch result.
pub fn build_turn_slice_index(result: &BatchSliceResult) -> BTreeMap<String, Vec<String>> {
    let mut index: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for slice in &result.slices {
        for turn in &slice.turns {
            let turn_id = turn.id.as_uuid().to_string();
            index
                .entry(turn_id)
                .or_default()
                .push(slice.slice_id.to_string());
        }
    }

    // Sort slice IDs for determinism
    for slices in index.values_mut() {
        slices.sort();
    }

    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::memory::InMemoryGraphStore;
    use crate::types::{TurnSnapshot, Edge, EdgeType, Phase, Role};
    use crate::policy::PhaseWeights;
    use uuid::Uuid;

    fn make_test_store() -> Arc<InMemoryGraphStore> {
        let mut store = InMemoryGraphStore::new();

        // Create a small graph
        let turn1 = TurnSnapshot::new(
            TurnId::new(Uuid::new_v4()),
            "session_1".to_string(),
            Role::User,
            Phase::Exploration,
            0.8,
            0, 0, 0.5, 0.1, 1.0,
            1000,
        );
        let turn2 = TurnSnapshot::new(
            TurnId::new(Uuid::new_v4()),
            "session_1".to_string(),
            Role::Assistant,
            Phase::Exploration,
            0.7,
            1, 0, 0.6, 0.2, 1.0,
            2000,
        );
        let turn3 = TurnSnapshot::new(
            TurnId::new(Uuid::new_v4()),
            "session_1".to_string(),
            Role::User,
            Phase::Synthesis,
            0.9,
            2, 0, 0.7, 0.3, 1.0,
            3000,
        );

        let id1 = turn1.id.clone();
        let id2 = turn2.id.clone();
        let id3 = turn3.id.clone();

        store.add_turn(turn1);
        store.add_turn(turn2);
        store.add_turn(turn3);
        store.add_edge(Edge::new(id1.clone(), id2.clone(), EdgeType::Reply));
        store.add_edge(Edge::new(id2.clone(), id3.clone(), EdgeType::Reply));

        Arc::new(store)
    }

    #[tokio::test]
    async fn test_batch_slice() {
        let store = make_test_store();
        let turns: Vec<_> = store.all_turns().iter().map(|t| t.id.clone()).collect();

        let policy = SlicePolicyV1 {
            max_nodes: 10,
            max_radius: 3,
            phase_weights: PhaseWeights::default(),
            salience_weight: 1.0,
            distance_decay: 0.8,
            include_siblings: true,
            max_siblings_per_node: 3,
            version: "slice_policy_v1".to_string(),
        };

        let slicer = BatchSlicer::new_for_test(store, policy);
        let anchors = vec![turns[0].clone(), turns[2].clone()];

        let result = slicer
            .slice_all(&anchors, "snapshot_test", "anchor_hash_test")
            .await
            .unwrap();

        assert_eq!(result.slices.len(), 2);
        assert_eq!(result.registry.entries.len(), 2);
        assert_eq!(result.snapshot_id, "snapshot_test");
    }

    #[test]
    fn test_anchor_set_determinism() {
        let id1 = TurnId::new(Uuid::new_v4());
        let id2 = TurnId::new(Uuid::new_v4());
        let id3 = TurnId::new(Uuid::new_v4());

        // Different order, same content
        let set1 = AnchorSet::new(vec![id1.clone(), id2.clone(), id3.clone()], "policy_v1");
        let set2 = AnchorSet::new(vec![id3.clone(), id1.clone(), id2.clone()], "policy_v1");

        assert_eq!(set1.anchor_set_hash, set2.anchor_set_hash);
    }

    #[tokio::test]
    async fn test_turn_slice_index() {
        let store = make_test_store();
        let turns: Vec<_> = store.all_turns().iter().map(|t| t.id.clone()).collect();

        let policy = SlicePolicyV1 {
            max_nodes: 10,
            max_radius: 3,
            phase_weights: PhaseWeights::default(),
            salience_weight: 1.0,
            distance_decay: 0.8,
            include_siblings: true,
            max_siblings_per_node: 3,
            version: "slice_policy_v1".to_string(),
        };

        let slicer = BatchSlicer::new_for_test(store, policy);
        let result = slicer
            .slice_all(&turns, "snapshot", "anchors")
            .await
            .unwrap();

        let index = build_turn_slice_index(&result);

        // Each turn should appear in at least one slice
        for turn in &turns {
            let turn_str = turn.as_uuid().to_string();
            assert!(index.contains_key(&turn_str), "Turn {} not in index", turn_str);
        }
    }
}

