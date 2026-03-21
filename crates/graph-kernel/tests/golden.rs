//! Golden tests for the Graph Kernel.
//!
//! These tests verify determinism and correctness of the context slicer.

use std::sync::Arc;
use cc_graph_kernel::{
    TurnId, TurnSnapshot, Edge, EdgeType, Role, Phase,
    SlicePolicyV1, PhaseWeights,
    ContextSlicer,
    canonical_hash_hex,
};
use cc_graph_kernel::store::InMemoryGraphStore;
use uuid::Uuid;

/// Test HMAC secret for unit tests
const TEST_HMAC_SECRET: &[u8] = b"test_hmac_secret_for_golden_tests";

// ─────────────────────────────────────────────────────────────────────────────
// Test Helpers
// ─────────────────────────────────────────────────────────────────────────────

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

fn build_branching_graph() -> Arc<InMemoryGraphStore> {
    //         1
    //        /|\
    //       2 3 4
    //      / \
    //     5   6
    let mut store = InMemoryGraphStore::new();
    
    // Root
    store.add_turn(make_turn(1, 0.9, Phase::Synthesis, 0));
    
    // First level children
    store.add_turn(make_turn(2, 0.7, Phase::Planning, 1));
    store.add_turn(make_turn(3, 0.5, Phase::Consolidation, 1));
    store.add_turn(make_turn(4, 0.3, Phase::Exploration, 1));
    
    // Second level children (from node 2)
    store.add_turn(make_turn(5, 0.8, Phase::Synthesis, 2));
    store.add_turn(make_turn(6, 0.6, Phase::Debugging, 2));
    
    // Edges
    store.add_edge(Edge::new(TurnId::new(Uuid::from_u128(1)), TurnId::new(Uuid::from_u128(2)), EdgeType::Reply));
    store.add_edge(Edge::new(TurnId::new(Uuid::from_u128(1)), TurnId::new(Uuid::from_u128(3)), EdgeType::Branch));
    store.add_edge(Edge::new(TurnId::new(Uuid::from_u128(1)), TurnId::new(Uuid::from_u128(4)), EdgeType::Branch));
    store.add_edge(Edge::new(TurnId::new(Uuid::from_u128(2)), TurnId::new(Uuid::from_u128(5)), EdgeType::Reply));
    store.add_edge(Edge::new(TurnId::new(Uuid::from_u128(2)), TurnId::new(Uuid::from_u128(6)), EdgeType::Reply));
    
    Arc::new(store)
}

// ─────────────────────────────────────────────────────────────────────────────
// DETERMINISM TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_same_anchor_same_slice_id_100_runs() {
    let store = build_branching_graph();
    let policy = SlicePolicyV1::default();
    let slicer = ContextSlicer::new(store, policy, TEST_HMAC_SECRET.to_vec());

    let anchor_id = TurnId::new(Uuid::from_u128(1));

    let mut slice_ids: Vec<String> = Vec::with_capacity(100);

    for _ in 0..100 {
        let bundle = slicer.slice(anchor_id).await.unwrap();
        slice_ids.push(bundle.slice_id().as_str().to_string());
    }

    // All slice_ids must be identical
    for i in 1..100 {
        assert_eq!(
            slice_ids[0], slice_ids[i],
            "Slice ID must be deterministic (run {} differs from run 0)",
            i
        );
    }

    eprintln!("Deterministic slice_id: {}", slice_ids[0]);
}

#[tokio::test]
async fn test_policy_param_change_changes_slice_id() {
    let store = build_branching_graph();

    let policy1 = SlicePolicyV1::default();
    let mut policy2 = SlicePolicyV1::default();
    policy2.max_nodes = 3; // Change max_nodes

    let slicer1 = ContextSlicer::new(Arc::clone(&store), policy1, TEST_HMAC_SECRET.to_vec());
    let slicer2 = ContextSlicer::new(store, policy2, TEST_HMAC_SECRET.to_vec());

    let anchor_id = TurnId::new(Uuid::from_u128(1));

    let bundle1 = slicer1.slice(anchor_id).await.unwrap();
    let bundle2 = slicer2.slice(anchor_id).await.unwrap();

    // Different policy params → different slice_id (or different content)
    // Even if the turn set is the same, the policy_params_hash differs
    assert_ne!(
        bundle1.policy_params_hash(), bundle2.policy_params_hash(),
        "Different policy params must produce different params_hash"
    );
    assert_ne!(
        bundle1.slice_id().as_str(), bundle2.slice_id().as_str(),
        "Different policy params must produce different slice_id"
    );
}

#[tokio::test]
async fn test_edge_ordering_determinism() {
    let store = build_branching_graph();
    let policy = SlicePolicyV1::default();
    let slicer = ContextSlicer::new(store, policy, TEST_HMAC_SECRET.to_vec());

    let anchor_id = TurnId::new(Uuid::from_u128(1));

    let bundle1 = slicer.slice(anchor_id).await.unwrap();
    let bundle2 = slicer.slice(anchor_id).await.unwrap();

    // Edges must be in the same order
    let slice1 = bundle1.slice();
    let slice2 = bundle2.slice();
    assert_eq!(slice1.edges.len(), slice2.edges.len());
    for (e1, e2) in slice1.edges.iter().zip(slice2.edges.iter()) {
        assert_eq!(e1.parent, e2.parent);
        assert_eq!(e1.child, e2.child);
        assert_eq!(e1.edge_type, e2.edge_type);
    }
}

#[tokio::test]
async fn test_turn_ordering_determinism() {
    let store = build_branching_graph();
    let policy = SlicePolicyV1::default();
    let slicer = ContextSlicer::new(store, policy, TEST_HMAC_SECRET.to_vec());

    let anchor_id = TurnId::new(Uuid::from_u128(1));

    let bundle1 = slicer.slice(anchor_id).await.unwrap();
    let bundle2 = slicer.slice(anchor_id).await.unwrap();

    // Turns must be in the same order (by TurnId)
    let slice1 = bundle1.slice();
    let slice2 = bundle2.slice();
    assert_eq!(slice1.turns.len(), slice2.turns.len());
    for (t1, t2) in slice1.turns.iter().zip(slice2.turns.iter()) {
        assert_eq!(t1.id, t2.id);
    }

    // Verify sorted order
    for i in 1..slice1.turns.len() {
        assert!(
            slice1.turns[i - 1].id < slice1.turns[i].id,
            "Turns must be sorted by TurnId"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CORRECTNESS TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_anchor_always_included() {
    let store = build_linear_graph(10);
    let policy = SlicePolicyV1::default();
    let slicer = ContextSlicer::new(store, policy, TEST_HMAC_SECRET.to_vec());

    for i in 1..=10 {
        let anchor_id = TurnId::new(Uuid::from_u128(i));
        let bundle = slicer.slice(anchor_id).await.unwrap();

        assert!(
            bundle.is_turn_admissible(&anchor_id),
            "Slice must contain anchor turn"
        );
    }
}

#[tokio::test]
async fn test_max_nodes_respected() {
    let store = build_linear_graph(100);
    let mut policy = SlicePolicyV1::default();
    policy.max_nodes = 10;
    policy.max_radius = 50; // High enough to test node limit

    let slicer = ContextSlicer::new(store, policy, TEST_HMAC_SECRET.to_vec());

    let anchor_id = TurnId::new(Uuid::from_u128(50));
    let bundle = slicer.slice(anchor_id).await.unwrap();

    assert!(
        bundle.num_turns() <= 10,
        "Slice has {} turns but max is 10",
        bundle.num_turns()
    );
}

#[tokio::test]
async fn test_max_radius_respected() {
    let store = build_linear_graph(100);
    let mut policy = SlicePolicyV1::default();
    policy.max_nodes = 100; // High enough to test radius limit
    policy.max_radius = 3;

    let slicer = ContextSlicer::new(store, policy, TEST_HMAC_SECRET.to_vec());

    let anchor_id = TurnId::new(Uuid::from_u128(50));
    let bundle = slicer.slice(anchor_id).await.unwrap();

    // With radius 3, max possible turns: anchor + 3 in each direction = 7
    assert!(
        bundle.num_turns() <= 7,
        "Slice has {} turns but max radius 3 allows at most 7",
        bundle.num_turns()
    );
}

#[tokio::test]
async fn test_phase_weighting_affects_selection() {
    let mut store = InMemoryGraphStore::new();
    
    // Root
    store.add_turn(make_turn(1, 0.5, Phase::Consolidation, 0));
    
    // Two children: one Synthesis (high priority), one Exploration (low priority)
    store.add_turn(make_turn(2, 0.5, Phase::Synthesis, 1));
    store.add_turn(make_turn(3, 0.5, Phase::Exploration, 1));
    
    // Grandchildren - more turns to fill budget
    for i in 4..=20 {
        store.add_turn(make_turn(i, 0.5, Phase::Consolidation, 2));
    }
    
    store.add_edge(Edge::new(TurnId::new(Uuid::from_u128(1)), TurnId::new(Uuid::from_u128(2)), EdgeType::Reply));
    store.add_edge(Edge::new(TurnId::new(Uuid::from_u128(1)), TurnId::new(Uuid::from_u128(3)), EdgeType::Reply));
    
    for i in 4..=11 {
        store.add_edge(Edge::new(TurnId::new(Uuid::from_u128(2)), TurnId::new(Uuid::from_u128(i)), EdgeType::Reply));
    }
    for i in 12..=20 {
        store.add_edge(Edge::new(TurnId::new(Uuid::from_u128(3)), TurnId::new(Uuid::from_u128(i)), EdgeType::Reply));
    }
    
    let mut policy = SlicePolicyV1::default();
    policy.max_nodes = 5;
    policy.include_siblings = false;
    
    let slicer = ContextSlicer::new(Arc::new(store), policy, TEST_HMAC_SECRET.to_vec());

    let anchor_id = TurnId::new(Uuid::from_u128(1));
    let bundle = slicer.slice(anchor_id).await.unwrap();

    // Synthesis turn (id=2) should be included before Exploration (id=3)
    let has_synthesis = bundle.is_turn_admissible(&TurnId::new(Uuid::from_u128(2)));

    assert!(has_synthesis, "High-priority Synthesis turn should be selected");
}

// ─────────────────────────────────────────────────────────────────────────────
// CANONICAL SERIALIZATION TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_canonical_hash_determinism() {
    let policy = SlicePolicyV1::default();
    
    let mut hashes: Vec<String> = Vec::with_capacity(100);
    
    for _ in 0..100 {
        hashes.push(policy.params_hash());
    }
    
    for i in 1..100 {
        assert_eq!(
            hashes[0], hashes[i],
            "Policy params_hash must be deterministic"
        );
    }
}

#[tokio::test]
async fn test_slice_fingerprint_byte_level() {
    let store = build_branching_graph();
    let policy = SlicePolicyV1::default();
    let slicer = ContextSlicer::new(store, policy, TEST_HMAC_SECRET.to_vec());

    let anchor_id = TurnId::new(Uuid::from_u128(1));

    // Generate slices 100 times and verify byte-level determinism
    let mut all_slice_ids: Vec<String> = Vec::with_capacity(100);

    for _ in 0..100 {
        let bundle = slicer.slice(anchor_id).await.unwrap();
        all_slice_ids.push(bundle.slice_id().as_str().to_string());
    }

    // All must be identical
    for i in 1..100 {
        assert_eq!(
            all_slice_ids[0], all_slice_ids[i],
            "Slice fingerprint must be byte-level deterministic"
        );
    }
}

#[test]
fn test_phase_weights_hash_stability() {
    let weights1 = PhaseWeights::default();
    let weights2 = PhaseWeights::default();
    
    let h1 = canonical_hash_hex(&weights1);
    let h2 = canonical_hash_hex(&weights2);
    
    assert_eq!(h1, h2, "Identical PhaseWeights must have identical hash");
}

// ─────────────────────────────────────────────────────────────────────────────
// SIBLING TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_sibling_inclusion() {
    let store = build_branching_graph();

    let mut policy = SlicePolicyV1::default();
    policy.include_siblings = true;
    policy.max_siblings_per_node = 10;

    let slicer = ContextSlicer::new(store, policy, TEST_HMAC_SECRET.to_vec());

    // Start from node 2, which has siblings 3 and 4
    let anchor_id = TurnId::new(Uuid::from_u128(2));
    let bundle = slicer.slice(anchor_id).await.unwrap();

    // Should include siblings 3 and 4
    assert!(
        bundle.slice().contains_turn(&TurnId::new(Uuid::from_u128(3))),
        "Should include sibling 3"
    );
    assert!(
        bundle.slice().contains_turn(&TurnId::new(Uuid::from_u128(4))),
        "Should include sibling 4"
    );
}

#[tokio::test]
async fn test_sibling_limit() {
    let mut store = InMemoryGraphStore::new();

    // Root with many children
    store.add_turn(make_turn(1, 0.9, Phase::Synthesis, 0));

    for i in 2..=20 {
        store.add_turn(make_turn(i, (20 - i) as f32 / 20.0, Phase::Consolidation, 1));
        store.add_edge(Edge::new(
            TurnId::new(Uuid::from_u128(1)),
            TurnId::new(Uuid::from_u128(i)),
            EdgeType::Reply,
        ));
    }

    let mut policy = SlicePolicyV1::default();
    policy.include_siblings = true;
    policy.max_siblings_per_node = 3;
    policy.max_nodes = 10; // Limit nodes to test sibling behavior

    let slicer = ContextSlicer::new(Arc::new(store), policy, TEST_HMAC_SECRET.to_vec());

    let anchor_id = TurnId::new(Uuid::from_u128(10));
    let bundle = slicer.slice(anchor_id).await.unwrap();

    // With max_nodes = 10, we should respect the budget
    assert!(
        bundle.num_turns() <= 10,
        "Slice has {} turns but max is 10",
        bundle.num_turns()
    );

    // Anchor should always be included
    assert!(bundle.slice().contains_turn(&anchor_id), "Anchor must be in slice");
}

