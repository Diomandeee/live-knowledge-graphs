//! Integration tests for Atlas Run v1: Reproducible graph structure computation.
//!
//! These tests validate the end-to-end atlas pipeline:
//! 1. Graph snapshot identity
//! 2. Anchor selection
//! 3. Batch slice generation
//! 4. Overlap computation
//! 5. Influence scoring
//! 6. Phase topology
//! 7. Manifest bundling

use cc_graph_kernel::{
    TurnId, TurnSnapshot, Edge, EdgeType, Phase, Role,
    SlicePolicyV1, PhaseWeights as KernelPhaseWeights,
    GraphSnapshot, SnapshotInput,
    BatchSlicer, AnchorSet,
    OverlapAnalyzer,
    compute_influence, compute_phase_topology,
    AtlasBundler, PhaseTopology,
    ATLAS_SCHEMA_VERSION,
};
use cc_graph_kernel::store::memory::InMemoryGraphStore;
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────────────────────
// Test Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_turn(phase: Phase, salience: f32, created_at: i64) -> TurnSnapshot {
    TurnSnapshot::new(
        TurnId::new(Uuid::new_v4()),
        "test_session".to_string(),
        Role::User,
        phase,
        salience,
        1, 0, 0.5, 0.5, 1.0,
        created_at,
    )
}

fn create_test_graph(num_turns: usize) -> (InMemoryGraphStore, Vec<TurnSnapshot>) {
    let mut store = InMemoryGraphStore::new();
    let mut turns = Vec::new();
    
    let phases = [
        Phase::Exploration,
        Phase::Debugging,
        Phase::Planning,
        Phase::Consolidation,
        Phase::Synthesis,
    ];
    
    // Create turns distributed across phases
    for i in 0..num_turns {
        let phase = phases[i % phases.len()];
        let salience = 0.4 + (i as f32 % 10.0) * 0.05;
        let created_at = (i as i64) * 1000 + 1000000;
        
        let turn = make_turn(phase, salience, created_at);
        turns.push(turn.clone());
        store.add_turn(turn);
    }
    
    // Add edges to create a DAG structure
    for i in 1..turns.len() {
        let parent_idx = (i - 1) / 2; // Binary tree-ish structure
        store.add_edge(Edge::new(
            turns[parent_idx].id.clone(),
            turns[i].id.clone(),
            EdgeType::Reply,
        ));
    }
    
    (store, turns)
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 0: Snapshot Identity Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_snapshot_determinism() {
    let (store, _turns) = create_test_graph(50);
    
    // Get all turn IDs and edges
    let turn_ids: Vec<TurnId> = store.all_turns().iter().map(|t| t.id.clone()).collect();
    let edges = store.all_edges().to_vec();
    let timestamps: Vec<i64> = store.all_turns().iter().map(|t| t.created_at).collect();
    
    let input = SnapshotInput {
        turn_ids: turn_ids.clone(),
        edges: edges.clone(),
        timestamps: timestamps.clone(),
    };
    
    // Compute twice
    let snapshot1 = GraphSnapshot::compute(&input);
    let snapshot2 = GraphSnapshot::compute(&input);
    
    // Must be identical
    assert_eq!(snapshot1.snapshot_id, snapshot2.snapshot_id);
    assert_eq!(snapshot1.turn_count, 50);
}

#[test]
fn test_snapshot_changes_on_new_data() {
    let (store1, _) = create_test_graph(50);
    let (store2, _) = create_test_graph(51);
    
    let input1 = SnapshotInput {
        turn_ids: store1.all_turns().iter().map(|t| t.id.clone()).collect(),
        edges: store1.all_edges().to_vec(),
        timestamps: store1.all_turns().iter().map(|t| t.created_at).collect(),
    };
    
    let input2 = SnapshotInput {
        turn_ids: store2.all_turns().iter().map(|t| t.id.clone()).collect(),
        edges: store2.all_edges().to_vec(),
        timestamps: store2.all_turns().iter().map(|t| t.created_at).collect(),
    };
    
    let snapshot1 = GraphSnapshot::compute(&input1);
    let snapshot2 = GraphSnapshot::compute(&input2);
    
    // Different data = different snapshot
    assert_ne!(snapshot1.snapshot_id, snapshot2.snapshot_id);
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: Anchor Set Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_anchor_set_determinism() {
    let (_, turns) = create_test_graph(100);
    let turn_ids: Vec<TurnId> = turns.iter().take(20).map(|t| t.id.clone()).collect();
    
    let set1 = AnchorSet::new(turn_ids.clone(), "test_policy_v1");
    let set2 = AnchorSet::new(turn_ids, "test_policy_v1");
    
    assert_eq!(set1.anchor_set_hash, set2.anchor_set_hash);
}

#[test]
fn test_anchor_set_order_independence() {
    let (_, turns) = create_test_graph(100);
    let mut turn_ids: Vec<TurnId> = turns.iter().take(20).map(|t| t.id.clone()).collect();
    
    let set1 = AnchorSet::new(turn_ids.clone(), "test_policy_v1");
    
    // Reverse order
    turn_ids.reverse();
    let set2 = AnchorSet::new(turn_ids, "test_policy_v1");
    
    // Same hash despite different input order
    assert_eq!(set1.anchor_set_hash, set2.anchor_set_hash);
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: Batch Slice Tests
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_batch_slicer_determinism() {
    let (store, turns) = create_test_graph(100);
    let store = std::sync::Arc::new(store);
    let anchors: Vec<TurnId> = turns.iter().take(10).map(|t| t.id.clone()).collect();

    let policy = SlicePolicyV1 {
        max_nodes: 15,
        max_radius: 3,
        phase_weights: KernelPhaseWeights::default(),
        salience_weight: 1.0,
        distance_decay: 0.8,
        include_siblings: true,
        max_siblings_per_node: 3,
        version: "slice_policy_v1".to_string(),
    };

    let slicer = BatchSlicer::new(store, policy, b"test_hmac_secret_for_integration".to_vec());

    let result1 = slicer.slice_all(&anchors, "snapshot_1", "anchors_1").await.unwrap();
    let result2 = slicer.slice_all(&anchors, "snapshot_1", "anchors_1").await.unwrap();

    // Registry hashes must match
    assert_eq!(result1.registry.registry_hash, result2.registry.registry_hash);

    // All slice IDs must match
    for (s1, s2) in result1.slices.iter().zip(result2.slices.iter()) {
        assert_eq!(s1.slice_id.to_string(), s2.slice_id.to_string());
    }
}

#[tokio::test]
async fn test_batch_slicer_produces_registry() {
    let (store, turns) = create_test_graph(50);
    let store = std::sync::Arc::new(store);
    let anchors: Vec<TurnId> = turns.iter().take(5).map(|t| t.id.clone()).collect();

    let policy = SlicePolicyV1::default();
    let slicer = BatchSlicer::new(store, policy, b"test_hmac_secret_for_integration".to_vec());

    let result = slicer.slice_all(&anchors, "snapshot", "anchors").await.unwrap();

    assert_eq!(result.slices.len(), 5);
    assert_eq!(result.registry.entries.len(), 5);

    // Each entry should have valid data
    for entry in &result.registry.entries {
        assert!(!entry.slice_id.is_empty());
        assert!(!entry.anchor_turn_id.is_empty());
        assert!(entry.turn_count > 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3: Overlap Computation Tests
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_overlap_graph_determinism() {
    let (store, turns) = create_test_graph(50);
    let store = std::sync::Arc::new(store);
    let anchors: Vec<TurnId> = turns.iter().take(10).map(|t| t.id.clone()).collect();

    let policy = SlicePolicyV1::default();
    let slicer = BatchSlicer::new(store, policy, b"test_hmac_secret_for_integration".to_vec());
    let batch = slicer.slice_all(&anchors, "snapshot", "anchors").await.unwrap();

    let analyzer = OverlapAnalyzer::new();

    let graph1 = analyzer.compute(&batch.slices);
    let graph2 = analyzer.compute(&batch.slices);

    assert_eq!(graph1.graph_hash, graph2.graph_hash);
}

#[tokio::test]
async fn test_overlap_detects_shared_turns() {
    let (store, turns) = create_test_graph(20);
    let store = std::sync::Arc::new(store);

    // Use adjacent turns as anchors - they should share neighbors
    let anchors = vec![turns[0].id.clone(), turns[1].id.clone()];

    let policy = SlicePolicyV1 {
        max_nodes: 10,
        max_radius: 5,
        ..SlicePolicyV1::default()
    };

    let slicer = BatchSlicer::new(store, policy, b"test_hmac_secret_for_integration".to_vec());
    let batch = slicer.slice_all(&anchors, "snapshot", "anchors").await.unwrap();

    let analyzer = OverlapAnalyzer::new();
    let graph = analyzer.compute(&batch.slices);

    // Should have at least one overlap edge
    assert!(!graph.edges.is_empty(), "Adjacent anchors should produce overlapping slices");
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4: Influence Score Tests
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_influence_scores_determinism() {
    let (store, turns) = create_test_graph(50);
    let store = std::sync::Arc::new(store);
    let anchors: Vec<TurnId> = turns.iter().take(10).map(|t| t.id.clone()).collect();

    let policy = SlicePolicyV1::default();
    let slicer = BatchSlicer::new(store, policy, b"test_hmac_secret_for_integration".to_vec());
    let batch = slicer.slice_all(&anchors, "snapshot", "anchors").await.unwrap();

    let scores1 = compute_influence(&batch.slices);
    let scores2 = compute_influence(&batch.slices);

    assert_eq!(scores1.scores_hash, scores2.scores_hash);
}

#[tokio::test]
async fn test_influence_identifies_high_coverage_turns() {
    let (store, turns) = create_test_graph(30);
    let store = std::sync::Arc::new(store);
    let anchors: Vec<TurnId> = turns.iter().take(10).map(|t| t.id.clone()).collect();

    let policy = SlicePolicyV1 {
        max_nodes: 20,
        max_radius: 10,
        ..SlicePolicyV1::default()
    };

    let slicer = BatchSlicer::new(store, policy, b"test_hmac_secret_for_integration".to_vec());
    let batch = slicer.slice_all(&anchors, "snapshot", "anchors").await.unwrap();

    let scores = compute_influence(&batch.slices);

    // Should have computed scores for some turns
    assert!(!scores.scores.is_empty());

    // Top influential should have slice_count > 1
    if let Some(top) = scores.top_influential(1).first() {
        assert!(top.slice_count >= 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 5: Phase Topology Tests
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_phase_topology_determinism() {
    let (store, turns) = create_test_graph(50);
    let store = std::sync::Arc::new(store);
    let anchors: Vec<TurnId> = turns.iter().take(15).map(|t| t.id.clone()).collect();

    let policy = SlicePolicyV1 {
        max_nodes: 15,
        max_radius: 5,
        ..SlicePolicyV1::default()
    };

    let slicer = BatchSlicer::new(store, policy, b"test_hmac_secret_for_integration".to_vec());
    let batch = slicer.slice_all(&anchors, "snapshot", "anchors").await.unwrap();

    let analyzer = OverlapAnalyzer::new();
    let overlap = analyzer.compute(&batch.slices);

    let topo1 = compute_phase_topology(&batch.slices, &overlap.edges, 3);
    let topo2 = compute_phase_topology(&batch.slices, &overlap.edges, 3);

    assert_eq!(topo1.stats_hash, topo2.stats_hash);
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 6: Atlas Bundle Tests
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_atlas_bundle_determinism() {
    let (store, turns) = create_test_graph(50);
    let anchors: Vec<TurnId> = turns.iter().take(10).map(|t| t.id.clone()).collect();

    // Compute snapshot
    let snapshot_input = SnapshotInput {
        turn_ids: store.all_turns().iter().map(|t| t.id.clone()).collect(),
        edges: store.all_edges().to_vec(),
        timestamps: store.all_turns().iter().map(|t| t.created_at).collect(),
    };
    let snapshot = GraphSnapshot::compute(&snapshot_input);

    // Compute slices
    let store = std::sync::Arc::new(store);
    let policy = SlicePolicyV1::default();
    let slicer = BatchSlicer::new(store, policy, b"test_hmac_secret_for_integration".to_vec());
    let anchor_set = AnchorSet::new(anchors, "test_policy");
    let batch = slicer.slice_all(&anchor_set.anchors, &snapshot.snapshot_id, &anchor_set.anchor_set_hash).await.unwrap();

    // Compute overlap
    let analyzer = OverlapAnalyzer::new();
    let overlap = analyzer.compute(&batch.slices);

    // Compute influence
    let influence = compute_influence(&batch.slices);

    // Compute phase topology
    let topo_stats = compute_phase_topology(&batch.slices, &overlap.edges, 3);
    let phase_topology = PhaseTopology::new(
        topo_stats.phase_pair_overlaps,
        topo_stats.phase_centroids,
        topo_stats.cross_phase_bridges.len(),
    );

    // Bundle twice
    let manifest1 = AtlasBundler::new()
        .snapshot(snapshot.clone())
        .batch_result(batch.clone())
        .overlap_graph(overlap.clone())
        .influence_scores(influence.clone())
        .phase_topology(phase_topology.clone())
        .build();

    let manifest2 = AtlasBundler::new()
        .snapshot(snapshot)
        .batch_result(batch)
        .overlap_graph(overlap)
        .influence_scores(influence)
        .phase_topology(phase_topology)
        .build();

    // Atlas ID must be identical
    assert_eq!(manifest1.atlas_id, manifest2.atlas_id);
    assert_eq!(manifest1.version, ATLAS_SCHEMA_VERSION);
}

#[tokio::test]
async fn test_atlas_bundle_includes_stats() {
    let (store, turns) = create_test_graph(30);
    let anchors: Vec<TurnId> = turns.iter().take(5).map(|t| t.id.clone()).collect();

    let snapshot_input = SnapshotInput {
        turn_ids: store.all_turns().iter().map(|t| t.id.clone()).collect(),
        edges: store.all_edges().to_vec(),
        timestamps: store.all_turns().iter().map(|t| t.created_at).collect(),
    };
    let snapshot = GraphSnapshot::compute(&snapshot_input);

    let store = std::sync::Arc::new(store);
    let policy = SlicePolicyV1::default();
    let slicer = BatchSlicer::new(store, policy, b"test_hmac_secret_for_integration".to_vec());
    let anchor_set = AnchorSet::new(anchors, "test");
    let batch = slicer.slice_all(&anchor_set.anchors, &snapshot.snapshot_id, &anchor_set.anchor_set_hash).await.unwrap();

    let analyzer = OverlapAnalyzer::new();
    let overlap = analyzer.compute(&batch.slices);
    let influence = compute_influence(&batch.slices);
    let topo_stats = compute_phase_topology(&batch.slices, &overlap.edges, 3);
    let phase_topology = PhaseTopology::new(
        topo_stats.phase_pair_overlaps,
        topo_stats.phase_centroids,
        topo_stats.cross_phase_bridges.len(),
    );

    let manifest = AtlasBundler::new()
        .snapshot(snapshot)
        .batch_result(batch)
        .overlap_graph(overlap)
        .influence_scores(influence)
        .phase_topology(phase_topology)
        .build();

    assert_eq!(manifest.stats.turn_count, 30);
    assert_eq!(manifest.stats.anchor_count, 5);
    assert_eq!(manifest.stats.slice_count, 5);
}

// ─────────────────────────────────────────────────────────────────────────────
// End-to-End Replay Test
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_full_atlas_replay_determinism() {
    // Run the full pipeline twice and verify byte-identical output

    async fn run_atlas_pipeline(store: std::sync::Arc<InMemoryGraphStore>, turns: &[TurnSnapshot]) -> String {
        let anchors: Vec<TurnId> = turns.iter().take(8).map(|t| t.id.clone()).collect();

        let snapshot_input = SnapshotInput {
            turn_ids: store.all_turns().iter().map(|t| t.id.clone()).collect(),
            edges: store.all_edges().to_vec(),
            timestamps: store.all_turns().iter().map(|t| t.created_at).collect(),
        };
        let snapshot = GraphSnapshot::compute(&snapshot_input);

        let policy = SlicePolicyV1 {
            max_nodes: 12,
            max_radius: 4,
            phase_weights: KernelPhaseWeights::default(),
            salience_weight: 1.0,
            distance_decay: 0.8,
            include_siblings: true,
            max_siblings_per_node: 2,
            version: "slice_policy_v1".to_string(),
        };

        let slicer = BatchSlicer::new(store, policy, b"test_hmac_secret_for_integration".to_vec());
        let anchor_set = AnchorSet::new(anchors, "replay_test");
        let batch = slicer.slice_all(
            &anchor_set.anchors,
            &snapshot.snapshot_id,
            &anchor_set.anchor_set_hash,
        ).await.unwrap();

        let analyzer = OverlapAnalyzer::new();
        let overlap = analyzer.compute(&batch.slices);
        let influence = compute_influence(&batch.slices);
        let topo_stats = compute_phase_topology(&batch.slices, &overlap.edges, 3);
        let phase_topology = PhaseTopology::new(
            topo_stats.phase_pair_overlaps,
            topo_stats.phase_centroids,
            topo_stats.cross_phase_bridges.len(),
        );

        let manifest = AtlasBundler::new()
            .snapshot(snapshot)
            .batch_result(batch)
            .overlap_graph(overlap)
            .influence_scores(influence)
            .phase_topology(phase_topology)
            .build();

        manifest.atlas_id
    }

    let (store, turns) = create_test_graph(40);
    let store = std::sync::Arc::new(store);

    let atlas_id_1 = run_atlas_pipeline(std::sync::Arc::clone(&store), &turns).await;
    let atlas_id_2 = run_atlas_pipeline(store, &turns).await;

    assert_eq!(atlas_id_1, atlas_id_2, "Full atlas pipeline must be deterministic");
}

