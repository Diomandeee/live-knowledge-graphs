#!/usr/bin/env python3
"""Example: BFS traversal through the knowledge graph.

Demonstrates how to use the Graph Kernel's traversal API to explore
relationships between entities, score paths with the reward function,
and build a local subgraph for analysis.

Prerequisites:
    - Graph Kernel service running at localhost:8001
    - Some knowledge triples already ingested

Usage:
    python examples/traverse.py
"""

import json
import sys
import os

# Add parent directory so we can import from python/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from kg_client import KGClient, Triple
from kg_reward import KGPath, PathEdge, compute_path_reward, compute_trajectory_reward, RewardWeights


def seed_example_graph(kg: KGClient) -> None:
    """Seed a small example knowledge graph for demonstration."""
    triples = [
        # Project relationships
        {"subject": "spore", "predicate": "built_with", "object": "swiftui", "confidence": 0.95, "source": "codebase"},
        {"subject": "spore", "predicate": "built_with", "object": "cloudkit", "confidence": 0.90, "source": "codebase"},
        {"subject": "spore", "predicate": "is_a", "object": "ios_app", "confidence": 0.99, "source": "codebase"},
        {"subject": "spore", "predicate": "has_feature", "object": "garden_metaphor", "confidence": 0.95, "source": "design"},
        {"subject": "spore", "predicate": "has_feature", "object": "voice_capture", "confidence": 0.85, "source": "codebase"},

        # Framework relationships
        {"subject": "swiftui", "predicate": "is_a", "object": "framework", "confidence": 0.99, "source": "docs"},
        {"subject": "swiftui", "predicate": "used_by", "object": "creativedirector", "confidence": 0.90, "source": "codebase"},
        {"subject": "swiftui", "predicate": "used_by", "object": "openclawhub", "confidence": 0.92, "source": "codebase"},
        {"subject": "cloudkit", "predicate": "is_a", "object": "framework", "confidence": 0.99, "source": "docs"},
        {"subject": "cloudkit", "predicate": "provides", "object": "sync", "confidence": 0.95, "source": "docs"},

        # Cross-project connections
        {"subject": "creativedirector", "predicate": "is_a", "object": "ios_app", "confidence": 0.99, "source": "codebase"},
        {"subject": "creativedirector", "predicate": "has_feature", "object": "teleprompter", "confidence": 0.90, "source": "codebase"},
        {"subject": "creativedirector", "predicate": "has_feature", "object": "video_editing", "confidence": 0.88, "source": "codebase"},

        # Infrastructure
        {"subject": "graph_kernel", "predicate": "built_with", "object": "rust", "confidence": 0.99, "source": "codebase"},
        {"subject": "graph_kernel", "predicate": "built_with", "object": "axum", "confidence": 0.95, "source": "codebase"},
        {"subject": "graph_kernel", "predicate": "provides", "object": "context_slicing", "confidence": 0.98, "source": "architecture"},
        {"subject": "rag_plus_plus", "predicate": "uses", "object": "graph_kernel", "confidence": 0.95, "source": "codebase"},
        {"subject": "rag_plus_plus", "predicate": "built_with", "object": "python", "confidence": 0.99, "source": "codebase"},
        {"subject": "rag_plus_plus", "predicate": "provides", "object": "retrieval", "confidence": 0.97, "source": "architecture"},

        # Semantic layer
        {"subject": "ios_app", "predicate": "deployed_on", "object": "testflight", "confidence": 0.95, "source": "ops"},
        {"subject": "testflight", "predicate": "is_a", "object": "distribution", "confidence": 0.99, "source": "docs"},
    ]

    result = kg.add_batch(triples)
    print(f"Seeded {result.get('added', 0)} triples ({result.get('updated', 0)} updated)")


def explore_entity(kg: KGClient, entity: str, max_hops: int = 3) -> None:
    """Explore the neighborhood of an entity using BFS traversal."""
    print(f"\n{'='*60}")
    print(f"Exploring: {entity} (max {max_hops} hops)")
    print(f"{'='*60}")

    paths = kg.traverse(
        start=entity,
        max_hops=max_hops,
        max_results=20,
        return_paths=True,
    )

    if not paths:
        print("  No paths found.")
        return

    print(f"  Found {len(paths)} paths:\n")

    for i, path in enumerate(paths):
        entity_chain = " -> ".join(path.entities)
        print(f"  [{i+1}] {entity_chain}")
        print(f"      Hops: {path.hops}, Min confidence: {path.min_confidence:.2f}")


def score_paths(kg: KGClient, entity: str) -> None:
    """Score traversal paths using the reward function."""
    print(f"\n{'='*60}")
    print(f"Scoring paths from: {entity}")
    print(f"{'='*60}")

    traversal_paths = kg.traverse(
        start=entity,
        max_hops=3,
        max_results=15,
        return_paths=True,
    )

    if not traversal_paths:
        print("  No paths to score.")
        return

    # Convert to KGPath objects for scoring
    kg_paths = []
    for tp in traversal_paths:
        edges = []
        for edge in tp.edges:
            edges.append(PathEdge(
                subject=edge.get("subject", ""),
                predicate=edge.get("predicate", ""),
                object=edge.get("object", ""),
                confidence=edge.get("confidence", 0.5),
            ))
        kg_paths.append(KGPath(edges=edges))

    # Score each path
    weights = RewardWeights()
    print(f"\n  {'Path':<50} {'Reward':>8}")
    print(f"  {'-'*50} {'-'*8}")

    scored = []
    for tp, kp in zip(traversal_paths, kg_paths):
        reward = compute_path_reward(kp, weights)
        entity_chain = " -> ".join(tp.entities[:4])
        if len(tp.entities) > 4:
            entity_chain += f" -> ... ({len(tp.entities)} total)"
        scored.append((entity_chain, reward))

    # Sort by reward descending
    scored.sort(key=lambda x: x[1], reverse=True)
    for chain, reward in scored:
        print(f"  {chain:<50} {reward:>+.4f}")

    # Aggregate trajectory reward
    trajectory_reward = compute_trajectory_reward(kg_paths, weights)
    print(f"\n  Trajectory reward (aggregate): {trajectory_reward:+.4f}")


def show_graph_stats(kg: KGClient) -> None:
    """Display knowledge graph statistics."""
    stats = kg.stats()
    print(f"\n{'='*60}")
    print("Knowledge Graph Statistics")
    print(f"{'='*60}")
    print(f"  Total triples:     {stats.get('total_triples', 0)}")
    print(f"  Unique subjects:   {stats.get('unique_subjects', 0)}")
    print(f"  Unique predicates: {stats.get('unique_predicates', 0)}")

    top = stats.get("top_predicates", [])
    if top:
        print(f"\n  Top predicates:")
        for p in top[:10]:
            name = p.get("predicate", p) if isinstance(p, dict) else p
            count = p.get("count", "?") if isinstance(p, dict) else "?"
            print(f"    {name}: {count}")


def main():
    kg = KGClient("http://localhost:8001")

    # Check connectivity
    if not kg.is_healthy():
        print("Graph Kernel is not reachable at localhost:8001")
        print("Start it with: cargo run --features service")
        print("\nRunning in demo mode with local-only reward scoring...\n")

        # Demo the reward function without a live server
        demo_path = KGPath(edges=[
            PathEdge("spore", "built_with", "swiftui", confidence=0.95),
            PathEdge("swiftui", "is_a", "framework", confidence=0.99),
            PathEdge("framework", "used_by", "creativedirector", confidence=0.85),
        ])

        reward = compute_path_reward(
            demo_path,
            entity_domains={
                "spore": "mobile",
                "swiftui": "platform",
                "framework": "platform",
                "creativedirector": "mobile",
            },
        )
        print(f"Demo path: {' -> '.join(demo_path.entities)}")
        print(f"Length: {demo_path.length}, Mean confidence: {demo_path.mean_confidence:.2f}")
        print(f"Predicate diversity: {demo_path.predicate_diversity:.2f}")
        print(f"Reward: {reward:+.4f}")
        return

    print("Connected to Graph Kernel")

    # Seed example data
    seed_example_graph(kg)

    # Show stats
    show_graph_stats(kg)

    # Explore entities
    for entity in ["spore", "swiftui", "graph_kernel"]:
        explore_entity(kg, entity)

    # Score paths
    score_paths(kg, "spore")

    print("\nDone.")


if __name__ == "__main__":
    main()
