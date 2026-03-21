#!/usr/bin/env python3
"""KG Path Reward Function for trajectory scoring.

Scores knowledge-graph paths by structural properties that correlate
with useful agent trajectories: path length, branching factor,
confidence decay, and phase diversity.

Used in KARL trajectory intelligence to compute advantage-weighted
rewards for fine-tuning.

The key insight: paths through the knowledge graph that an agent
actually traversed during a successful task carry structural signal.
Short, high-confidence, diverse-phase paths correlate with efficient
problem-solving. Long, low-confidence, single-phase paths correlate
with thrashing.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathEdge:
    """A single edge in a KG path."""
    subject: str
    predicate: str
    object: str
    confidence: float


@dataclass
class KGPath:
    """A path through the knowledge graph."""
    edges: list[PathEdge] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.edges)

    @property
    def entities(self) -> list[str]:
        if not self.edges:
            return []
        result = [self.edges[0].subject]
        for e in self.edges:
            result.append(e.object)
        return result

    @property
    def predicates(self) -> list[str]:
        return [e.predicate for e in self.edges]

    @property
    def min_confidence(self) -> float:
        if not self.edges:
            return 0.0
        return min(e.confidence for e in self.edges)

    @property
    def mean_confidence(self) -> float:
        if not self.edges:
            return 0.0
        return sum(e.confidence for e in self.edges) / len(self.edges)

    @property
    def predicate_diversity(self) -> float:
        """Ratio of unique predicates to total edges (0-1)."""
        if not self.edges:
            return 0.0
        return len(set(self.predicates)) / len(self.edges)


@dataclass
class RewardWeights:
    """Configurable weights for the path reward function."""
    length_penalty: float = -0.05      # Penalty per hop (shorter = better)
    confidence_weight: float = 0.3     # Reward for high mean confidence
    min_conf_weight: float = 0.15      # Reward for high minimum confidence
    diversity_weight: float = 0.25     # Reward for predicate diversity
    branching_weight: float = 0.15     # Reward for intermediate branching
    cycle_penalty: float = -0.3        # Penalty for revisiting entities
    bridge_bonus: float = 0.2          # Bonus for cross-domain connections


def compute_path_reward(
    path: KGPath,
    weights: Optional[RewardWeights] = None,
    branching_factors: Optional[dict[str, int]] = None,
    entity_domains: Optional[dict[str, str]] = None,
) -> float:
    """Compute a scalar reward for a KG path.

    Args:
        path: The knowledge graph path to score
        weights: Configurable reward weights (uses defaults if None)
        branching_factors: Map of entity -> outgoing edge count
            (from graph stats). Used for branching reward.
        entity_domains: Map of entity -> domain label.
            Used for bridge bonus (cross-domain connections).

    Returns:
        Float reward in roughly [-1, 1] range. Higher is better.
        A reward of 0.0 indicates a neutral/average path.
    """
    if path.length == 0:
        return 0.0

    w = weights or RewardWeights()

    # 1. Length penalty: shorter paths preferred
    length_score = w.length_penalty * path.length

    # 2. Confidence reward: high-confidence paths preferred
    conf_score = w.confidence_weight * path.mean_confidence
    min_conf_score = w.min_conf_weight * path.min_confidence

    # 3. Predicate diversity: varied relationships preferred
    div_score = w.diversity_weight * path.predicate_diversity

    # 4. Branching factor: paths through well-connected nodes preferred
    branch_score = 0.0
    if branching_factors:
        internal_entities = path.entities[1:-1]  # skip start/end
        if internal_entities:
            avg_branching = sum(
                branching_factors.get(e, 1) for e in internal_entities
            ) / len(internal_entities)
            # Log-scale branching (diminishing returns)
            branch_score = w.branching_weight * math.log1p(avg_branching) / 5.0

    # 5. Cycle penalty: revisiting entities is usually thrashing
    entities = path.entities
    unique_ratio = len(set(entities)) / len(entities) if entities else 1.0
    cycle_score = w.cycle_penalty * (1.0 - unique_ratio)

    # 6. Bridge bonus: crossing domain boundaries indicates synthesis
    bridge_score = 0.0
    if entity_domains and path.length >= 2:
        domains_traversed = set()
        for e in path.entities:
            if e in entity_domains:
                domains_traversed.add(entity_domains[e])
        if len(domains_traversed) > 1:
            # Bonus scales with number of domains crossed
            bridge_score = w.bridge_bonus * min(len(domains_traversed) / 3.0, 1.0)

    total = (
        length_score
        + conf_score
        + min_conf_score
        + div_score
        + branch_score
        + cycle_score
        + bridge_score
    )

    # Clamp to [-1, 1] for stability
    return max(-1.0, min(1.0, total))


def compute_trajectory_reward(
    paths: list[KGPath],
    weights: Optional[RewardWeights] = None,
    branching_factors: Optional[dict[str, int]] = None,
    entity_domains: Optional[dict[str, str]] = None,
) -> float:
    """Compute aggregate reward for a sequence of KG paths in a trajectory.

    Takes the mean of individual path rewards, with a small bonus
    for path diversity (different starting entities = broader exploration).
    """
    if not paths:
        return 0.0

    individual_rewards = [
        compute_path_reward(p, weights, branching_factors, entity_domains)
        for p in paths
    ]

    mean_reward = sum(individual_rewards) / len(individual_rewards)

    # Diversity bonus: unique starting entities
    start_entities = set(p.entities[0] for p in paths if p.entities)
    diversity_bonus = 0.05 * min(len(start_entities) / max(len(paths), 1), 1.0)

    return max(-1.0, min(1.0, mean_reward + diversity_bonus))


def compute_advantage(
    trajectory_reward: float,
    baseline_mean: float,
    baseline_std: float,
) -> float:
    """Compute z-score advantage for advantage-weighted fine-tuning.

    advantage = (reward - baseline_mean) / baseline_std

    Positive advantage means this trajectory is above average.
    Used in KARL FlowRL sampling to weight SFT examples.
    """
    if baseline_std < 1e-8:
        return 0.0
    return (trajectory_reward - baseline_mean) / baseline_std


if __name__ == "__main__":
    # Example: score a sample path
    path = KGPath(edges=[
        PathEdge("spore", "built_with", "swiftui", confidence=0.95),
        PathEdge("swiftui", "is_a", "framework", confidence=0.99),
        PathEdge("framework", "used_by", "creativedirector", confidence=0.85),
    ])

    reward = compute_path_reward(
        path,
        entity_domains={
            "spore": "mobile",
            "swiftui": "platform",
            "framework": "platform",
            "creativedirector": "mobile",
        },
    )
    print(f"Path: {' -> '.join(path.entities)}")
    print(f"Length: {path.length}, Confidence: {path.mean_confidence:.2f}")
    print(f"Diversity: {path.predicate_diversity:.2f}")
    print(f"Reward: {reward:.4f}")

    advantage = compute_advantage(reward, baseline_mean=0.15, baseline_std=0.12)
    print(f"Advantage (z-score): {advantage:.4f}")
