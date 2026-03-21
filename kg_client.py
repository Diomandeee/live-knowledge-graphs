#!/usr/bin/env python3
"""Minimal Knowledge Graph client for runtime integration.

Demonstrates the core pattern: live graph queries during inference,
not just at training time.

The full Rust implementation (cc-graph-kernel) is at:
  github.com/Diomandeee/Comp-Core/core/semantic/cc-graph-kernel/
"""
import json
import requests
from dataclasses import dataclass
from typing import Optional

@dataclass
class TraversalPath:
    entities: list[str]
    edges: list[dict]
    hops: int
    min_confidence: float

@dataclass
class ContextSlice:
    turns: list[dict]
    edges: list[dict]
    admissibility_token: Optional[str]

class LiveKnowledgeGraph:
    """Runtime KG client — queries graph DURING inference, not just training."""
    
    def __init__(self, gk_url: str = "http://localhost:8001"):
        self.url = gk_url
        self._edge_cache: dict[tuple, bool] = {}
    
    def health(self) -> dict:
        return requests.get(f"{self.url}/health", timeout=5).json()
    
    def traverse(self, start: str, max_hops: int = 2, 
                 predicates: Optional[list[str]] = None,
                 min_confidence: float = 0.0) -> list[TraversalPath]:
        """BFS traversal from entity — the core runtime operation."""
        payload = {"start": start, "max_hops": max_hops}
        if predicates:
            payload["predicates"] = predicates
        if min_confidence > 0:
            payload["min_confidence"] = min_confidence
        
        resp = requests.post(f"{self.url}/api/knowledge/traverse",
                           json=payload, timeout=10)
        resp.raise_for_status()
        
        paths = []
        for p in resp.json().get("paths", []):
            paths.append(TraversalPath(
                entities=p["entities"],
                edges=p["edges"],
                hops=p["hops"],
                min_confidence=p["min_confidence"],
            ))
        return paths
    
    def edge_exists(self, subject: str, predicate: str, obj: str) -> bool:
        """Check if a specific edge exists — used for reward computation."""
        key = (subject, predicate, obj)
        if key in self._edge_cache:
            return self._edge_cache[key]
        
        resp = requests.get(f"{self.url}/api/knowledge",
            params={"subject": subject, "predicate": predicate, "object": obj},
            timeout=5)
        exists = resp.json().get("total", 0) > 0
        self._edge_cache[key] = exists
        return exists
    
    def context_slice(self, anchor_turn: str, max_nodes: int = 50,
                      max_radius: int = 3) -> ContextSlice:
        """Get a provenance-tracked context slice for prompt injection."""
        resp = requests.post(f"{self.url}/api/slice",
            json={"anchor": anchor_turn, "max_nodes": max_nodes, 
                  "max_radius": max_radius},
            timeout=10)
        data = resp.json()
        return ContextSlice(
            turns=data.get("turns", []),
            edges=data.get("edges", []),
            admissibility_token=data.get("admissibility_token"),
        )

if __name__ == "__main__":
    kg = LiveKnowledgeGraph()
    
    try:
        print(f"Health: {kg.health()}")
        
        # Traverse from a known entity
        paths = kg.traverse("comp-core", max_hops=2)
        print(f"\nPaths from 'comp-core': {len(paths)}")
        for p in paths[:5]:
            chain = " -> ".join(p.entities)
            print(f"  [{p.hops} hops, conf={p.min_confidence:.2f}] {chain}")
        
        # Check specific edge
        exists = kg.edge_exists("spore", "is_a", "product")
        print(f"\nspore --[is_a]--> product: {exists}")
        
    except requests.ConnectionError:
        print("Graph Kernel not running at localhost:8001")
        print("Start it or SSH tunnel: ssh -L 8001:localhost:8001 cloud-vm")
