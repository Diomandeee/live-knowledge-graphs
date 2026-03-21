#!/usr/bin/env python3
"""Minimal Knowledge Graph client for runtime integration.

Demonstrates the core pattern: live graph queries during inference,
not just at training time.

Usage:
    from kg_client import KGClient
    kg = KGClient("http://localhost:8001")
    kg.add_triple("spore", "built_with", "swiftui", confidence=0.95)
    results = kg.query(subject="spore")
    paths = kg.traverse("spore", max_hops=3)
"""

import json
import time
from dataclasses import dataclass
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode


@dataclass
class Triple:
    """A knowledge triple (subject-predicate-object)."""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.5
    source: str = "unknown"
    id: Optional[int] = None
    created_at: Optional[str] = None


@dataclass
class TraversalPath:
    """A path through the knowledge graph."""
    entities: list
    edges: list
    hops: int
    min_confidence: float


class KGClient:
    """Client for the Graph Kernel knowledge API.

    All methods use stdlib urllib (no dependencies beyond Python 3.9+).
    """

    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Knowledge CRUD
    # ------------------------------------------------------------------

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.5,
        source: str = "kg_client",
    ) -> dict:
        """Insert a single knowledge triple."""
        payload = {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "source": source,
        }
        return self._post("/api/knowledge", payload)

    def add_batch(self, triples: list[dict]) -> dict:
        """Insert multiple triples in one call.

        Each dict should have: subject, predicate, object, confidence, source.
        """
        return self._post("/api/knowledge/batch", {"triples": triples})

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 100,
    ) -> list[Triple]:
        """Query knowledge triples with optional filters."""
        params = {"limit": str(limit)}
        if subject:
            params["subject"] = subject
        if predicate:
            params["predicate"] = predicate
        if obj:
            params["object"] = obj
        if min_confidence is not None:
            params["min_confidence"] = str(min_confidence)

        data = self._get("/api/knowledge", params)
        return [
            Triple(
                id=t.get("id"),
                subject=t["subject"],
                predicate=t["predicate"],
                object=t["object"],
                confidence=t["confidence"],
                source=t.get("source", "unknown"),
                created_at=t.get("created_at"),
            )
            for t in data.get("triples", [])
        ]

    def stats(self) -> dict:
        """Get knowledge graph statistics."""
        return self._get("/api/knowledge/stats")

    def delete(
        self,
        id: Optional[int] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> dict:
        """Delete triples matching the given filters."""
        payload = {}
        if id is not None:
            payload["id"] = id
        if subject:
            payload["subject"] = subject
        if predicate:
            payload["predicate"] = predicate
        if obj:
            payload["object"] = obj
        return self._delete("/api/knowledge", payload)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def traverse(
        self,
        start: str,
        predicates: Optional[list[str]] = None,
        direction: str = "both",
        max_hops: int = 3,
        max_results: int = 50,
        min_confidence: Optional[float] = None,
        return_paths: bool = True,
    ) -> list[TraversalPath]:
        """Multi-hop graph traversal from a starting entity.

        Args:
            start: Starting entity name
            predicates: Filter to specific predicate types (None = all)
            direction: "outgoing", "incoming", or "both"
            max_hops: Maximum traversal depth
            max_results: Maximum paths to return
            min_confidence: Minimum confidence threshold
            return_paths: Whether to return full paths
        """
        payload = {
            "start": start,
            "direction": direction,
            "max_hops": max_hops,
            "max_results": max_results,
            "return_paths": return_paths,
        }
        if predicates:
            payload["predicates"] = predicates
        if min_confidence is not None:
            payload["min_confidence"] = min_confidence

        data = self._post("/api/knowledge/traverse", payload)
        return [
            TraversalPath(
                entities=p["entities"],
                edges=p["edges"],
                hops=p["hops"],
                min_confidence=p["min_confidence"],
            )
            for p in data.get("paths", [])
        ]

    # ------------------------------------------------------------------
    # Context Slicing
    # ------------------------------------------------------------------

    def slice(self, anchor_turn_id: str, policy_ref: Optional[dict] = None) -> dict:
        """Construct a context slice around an anchor turn.

        Returns the full slice export including fingerprint,
        admissibility token, and turn list.
        """
        payload = {"anchor_turn_id": anchor_turn_id}
        if policy_ref:
            payload["policy_ref"] = policy_ref
        return self._post("/api/slice", payload)

    def batch_slice(self, anchor_turn_ids: list[str], policy_ref: Optional[dict] = None) -> dict:
        """Construct multiple context slices in batch."""
        payload = {"anchor_turn_ids": anchor_turn_ids}
        if policy_ref:
            payload["policy_ref"] = policy_ref
        return self._post("/api/slice/batch", payload)

    def verify_token(
        self,
        admissibility_token: str,
        slice_id: str,
        anchor_turn_id: str,
        policy_id: str,
        policy_params_hash: str,
        graph_snapshot_hash: str,
        schema_version: str,
    ) -> dict:
        """Verify an admissibility token without access to the HMAC secret."""
        return self._post("/api/verify_token", {
            "admissibility_token": admissibility_token,
            "slice_id": slice_id,
            "anchor_turn_id": anchor_turn_id,
            "policy_id": policy_id,
            "policy_params_hash": policy_params_hash,
            "graph_snapshot_hash": graph_snapshot_hash,
            "schema_version": schema_version,
        })

    # ------------------------------------------------------------------
    # Policy Management
    # ------------------------------------------------------------------

    def list_policies(self) -> dict:
        """List registered slice policies."""
        return self._get("/api/policies")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """Full health check with database status."""
        return self._get("/health")

    def is_healthy(self) -> bool:
        """Quick liveness check."""
        try:
            data = self._get("/health/live")
            return data.get("status") == "alive"
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def graph_d3(self, subject: Optional[str] = None) -> dict:
        """Get D3-compatible graph JSON."""
        params = {}
        if subject:
            params["subject"] = subject
        return self._get("/api/knowledge/graph", params)

    def graph_mermaid(self, subject: Optional[str] = None) -> str:
        """Get Mermaid diagram of the graph."""
        params = {}
        if subject:
            params["subject"] = subject
        return self._get_text("/api/knowledge/graph.mermaid", params)

    # ------------------------------------------------------------------
    # HTTP internals
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urlencode(params)
        req = Request(url, method="GET")
        req.add_header("Accept", "application/json")
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _get_text(self, path: str, params: Optional[dict] = None) -> str:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urlencode(params)
        req = Request(url, method="GET")
        with urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode()

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode()
        req = Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _delete(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode()
        req = Request(url, data=data, method="DELETE")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())


if __name__ == "__main__":
    kg = KGClient()
    if kg.is_healthy():
        print("Graph Kernel is healthy")
        stats = kg.stats()
        print(f"Stats: {json.dumps(stats, indent=2)}")
    else:
        print("Graph Kernel is not reachable at localhost:8001")
