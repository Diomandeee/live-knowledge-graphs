//! In-memory graph store for testing.

use std::collections::{BTreeMap, BTreeSet};

use crate::types::{TurnId, TurnSnapshot, Edge};
use super::GraphStore;

/// Error type for in-memory store.
#[derive(Debug, Clone, thiserror::Error)]
pub enum InMemoryError {
    /// Turn not found.
    #[error("Turn not found: {0}")]
    TurnNotFound(TurnId),
}

/// In-memory graph store for testing.
///
/// Uses BTreeMap/BTreeSet for deterministic iteration order.
#[derive(Debug, Clone, Default)]
pub struct InMemoryGraphStore {
    /// Turns by ID.
    turns: BTreeMap<TurnId, TurnSnapshot>,
    /// Parent -> Children mapping.
    children: BTreeMap<TurnId, BTreeSet<TurnId>>,
    /// Child -> Parents mapping.
    parents: BTreeMap<TurnId, BTreeSet<TurnId>>,
    /// All edges.
    edges: Vec<Edge>,
}

impl InMemoryGraphStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a turn to the store.
    pub fn add_turn(&mut self, turn: TurnSnapshot) {
        self.turns.insert(turn.id, turn);
    }

    /// Add an edge to the store.
    pub fn add_edge(&mut self, edge: Edge) {
        // Update parent -> child mapping
        self.children
            .entry(edge.parent)
            .or_default()
            .insert(edge.child);
        
        // Update child -> parent mapping
        self.parents
            .entry(edge.child)
            .or_default()
            .insert(edge.parent);
        
        self.edges.push(edge);
    }

    /// Get all turns.
    pub fn all_turns(&self) -> Vec<&TurnSnapshot> {
        self.turns.values().collect()
    }

    /// Get number of turns.
    pub fn num_turns(&self) -> usize {
        self.turns.len()
    }

    /// Get number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get all edges.
    pub fn all_edges(&self) -> &[Edge] {
        &self.edges
    }
}

impl GraphStore for InMemoryGraphStore {
    type Error = InMemoryError;

    async fn get_turn(&self, id: &TurnId) -> Result<Option<TurnSnapshot>, Self::Error> {
        Ok(self.turns.get(id).cloned())
    }

    async fn get_turns(&self, ids: &[TurnId]) -> Result<Vec<TurnSnapshot>, Self::Error> {
        Ok(ids.iter()
            .filter_map(|id| self.turns.get(id).cloned())
            .collect())
    }

    async fn get_parents(&self, id: &TurnId) -> Result<Vec<TurnId>, Self::Error> {
        Ok(self.parents
            .get(id)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default())
    }

    async fn get_children(&self, id: &TurnId) -> Result<Vec<TurnId>, Self::Error> {
        Ok(self.children
            .get(id)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default())
    }

    async fn get_siblings(&self, id: &TurnId, limit: usize) -> Result<Vec<TurnId>, Self::Error> {
        // Get parents of this turn
        let parents: Vec<TurnId> = self.parents
            .get(id)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default();
        
        let mut siblings: BTreeSet<TurnId> = BTreeSet::new();
        
        // For each parent, get their children (our siblings)
        for parent_id in parents {
            if let Some(children) = self.children.get(&parent_id) {
                for child_id in children {
                    if child_id != id {
                        siblings.insert(*child_id);
                    }
                }
            }
        }
        
        // Sort by salience (desc) then by TurnId for determinism
        let mut sibling_list: Vec<_> = siblings.into_iter()
            .filter_map(|sid| self.turns.get(&sid).map(|t| (sid, t.salience)))
            .collect();
        
        sibling_list.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        
        Ok(sibling_list.into_iter()
            .take(limit)
            .map(|(id, _)| id)
            .collect())
    }

    async fn get_edges(&self, turn_ids: &[TurnId]) -> Result<Vec<Edge>, Self::Error> {
        let id_set: BTreeSet<_> = turn_ids.iter().copied().collect();
        
        let mut result: Vec<Edge> = self.edges.iter()
            .filter(|e| id_set.contains(&e.parent) && id_set.contains(&e.child))
            .cloned()
            .collect();
        
        // Sort for determinism
        result.sort();
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Role, Phase, EdgeType};
    use uuid::Uuid;

    fn make_turn(id: u128, salience: f32) -> TurnSnapshot {
        TurnSnapshot::new(
            TurnId::new(Uuid::from_u128(id)),
            "session_1".to_string(),
            Role::User,
            Phase::Consolidation,
            salience,
            1,
            0,
            0.5,
            0.5,
            1.0,
            1000,
        )
    }

    #[tokio::test]
    async fn test_add_and_get_turn() {
        let mut store = InMemoryGraphStore::new();
        let turn = make_turn(1, 0.5);
        let id = turn.id;
        
        store.add_turn(turn);
        
        let retrieved = store.get_turn(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, id);
    }

    #[tokio::test]
    async fn test_parents_and_children() {
        let mut store = InMemoryGraphStore::new();
        
        let t1 = make_turn(1, 0.5);
        let t2 = make_turn(2, 0.5);
        let t3 = make_turn(3, 0.5);
        
        let id1 = t1.id;
        let id2 = t2.id;
        let id3 = t3.id;
        
        store.add_turn(t1);
        store.add_turn(t2);
        store.add_turn(t3);
        
        store.add_edge(Edge::new(id1, id2, EdgeType::Reply));
        store.add_edge(Edge::new(id2, id3, EdgeType::Reply));
        
        // Test children
        let children_of_1 = store.get_children(&id1).await.unwrap();
        assert_eq!(children_of_1.len(), 1);
        assert_eq!(children_of_1[0], id2);
        
        // Test parents
        let parents_of_3 = store.get_parents(&id3).await.unwrap();
        assert_eq!(parents_of_3.len(), 1);
        assert_eq!(parents_of_3[0], id2);
    }

    #[tokio::test]
    async fn test_siblings() {
        let mut store = InMemoryGraphStore::new();
        
        let parent = make_turn(1, 0.5);
        let child1 = make_turn(2, 0.8); // Higher salience
        let child2 = make_turn(3, 0.3); // Lower salience
        let child3 = make_turn(4, 0.5);
        
        let parent_id = parent.id;
        let id1 = child1.id;
        let id2 = child2.id;
        let id3 = child3.id;
        
        store.add_turn(parent);
        store.add_turn(child1);
        store.add_turn(child2);
        store.add_turn(child3);
        
        store.add_edge(Edge::new(parent_id, id1, EdgeType::Reply));
        store.add_edge(Edge::new(parent_id, id2, EdgeType::Reply));
        store.add_edge(Edge::new(parent_id, id3, EdgeType::Reply));
        
        // Get siblings of child1
        let siblings = store.get_siblings(&id1, 10).await.unwrap();
        
        // Should be sorted by salience desc
        assert_eq!(siblings.len(), 2);
        assert_eq!(siblings[0], id3); // 0.5 salience (higher than 0.3)
        assert_eq!(siblings[1], id2); // 0.3 salience
    }

    #[tokio::test]
    async fn test_get_edges() {
        let mut store = InMemoryGraphStore::new();
        
        let t1 = make_turn(1, 0.5);
        let t2 = make_turn(2, 0.5);
        let t3 = make_turn(3, 0.5);
        
        let id1 = t1.id;
        let id2 = t2.id;
        let id3 = t3.id;
        
        store.add_turn(t1);
        store.add_turn(t2);
        store.add_turn(t3);
        
        store.add_edge(Edge::new(id1, id2, EdgeType::Reply));
        store.add_edge(Edge::new(id2, id3, EdgeType::Reply));
        
        // Get edges for subset
        let edges = store.get_edges(&[id1, id2]).await.unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].parent, id1);
        assert_eq!(edges[0].child, id2);
    }
}

