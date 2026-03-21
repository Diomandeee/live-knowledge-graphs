//! Edge types for the graph kernel.

use serde::{Deserialize, Serialize};
use super::turn::TurnId;

/// Type of edge in the conversation DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Direct reply/continuation.
    Reply,
    /// Branch/fork in conversation.
    Branch,
    /// Reference to earlier turn.
    Reference,
    /// Default/unspecified.
    Default,
}

impl EdgeType {
    /// Parse edge type from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "reply" => Some(Self::Reply),
            "branch" => Some(Self::Branch),
            "reference" => Some(Self::Reference),
            "default" | "" => Some(Self::Default),
            _ => None,
        }
    }
}

impl Default for EdgeType {
    fn default() -> Self {
        Self::Default
    }
}

impl std::fmt::Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reply => write!(f, "reply"),
            Self::Branch => write!(f, "branch"),
            Self::Reference => write!(f, "reference"),
            Self::Default => write!(f, "default"),
        }
    }
}

/// Edge in the conversation DAG.
///
/// Represents a directed connection from parent to child.
/// Implements `Ord` for deterministic ordering: (parent, child, edge_type).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Edge {
    /// Parent turn (source).
    pub parent: TurnId,
    /// Child turn (target).
    pub child: TurnId,
    /// Type of edge.
    pub edge_type: EdgeType,
}

impl Edge {
    /// Create a new edge.
    pub fn new(parent: TurnId, child: TurnId, edge_type: EdgeType) -> Self {
        Self {
            parent,
            child,
            edge_type,
        }
    }

    /// Create a default edge (reply type).
    pub fn reply(parent: TurnId, child: TurnId) -> Self {
        Self::new(parent, child, EdgeType::Reply)
    }
}

// Canonical ordering: parent, then child, then edge_type
impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.parent.cmp(&other.parent) {
            std::cmp::Ordering::Equal => match self.child.cmp(&other.child) {
                std::cmp::Ordering::Equal => self.edge_type.cmp(&other.edge_type),
                ord => ord,
            },
            ord => ord,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_edge_ordering() {
        let id1 = TurnId::new(Uuid::from_u128(1));
        let id2 = TurnId::new(Uuid::from_u128(2));
        let id3 = TurnId::new(Uuid::from_u128(3));

        let e1 = Edge::new(id1, id2, EdgeType::Reply);
        let e2 = Edge::new(id1, id3, EdgeType::Reply);
        let e3 = Edge::new(id2, id3, EdgeType::Reply);

        // Same parent, different child
        assert!(e1 < e2);
        // Different parent
        assert!(e1 < e3);
        assert!(e2 < e3);
    }

    #[test]
    fn test_edge_type_ordering() {
        let id1 = TurnId::new(Uuid::from_u128(1));
        let id2 = TurnId::new(Uuid::from_u128(2));

        let e1 = Edge::new(id1, id2, EdgeType::Reply);
        let e2 = Edge::new(id1, id2, EdgeType::Branch);

        // EdgeType ordering matters when parent and child are equal
        assert!(e1 != e2);
    }
}

