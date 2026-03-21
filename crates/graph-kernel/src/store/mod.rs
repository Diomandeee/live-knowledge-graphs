//! Graph storage backends.

pub mod memory;

#[cfg(feature = "postgres")]
pub mod postgres;

#[cfg(feature = "sqlite")]
pub mod sqlite;

/// Knowledge graph database abstraction (for service layer).
#[cfg(any(feature = "postgres", feature = "sqlite"))]
pub mod knowledge_db;

#[cfg(any(feature = "postgres", feature = "sqlite"))]
pub mod knowledge_db_dyn;

#[cfg(feature = "postgres")]
pub mod knowledge_db_pg;

#[cfg(feature = "sqlite")]
pub mod knowledge_db_sqlite;

use crate::types::{TurnId, TurnSnapshot, Edge};

/// Trait for graph storage backends.
///
/// Implementations must guarantee deterministic ordering of results.
/// All methods are async to support async database access.
///
/// Uses native async fn in traits (Rust 1.75+), no `async-trait` crate needed.
pub trait GraphStore: Send + Sync {
    /// Error type for store operations.
    type Error: std::error::Error + Send + Sync;

    /// Fetch a turn by ID.
    fn get_turn(&self, id: &TurnId) -> impl std::future::Future<Output = Result<Option<TurnSnapshot>, Self::Error>> + Send;

    /// Fetch multiple turns by ID.
    fn get_turns(&self, ids: &[TurnId]) -> impl std::future::Future<Output = Result<Vec<TurnSnapshot>, Self::Error>> + Send;

    /// Fetch parent turn IDs (ordered by TurnId for determinism).
    fn get_parents(&self, id: &TurnId) -> impl std::future::Future<Output = Result<Vec<TurnId>, Self::Error>> + Send;

    /// Fetch child turn IDs (ordered by TurnId for determinism).
    fn get_children(&self, id: &TurnId) -> impl std::future::Future<Output = Result<Vec<TurnId>, Self::Error>> + Send;

    /// Fetch sibling turn IDs (same parent, ordered by salience desc then TurnId).
    fn get_siblings(&self, id: &TurnId, limit: usize) -> impl std::future::Future<Output = Result<Vec<TurnId>, Self::Error>> + Send;

    /// Fetch edges between a set of turns.
    fn get_edges(&self, turn_ids: &[TurnId]) -> impl std::future::Future<Output = Result<Vec<Edge>, Self::Error>> + Send;
}

pub use memory::InMemoryGraphStore;

#[cfg(feature = "postgres")]
pub use postgres::PostgresGraphStore;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteGraphStore;

#[cfg(any(feature = "postgres", feature = "sqlite"))]
pub use knowledge_db::KnowledgeDb;

#[cfg(any(feature = "postgres", feature = "sqlite"))]
pub use knowledge_db_dyn::DynKnowledgeDb;

#[cfg(feature = "postgres")]
pub use knowledge_db_pg::PgKnowledgeDb;

#[cfg(feature = "sqlite")]
pub use knowledge_db_sqlite::SqliteKnowledgeDb;

