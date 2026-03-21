//! Abstraction over knowledge graph database operations.
//!
//! This trait provides a unified interface for knowledge graph CRUD operations
//! across different database backends (PostgreSQL, SQLite). Unlike the `GraphStore`
//! trait which handles turn/edge DAG operations, `KnowledgeDb` handles the
//! subject-predicate-object knowledge triple store.

use serde::{Deserialize, Serialize};
use std::future::Future;

/// A stored knowledge triple with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTriple {
    pub id: i64,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub source: String,
    pub created_at: String,
}

/// Result of an insert/upsert operation.
#[derive(Debug, Clone)]
pub struct UpsertResult {
    /// True if the row was newly inserted (not an update).
    pub inserted: bool,
}

/// Result of a delete operation.
#[derive(Debug, Clone)]
pub struct DeleteResult {
    pub rows_affected: u64,
}

/// Query parameters for knowledge search.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeQuery {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub min_confidence: Option<f64>,
    pub limit: i64,
}

/// Aggregate statistics for the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeStats {
    pub total_triples: i64,
    pub unique_subjects: i64,
    pub unique_predicates: i64,
    pub top_predicates: Vec<(String, i64)>,
}

/// Result of an adjacent triples query (for traversal).
#[derive(Debug, Clone)]
pub struct AdjacentTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
}

/// Trait for knowledge graph database operations.
///
/// Implemented by both PostgreSQL and SQLite backends.
/// All SQL dialect differences are encapsulated here.
pub trait KnowledgeDb: Send + Sync {
    /// Error type for database operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Insert or update a knowledge triple. Returns whether it was inserted (vs updated).
    fn upsert_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
        source: &str,
    ) -> impl Future<Output = Result<UpsertResult, Self::Error>> + Send;

    /// Insert or update a batch of triples in a transaction.
    fn upsert_batch(
        &self,
        triples: &[(String, String, String, f64, String)],
    ) -> impl Future<Output = Result<(usize, usize), Self::Error>> + Send;

    /// Query knowledge triples with optional filters.
    fn query_triples(
        &self,
        query: &KnowledgeQuery,
    ) -> impl Future<Output = Result<(Vec<StoredTriple>, i64), Self::Error>> + Send;

    /// Delete triples matching the given filters. Returns rows affected.
    fn delete_triples(
        &self,
        id: Option<i64>,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> impl Future<Output = Result<DeleteResult, Self::Error>> + Send;

    /// Get aggregate statistics.
    fn stats(&self) -> impl Future<Output = Result<KnowledgeStats, Self::Error>> + Send;

    /// Query adjacent triples for graph traversal.
    fn query_adjacent(
        &self,
        entity: &str,
        direction: &str,
    ) -> impl Future<Output = Result<Vec<AdjacentTriple>, Self::Error>> + Send;

    /// Check if the database is reachable.
    fn is_healthy(&self) -> impl Future<Output = bool> + Send;
}
