//! Dynamic dispatch wrapper for `KnowledgeDb`.
//!
//! Allows runtime backend selection between PostgreSQL and SQLite
//! without making the entire service generic.

use std::sync::Arc;
use super::knowledge_db::*;

/// Dynamic dispatch wrapper for `KnowledgeDb`.
///
/// This enables runtime backend selection via `GK_BACKEND` env var
/// without requiring compile-time generics throughout the service.
pub struct DynKnowledgeDb {
    inner: Arc<dyn KnowledgeDbDyn + Send + Sync>,
}

/// Object-safe version of `KnowledgeDb` using boxed futures.
#[allow(dead_code)]
trait KnowledgeDbDyn: Send + Sync {
    fn upsert_triple_dyn(
        &self,
        subject: String,
        predicate: String,
        object: String,
        confidence: f64,
        source: String,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<UpsertResult, Box<dyn std::error::Error + Send + Sync>>> + Send + '_>>;

    fn upsert_batch_dyn(
        &self,
        triples: Vec<(String, String, String, f64, String)>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(usize, usize), Box<dyn std::error::Error + Send + Sync>>> + Send + '_>>;

    fn query_triples_dyn(
        &self,
        query: KnowledgeQuery,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(Vec<StoredTriple>, i64), Box<dyn std::error::Error + Send + Sync>>> + Send + '_>>;

    fn delete_triples_dyn(
        &self,
        id: Option<i64>,
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<DeleteResult, Box<dyn std::error::Error + Send + Sync>>> + Send + '_>>;

    fn stats_dyn(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<KnowledgeStats, Box<dyn std::error::Error + Send + Sync>>> + Send + '_>>;

    fn query_adjacent_dyn(
        &self,
        entity: String,
        direction: String,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<AdjacentTriple>, Box<dyn std::error::Error + Send + Sync>>> + Send + '_>>;

    fn is_healthy_dyn(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + '_>>;
}

/// Blanket impl for any KnowledgeDb with 'static Error
impl<T: KnowledgeDb + Send + Sync + 'static> KnowledgeDbDyn for T
where
    T::Error: 'static,
{
    fn upsert_triple_dyn(
        &self,
        subject: String,
        predicate: String,
        object: String,
        confidence: f64,
        source: String,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<UpsertResult, Box<dyn std::error::Error + Send + Sync>>> + Send + '_>> {
        Box::pin(async move {
            self.upsert_triple(&subject, &predicate, &object, confidence, &source)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
    }

    fn upsert_batch_dyn(
        &self,
        triples: Vec<(String, String, String, f64, String)>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(usize, usize), Box<dyn std::error::Error + Send + Sync>>> + Send + '_>> {
        Box::pin(async move {
            self.upsert_batch(&triples)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
    }

    fn query_triples_dyn(
        &self,
        query: KnowledgeQuery,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(Vec<StoredTriple>, i64), Box<dyn std::error::Error + Send + Sync>>> + Send + '_>> {
        Box::pin(async move {
            self.query_triples(&query)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
    }

    fn delete_triples_dyn(
        &self,
        id: Option<i64>,
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<DeleteResult, Box<dyn std::error::Error + Send + Sync>>> + Send + '_>> {
        Box::pin(async move {
            self.delete_triples(id, subject.as_deref(), predicate.as_deref(), object.as_deref())
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
    }

    fn stats_dyn(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<KnowledgeStats, Box<dyn std::error::Error + Send + Sync>>> + Send + '_>> {
        Box::pin(async move {
            self.stats()
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
    }

    fn query_adjacent_dyn(
        &self,
        entity: String,
        direction: String,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<AdjacentTriple>, Box<dyn std::error::Error + Send + Sync>>> + Send + '_>> {
        Box::pin(async move {
            self.query_adjacent(&entity, &direction)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
        })
    }

    fn is_healthy_dyn(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + '_>> {
        Box::pin(self.is_healthy())
    }
}

type DynError = Box<dyn std::error::Error + Send + Sync>;

/// No-op implementation for legacy constructors that don't provide a knowledge DB.
struct NoopKnowledgeDb;

impl KnowledgeDbDyn for NoopKnowledgeDb {
    fn upsert_triple_dyn(&self, _: String, _: String, _: String, _: f64, _: String)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<UpsertResult, Box<dyn std::error::Error + Send + Sync>>> + Send + '_>> {
        Box::pin(async { Err("No knowledge DB backend configured".into()) })
    }
    fn upsert_batch_dyn(&self, _: Vec<(String, String, String, f64, String)>)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(usize, usize), Box<dyn std::error::Error + Send + Sync>>> + Send + '_>> {
        Box::pin(async { Err("No knowledge DB backend configured".into()) })
    }
    fn query_triples_dyn(&self, _: KnowledgeQuery)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(Vec<StoredTriple>, i64), Box<dyn std::error::Error + Send + Sync>>> + Send + '_>> {
        Box::pin(async { Err("No knowledge DB backend configured".into()) })
    }
    fn delete_triples_dyn(&self, _: Option<i64>, _: Option<String>, _: Option<String>, _: Option<String>)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<DeleteResult, Box<dyn std::error::Error + Send + Sync>>> + Send + '_>> {
        Box::pin(async { Err("No knowledge DB backend configured".into()) })
    }
    fn stats_dyn(&self)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<KnowledgeStats, Box<dyn std::error::Error + Send + Sync>>> + Send + '_>> {
        Box::pin(async { Err("No knowledge DB backend configured".into()) })
    }
    fn query_adjacent_dyn(&self, _: String, _: String)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<AdjacentTriple>, Box<dyn std::error::Error + Send + Sync>>> + Send + '_>> {
        Box::pin(async { Err("No knowledge DB backend configured".into()) })
    }
    fn is_healthy_dyn(&self)
        -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + '_>> {
        Box::pin(async { false })
    }
}

impl DynKnowledgeDb {
    /// Create a no-op knowledge DB (for legacy constructors).
    pub fn noop() -> Self {
        Self {
            inner: Arc::new(NoopKnowledgeDb),
        }
    }

    /// Create a new DynKnowledgeDb from any KnowledgeDb implementation.
    pub fn new<T: KnowledgeDb + Send + Sync + 'static>(inner: T) -> Self
    where
        T::Error: 'static,
    {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// Upsert a triple.
    pub async fn upsert_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
        source: &str,
    ) -> Result<UpsertResult, DynError> {
        self.inner.upsert_triple_dyn(
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
            confidence,
            source.to_string(),
        ).await
    }

    /// Upsert a batch of triples.
    pub async fn upsert_batch(
        &self,
        triples: &[(String, String, String, f64, String)],
    ) -> Result<(usize, usize), DynError> {
        self.inner.upsert_batch_dyn(triples.to_vec()).await
    }

    /// Query triples.
    pub async fn query_triples(
        &self,
        query: &KnowledgeQuery,
    ) -> Result<(Vec<StoredTriple>, i64), DynError> {
        self.inner.query_triples_dyn(query.clone()).await
    }

    /// Delete triples.
    pub async fn delete_triples(
        &self,
        id: Option<i64>,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Result<DeleteResult, DynError> {
        self.inner.delete_triples_dyn(
            id,
            subject.map(|s| s.to_string()),
            predicate.map(|s| s.to_string()),
            object.map(|s| s.to_string()),
        ).await
    }

    /// Get stats.
    pub async fn stats(&self) -> Result<KnowledgeStats, DynError> {
        self.inner.stats_dyn().await
    }

    /// Query adjacent triples for traversal.
    pub async fn query_adjacent(
        &self,
        entity: &str,
        direction: &str,
    ) -> Result<Vec<AdjacentTriple>, DynError> {
        self.inner.query_adjacent_dyn(entity.to_string(), direction.to_string()).await
    }

    /// Health check.
    pub async fn is_healthy(&self) -> bool {
        self.inner.is_healthy_dyn().await
    }
}

impl Clone for DynKnowledgeDb {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}
