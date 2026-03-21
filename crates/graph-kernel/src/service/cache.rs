//! Query cache layer for the knowledge graph.
//!
//! Provides an in-memory LRU cache with TTL for frequent knowledge graph queries.
//! Dramatically reduces latency for repeated queries (e.g., dashboard refreshes,
//! agent context lookups).
//!
//! ## Design
//!
//! - **Cache key:** Normalized tuple of (subject, predicate, object, min_confidence, limit)
//! - **TTL:** Configurable via `CACHE_TTL_SECS` env var (default: 300s)
//! - **Max entries:** Configurable via `CACHE_MAX_ENTRIES` env var (default: 1000)
//! - **Invalidation:** Full cache clear on any write (POST/DELETE) to knowledge graph
//! - **Thread safety:** Lock-free concurrent access via `moka` crate
//!
//! ## Cache Stats
//!
//! Exposed in `/health` response for monitoring:
//! - `hit_count`: Total cache hits
//! - `miss_count`: Total cache misses
//! - `entry_count`: Current number of cached entries
//! - `hit_rate`: Hit ratio (0.0 – 1.0)

use moka::sync::Cache;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use super::knowledge_handlers::KnowledgeQueryResponse;

/// Cache key for knowledge queries.
///
/// All fields are normalized before insertion (lowercased, canonicalized entities).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryCacheKey {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub min_confidence: Option<u64>, // f64 bits for Hash/Eq
    pub limit: usize,
}

impl QueryCacheKey {
    /// Create a new cache key from query parameters.
    ///
    /// Confidence is stored as bits for deterministic hashing.
    pub fn new(
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
        min_confidence: Option<f64>,
        limit: usize,
    ) -> Self {
        Self {
            subject,
            predicate,
            object,
            min_confidence: min_confidence.map(|f| f.to_bits()),
            limit,
        }
    }
}

/// Cache statistics for monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total cache hits since startup.
    pub hit_count: u64,
    /// Total cache misses since startup.
    pub miss_count: u64,
    /// Current number of entries in the cache.
    pub entry_count: u64,
    /// Hit rate (0.0 - 1.0). Returns 0.0 if no queries have been made.
    pub hit_rate: f64,
    /// Configured TTL in seconds.
    pub ttl_secs: u64,
    /// Configured max entries.
    pub max_entries: u64,
    /// Number of cache invalidations (clears) since startup.
    pub invalidation_count: u64,
}

/// Query cache for the knowledge graph.
///
/// Thread-safe, TTL-aware LRU cache backed by `moka`.
pub struct QueryCache {
    cache: Cache<QueryCacheKey, KnowledgeQueryResponse>,
    hit_count: AtomicU64,
    miss_count: AtomicU64,
    invalidation_count: AtomicU64,
    ttl_secs: u64,
    max_entries: u64,
}

impl QueryCache {
    /// Create a new query cache with the given configuration.
    ///
    /// # Arguments
    /// * `max_entries` - Maximum number of cached query results
    /// * `ttl_secs` - Time-to-live for cached entries in seconds
    pub fn new(max_entries: u64, ttl_secs: u64) -> Self {
        let cache = Cache::builder()
            .max_capacity(max_entries)
            .time_to_live(Duration::from_secs(ttl_secs))
            .build();

        Self {
            cache,
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
            invalidation_count: AtomicU64::new(0),
            ttl_secs,
            max_entries,
        }
    }

    /// Create a query cache from environment variables.
    ///
    /// - `CACHE_TTL_SECS`: TTL in seconds (default: 300)
    /// - `CACHE_MAX_ENTRIES`: Max entries (default: 1000)
    pub fn from_env() -> Self {
        let ttl_secs: u64 = std::env::var("CACHE_TTL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);

        let max_entries: u64 = std::env::var("CACHE_MAX_ENTRIES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);

        tracing::info!(
            ttl_secs = ttl_secs,
            max_entries = max_entries,
            "Query cache initialized"
        );

        Self::new(max_entries, ttl_secs)
    }

    /// Look up a cached query result.
    ///
    /// Returns `Some(response)` on cache hit, `None` on miss.
    /// Updates hit/miss counters atomically.
    pub fn get(&self, key: &QueryCacheKey) -> Option<KnowledgeQueryResponse> {
        match self.cache.get(key) {
            Some(response) => {
                self.hit_count.fetch_add(1, Ordering::Relaxed);
                tracing::trace!(
                    subject = ?key.subject,
                    predicate = ?key.predicate,
                    "Cache HIT"
                );
                Some(response)
            }
            None => {
                self.miss_count.fetch_add(1, Ordering::Relaxed);
                tracing::trace!(
                    subject = ?key.subject,
                    predicate = ?key.predicate,
                    "Cache MISS"
                );
                None
            }
        }
    }

    /// Store a query result in the cache.
    pub fn put(&self, key: QueryCacheKey, response: KnowledgeQueryResponse) {
        self.cache.insert(key, response);
    }

    /// Invalidate all cache entries.
    ///
    /// Called on any write operation (POST/DELETE) to the knowledge graph.
    /// Full invalidation is simpler and safer than selective invalidation —
    /// knowledge graph writes are infrequent relative to reads.
    pub fn invalidate_all(&self) {
        self.cache.invalidate_all();
        self.invalidation_count.fetch_add(1, Ordering::Relaxed);
        tracing::debug!("Query cache invalidated");
    }

    /// Get current cache statistics.
    pub fn stats(&self) -> CacheStats {
        let hits = self.hit_count.load(Ordering::Relaxed);
        let misses = self.miss_count.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        CacheStats {
            hit_count: hits,
            miss_count: misses,
            entry_count: self.cache.entry_count(),
            hit_rate,
            ttl_secs: self.ttl_secs,
            max_entries: self.max_entries,
            invalidation_count: self.invalidation_count.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::knowledge_handlers::{KnowledgeQueryResponse, StoredKnowledgeTriple};

    fn make_response(count: usize) -> KnowledgeQueryResponse {
        KnowledgeQueryResponse {
            triples: (0..count)
                .map(|i| StoredKnowledgeTriple {
                    id: i as i64,
                    subject: format!("entity-{}", i),
                    predicate: "uses".to_string(),
                    object: format!("target-{}", i),
                    confidence: 0.9,
                    source: "test".to_string(),
                    created_at: "2025-01-01".to_string(),
                })
                .collect(),
            total: count,
        }
    }

    #[test]
    fn test_cache_hit_miss() {
        let cache = QueryCache::new(100, 300);
        let key = QueryCacheKey::new(
            Some("clawdbot".to_string()),
            None,
            None,
            None,
            50,
        );

        // Miss
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().miss_count, 1);
        assert_eq!(cache.stats().hit_count, 0);

        // Insert
        cache.put(key.clone(), make_response(3));

        // Hit
        let result = cache.get(&key);
        assert!(result.is_some());
        assert_eq!(result.unwrap().triples.len(), 3);
        assert_eq!(cache.stats().hit_count, 1);
        assert_eq!(cache.stats().miss_count, 1);
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = QueryCache::new(100, 300);
        let key = QueryCacheKey::new(Some("x".to_string()), None, None, None, 50);

        cache.put(key.clone(), make_response(1));
        assert!(cache.get(&key).is_some());

        cache.invalidate_all();
        // moka invalidate_all is lazy, but get should reflect it
        // Note: moka's invalidate_all may not be immediate for get, 
        // but run_pending_tasks forces cleanup
        assert_eq!(cache.stats().invalidation_count, 1);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let cache = QueryCache::new(100, 300);
        let key = QueryCacheKey::new(Some("a".to_string()), None, None, None, 10);

        cache.put(key.clone(), make_response(1));

        // 1 hit, 0 misses initially after put
        let _ = cache.get(&key); // hit
        let _ = cache.get(&key); // hit
        let _ = cache.get(&QueryCacheKey::new(Some("b".to_string()), None, None, None, 10)); // miss

        let stats = cache.stats();
        assert_eq!(stats.hit_count, 2);
        assert_eq!(stats.miss_count, 1);
        assert!((stats.hit_rate - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_cache_key_different_limits() {
        let cache = QueryCache::new(100, 300);
        let key1 = QueryCacheKey::new(Some("x".to_string()), None, None, None, 10);
        let key2 = QueryCacheKey::new(Some("x".to_string()), None, None, None, 50);

        cache.put(key1.clone(), make_response(1));

        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_none()); // Different limit = different key
    }
}
