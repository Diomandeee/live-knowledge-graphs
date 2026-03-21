//! Token verification modes for production deployment.
//!
//! ## Purpose
//!
//! This module provides configurable verification strategies for admissibility tokens.
//! The goal is to make verification **cheap enough that it's never bypassed**.
//!
//! ## Performance Target
//!
//! - **Cached verification**: < 5ms p99
//! - **Local verification**: < 1ms p99 (HMAC computation)
//!
//! ## Verification Modes
//!
//! | Mode | Use Case | Performance | Security |
//! |------|----------|-------------|----------|
//! | `LocalSecret` | Single-node deployment | ~100μs | Full HMAC verification |
//! | `Cached` | High-throughput services | ~10μs (cache hit) | Full HMAC + LRU cache |
//!
//! ## Cache Key Design
//!
//! The cache key is derived from all fields that affect token validity:
//! - `slice_id`
//! - `anchor_turn_id`
//! - `policy_id`
//! - `policy_params_hash`
//! - `graph_snapshot_hash`
//! - `schema_version`
//! - `admissibility_token`
//!
//! This ensures that any parameter change results in a cache miss and full verification.

use std::sync::Arc;
use parking_lot::RwLock;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::hash::{Hash, Hasher};
use xxhash_rust::xxh64::Xxh64;

use super::slice::{SliceFingerprint, GraphSnapshotHash, AdmissibilityToken};
use super::turn::TurnId;

/// Configuration for the token verification cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,
    /// Whether to enable the cache.
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            enabled: true,
        }
    }
}

/// Verification mode for admissibility tokens.
#[derive(Debug, Clone)]
pub enum VerificationMode {
    /// Verify using a local HMAC secret (no network calls).
    ///
    /// Best for: Single-node deployments, testing, low-latency requirements.
    LocalSecret {
        /// The HMAC secret shared with the kernel.
        secret: Vec<u8>,
    },

    /// Verify with LRU caching (reduces repeated verification overhead).
    ///
    /// Best for: High-throughput services where the same slices are verified repeatedly.
    Cached {
        /// The HMAC secret shared with the kernel.
        secret: Vec<u8>,
        /// Cache configuration.
        config: CacheConfig,
    },
}

impl VerificationMode {
    /// Create a local secret verification mode.
    pub fn local_secret(secret: Vec<u8>) -> Self {
        Self::LocalSecret { secret }
    }

    /// Create a cached verification mode with default configuration.
    pub fn cached(secret: Vec<u8>) -> Self {
        Self::Cached {
            secret,
            config: CacheConfig::default(),
        }
    }

    /// Create a cached verification mode with custom configuration.
    pub fn cached_with_config(secret: Vec<u8>, config: CacheConfig) -> Self {
        Self::Cached { secret, config }
    }
}

/// Cache key for token verification.
///
/// Computed from all fields that affect token validity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VerificationCacheKey(u64);

impl VerificationCacheKey {
    /// Compute the cache key from verification parameters.
    fn compute(
        slice_id: &SliceFingerprint,
        anchor_turn_id: &TurnId,
        policy_id: &str,
        policy_params_hash: &str,
        graph_snapshot_hash: &GraphSnapshotHash,
        schema_version: &str,
        token: &AdmissibilityToken,
    ) -> Self {
        let mut hasher = Xxh64::new(0);

        hasher.write(slice_id.as_str().as_bytes());
        hasher.write(anchor_turn_id.as_uuid().as_bytes());
        hasher.write(policy_id.as_bytes());
        hasher.write(policy_params_hash.as_bytes());
        hasher.write(graph_snapshot_hash.as_str().as_bytes());
        hasher.write(schema_version.as_bytes());
        hasher.write(token.as_str().as_bytes());

        Self(hasher.finish())
    }
}

/// Result of a cached verification.
#[derive(Debug, Clone, Copy)]
pub struct VerificationResult {
    /// Whether the token is valid.
    pub is_valid: bool,
    /// Whether this result came from cache.
    pub cache_hit: bool,
}

/// Token verifier with optional caching.
///
/// Thread-safe and suitable for use in async services.
///
/// # Example
///
/// ```rust,ignore
/// use cc_graph_kernel::types::verification::{TokenVerifier, VerificationMode};
///
/// // Create a cached verifier
/// let verifier = TokenVerifier::new(VerificationMode::cached(secret.to_vec()));
///
/// // Verify a slice
/// let result = verifier.verify_slice(&slice);
/// if result.is_valid {
///     println!("Token verified (cache hit: {})", result.cache_hit);
/// }
/// ```
pub struct TokenVerifier {
    mode: VerificationMode,
    cache: Option<Arc<RwLock<LruCache<VerificationCacheKey, bool>>>>,
}

impl TokenVerifier {
    /// Create a new token verifier with the specified mode.
    pub fn new(mode: VerificationMode) -> Self {
        let cache = match &mode {
            VerificationMode::Cached { config, .. } if config.enabled => {
                let size = NonZeroUsize::new(config.max_entries).unwrap_or(NonZeroUsize::new(1000).unwrap());
                Some(Arc::new(RwLock::new(LruCache::new(size))))
            }
            _ => None,
        };

        Self { mode, cache }
    }

    /// Get the HMAC secret from the verification mode.
    fn secret(&self) -> &[u8] {
        match &self.mode {
            VerificationMode::LocalSecret { secret } => secret,
            VerificationMode::Cached { secret, .. } => secret,
        }
    }

    /// Verify an admissibility token.
    ///
    /// This is the low-level verification method that takes all parameters explicitly.
    ///
    /// # Arguments
    /// * All fields that were used to issue the token
    ///
    /// # Returns
    /// `VerificationResult` with validity and cache hit status
    pub fn verify_token(
        &self,
        token: &AdmissibilityToken,
        slice_id: &SliceFingerprint,
        anchor_turn_id: &TurnId,
        policy_id: &str,
        policy_params_hash: &str,
        graph_snapshot_hash: &GraphSnapshotHash,
        schema_version: &str,
    ) -> VerificationResult {
        // Compute cache key
        let cache_key = VerificationCacheKey::compute(
            slice_id,
            anchor_turn_id,
            policy_id,
            policy_params_hash,
            graph_snapshot_hash,
            schema_version,
            token,
        );

        // Check cache first (if enabled)
        if let Some(cache) = &self.cache {
            // Try read lock first (non-blocking for other readers)
            if let Some(&is_valid) = cache.read().peek(&cache_key) {
                return VerificationResult {
                    is_valid,
                    cache_hit: true,
                };
            }
        }

        // Cache miss - perform full HMAC verification
        let is_valid = token.verify_hmac(
            self.secret(),
            slice_id,
            anchor_turn_id,
            policy_id,
            policy_params_hash,
            graph_snapshot_hash,
            schema_version,
        );

        // Update cache (if enabled)
        if let Some(cache) = &self.cache {
            cache.write().put(cache_key, is_valid);
        }

        VerificationResult {
            is_valid,
            cache_hit: false,
        }
    }

    /// Verify a `SliceExport` using its embedded token.
    ///
    /// This is the high-level verification method for typical use cases.
    pub fn verify_slice(&self, slice: &super::slice::SliceExport) -> VerificationResult {
        self.verify_token(
            &slice.admissibility_token,
            &slice.slice_id,
            &slice.anchor_turn_id,
            &slice.policy_id,
            &slice.policy_params_hash,
            &slice.graph_snapshot_hash,
            &slice.schema_version,
        )
    }

    /// Get cache statistics.
    ///
    /// Returns `None` if caching is disabled.
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(|cache| {
            let cache = cache.read();
            CacheStats {
                len: cache.len(),
                cap: cache.cap().get(),
            }
        })
    }

    /// Clear the verification cache.
    ///
    /// Does nothing if caching is disabled.
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.write().clear();
        }
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Current number of entries in the cache.
    pub len: usize,
    /// Maximum capacity of the cache.
    pub cap: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TurnSnapshot, Role, Phase, SliceExport};
    use uuid::Uuid;

    fn make_turn(id: u128) -> TurnSnapshot {
        TurnSnapshot::new(
            TurnId::new(Uuid::from_u128(id)),
            "session_test".to_string(),
            Role::User,
            Phase::Synthesis,
            0.8,
            1,
            0,
            0.5,
            0.5,
            1.0,
            1000,
        )
    }

    fn make_slice(secret: &[u8]) -> SliceExport {
        let anchor = TurnId::new(Uuid::from_u128(1));
        let turns = vec![make_turn(1)];
        let snapshot = GraphSnapshotHash::new("test_snapshot".to_string());

        SliceExport::new_with_secret(
            secret,
            anchor,
            turns,
            vec![],
            "test_policy".to_string(),
            "params_hash".to_string(),
            snapshot,
        )
    }

    #[test]
    fn test_local_verification() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let verifier = TokenVerifier::new(VerificationMode::local_secret(secret.to_vec()));
        let slice = make_slice(secret);

        let result = verifier.verify_slice(&slice);
        assert!(result.is_valid);
        assert!(!result.cache_hit); // No cache in local mode
    }

    #[test]
    fn test_cached_verification_miss() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let verifier = TokenVerifier::new(VerificationMode::cached(secret.to_vec()));
        let slice = make_slice(secret);

        // First verification should be a cache miss
        let result = verifier.verify_slice(&slice);
        assert!(result.is_valid);
        assert!(!result.cache_hit);

        // Check cache stats
        let stats = verifier.cache_stats().unwrap();
        assert_eq!(stats.len, 1);
    }

    #[test]
    fn test_cached_verification_hit() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let verifier = TokenVerifier::new(VerificationMode::cached(secret.to_vec()));
        let slice = make_slice(secret);

        // First verification - cache miss
        let result1 = verifier.verify_slice(&slice);
        assert!(result1.is_valid);
        assert!(!result1.cache_hit);

        // Second verification - cache hit
        let result2 = verifier.verify_slice(&slice);
        assert!(result2.is_valid);
        assert!(result2.cache_hit);
    }

    #[test]
    fn test_verification_failure_wrong_secret() {
        let correct_secret = b"correct_secret_32_bytes_minimum!";
        let wrong_secret = b"wrong_secret_totally_different!!";

        let slice = make_slice(correct_secret);
        let verifier = TokenVerifier::new(VerificationMode::local_secret(wrong_secret.to_vec()));

        let result = verifier.verify_slice(&slice);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_cache_clear() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let verifier = TokenVerifier::new(VerificationMode::cached(secret.to_vec()));
        let slice = make_slice(secret);

        // Populate cache
        verifier.verify_slice(&slice);
        assert_eq!(verifier.cache_stats().unwrap().len, 1);

        // Clear cache
        verifier.clear_cache();
        assert_eq!(verifier.cache_stats().unwrap().len, 0);

        // Next verification should be a cache miss again
        let result = verifier.verify_slice(&slice);
        assert!(result.is_valid);
        assert!(!result.cache_hit);
    }

    #[test]
    fn test_custom_cache_config() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let config = CacheConfig {
            max_entries: 5,
            enabled: true,
        };
        let verifier = TokenVerifier::new(VerificationMode::cached_with_config(
            secret.to_vec(),
            config,
        ));

        let stats = verifier.cache_stats().unwrap();
        assert_eq!(stats.cap, 5);
    }

    #[test]
    fn test_cache_disabled() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let config = CacheConfig {
            max_entries: 100,
            enabled: false,
        };
        let verifier = TokenVerifier::new(VerificationMode::cached_with_config(
            secret.to_vec(),
            config,
        ));

        // Cache should be None
        assert!(verifier.cache_stats().is_none());

        let slice = make_slice(secret);
        let result = verifier.verify_slice(&slice);
        assert!(result.is_valid);
        assert!(!result.cache_hit); // No cache
    }

    #[test]
    fn test_cache_key_uniqueness() {
        // Different slice parameters should produce different cache keys
        let slice_id_1 = SliceFingerprint::new("slice_1".to_string());
        let slice_id_2 = SliceFingerprint::new("slice_2".to_string());
        let anchor = TurnId::new(Uuid::from_u128(1));
        let snapshot = GraphSnapshotHash::new("test".to_string());
        let token = AdmissibilityToken::from_string("00000000000000000000000000000000".to_string());

        let key1 = VerificationCacheKey::compute(
            &slice_id_1,
            &anchor,
            "policy",
            "params",
            &snapshot,
            "1.0.0",
            &token,
        );

        let key2 = VerificationCacheKey::compute(
            &slice_id_2, // Different slice ID
            &anchor,
            "policy",
            "params",
            &snapshot,
            "1.0.0",
            &token,
        );

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_invalid_token_cached() {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let verifier = TokenVerifier::new(VerificationMode::cached(secret.to_vec()));
        let mut slice = make_slice(secret);

        // Tamper with the token
        slice.admissibility_token = AdmissibilityToken::from_string(
            "00000000000000000000000000000000".to_string()
        );

        // First verification - cache miss, invalid
        let result1 = verifier.verify_slice(&slice);
        assert!(!result1.is_valid);
        assert!(!result1.cache_hit);

        // Second verification - cache hit, still invalid
        let result2 = verifier.verify_slice(&slice);
        assert!(!result2.is_valid);
        assert!(result2.cache_hit); // Invalid results are also cached
    }
}
