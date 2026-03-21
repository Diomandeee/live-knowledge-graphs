//! Service state management.
//!
//! Contains the PolicyRegistry and shared service state.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

use crate::canonical::canonical_hash_hex;
use crate::policy::SlicePolicyV1;
use crate::store::GraphStore;

#[cfg(any(feature = "postgres", feature = "sqlite"))]
use crate::store::DynKnowledgeDb;
use super::cache::QueryCache;

/// Reference to a registered policy by hash.
///
/// This enables hash-stable policy references across requests.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PolicyRef {
    /// Policy type identifier (e.g., "slice_policy_v1")
    pub policy_id: String,
    /// xxHash64 of canonical policy JSON
    pub params_hash: String,
}

impl PolicyRef {
    /// Create a policy reference from a SlicePolicyV1.
    pub fn from_policy(policy: &SlicePolicyV1) -> Self {
        Self {
            policy_id: policy.policy_id().to_string(),
            params_hash: policy.params_hash(),
        }
    }

    /// Create a reference with explicit values.
    pub fn new(policy_id: impl Into<String>, params_hash: impl Into<String>) -> Self {
        Self {
            policy_id: policy_id.into(),
            params_hash: params_hash.into(),
        }
    }
}

/// Registry of immutable policies with stable hashes.
///
/// Policies are registered once and referenced by PolicyRef.
/// The registry itself has a fingerprint that changes when policies change.
#[derive(Debug, Clone)]
pub struct PolicyRegistry {
    policies: BTreeMap<PolicyRef, SlicePolicyV1>,
    registry_fingerprint: String,
}

impl PolicyRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        let mut registry = Self {
            policies: BTreeMap::new(),
            registry_fingerprint: String::new(),
        };
        registry.update_fingerprint();
        registry
    }

    /// Create a registry with a default policy pre-registered.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(SlicePolicyV1::default());
        registry
    }

    /// Register a policy and return its reference.
    ///
    /// If the policy already exists (same hash), returns the existing reference.
    pub fn register(&mut self, policy: SlicePolicyV1) -> PolicyRef {
        let policy_ref = PolicyRef::from_policy(&policy);
        
        if !self.policies.contains_key(&policy_ref) {
            self.policies.insert(policy_ref.clone(), policy);
            self.update_fingerprint();
        }
        
        policy_ref
    }

    /// Resolve a policy reference to the actual policy.
    pub fn resolve(&self, policy_ref: &PolicyRef) -> Option<&SlicePolicyV1> {
        self.policies.get(policy_ref)
    }

    /// Get all registered policy references.
    pub fn list(&self) -> Vec<PolicyRef> {
        self.policies.keys().cloned().collect()
    }

    /// Get the registry fingerprint.
    ///
    /// This changes whenever policies are added/removed.
    pub fn fingerprint(&self) -> &str {
        &self.registry_fingerprint
    }

    /// Get the number of registered policies.
    pub fn len(&self) -> usize {
        self.policies.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.policies.is_empty()
    }

    /// Update the registry fingerprint.
    fn update_fingerprint(&mut self) {
        let refs: Vec<_> = self.policies.keys().collect();
        self.registry_fingerprint = canonical_hash_hex(&refs);
    }
}

impl Default for PolicyRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Shared service state.
///
/// Contains the graph store, policy registry, query cache, and HMAC secret for token signing.
pub struct ServiceState<S: GraphStore + Send + Sync + 'static> {
    /// The graph store for turn/edge lookups (context slicer DAG).
    pub store: Arc<S>,
    /// Registry of available policies.
    pub policy_registry: Arc<RwLock<PolicyRegistry>>,
    /// HMAC secret for signing admissibility tokens.
    hmac_secret: Arc<Vec<u8>>,
    /// Query cache for knowledge graph queries.
    pub query_cache: Arc<QueryCache>,
    /// Knowledge graph database abstraction (for SPO triple CRUD).
    /// Supports both PostgreSQL and SQLite backends via dynamic dispatch.
    #[cfg(any(feature = "postgres", feature = "sqlite"))]
    pub knowledge_db: Arc<DynKnowledgeDb>,
    /// Name of the active backend ("postgres" or "sqlite").
    pub backend_name: String,
}

impl<S: GraphStore + Send + Sync + 'static> ServiceState<S> {
    /// Create new service state with a graph store, HMAC secret, and knowledge DB.
    ///
    /// # Arguments
    /// * `store` - The graph store backend (for context slicer DAG)
    /// * `hmac_secret` - Secret key for signing admissibility tokens (32+ bytes recommended)
    /// * `knowledge_db` - Knowledge graph database (for SPO triple CRUD)
    /// * `backend_name` - Name of the active backend ("postgres" or "sqlite")
    #[cfg(any(feature = "postgres", feature = "sqlite"))]
    pub fn new_with_knowledge_db(
        store: S,
        hmac_secret: Vec<u8>,
        knowledge_db: DynKnowledgeDb,
        backend_name: impl Into<String>,
    ) -> Self {
        Self {
            store: Arc::new(store),
            policy_registry: Arc::new(RwLock::new(PolicyRegistry::with_defaults())),
            hmac_secret: Arc::new(hmac_secret),
            query_cache: Arc::new(QueryCache::from_env()),
            knowledge_db: Arc::new(knowledge_db),
            backend_name: backend_name.into(),
        }
    }

    /// Create service state with a custom policy registry and knowledge DB.
    #[cfg(any(feature = "postgres", feature = "sqlite"))]
    pub fn with_registry_and_knowledge_db(
        store: S,
        registry: PolicyRegistry,
        hmac_secret: Vec<u8>,
        knowledge_db: DynKnowledgeDb,
        backend_name: impl Into<String>,
    ) -> Self {
        Self {
            store: Arc::new(store),
            policy_registry: Arc::new(RwLock::new(registry)),
            hmac_secret: Arc::new(hmac_secret),
            query_cache: Arc::new(QueryCache::from_env()),
            knowledge_db: Arc::new(knowledge_db),
            backend_name: backend_name.into(),
        }
    }

    /// Create service state from environment variables.
    ///
    /// Reads `KERNEL_HMAC_SECRET` from environment.
    /// Falls back to a random secret if not set (development mode).
    pub fn from_env(store: S) -> Self {
        let hmac_secret = std::env::var("KERNEL_HMAC_SECRET")
            .map(|s| s.into_bytes())
            .unwrap_or_else(|_| {
                tracing::warn!(
                    "KERNEL_HMAC_SECRET not set, using development secret. \
                     Set this for production!"
                );
                b"development_only_secret_not_for_production".to_vec()
            });
        
        Self::new(store, hmac_secret)
    }

    /// Legacy constructor — creates state without knowledge_db.
    /// Knowledge graph endpoints will panic if called.
    pub fn new(store: S, hmac_secret: Vec<u8>) -> Self {
        Self {
            store: Arc::new(store),
            policy_registry: Arc::new(RwLock::new(PolicyRegistry::with_defaults())),
            hmac_secret: Arc::new(hmac_secret),
            query_cache: Arc::new(QueryCache::from_env()),
            #[cfg(any(feature = "postgres", feature = "sqlite"))]
            knowledge_db: Arc::new(DynKnowledgeDb::noop()),
            backend_name: "none".to_string(),
        }
    }

    /// Legacy constructor with registry.
    pub fn with_registry(store: S, registry: PolicyRegistry, hmac_secret: Vec<u8>) -> Self {
        Self {
            store: Arc::new(store),
            policy_registry: Arc::new(RwLock::new(registry)),
            hmac_secret: Arc::new(hmac_secret),
            query_cache: Arc::new(QueryCache::from_env()),
            #[cfg(any(feature = "postgres", feature = "sqlite"))]
            knowledge_db: Arc::new(DynKnowledgeDb::noop()),
            backend_name: "none".to_string(),
        }
    }

    /// Get the HMAC secret for signing tokens.
    ///
    /// This is kernel-internal; downstream services should not access this.
    pub(crate) fn hmac_secret(&self) -> &[u8] {
        &self.hmac_secret
    }
}

impl<S: GraphStore + Send + Sync + 'static> Clone for ServiceState<S> {
    fn clone(&self) -> Self {
        Self {
            store: Arc::clone(&self.store),
            policy_registry: Arc::clone(&self.policy_registry),
            hmac_secret: Arc::clone(&self.hmac_secret),
            query_cache: Arc::clone(&self.query_cache),
            #[cfg(any(feature = "postgres", feature = "sqlite"))]
            knowledge_db: Arc::clone(&self.knowledge_db),
            backend_name: self.backend_name.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_registry_register() {
        let mut registry = PolicyRegistry::new();
        let policy = SlicePolicyV1::default();
        
        let ref1 = registry.register(policy.clone());
        let ref2 = registry.register(policy);
        
        // Same policy should return same reference
        assert_eq!(ref1, ref2);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_policy_registry_resolve() {
        let mut registry = PolicyRegistry::new();
        let policy = SlicePolicyV1::default();
        
        let policy_ref = registry.register(policy.clone());
        let resolved = registry.resolve(&policy_ref);
        
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap().max_nodes, policy.max_nodes);
    }

    #[test]
    fn test_policy_registry_fingerprint_changes() {
        let mut registry = PolicyRegistry::new();
        let initial_fingerprint = registry.fingerprint().to_string();
        
        let policy = SlicePolicyV1::default();
        registry.register(policy);
        
        assert_ne!(registry.fingerprint(), initial_fingerprint);
    }

    #[test]
    fn test_policy_ref_from_policy() {
        let policy = SlicePolicyV1::default();
        let ref1 = PolicyRef::from_policy(&policy);
        let ref2 = PolicyRef::from_policy(&policy);
        
        assert_eq!(ref1, ref2);
        assert_eq!(ref1.policy_id, "slice_policy_v1");
    }
}

