//! Axum routes for the Graph Kernel service.

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::policy::SlicePolicyV1;
use crate::slicer::ContextSlicer;
use crate::store::PostgresGraphStore;
use crate::types::slice::SliceExport;
use crate::types::TurnId;
use crate::GRAPH_KERNEL_SCHEMA_VERSION;

use super::state::{PolicyRef, ServiceState};

/// Type alias for the service state with PostgresGraphStore.
pub type AppState = ServiceState<PostgresGraphStore>;

// ============================================================================
// Request/Response Types
// ============================================================================

/// Request to construct a context slice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceRequest {
    /// The anchor turn ID to slice around.
    pub anchor_turn_id: String,
    /// Optional policy reference. If not provided, uses default policy.
    pub policy_ref: Option<PolicyRef>,
}

/// Request to construct multiple slices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSliceRequest {
    /// List of anchor turn IDs.
    pub anchor_turn_ids: Vec<String>,
    /// Policy reference (applies to all).
    pub policy_ref: Option<PolicyRef>,
}

/// Response containing a slice export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceResponse {
    /// The constructed slice.
    pub slice: SliceExportDto,
    /// Policy used.
    pub policy_ref: PolicyRef,
}

/// Batch slice response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSliceResponse {
    /// List of constructed slices.
    pub slices: Vec<SliceExportDto>,
    /// Policy used.
    pub policy_ref: PolicyRef,
    /// Number of successful slices.
    pub success_count: usize,
    /// Errors (anchor_id -> error message).
    pub errors: Vec<SliceError>,
}

/// Slice error for a specific anchor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceError {
    pub anchor_turn_id: String,
    pub error: String,
}

/// Serializable slice export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceExportDto {
    /// Unique slice fingerprint.
    pub slice_id: String,
    /// The anchor turn this slice was built around.
    pub anchor_turn_id: String,
    /// Turn IDs in the slice (sorted).
    pub turn_ids: Vec<String>,
    /// Number of edges in the slice.
    pub edge_count: usize,
    /// Policy identifier.
    pub policy_id: String,
    /// Policy parameters hash.
    pub policy_params_hash: String,
    /// Schema version.
    pub schema_version: String,
    /// Graph snapshot hash for content immutability.
    pub graph_snapshot_hash: String,
    /// HMAC-signed admissibility token.
    pub admissibility_token: String,
}

impl From<SliceExport> for SliceExportDto {
    fn from(slice: SliceExport) -> Self {
        Self {
            slice_id: slice.slice_id.to_string(),
            anchor_turn_id: slice.anchor_turn_id.to_string(),
            turn_ids: slice.turns.iter().map(|t| t.id.to_string()).collect(),
            edge_count: slice.edges.len(),
            policy_id: slice.policy_id,
            policy_params_hash: slice.policy_params_hash,
            schema_version: slice.schema_version,
            graph_snapshot_hash: slice.graph_snapshot_hash.to_string(),
            admissibility_token: slice.admissibility_token.to_string(),
        }
    }
}

/// Request to verify an admissibility token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyTokenRequest {
    /// The admissibility token to verify.
    pub admissibility_token: String,
    /// The slice ID the token claims to authorize.
    pub slice_id: String,
    /// The anchor turn ID.
    pub anchor_turn_id: String,
    /// Policy identifier.
    pub policy_id: String,
    /// Policy parameters hash.
    pub policy_params_hash: String,
    /// Graph snapshot hash.
    pub graph_snapshot_hash: String,
    /// Schema version.
    pub schema_version: String,
}

/// Response from token verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyTokenResponse {
    /// Whether the token is valid.
    pub valid: bool,
    /// Reason if invalid.
    pub reason: Option<String>,
}

/// Request to register a new policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterPolicyRequest {
    pub policy: SlicePolicyV1,
}

/// Response containing a policy reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRefResponse {
    pub policy_ref: PolicyRef,
}

/// List of registered policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyListResponse {
    pub policies: Vec<PolicyRef>,
    pub registry_fingerprint: String,
}

/// Service health response (detailed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub schema_version: String,
    pub policy_count: usize,
    pub registry_fingerprint: String,
    /// Active database backend ("postgres" or "sqlite").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// Database connectivity status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub database: Option<DatabaseHealth>,
    /// Query cache statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<crate::service::cache::CacheStats>,
}

/// Database health information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseHealth {
    pub connected: bool,
}

/// Simple liveness response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessResponse {
    pub status: String,
}

/// Readiness response with dependency status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessResponse {
    pub ready: bool,
    pub database: bool,
    pub details: Option<String>,
}

/// Structured error response with correlation ID for tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Human-readable error message.
    pub error: String,
    /// Machine-readable error code.
    pub code: String,
    /// Correlation ID for request tracing (matches X-Cloud-Trace-Context or generated UUID).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    /// Additional error details (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl ErrorResponse {
    /// Create a new error response with code and message.
    pub fn new(code: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            code: code.into(),
            correlation_id: None,
            details: None,
        }
    }
    
    /// Add a correlation ID to the error.
    pub fn with_correlation_id(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }
    
    /// Add details to the error.
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> axum::response::Response {
        // Log the error for debugging
        tracing::warn!(
            code = %self.code,
            error = %self.error,
            correlation_id = ?self.correlation_id,
            "Request error"
        );
        (StatusCode::BAD_REQUEST, Json(self)).into_response()
    }
}

// ============================================================================
// Route Handlers
// ============================================================================

/// Construct a context slice around an anchor turn.
async fn slice_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SliceRequest>,
) -> Result<Json<SliceResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Parse anchor turn ID
    let anchor_id = TurnId::from_str(&request.anchor_turn_id).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "INVALID_TURN_ID",
                format!("Invalid anchor turn ID: {}", e),
            ).with_details(request.anchor_turn_id.clone())),
        )
    })?;

    // Resolve policy (in a block to ensure guard is dropped before await)
    let (policy, policy_ref, hmac_secret, store) = {
        let registry = state.policy_registry.read().unwrap();
        let (policy, policy_ref) = if let Some(ref pref) = request.policy_ref {
            let policy = registry.resolve(pref).ok_or_else(|| {
                (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse::new(
                        "POLICY_NOT_FOUND",
                        format!("Policy not found: {:?}", pref),
                    )),
                )
            })?;
            (policy.clone(), pref.clone())
        } else {
            let default_policy = SlicePolicyV1::default();
            let pref = PolicyRef::from_policy(&default_policy);
            (default_policy, pref)
        };
        (policy, policy_ref, state.hmac_secret().to_vec(), Arc::clone(&state.store))
    };

    // Create slicer with HMAC secret and generate verified slice bundle
    let slicer = ContextSlicer::new(store, policy, hmac_secret);
    let bundle = slicer.slice(anchor_id).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(
                "SLICE_FAILED",
                format!("Slice generation failed: {}", e),
            )),
        )
    })?;

    // Extract the verified slice for serialization
    // The bundle proves verification occurred - we serialize just the slice data
    Ok(Json(SliceResponse {
        slice: bundle.slice().clone().into(),
        policy_ref,
    }))
}

/// Construct multiple slices in batch.
async fn batch_slice_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BatchSliceRequest>,
) -> Result<Json<BatchSliceResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Resolve policy (in a block to ensure guard is dropped before await)
    let (policy, policy_ref, hmac_secret, store) = {
        let registry = state.policy_registry.read().unwrap();
        let (policy, policy_ref) = if let Some(ref pref) = request.policy_ref {
            let policy = registry.resolve(pref).ok_or_else(|| {
                (
                    StatusCode::NOT_FOUND,
                    Json(ErrorResponse::new(
                        "POLICY_NOT_FOUND",
                        format!("Policy not found: {:?}", pref),
                    )),
                )
            })?;
            (policy.clone(), pref.clone())
        } else {
            let default_policy = SlicePolicyV1::default();
            let pref = PolicyRef::from_policy(&default_policy);
            (default_policy, pref)
        };
        (policy, policy_ref, state.hmac_secret().to_vec(), Arc::clone(&state.store))
    };

    // Create slicer with HMAC secret
    let slicer = ContextSlicer::new(store, policy, hmac_secret);

    // Process each anchor
    let mut slices = Vec::new();
    let mut errors = Vec::new();

    for anchor_str in &request.anchor_turn_ids {
        match TurnId::from_str(anchor_str) {
            Ok(anchor_id) => match slicer.slice(anchor_id).await {
                Ok(bundle) => {
                    // Extract verified slice for serialization
                    slices.push(bundle.slice().clone().into());
                }
                Err(e) => errors.push(SliceError {
                    anchor_turn_id: anchor_str.clone(),
                    error: e.to_string(),
                }),
            },
            Err(e) => errors.push(SliceError {
                anchor_turn_id: anchor_str.clone(),
                error: format!("Invalid turn ID: {}", e),
            }),
        }
    }

    Ok(Json(BatchSliceResponse {
        success_count: slices.len(),
        slices,
        policy_ref,
        errors,
    }))
}

/// List registered policies.
async fn list_policies_handler(
    State(state): State<Arc<AppState>>,
) -> Json<PolicyListResponse> {
    let registry = state.policy_registry.read().unwrap();
    Json(PolicyListResponse {
        policies: registry.list(),
        registry_fingerprint: registry.fingerprint().to_string(),
    })
}

/// Register a new policy.
async fn register_policy_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RegisterPolicyRequest>,
) -> Json<PolicyRefResponse> {
    let mut registry = state.policy_registry.write().unwrap();
    let policy_ref = registry.register(request.policy);
    Json(PolicyRefResponse { policy_ref })
}

/// Health check endpoint (detailed).
///
/// Returns full service status including database health.
async fn health_handler(
    State(state): State<Arc<AppState>>,
) -> Json<HealthResponse> {
    // Get registry info first (release lock before await)
    let (policy_count, registry_fingerprint) = {
        let registry = state.policy_registry.read().unwrap();
        (registry.len(), registry.fingerprint().to_string())
    };
    
    // Check database health via knowledge_db (works with both PG and SQLite)
    let db_healthy = state.knowledge_db.is_healthy().await;
    
    // Get cache stats
    let cache_stats = state.query_cache.stats();

    Json(HealthResponse {
        status: if db_healthy { "healthy" } else { "degraded" }.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        schema_version: GRAPH_KERNEL_SCHEMA_VERSION.to_string(),
        policy_count,
        registry_fingerprint,
        backend: Some(state.backend_name.clone()),
        database: Some(DatabaseHealth {
            connected: db_healthy,
        }),
        cache: Some(cache_stats),
    })
}

/// Liveness probe endpoint.
///
/// Simple check that the service is running. Does NOT check dependencies.
/// Returns 200 if the process is alive.
async fn liveness_handler() -> Json<LivenessResponse> {
    Json(LivenessResponse {
        status: "alive".to_string(),
    })
}

/// Readiness probe endpoint.
///
/// Checks if the service is ready to accept traffic.
/// Returns 200 if database is connected, 503 otherwise.
async fn readiness_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ReadinessResponse>, (StatusCode, Json<ReadinessResponse>)> {
    let db_healthy = state.knowledge_db.is_healthy().await;
    
    if db_healthy {
        Ok(Json(ReadinessResponse {
            ready: true,
            database: true,
            details: None,
        }))
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadinessResponse {
                ready: false,
                database: false,
                details: Some("Database connection failed".to_string()),
            }),
        ))
    }
}

/// Startup probe endpoint.
///
/// Checks if the service has started up successfully.
/// Cloud Run uses this to determine when the container is ready.
async fn startup_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ReadinessResponse>, (StatusCode, Json<ReadinessResponse>)> {
    // For startup, we check database connectivity via knowledge_db
    let db_healthy = state.knowledge_db.is_healthy().await;
    
    if db_healthy {
        Ok(Json(ReadinessResponse {
            ready: true,
            database: true,
            details: Some("Service started successfully".to_string()),
        }))
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ReadinessResponse {
                ready: false,
                database: false,
                details: Some("Database not yet available".to_string()),
            }),
        ))
    }
}

/// Verify an admissibility token.
///
/// Downstream services can call this to verify a token is valid
/// without needing access to the HMAC secret.
async fn verify_token_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<VerifyTokenRequest>,
) -> Json<VerifyTokenResponse> {
    use crate::types::slice::{AdmissibilityToken, SliceFingerprint, GraphSnapshotHash};

    // Parse fields
    let slice_id = SliceFingerprint::new(request.slice_id.clone());
    let anchor_id = match TurnId::from_str(&request.anchor_turn_id) {
        Ok(id) => id,
        Err(_) => {
            return Json(VerifyTokenResponse {
                valid: false,
                reason: Some("Invalid anchor_turn_id format".to_string()),
            });
        }
    };
    let graph_snapshot_hash = GraphSnapshotHash::new(request.graph_snapshot_hash.clone());
    
    // Create token and verify
    let token = AdmissibilityToken::from_string(request.admissibility_token.clone());
    let valid = token.verify_hmac(
        state.hmac_secret(),
        &slice_id,
        &anchor_id,
        &request.policy_id,
        &request.policy_params_hash,
        &graph_snapshot_hash,
        &request.schema_version,
    );

    Json(VerifyTokenResponse {
        valid,
        reason: if valid { None } else { Some("Token does not match expected HMAC".to_string()) },
    })
}


// ============================================================================
// Knowledge Graph Types & Handlers (re-exported from knowledge_handlers module)
// ============================================================================

// Re-export types for backward compatibility
pub use super::knowledge_handlers::{
    KnowledgeTriple, StoredKnowledgeTriple, KnowledgeBatchResponse, BatchTripleError,
    KnowledgeQueryParams, KnowledgeQueryResponse, KnowledgeDeleteRequest, KnowledgeDeleteResponse,
    KnowledgeStatsResponse, PredicateCount,
    TraversalRequest, TraversalEdge, TraversalPath, TraversalStats, TraversalResponse,
    AliasQueryParams, AliasEntry, AliasResponse,
    BatchInput,
};

// Use handler functions from the backend-agnostic module
use super::knowledge_handlers::{
    add_knowledge_handler, add_knowledge_batch_handler,
    query_knowledge_handler, knowledge_stats_handler,
    delete_knowledge_handler, traverse_knowledge_handler,
    aliases_handler,
};

// ============================================================================
// Router Construction
// ============================================================================

/// Create the Axum router for the Graph Kernel service.
pub fn create_router(state: AppState) -> Router {
    use super::visualization;
    let state = Arc::new(state);

    Router::new()
        // Slice operations
        .route("/api/slice", post(slice_handler))
        .route("/api/slice/batch", post(batch_slice_handler))
        // Token verification
        .route("/api/verify_token", post(verify_token_handler))
        // Policy management
        .route("/api/policies", get(list_policies_handler))
        .route("/api/policies", post(register_policy_handler))
        // Knowledge graph CRUD
        .route("/api/knowledge", get(query_knowledge_handler))
        .route("/api/knowledge", post(add_knowledge_handler))
        .route("/api/knowledge", delete(delete_knowledge_handler))
        .route("/api/knowledge/batch", post(add_knowledge_batch_handler))
        .route("/api/knowledge/stats", get(knowledge_stats_handler))
        // Entity alias lookup (for RAG++ query expansion)
        .route("/api/knowledge/aliases", get(aliases_handler))
        // Server-side multi-hop traversal
        .route("/api/knowledge/traverse", post(traverse_knowledge_handler))
        // Graph visualization (D3, Mermaid, DOT)
        .route("/api/knowledge/graph", get(visualization::graph_d3_handler))
        .route("/api/knowledge/graph.mermaid", get(visualization::graph_mermaid_handler))
        .route("/api/knowledge/graph.dot", get(visualization::graph_dot_handler))
        // Health checks (Cloud Run compatible)
        .route("/health", get(health_handler))           // Detailed health
        .route("/health/live", get(liveness_handler))    // Liveness probe
        .route("/health/ready", get(readiness_handler))  // Readiness probe
        .route("/health/startup", get(startup_handler))  // Startup probe
        .with_state(state)
}

