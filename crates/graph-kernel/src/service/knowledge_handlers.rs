//! Knowledge graph CRUD handlers using the backend-agnostic `DynKnowledgeDb`.
//!
//! These handlers work with both PostgreSQL and SQLite backends.
//! All SQL dialect differences are encapsulated in the `KnowledgeDb` implementations.

use axum::{
    extract::{Json, Query, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::store::knowledge_db::KnowledgeQuery;
use super::routes::{AppState, ErrorResponse};

// ============================================================================
// Request/Response Types
// ============================================================================

/// A knowledge triple (subject-predicate-object).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    #[serde(default = "default_source")]
    pub source: String,
}

fn default_confidence() -> f64 { 0.5 }
fn default_source() -> String { "unknown".to_string() }

/// Stored knowledge triple with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredKnowledgeTriple {
    pub id: i64,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
    pub source: String,
    pub created_at: String,
}

/// Response from knowledge batch insert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBatchResponse {
    pub added: usize,
    pub updated: usize,
    pub total: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<BatchTripleError>,
}

/// Flexible batch input: accepts either a bare array or `{"triples": [...]}`.
/// This prevents 422 errors from clients that wrap the array in an object.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum BatchInput {
    Wrapped { triples: Vec<KnowledgeTriple> },
    Bare(Vec<KnowledgeTriple>),
}

impl BatchInput {
    pub fn into_triples(self) -> Vec<KnowledgeTriple> {
        match self {
            BatchInput::Wrapped { triples } => triples,
            BatchInput::Bare(triples) => triples,
        }
    }
}

/// Error for a specific triple in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTripleError {
    pub index: usize,
    pub error: String,
}

/// Query parameters for knowledge search.
#[derive(Debug, Clone, Deserialize)]
pub struct KnowledgeQueryParams {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub min_confidence: Option<f64>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize { 50 }

/// Response containing knowledge query results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeQueryResponse {
    pub triples: Vec<StoredKnowledgeTriple>,
    pub total: usize,
}

/// Request to delete knowledge triples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDeleteRequest {
    pub id: Option<i64>,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
}

/// Response from knowledge delete.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDeleteResponse {
    pub deleted: usize,
}

/// Knowledge graph stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeStatsResponse {
    pub total_triples: usize,
    pub unique_subjects: usize,
    pub unique_predicates: usize,
    pub top_predicates: Vec<PredicateCount>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateCount {
    pub predicate: String,
    pub count: usize,
}

// ============================================================================
// Alias Types
// ============================================================================

/// Query parameters for alias lookup.
#[derive(Debug, Clone, Deserialize)]
pub struct AliasQueryParams {
    pub query: String,
}

/// A single alias entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasEntry {
    pub original: String,
    pub alias: String,
    pub canonical: String,
}

/// Response containing alias expansions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasResponse {
    pub aliases: Vec<AliasEntry>,
}

// ============================================================================
// Traversal Types
// ============================================================================

/// Request for server-side multi-hop graph traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalRequest {
    pub start: String,
    pub predicates: Option<Vec<String>>,
    #[serde(default = "default_direction")]
    pub direction: String,
    #[serde(default = "default_max_hops")]
    pub max_hops: u32,
    #[serde(default = "default_max_results")]
    pub max_results: usize,
    #[serde(default)]
    pub min_confidence: Option<f64>,
    #[serde(default = "default_true")]
    pub return_paths: bool,
}

fn default_direction() -> String { "outgoing".to_string() }
fn default_max_hops() -> u32 { 3 }
fn default_max_results() -> usize { 100 }
fn default_true() -> bool { true }

/// A single edge in a traversal path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalEdge {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
}

/// A complete traversal path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalPath {
    pub entities: Vec<String>,
    pub edges: Vec<TraversalEdge>,
    pub hops: u32,
    pub min_confidence: f64,
}

/// Traversal statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalStats {
    pub entities_visited: usize,
    pub edges_traversed: usize,
    pub elapsed_ms: u64,
}

/// Response from graph traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalResponse {
    pub paths: Vec<TraversalPath>,
    pub stats: TraversalStats,
}

// ============================================================================
// Handlers (backend-agnostic via DynKnowledgeDb)
// ============================================================================

/// Add a single knowledge triple.
pub async fn add_knowledge_handler(
    State(state): State<Arc<AppState>>,
    Json(triple): Json<KnowledgeTriple>,
) -> Result<Json<KnowledgeBatchResponse>, (StatusCode, Json<ErrorResponse>)> {
    use crate::service::normalize::canonicalize_entity;

    let subject = canonicalize_entity(&triple.subject);
    let object = canonicalize_entity(&triple.object);
    let predicate = triple.predicate.to_lowercase();

    let result = state.knowledge_db.upsert_triple(
        &subject, &predicate, &object, triple.confidence, &triple.source,
    ).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new("DB_ERROR", format!("Failed to insert triple: {}", e))),
        )
    })?;

    // Invalidate query cache on write
    state.query_cache.invalidate_all();

    Ok(Json(KnowledgeBatchResponse {
        added: if result.inserted { 1 } else { 0 },
        updated: if result.inserted { 0 } else { 1 },
        total: 1,
        errors: vec![],
    }))
}

/// Add a batch of knowledge triples.
///
/// Accepts either a bare JSON array `[{...}, ...]` or a wrapped object `{"triples": [{...}, ...]}`.
pub async fn add_knowledge_batch_handler(
    State(state): State<Arc<AppState>>,
    Json(input): Json<BatchInput>,
) -> Result<Json<KnowledgeBatchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let triples = input.into_triples();
    use crate::service::normalize::canonicalize_entity;

    if triples.is_empty() {
        return Ok(Json(KnowledgeBatchResponse {
            added: 0, updated: 0, total: 0, errors: vec![],
        }));
    }

    if triples.len() > 10_000 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "BATCH_TOO_LARGE",
                format!("Batch size {} exceeds maximum of 10,000 triples", triples.len()),
            )),
        ));
    }

    // Pre-normalize all triples
    let normalized: Vec<(String, String, String, f64, String)> = triples
        .iter()
        .map(|t| (
            canonicalize_entity(&t.subject),
            t.predicate.to_lowercase(),
            canonicalize_entity(&t.object),
            t.confidence,
            t.source.clone(),
        ))
        .collect();

    // Validate
    let mut errors = Vec::new();
    for (i, (subject, predicate, object, confidence, _)) in normalized.iter().enumerate() {
        if subject.is_empty() {
            errors.push(BatchTripleError { index: i, error: "Subject cannot be empty".to_string() });
        }
        if predicate.is_empty() {
            errors.push(BatchTripleError { index: i, error: "Predicate cannot be empty".to_string() });
        }
        if object.is_empty() {
            errors.push(BatchTripleError { index: i, error: "Object cannot be empty".to_string() });
        }
        if !confidence.is_finite() || *confidence < 0.0 || *confidence > 1.0 {
            errors.push(BatchTripleError {
                index: i,
                error: format!("Confidence must be between 0.0 and 1.0, got {}", confidence),
            });
        }
    }

    if !errors.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "VALIDATION_ERROR",
                format!("{} triple(s) failed validation", errors.len()),
            ).with_details(serde_json::to_string(&errors).unwrap_or_default())),
        ));
    }

    let (added, updated) = state.knowledge_db.upsert_batch(&normalized)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new("DB_ERROR", format!("Batch insert failed: {}", e))),
            )
        })?;

    state.query_cache.invalidate_all();

    tracing::info!(
        total = triples.len(),
        added = added,
        updated = updated,
        "Batch ingest completed"
    );

    Ok(Json(KnowledgeBatchResponse {
        added,
        updated,
        total: added + updated,
        errors: vec![],
    }))
}

/// Query knowledge triples.
pub async fn query_knowledge_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<KnowledgeQueryParams>,
) -> Result<Json<KnowledgeQueryResponse>, (StatusCode, Json<ErrorResponse>)> {
    let subject = params.subject.as_deref().map(crate::service::normalize::canonicalize_entity);
    let object = params.object.as_deref().map(crate::service::normalize::canonicalize_entity);

    // Check cache first
    let cache_key = crate::service::cache::QueryCacheKey::new(
        subject.clone(),
        params.predicate.clone(),
        object.clone(),
        params.min_confidence,
        params.limit,
    );

    if let Some(cached) = state.query_cache.get(&cache_key) {
        return Ok(Json(cached));
    }

    let limit = params.limit.min(500) as i64;

    let query = KnowledgeQuery {
        subject,
        predicate: params.predicate,
        object,
        min_confidence: params.min_confidence,
        limit,
    };

    let (triples, total) = state.knowledge_db.query_triples(&query)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new("DB_ERROR", format!("Query failed: {}", e))),
            )
        })?;

    let response = KnowledgeQueryResponse {
        triples: triples.into_iter().map(|t| StoredKnowledgeTriple {
            id: t.id,
            subject: t.subject,
            predicate: t.predicate,
            object: t.object,
            confidence: t.confidence,
            source: t.source,
            created_at: t.created_at,
        }).collect(),
        total: total as usize,
    };

    state.query_cache.put(cache_key, response.clone());

    Ok(Json(response))
}

/// Get knowledge graph stats.
pub async fn knowledge_stats_handler(
    State(state): State<Arc<AppState>>,
) -> Result<Json<KnowledgeStatsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let stats = state.knowledge_db.stats()
        .await
        .map_err(|e| {
            (StatusCode::INTERNAL_SERVER_ERROR,
             Json(ErrorResponse::new("DB_ERROR", format!("{}", e))))
        })?;

    Ok(Json(KnowledgeStatsResponse {
        total_triples: stats.total_triples as usize,
        unique_subjects: stats.unique_subjects as usize,
        unique_predicates: stats.unique_predicates as usize,
        top_predicates: stats.top_predicates.into_iter()
            .map(|(p, c)| PredicateCount { predicate: p, count: c as usize })
            .collect(),
    }))
}

/// Delete knowledge triples.
pub async fn delete_knowledge_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<KnowledgeDeleteRequest>,
) -> Result<Json<KnowledgeDeleteResponse>, (StatusCode, Json<ErrorResponse>)> {
    use crate::service::normalize::canonicalize_entity;

    if request.id.is_none()
        && request.subject.is_none()
        && request.predicate.is_none()
        && request.object.is_none()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "MISSING_FILTER",
                "At least one filter (id, subject, predicate, or object) is required for deletion",
            )),
        ));
    }

    let subject = request.subject.as_deref().map(canonicalize_entity);
    let predicate = request.predicate.as_deref().map(|p| p.to_lowercase());
    let object = request.object.as_deref().map(canonicalize_entity);

    let result = state.knowledge_db.delete_triples(
        request.id,
        subject.as_deref(),
        predicate.as_deref(),
        object.as_deref(),
    ).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new("DB_ERROR", format!("Delete failed: {}", e))),
        )
    })?;

    let deleted = result.rows_affected as usize;
    if deleted > 0 {
        state.query_cache.invalidate_all();
    }

    Ok(Json(KnowledgeDeleteResponse { deleted }))
}

/// Server-side multi-hop BFS graph traversal.
pub async fn traverse_knowledge_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TraversalRequest>,
) -> Result<Json<TraversalResponse>, (StatusCode, Json<ErrorResponse>)> {
    use crate::service::normalize::canonicalize_entity;
    use std::collections::{HashSet, VecDeque};
    use std::time::Instant;

    let start_time = Instant::now();
    let start_entity = canonicalize_entity(&request.start);
    let mut visited: HashSet<String> = HashSet::new();
    let mut frontier: VecDeque<(String, Vec<TraversalEdge>, u32)> = VecDeque::new();
    let mut paths: Vec<TraversalPath> = Vec::new();
    let mut total_edges_traversed = 0usize;

    frontier.push_back((start_entity.clone(), vec![], 0));
    visited.insert(start_entity.clone());

    while let Some((entity, path, depth)) = frontier.pop_front() {
        if depth >= request.max_hops || paths.len() >= request.max_results {
            break;
        }

        let adjacent = state.knowledge_db.query_adjacent(&entity, &request.direction)
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::new("DB_ERROR", format!("Traversal query failed: {}", e))),
                )
            })?;

        for triple in adjacent {
            total_edges_traversed += 1;

            // Filter by predicates if specified
            if let Some(ref allowed_preds) = request.predicates {
                if !allowed_preds.iter().any(|p| p == &triple.predicate) {
                    continue;
                }
            }

            // Filter by confidence
            if let Some(min_conf) = request.min_confidence {
                if triple.confidence < min_conf {
                    continue;
                }
            }

            let next_entity = if entity == triple.subject {
                triple.object.clone()
            } else {
                triple.subject.clone()
            };

            let edge = TraversalEdge {
                subject: triple.subject,
                predicate: triple.predicate,
                object: triple.object,
                confidence: triple.confidence,
            };

            let mut new_path = path.clone();
            new_path.push(edge);

            if !visited.contains(&next_entity) {
                visited.insert(next_entity.clone());

                let mut entities = vec![start_entity.clone()];
                for e in &new_path {
                    let end = if entities.last().map(|l| l.as_str()) == Some(&e.subject) {
                        &e.object
                    } else {
                        &e.subject
                    };
                    entities.push(end.clone());
                }

                let min_conf = new_path.iter()
                    .map(|e| e.confidence)
                    .fold(f64::INFINITY, f64::min);

                paths.push(TraversalPath {
                    entities,
                    edges: new_path.clone(),
                    hops: depth + 1,
                    min_confidence: min_conf,
                });

                if paths.len() < request.max_results {
                    frontier.push_back((next_entity, new_path, depth + 1));
                }
            }
        }
    }

    let elapsed = start_time.elapsed();

    Ok(Json(TraversalResponse {
        paths,
        stats: TraversalStats {
            entities_visited: visited.len(),
            edges_traversed: total_edges_traversed,
            elapsed_ms: elapsed.as_millis() as u64,
        },
    }))
}

/// Return known aliases for entities found in a query string.
///
/// RAG++ calls this endpoint to expand search queries with alternative entity names.
/// For each word/phrase in the query that matches a known entity, returns all known aliases.
pub async fn aliases_handler(
    Query(params): Query<AliasQueryParams>,
) -> Json<AliasResponse> {
    use crate::service::normalize::{canonicalize_entity, get_aliases, is_known_entity};

    let query = &params.query;
    let mut aliases = Vec::new();

    // Split query into words and try progressively longer n-grams
    let words: Vec<&str> = query.split_whitespace().collect();

    for window_size in (1..=words.len().min(4)).rev() {
        for window in words.windows(window_size) {
            let phrase = window.join(" ");
            if is_known_entity(&phrase) {
                let canonical = canonicalize_entity(&phrase);
                let known_aliases = get_aliases(&canonical);
                for &alias in known_aliases {
                    aliases.push(AliasEntry {
                        original: phrase.clone(),
                        alias: alias.to_string(),
                        canonical: canonical.clone(),
                    });
                }
                // Also include the canonical form itself if different from original
                if canonical != phrase.to_lowercase() {
                    aliases.push(AliasEntry {
                        original: phrase.clone(),
                        alias: canonical.clone(),
                        canonical: canonical.clone(),
                    });
                }
            }
        }
    }

    Json(AliasResponse { aliases })
}
