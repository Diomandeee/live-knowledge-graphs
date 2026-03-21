//! Graph visualization endpoints for the knowledge graph.
//!
//! Renders the knowledge graph in formats consumable by visualization tools:
//! - **D3.js** — Force-directed graph JSON with nodes and edges
//! - **Mermaid** — Markdown-embeddable diagram syntax
//! - **Graphviz DOT** — Standard graph description language
//!
//! ## Endpoints
//!
//! - `GET /api/knowledge/graph` — D3-compatible JSON
//! - `GET /api/knowledge/graph.mermaid` — Mermaid diagram text
//! - `GET /api/knowledge/graph.dot` — Graphviz DOT format
//!
//! ## Filtering
//!
//! All endpoints support optional query parameters:
//! - `subject` — Center the subgraph around this entity
//! - `depth` — BFS depth from the subject (default: 2)
//! - `min_confidence` — Minimum edge confidence to include
//! - `limit` — Maximum number of edges to return (default: 500)

use axum::{
    extract::{Json, Query, State},
    http::{header, StatusCode},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use super::routes::{AppState, ErrorResponse};

// ============================================================================
// Types
// ============================================================================

/// Query parameters for graph visualization endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct GraphQueryParams {
    /// Center the subgraph around this entity.
    pub subject: Option<String>,
    /// BFS depth from the subject (default: 2, max: 5).
    #[serde(default = "default_depth")]
    pub depth: u32,
    /// Minimum edge confidence to include.
    pub min_confidence: Option<f64>,
    /// Maximum number of edges (default: 500).
    #[serde(default = "default_graph_limit")]
    pub limit: usize,
}

fn default_depth() -> u32 { 2 }
fn default_graph_limit() -> usize { 500 }

/// A node in the D3-compatible graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D3Node {
    /// Unique entity identifier.
    pub id: String,
    /// Display label (same as id for now).
    pub label: String,
    /// Entity type (inferred from predicates).
    #[serde(rename = "type")]
    pub node_type: String,
    /// Number of edges connected to this node.
    pub degree: usize,
}

/// An edge in the D3-compatible graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D3Edge {
    /// Source node ID.
    pub source: String,
    /// Target node ID.
    pub target: String,
    /// Relationship predicate.
    pub predicate: String,
    /// Edge weight (confidence score).
    pub weight: f64,
}

/// D3-compatible graph response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D3GraphResponse {
    /// Graph nodes.
    pub nodes: Vec<D3Node>,
    /// Graph edges.
    pub edges: Vec<D3Edge>,
    /// Metadata about the graph.
    pub meta: GraphMeta,
}

/// Graph metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMeta {
    /// Total nodes in response.
    pub node_count: usize,
    /// Total edges in response.
    pub edge_count: usize,
    /// Whether the graph was filtered (subgraph).
    pub filtered: bool,
    /// Filter center (if filtered).
    pub center: Option<String>,
    /// Filter depth (if filtered).
    pub depth: Option<u32>,
}

/// Raw triple from the database.
#[derive(Debug, Clone)]
struct RawTriple {
    subject: String,
    predicate: String,
    object: String,
    confidence: f64,
}

// ============================================================================
// Internal: Fetch graph data
// ============================================================================

/// Fetch triples from the knowledge graph, optionally filtered to a subgraph.
async fn fetch_graph_triples(
    state: &Arc<AppState>,
    params: &GraphQueryParams,
) -> Result<Vec<RawTriple>, (StatusCode, Json<ErrorResponse>)> {
    let pool = state.store.pool();

    // Normalize subject if provided
    let center = params
        .subject
        .as_deref()
        .map(crate::service::normalize::canonicalize_entity);

    if let Some(ref center_entity) = center {
        // BFS to find all entities within `depth` hops
        let depth = params.depth.min(5);
        let mut visited: HashSet<String> = HashSet::new();
        let mut frontier: VecDeque<(String, u32)> = VecDeque::new();

        frontier.push_back((center_entity.clone(), 0));
        visited.insert(center_entity.clone());

        while let Some((entity, d)) = frontier.pop_front() {
            if d >= depth {
                continue;
            }

            // Find all neighbors
            let rows = sqlx::query_as::<_, (String, String, String, f64)>(
                "SELECT subject, predicate, object, confidence FROM knowledge_graph \
                 WHERE subject = $1 OR object = $1 \
                 ORDER BY confidence DESC LIMIT 200",
            )
            .bind(&entity)
            .fetch_all(pool)
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::new("DB_ERROR", format!("Graph query failed: {}", e))),
                )
            })?;

            for (subj, _pred, obj, _conf) in &rows {
                let neighbor = if subj == &entity { obj } else { subj };
                if visited.insert(neighbor.clone()) {
                    frontier.push_back((neighbor.clone(), d + 1));
                }
            }
        }

        // Now fetch all triples between visited entities
        // Build IN clause
        if visited.is_empty() {
            return Ok(vec![]);
        }

        let visited_vec: Vec<String> = visited.into_iter().collect();
        let limit = params.limit.min(2000) as i64;

        let mut conf_condition = String::new();
        if let Some(min_conf) = params.min_confidence {
            conf_condition = format!(" AND confidence >= {}", min_conf);
        }

        let query_str = format!(
            "SELECT subject, predicate, object, confidence FROM knowledge_graph \
             WHERE subject = ANY($1) AND object = ANY($1){} \
             ORDER BY confidence DESC LIMIT $2",
            conf_condition
        );

        let rows = sqlx::query_as::<_, (String, String, String, f64)>(&query_str)
            .bind(&visited_vec)
            .bind(limit)
            .fetch_all(pool)
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::new("DB_ERROR", format!("Graph query failed: {}", e))),
                )
            })?;

        Ok(rows
            .into_iter()
            .map(|(subject, predicate, object, confidence)| RawTriple {
                subject,
                predicate,
                object,
                confidence,
            })
            .collect())
    } else {
        // Full graph (limited)
        let limit = params.limit.min(2000) as i64;

        let mut conf_condition = String::new();
        if let Some(min_conf) = params.min_confidence {
            conf_condition = format!(" WHERE confidence >= {}", min_conf);
        }

        let query_str = format!(
            "SELECT subject, predicate, object, confidence FROM knowledge_graph{} \
             ORDER BY confidence DESC LIMIT $1",
            conf_condition
        );

        let rows = sqlx::query_as::<_, (String, String, String, f64)>(&query_str)
            .bind(limit)
            .fetch_all(pool)
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse::new("DB_ERROR", format!("Graph query failed: {}", e))),
                )
            })?;

        Ok(rows
            .into_iter()
            .map(|(subject, predicate, object, confidence)| RawTriple {
                subject,
                predicate,
                object,
                confidence,
            })
            .collect())
    }
}

/// Build D3 graph structure from raw triples.
fn build_d3_graph(triples: &[RawTriple], center: Option<&str>, depth: Option<u32>) -> D3GraphResponse {
    let mut node_degrees: HashMap<String, usize> = HashMap::new();
    let mut node_predicates: HashMap<String, HashSet<String>> = HashMap::new();
    let mut edges = Vec::new();
    let mut seen_edges: HashSet<(String, String, String)> = HashSet::new();

    for triple in triples {
        // Count degrees
        *node_degrees.entry(triple.subject.clone()).or_insert(0) += 1;
        *node_degrees.entry(triple.object.clone()).or_insert(0) += 1;

        // Track predicates per node for type inference
        node_predicates
            .entry(triple.subject.clone())
            .or_default()
            .insert(triple.predicate.clone());

        // Deduplicate edges
        let edge_key = (
            triple.subject.clone(),
            triple.object.clone(),
            triple.predicate.clone(),
        );
        if seen_edges.insert(edge_key) {
            edges.push(D3Edge {
                source: triple.subject.clone(),
                target: triple.object.clone(),
                predicate: triple.predicate.clone(),
                weight: triple.confidence,
            });
        }
    }

    let nodes: Vec<D3Node> = node_degrees
        .iter()
        .map(|(id, &degree)| {
            let node_type = infer_node_type(id, node_predicates.get(id));
            D3Node {
                id: id.clone(),
                label: id.clone(),
                node_type,
                degree,
            }
        })
        .collect();

    let meta = GraphMeta {
        node_count: nodes.len(),
        edge_count: edges.len(),
        filtered: center.is_some(),
        center: center.map(|s| s.to_string()),
        depth,
    };

    D3GraphResponse { nodes, edges, meta }
}

/// Infer a node's type based on its predicates and name.
fn infer_node_type(id: &str, predicates: Option<&HashSet<String>>) -> String {
    if let Some(preds) = predicates {
        if preds.contains("is_a") || preds.contains("instance_of") {
            return "entity".to_string();
        }
        if preds.contains("depends_on") || preds.contains("uses") {
            return "service".to_string();
        }
        if preds.contains("created_by") || preds.contains("authored_by") {
            return "artifact".to_string();
        }
    }

    // Heuristic by name
    if id.contains("engine") || id.contains("service") || id.contains("server") {
        "service".to_string()
    } else if id.contains("pipeline") || id.contains("flow") {
        "pipeline".to_string()
    } else {
        "entity".to_string()
    }
}

/// Render triples as Mermaid diagram text.
fn render_mermaid(triples: &[RawTriple]) -> String {
    let mut lines = vec!["graph LR".to_string()];
    let mut seen: HashSet<(String, String, String)> = HashSet::new();

    for triple in triples {
        let key = (
            triple.subject.clone(),
            triple.predicate.clone(),
            triple.object.clone(),
        );
        if seen.insert(key) {
            lines.push(format!(
                "    {} -->|{}| {}",
                sanitize_mermaid_id(&triple.subject),
                triple.predicate,
                sanitize_mermaid_id(&triple.object),
            ));
        }
    }

    lines.join("\n")
}

/// Sanitize an ID for Mermaid (replace hyphens, wrap special chars).
fn sanitize_mermaid_id(id: &str) -> String {
    // Mermaid IDs can't contain certain chars; use quoted form for safety
    let sanitized = id.replace('-', "_").replace(' ', "_");
    if sanitized.contains(|c: char| !c.is_alphanumeric() && c != '_') {
        format!("{}[\"{}\"]", sanitized.replace(|c: char| !c.is_alphanumeric() && c != '_', "_"), id)
    } else {
        format!("{}[\"{}\"]", sanitized, id)
    }
}

/// Render triples as Graphviz DOT format.
fn render_dot(triples: &[RawTriple]) -> String {
    let mut lines = vec![
        "digraph KnowledgeGraph {".to_string(),
        "    rankdir=LR;".to_string(),
        "    node [shape=box, style=rounded];".to_string(),
    ];

    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut nodes: HashSet<String> = HashSet::new();

    for triple in triples {
        nodes.insert(triple.subject.clone());
        nodes.insert(triple.object.clone());

        let key = (
            triple.subject.clone(),
            triple.predicate.clone(),
            triple.object.clone(),
        );
        if seen.insert(key) {
            lines.push(format!(
                "    \"{}\" -> \"{}\" [label=\"{}\", penwidth={:.1}];",
                escape_dot(&triple.subject),
                escape_dot(&triple.object),
                escape_dot(&triple.predicate),
                (triple.confidence * 3.0).max(0.5), // Scale confidence to line width
            ));
        }
    }

    lines.push("}".to_string());
    lines.join("\n")
}

/// Escape a string for DOT format.
fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

// ============================================================================
// Route Handlers
// ============================================================================

/// GET /api/knowledge/graph — D3-compatible JSON graph.
pub async fn graph_d3_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<GraphQueryParams>,
) -> Result<Json<D3GraphResponse>, (StatusCode, Json<ErrorResponse>)> {
    let center = params
        .subject
        .as_deref()
        .map(crate::service::normalize::canonicalize_entity);
    let depth = if center.is_some() {
        Some(params.depth)
    } else {
        None
    };

    let triples = fetch_graph_triples(&state, &params).await?;
    let graph = build_d3_graph(&triples, center.as_deref(), depth);

    Ok(Json(graph))
}

/// GET /api/knowledge/graph.mermaid — Mermaid diagram text.
pub async fn graph_mermaid_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<GraphQueryParams>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let triples = fetch_graph_triples(&state, &params).await?;
    let mermaid = render_mermaid(&triples);

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        mermaid,
    ))
}

/// GET /api/knowledge/graph.dot — Graphviz DOT format.
pub async fn graph_dot_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<GraphQueryParams>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let triples = fetch_graph_triples(&state, &params).await?;
    let dot = render_dot(&triples);

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/vnd.graphviz; charset=utf-8")],
        dot,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triples() -> Vec<RawTriple> {
        vec![
            RawTriple {
                subject: "clawdbot".to_string(),
                predicate: "uses".to_string(),
                object: "graph-kernel".to_string(),
                confidence: 0.95,
            },
            RawTriple {
                subject: "graph-kernel".to_string(),
                predicate: "uses".to_string(),
                object: "postgresql".to_string(),
                confidence: 0.90,
            },
            RawTriple {
                subject: "clawdbot".to_string(),
                predicate: "uses".to_string(),
                object: "rag-plusplus".to_string(),
                confidence: 0.85,
            },
        ]
    }

    #[test]
    fn test_build_d3_graph() {
        let triples = sample_triples();
        let graph = build_d3_graph(&triples, None, None);

        assert_eq!(graph.nodes.len(), 4); // clawdbot, graph-kernel, postgresql, rag-plusplus
        assert_eq!(graph.edges.len(), 3);
        assert_eq!(graph.meta.node_count, 4);
        assert_eq!(graph.meta.edge_count, 3);
        assert!(!graph.meta.filtered);

        // Check degree of clawdbot (2 outgoing edges)
        let clawdbot = graph.nodes.iter().find(|n| n.id == "clawdbot").unwrap();
        assert_eq!(clawdbot.degree, 2);
    }

    #[test]
    fn test_build_d3_graph_filtered() {
        let triples = sample_triples();
        let graph = build_d3_graph(&triples, Some("clawdbot"), Some(2));

        assert!(graph.meta.filtered);
        assert_eq!(graph.meta.center.as_deref(), Some("clawdbot"));
        assert_eq!(graph.meta.depth, Some(2));
    }

    #[test]
    fn test_render_mermaid() {
        let triples = sample_triples();
        let mermaid = render_mermaid(&triples);

        assert!(mermaid.starts_with("graph LR"));
        assert!(mermaid.contains("uses"));
        assert!(mermaid.contains("-->|"));
    }

    #[test]
    fn test_render_dot() {
        let triples = sample_triples();
        let dot = render_dot(&triples);

        assert!(dot.starts_with("digraph KnowledgeGraph {"));
        assert!(dot.contains("->"));
        assert!(dot.contains("[label=\"uses\""));
        assert!(dot.ends_with('}'));
    }

    #[test]
    fn test_mermaid_deduplication() {
        let triples = vec![
            RawTriple {
                subject: "a".to_string(),
                predicate: "uses".to_string(),
                object: "b".to_string(),
                confidence: 0.9,
            },
            RawTriple {
                subject: "a".to_string(),
                predicate: "uses".to_string(),
                object: "b".to_string(),
                confidence: 0.8,
            },
        ];
        let mermaid = render_mermaid(&triples);
        // Should only have one edge line (plus the header)
        let edge_lines: Vec<&str> = mermaid.lines().filter(|l| l.contains("-->")).collect();
        assert_eq!(edge_lines.len(), 1);
    }

    #[test]
    fn test_empty_graph() {
        let graph = build_d3_graph(&[], None, None);
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.edges.len(), 0);
    }

    #[test]
    fn test_infer_node_type() {
        let mut preds = HashSet::new();
        preds.insert("uses".to_string());
        assert_eq!(infer_node_type("clawdbot", Some(&preds)), "service");

        assert_eq!(infer_node_type("my-engine", None), "service");
        assert_eq!(infer_node_type("my-pipeline", None), "pipeline");
        assert_eq!(infer_node_type("random-thing", None), "entity");
    }
}
