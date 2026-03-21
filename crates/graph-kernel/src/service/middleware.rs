//! Service middleware for metrics, rate limiting, and request tracking.
//!
//! ## Metrics Exposed
//!
//! - `graph_kernel_requests_total` - Counter of total requests by path, method, status
//! - `graph_kernel_request_duration_seconds` - Histogram of request latency
//! - `graph_kernel_slice_turns_count` - Histogram of turns per slice
//! - `graph_kernel_token_verifications_total` - Counter of token verifications

use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
};
use std::time::Instant;
use tracing::info;

/// Metrics middleware that records request counts and latency.
///
/// Records:
/// - Total request count by path pattern, method, and status code
/// - Request duration as a histogram
///
/// Uses tracing for now - can be upgraded to prometheus metrics later.
pub async fn metrics_middleware(request: Request, next: Next) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let path = normalize_path(request.uri().path());
    
    let response = next.run(request).await;
    
    let latency = start.elapsed();
    let status = response.status().as_u16();
    
    // Log metrics for Cloud Monitoring (can be aggregated from logs)
    info!(
        target: "graph_kernel::metrics",
        metric_type = "request",
        path = %path,
        method = %method,
        status = status,
        latency_ms = latency.as_millis() as u64,
        "request_metric"
    );
    
    response
}

/// Normalize path for metrics to avoid high cardinality.
///
/// Replaces UUIDs and other dynamic path segments with placeholders.
fn normalize_path(path: &str) -> String {
    // Replace UUIDs with :id placeholder
    let uuid_regex = regex_lite::Regex::new(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    ).unwrap();
    
    uuid_regex.replace_all(path, ":id").to_string()
}

/// Record slice generation metrics.
///
/// Call this after generating a slice to track turn counts.
pub fn record_slice_metrics(turn_count: usize, edge_count: usize, latency_ms: u64) {
    info!(
        target: "graph_kernel::metrics",
        metric_type = "slice",
        turn_count = turn_count,
        edge_count = edge_count,
        latency_ms = latency_ms,
        "slice_metric"
    );
}

/// Record token verification metrics.
pub fn record_token_verification(valid: bool) {
    let result = if valid { "valid" } else { "invalid" };
    info!(
        target: "graph_kernel::metrics",
        metric_type = "token_verification",
        result = result,
        "token_verification_metric"
    );
}

/// Record database query metrics.
#[allow(dead_code)]
pub fn record_db_query(query_type: &str, latency_ms: u64, success: bool) {
    let status = if success { "success" } else { "error" };
    info!(
        target: "graph_kernel::metrics",
        metric_type = "db_query",
        query_type = query_type,
        status = status,
        latency_ms = latency_ms,
        "db_query_metric"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_normalize_path_replaces_uuid() {
        let path = "/api/slice/550e8400-e29b-41d4-a716-446655440000";
        let normalized = normalize_path(path);
        assert_eq!(normalized, "/api/slice/:id");
    }
    
    #[test]
    fn test_normalize_path_preserves_regular_path() {
        let path = "/health/ready";
        let normalized = normalize_path(path);
        assert_eq!(normalized, "/health/ready");
    }
}
