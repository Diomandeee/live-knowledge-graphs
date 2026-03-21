//! Graph Kernel Service Binary
//!
//! Runs the Graph Kernel as a REST API service with production-grade features:
//! - Structured JSON logging for Cloud Logging
//! - Request tracing with correlation IDs
//! - Graceful shutdown handling
//! - Health check endpoints
//!
//! ## Configuration
//!
//! Environment variables:
//! - `DATABASE_URL`: PostgreSQL connection string (required)
//! - `KERNEL_HMAC_SECRET`: HMAC secret for token signing (required in production)
//! - `PORT`: Service port (default: 8001)
//! - `HOST`: Service host (default: 0.0.0.0)
//! - `RUST_LOG`: Log level filter (default: info)
//! - `LOG_FORMAT`: "json" for structured logs, "pretty" for development (default: json)
//!
//! ## Usage
//!
//! ```bash
//! DATABASE_URL=postgresql://... KERNEL_HMAC_SECRET=... cargo run --bin graph_kernel_service --features service
//! ```

use std::net::SocketAddr;
use std::time::Instant;

use axum::{
    extract::Request,
    middleware::{self, Next},
    response::Response,
};
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, info_span, warn, Instrument};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};

use cc_graph_kernel::service::{create_router, PolicyRegistry, ServiceState};
use cc_graph_kernel::store::{DynKnowledgeDb, PgKnowledgeDb};
use cc_graph_kernel::PostgresGraphStore;

/// Initialize the tracing subscriber with JSON or pretty format
fn init_tracing() {
    let log_format = std::env::var("LOG_FORMAT").unwrap_or_else(|_| "json".to_string());
    
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "graph_kernel_service=info,tower_http=info,sqlx=warn".into());

    if log_format == "pretty" {
        // Pretty format for local development
        tracing_subscriber::registry()
            .with(filter)
            .with(
                fmt::layer()
                    .with_target(true)
                    .with_span_events(FmtSpan::CLOSE)
            )
            .init();
    } else {
        // JSON format for production (Cloud Logging compatible)
        tracing_subscriber::registry()
            .with(filter)
            .with(
                fmt::layer()
                    .json()
                    .with_target(true)
                    .with_current_span(true)
                    .with_span_events(FmtSpan::CLOSE)
                    .flatten_event(true)
            )
            .init();
    }
}

/// Request logging middleware that adds correlation ID and timing
async fn request_logging_middleware(request: Request, next: Next) -> Response {
    let start = Instant::now();
    
    // Extract Cloud Trace context if present
    let trace_id = request
        .headers()
        .get("X-Cloud-Trace-Context")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split('/').next().unwrap_or(s).to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    
    let method = request.method().clone();
    let uri = request.uri().path().to_string();
    
    // Create span with trace context
    let span = info_span!(
        "request",
        trace_id = %trace_id,
        method = %method,
        path = %uri,
        status = tracing::field::Empty,
        latency_ms = tracing::field::Empty,
    );
    
    let response = next.run(request).instrument(span.clone()).await;
    
    let latency = start.elapsed();
    let status = response.status().as_u16();
    
    span.record("status", status);
    span.record("latency_ms", latency.as_millis() as u64);
    
    // Log request completion
    info!(
        target: "graph_kernel_service::access",
        trace_id = %trace_id,
        method = %method,
        path = %uri,
        status = status,
        latency_ms = latency.as_millis() as u64,
        "request completed"
    );
    
    response
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize structured logging
    init_tracing();

    let version = env!("CARGO_PKG_VERSION");
    let build_sha = option_env!("BUILD_SHA").unwrap_or("dev");
    
    info!(
        version = version,
        build_sha = build_sha,
        "Starting Graph Kernel Service"
    );

    // Load configuration from environment
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8001);
    
    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());

    // Load HMAC secret for admissibility tokens
    let hmac_secret = match std::env::var("KERNEL_HMAC_SECRET") {
        Ok(s) if !s.is_empty() => {
            info!("HMAC secret loaded from environment");
            s.into_bytes()
        }
        _ => {
            warn!(
                "KERNEL_HMAC_SECRET not set or empty. Using development secret. \
                 This is a SECURITY RISK in production!"
            );
            b"development_only_secret_not_for_production".to_vec()
        }
    };

    // Connect to PostgreSQL with timeout
    info!("Connecting to PostgreSQL...");
    let connect_start = Instant::now();
    
    let store = match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        PostgresGraphStore::from_env()
    ).await {
        Ok(Ok(store)) => store,
        Ok(Err(e)) => {
            tracing::error!(error = %e, "Failed to connect to PostgreSQL");
            return Err(e.into());
        }
        Err(_) => {
            tracing::error!("PostgreSQL connection timeout after 30s");
            return Err("Database connection timeout".into());
        }
    };
    
    info!(
        latency_ms = connect_start.elapsed().as_millis() as u64,
        "PostgreSQL connection established"
    );

    // Ensure knowledge_graph table exists
    {
        let pool = store.pool();
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS knowledge_graph (\
                id BIGSERIAL PRIMARY KEY, \
                subject TEXT NOT NULL, \
                predicate TEXT NOT NULL, \
                object TEXT NOT NULL, \
                confidence DOUBLE PRECISION DEFAULT 0.5, \
                source TEXT DEFAULT 'unknown', \
                created_at TIMESTAMPTZ DEFAULT NOW(), \
                UNIQUE(subject, predicate, object))"
        ).execute(pool).await.map_err(|e| {
            tracing::error!(error = %e, "Failed to create knowledge_graph table");
            e
        })?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_knowledge_subject ON knowledge_graph(subject)")
            .execute(pool).await.ok();
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_knowledge_predicate ON knowledge_graph(predicate)")
            .execute(pool).await.ok();
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_knowledge_confidence ON knowledge_graph(confidence DESC)")
            .execute(pool).await.ok();
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_knowledge_object ON knowledge_graph(object)")
            .execute(pool).await.ok();
        info!("Knowledge graph table and indexes ensured");
    }

    // Create knowledge DB backed by the same PostgreSQL pool
    let knowledge_db = DynKnowledgeDb::new(PgKnowledgeDb::new(store.pool().clone()));
    info!("Knowledge DB initialized (postgres backend)");

    // Create service state with HMAC secret and knowledge DB
    let registry = PolicyRegistry::with_defaults();
    info!(
        policy_count = registry.len(),
        registry_fingerprint = %registry.fingerprint(),
        "Policy registry initialized"
    );

    let state = ServiceState::with_registry_and_knowledge_db(
        store, registry, hmac_secret, knowledge_db, "postgres",
    );

    // Build router with middleware
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Request timeout: 30s default, configurable via REQUEST_TIMEOUT_SECS
    let timeout_secs: u64 = std::env::var("REQUEST_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let app = create_router(state)
        .layer(middleware::from_fn(request_logging_middleware))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .layer(TimeoutLayer::new(std::time::Duration::from_secs(timeout_secs)));

    // Start server
    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    info!(
        address = %addr,
        version = version,
        "Graph Kernel Service listening"
    );

    let listener = TcpListener::bind(addr).await?;
    
    // Graceful shutdown handling
    let shutdown_signal = async {
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("Failed to install SIGTERM handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => info!("Received Ctrl+C, initiating graceful shutdown"),
            _ = terminate => info!("Received SIGTERM, initiating graceful shutdown"),
        }
    };
    
    info!("Ready to accept connections");
    
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;

    info!("Graph Kernel Service shutdown complete");
    
    Ok(())
}
