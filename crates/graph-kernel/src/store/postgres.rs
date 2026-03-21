//! PostgreSQL graph store for production use.
//!
//! ## Configuration
//!
//! All settings can be configured via environment variables:
//! - `DATABASE_URL`: PostgreSQL connection string (required)
//! - `DB_MAX_CONNECTIONS`: Maximum pool size (default: 10)
//! - `DB_MIN_CONNECTIONS`: Minimum idle connections (default: 2)
//! - `DB_CONNECT_TIMEOUT_SECS`: Connection timeout (default: 10)
//! - `DB_IDLE_TIMEOUT_SECS`: Idle connection timeout (default: 300)
//! - `DB_MAX_LIFETIME_SECS`: Max connection lifetime (default: 1800)

use sqlx::postgres::{PgPool, PgPoolOptions};
use sqlx::Row;
use std::time::Duration;
use uuid::Uuid;

use crate::types::{TurnId, TurnSnapshot, Edge, EdgeType, Role, Phase};
use super::GraphStore;

/// Configuration for PostgreSQL connection pool.
///
/// Production defaults are optimized for Cloud Run with Supabase:
/// - Pool size balances concurrency with connection limits
/// - Timeouts are aggressive to fail fast
/// - Idle timeout releases unused connections
/// - Max lifetime forces periodic reconnection for health
#[derive(Debug, Clone)]
pub struct PostgresConfig {
    /// Database connection URL.
    pub database_url: String,
    /// Maximum connections in pool (default: 10).
    pub max_connections: u32,
    /// Minimum idle connections to keep warm (default: 5).
    pub min_connections: u32,
    /// Connection acquire timeout in seconds (default: 10).
    pub connect_timeout_secs: u64,
    /// Idle connection timeout in seconds (default: 300 = 5 min).
    pub idle_timeout_secs: u64,
    /// Maximum connection lifetime in seconds (default: 1800 = 30 min).
    pub max_lifetime_secs: u64,
    /// Whether to test connections before acquiring from pool (default: false).
    /// Set to true for remote databases where connections may drop silently.
    /// Adds ~200ms RTT per query on remote databases when enabled.
    pub test_before_acquire: bool,
}

impl PostgresConfig {
    /// Load configuration from environment variables with production defaults.
    pub fn from_env() -> Self {
        Self {
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://localhost/orbit".to_string()),
            max_connections: std::env::var("DB_MAX_CONNECTIONS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
            min_connections: std::env::var("DB_MIN_CONNECTIONS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5),
            connect_timeout_secs: std::env::var("DB_CONNECT_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
            idle_timeout_secs: std::env::var("DB_IDLE_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(300),
            max_lifetime_secs: std::env::var("DB_MAX_LIFETIME_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1800),
            test_before_acquire: std::env::var("DB_TEST_BEFORE_ACQUIRE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(false),
        }
    }
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

/// PostgreSQL graph store.
///
/// Queries the Orbit database for turn and edge data.
/// Uses connection pooling with production-tuned settings.
pub struct PostgresGraphStore {
    pool: PgPool,
}

impl PostgresGraphStore {
    /// Create a new store with the given configuration.
    pub async fn new(config: PostgresConfig) -> Result<Self, sqlx::Error> {
        tracing::info!(
            max_connections = config.max_connections,
            min_connections = config.min_connections,
            connect_timeout_secs = config.connect_timeout_secs,
            idle_timeout_secs = config.idle_timeout_secs,
            max_lifetime_secs = config.max_lifetime_secs,
            "Initializing PostgreSQL connection pool"
        );

        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(config.min_connections)
            .acquire_timeout(Duration::from_secs(config.connect_timeout_secs))
            .idle_timeout(Duration::from_secs(config.idle_timeout_secs))
            .max_lifetime(Duration::from_secs(config.max_lifetime_secs))
            .test_before_acquire(config.test_before_acquire)
            .connect(&config.database_url)
            .await?;

        Ok(Self { pool })
    }

    /// Create a store from environment variables.
    pub async fn from_env() -> Result<Self, sqlx::Error> {
        Self::new(PostgresConfig::from_env()).await
    }

    /// Get the connection pool for health checks.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Check if the database is reachable.
    pub async fn is_healthy(&self) -> bool {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .is_ok()
    }

    /// Get pool statistics for monitoring.
    pub fn pool_stats(&self) -> PoolStats {
        PoolStats {
            size: self.pool.size(),
            idle: self.pool.num_idle(),
            max: self.pool.options().get_max_connections(),
        }
    }

    /// Fetch a turn with its content text and verify content hash.
    ///
    /// This enforces **INV-GK-004: Content Immutability** by verifying
    /// that the stored content hash matches the actual content.
    ///
    /// # Security
    /// - Returns error if content hash exists but doesn't match content
    /// - Logs warning if content hash is missing (legacy data)
    /// - Returns the turn with verified content
    ///
    /// # Returns
    /// A tuple of (TurnSnapshot, content_text) where the hash has been verified.
    pub async fn get_turn_with_verified_content(
        &self,
        id: &TurnId,
    ) -> Result<Option<(TurnSnapshot, String)>, PostgresError> {
        let row = sqlx::query(
            r#"
            SELECT id, conversation_id, role, phase, salience_score,
                   trajectory_depth, trajectory_sibling_order, trajectory_homogeneity,
                   trajectory_temporal, trajectory_complexity, created_at, content_hash,
                   content_text
            FROM memory_turns
            WHERE id = $1
            "#
        )
        .bind(id.as_uuid())
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(ref r) => {
                let turn = Self::parse_turn_row(r)?;
                let content_text: String = r.try_get("content_text").unwrap_or_default();

                // Verify content hash (INV-GK-004)
                if turn.has_content_hash() {
                    turn.verify_content_hash(&content_text)?;
                    tracing::trace!(
                        turn_id = %id,
                        "Content hash verified successfully"
                    );
                } else {
                    tracing::warn!(
                        turn_id = %id,
                        "Turn has no content hash (legacy data)"
                    );
                }

                Ok(Some((turn, content_text)))
            }
            None => Ok(None),
        }
    }

    /// Parse a turn from a database row.
    fn parse_turn_row(row: &sqlx::postgres::PgRow) -> Result<TurnSnapshot, sqlx::Error> {
        let id: Uuid = row.try_get("id")?;
        // Use conversation_id as session identifier (session_id column doesn't exist)
        let conversation_id: Option<Uuid> = row.try_get("conversation_id")?;
        let role_str: Option<String> = row.try_get("role")?;
        // Use 'phase' column, not 'trajectory_phase'
        let phase_str: Option<String> = row.try_get("phase")?;
        let salience: Option<f64> = row.try_get("salience_score")?;
        let depth: Option<i32> = row.try_get("trajectory_depth")?;
        let sibling_order: Option<i32> = row.try_get("trajectory_sibling_order")?;
        let homogeneity: Option<f64> = row.try_get("trajectory_homogeneity")?;
        let temporal: Option<f64> = row.try_get("trajectory_temporal")?;
        // trajectory_complexity is an integer in the schema
        let complexity: Option<i32> = row.try_get("trajectory_complexity")?;
        let created_at: chrono::DateTime<chrono::Utc> = row.try_get("created_at")?;
        // Get content_hash for graph snapshot computation
        let content_hash: Option<String> = row.try_get("content_hash")?;

        Ok(TurnSnapshot::new(
            TurnId::new(id),
            conversation_id.map(|u| u.to_string()).unwrap_or_default(),
            role_str.and_then(|s| Role::from_str(&s)).unwrap_or_default(),
            phase_str.and_then(|s| Phase::from_str(&s)).unwrap_or_default(),
            salience.unwrap_or(0.5) as f32,
            depth.unwrap_or(0) as u32,
            sibling_order.unwrap_or(0) as u32,
            homogeneity.unwrap_or(0.5) as f32,
            temporal.unwrap_or(0.5) as f32,
            complexity.unwrap_or(1) as f32,
            created_at.timestamp(),
        ).with_content_hash(content_hash))
    }
}

/// Pool statistics for monitoring.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PoolStats {
    /// Current pool size.
    pub size: u32,
    /// Number of idle connections.
    pub idle: usize,
    /// Maximum pool size.
    pub max: u32,
}

/// Error type for PostgreSQL store.
#[derive(Debug, thiserror::Error)]
pub enum PostgresError {
    /// Database error.
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    /// Content hash verification failed.
    #[error("Content hash verification failed: {0}")]
    ContentHashMismatch(#[from] crate::types::ContentHashError),
}

impl GraphStore for PostgresGraphStore {
    type Error = PostgresError;

    async fn get_turn(&self, id: &TurnId) -> Result<Option<TurnSnapshot>, Self::Error> {
        let row = sqlx::query(
            r#"
            SELECT id, conversation_id, role, phase, salience_score,
                   trajectory_depth, trajectory_sibling_order, trajectory_homogeneity,
                   trajectory_temporal, trajectory_complexity, created_at, content_hash
            FROM memory_turns
            WHERE id = $1
            "#
        )
        .bind(id.as_uuid())
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(ref r) => Ok(Some(Self::parse_turn_row(r)?)),
            None => Ok(None),
        }
    }

    async fn get_turns(&self, ids: &[TurnId]) -> Result<Vec<TurnSnapshot>, Self::Error> {
        let uuids: Vec<Uuid> = ids.iter().map(|id| id.as_uuid()).collect();
        let rows = sqlx::query(
            r#"
            SELECT id, conversation_id, role, phase, salience_score,
                   trajectory_depth, trajectory_sibling_order, trajectory_homogeneity,
                   trajectory_temporal, trajectory_complexity, created_at, content_hash
            FROM memory_turns
            WHERE id = ANY($1)
            ORDER BY id
            "#
        )
        .bind(&uuids)
        .fetch_all(&self.pool)
        .await?;

        rows.iter()
            .map(Self::parse_turn_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(PostgresError::from)
    }

    async fn get_parents(&self, id: &TurnId) -> Result<Vec<TurnId>, Self::Error> {
        let rows = sqlx::query(
            r#"
            SELECT parent_turn_id
            FROM memory_turn_edges
            WHERE child_turn_id = $1
            ORDER BY parent_turn_id
            "#
        )
        .bind(id.as_uuid())
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter()
            .map(|row| TurnId::new(row.get("parent_turn_id")))
            .collect())
    }

    async fn get_children(&self, id: &TurnId) -> Result<Vec<TurnId>, Self::Error> {
        let rows = sqlx::query(
            r#"
            SELECT child_turn_id
            FROM memory_turn_edges
            WHERE parent_turn_id = $1
            ORDER BY child_turn_id
            "#
        )
        .bind(id.as_uuid())
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter()
            .map(|row| TurnId::new(row.get("child_turn_id")))
            .collect())
    }

    async fn get_siblings(&self, id: &TurnId, limit: usize) -> Result<Vec<TurnId>, Self::Error> {
        let rows = sqlx::query(
            r#"
            SELECT mt.id, mt.salience_score
            FROM memory_turns mt
            JOIN memory_turn_edges e ON e.child_turn_id = mt.id
            WHERE e.parent_turn_id IN (
                SELECT parent_turn_id FROM memory_turn_edges WHERE child_turn_id = $1
            )
            AND mt.id != $1
            ORDER BY mt.salience_score DESC, mt.id
            LIMIT $2
            "#
        )
        .bind(id.as_uuid())
        .bind(limit as i32)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter()
            .map(|row| TurnId::new(row.get("id")))
            .collect())
    }

    async fn get_edges(&self, turn_ids: &[TurnId]) -> Result<Vec<Edge>, Self::Error> {
        let uuids: Vec<Uuid> = turn_ids.iter().map(|id| id.as_uuid()).collect();
        let rows = sqlx::query(
            r#"
            SELECT parent_turn_id, child_turn_id, edge_type
            FROM memory_turn_edges
            WHERE parent_turn_id = ANY($1) AND child_turn_id = ANY($1)
            ORDER BY parent_turn_id, child_turn_id
            "#
        )
        .bind(&uuids)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter()
            .map(|row| {
                let parent: Uuid = row.get("parent_turn_id");
                let child: Uuid = row.get("child_turn_id");
                let edge_type_str: Option<String> = row.get("edge_type");
                
                Edge::new(
                    TurnId::new(parent),
                    TurnId::new(child),
                    edge_type_str
                        .and_then(|s| EdgeType::from_str(&s))
                        .unwrap_or_default(),
                )
            })
            .collect())
    }
}

