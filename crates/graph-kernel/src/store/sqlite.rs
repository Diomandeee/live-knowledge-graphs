//! SQLite graph store for local/edge deployment.
//!
//! Provides sub-10ms query latency by eliminating network RTT to remote PostgreSQL.
//! Uses WAL mode for concurrent read access and file-based persistence.
//!
//! ## Configuration
//!
//! Environment variables:
//! - `GK_SQLITE_PATH`: Path to SQLite database file (default: ./graph-kernel.db)
//! - `GK_SQLITE_MAX_CONNECTIONS`: Maximum pool size (default: 5)
//!
//! ## Features
//!
//! - WAL journal mode for concurrent reads
//! - In-memory mode for testing (`:memory:`)
//! - Same schema as PostgreSQL `knowledge_graph` table
//! - Compatible with `GraphStore` trait for turn/edge DAG operations

use sqlx::sqlite::{SqlitePool, SqlitePoolOptions, SqliteConnectOptions, SqliteJournalMode, SqliteSynchronous};
use sqlx::Row;
use std::str::FromStr;
use std::time::Duration;
use uuid::Uuid;

use crate::types::{TurnId, TurnSnapshot, Edge, EdgeType, Role, Phase};
use super::GraphStore;

/// Configuration for SQLite connection.
#[derive(Debug, Clone)]
pub struct SqliteConfig {
    /// Path to the SQLite database file (or `:memory:` for in-memory).
    pub database_path: String,
    /// Maximum connections in pool (default: 5).
    pub max_connections: u32,
    /// Whether to create the database file if it doesn't exist (default: true).
    pub create_if_missing: bool,
}

impl SqliteConfig {
    /// Load configuration from environment variables with sane defaults.
    pub fn from_env() -> Self {
        Self {
            database_path: std::env::var("GK_SQLITE_PATH")
                .unwrap_or_else(|_| "./graph-kernel.db".to_string()),
            max_connections: std::env::var("GK_SQLITE_MAX_CONNECTIONS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5),
            create_if_missing: true,
        }
    }

    /// Create a config for in-memory database (testing).
    pub fn in_memory() -> Self {
        Self {
            database_path: ":memory:".to_string(),
            max_connections: 1,
            create_if_missing: true,
        }
    }
}

impl Default for SqliteConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

/// SQLite-backed graph store.
///
/// Provides the same interface as `PostgresGraphStore` but with local SQLite storage.
/// Query latency drops from ~291ms (remote PG) to ~5-15ms (local SQLite).
pub struct SqliteGraphStore {
    pool: SqlitePool,
}

impl SqliteGraphStore {
    /// Create a new store with the given configuration.
    pub async fn new(config: SqliteConfig) -> Result<Self, sqlx::Error> {
        tracing::info!(
            database_path = %config.database_path,
            max_connections = config.max_connections,
            "Initializing SQLite connection pool"
        );

        let connect_options = SqliteConnectOptions::from_str(&format!("sqlite:{}", config.database_path))?
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal)
            .create_if_missing(config.create_if_missing)
            .busy_timeout(Duration::from_secs(5))
            // Performance pragmas applied on each connection
            .pragma("cache_size", "-20000")      // 20MB cache
            .pragma("temp_store", "memory")      // Temp tables in memory
            .pragma("mmap_size", "268435456");    // 256MB mmap

        let pool = SqlitePoolOptions::new()
            .max_connections(config.max_connections)
            .acquire_timeout(Duration::from_secs(10))
            .connect_with(connect_options)
            .await?;

        let store = Self { pool };

        // Run migrations to ensure schema exists
        store.run_migrations().await?;

        Ok(store)
    }

    /// Create a store from environment variables.
    pub async fn from_env() -> Result<Self, sqlx::Error> {
        Self::new(SqliteConfig::from_env()).await
    }

    /// Create an in-memory store for testing.
    pub async fn in_memory() -> Result<Self, sqlx::Error> {
        Self::new(SqliteConfig::in_memory()).await
    }

    /// Get the connection pool (for knowledge graph CRUD operations in route handlers).
    pub fn pool(&self) -> &SqlitePool {
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
    pub fn pool_stats(&self) -> SqlitePoolStats {
        SqlitePoolStats {
            size: self.pool.size(),
            idle: self.pool.num_idle(),
            max: self.pool.options().get_max_connections(),
        }
    }

    /// Run schema migrations.
    ///
    /// Creates the knowledge_graph, memory_turns, and memory_turn_edges tables
    /// if they don't already exist.
    async fn run_migrations(&self) -> Result<(), sqlx::Error> {
        tracing::info!("Running SQLite schema migrations...");

        // Knowledge graph table (mirrors PostgreSQL schema)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS knowledge_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                source TEXT DEFAULT 'unknown',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(subject, predicate, object)
            )"
        ).execute(&self.pool).await?;

        // Knowledge graph indexes
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_knowledge_subject ON knowledge_graph(subject)")
            .execute(&self.pool).await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_knowledge_predicate ON knowledge_graph(predicate)")
            .execute(&self.pool).await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_knowledge_object ON knowledge_graph(object)")
            .execute(&self.pool).await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_knowledge_confidence ON knowledge_graph(confidence DESC)")
            .execute(&self.pool).await?;

        // Memory turns table (for GraphStore trait - context slicer DAG)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS memory_turns (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                role TEXT,
                phase TEXT,
                salience_score REAL,
                trajectory_depth INTEGER,
                trajectory_sibling_order INTEGER,
                trajectory_homogeneity REAL,
                trajectory_temporal REAL,
                trajectory_complexity INTEGER,
                created_at TEXT NOT NULL,
                content_hash TEXT,
                content_text TEXT
            )"
        ).execute(&self.pool).await?;

        // Memory turn edges table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS memory_turn_edges (
                parent_turn_id TEXT NOT NULL,
                child_turn_id TEXT NOT NULL,
                edge_type TEXT,
                PRIMARY KEY (parent_turn_id, child_turn_id)
            )"
        ).execute(&self.pool).await?;

        // Turn edges indexes
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_edges_parent ON memory_turn_edges(parent_turn_id)")
            .execute(&self.pool).await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_edges_child ON memory_turn_edges(child_turn_id)")
            .execute(&self.pool).await?;

        // Sync metadata table (for Supabase sync tracking)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS _sync_metadata (
                table_name TEXT PRIMARY KEY,
                last_sync_at TEXT,
                last_full_sync_at TEXT,
                rows_synced INTEGER DEFAULT 0
            )"
        ).execute(&self.pool).await?;

        tracing::info!("SQLite schema migrations complete");
        Ok(())
    }

    /// Parse a turn from a SQLite row.
    fn parse_turn_row(row: &sqlx::sqlite::SqliteRow) -> Result<TurnSnapshot, sqlx::Error> {
        let id_str: String = row.try_get("id")?;
        let id = Uuid::parse_str(&id_str).map_err(|e| sqlx::Error::Decode(Box::new(e)))?;
        let conversation_id: Option<String> = row.try_get("conversation_id")?;
        let role_str: Option<String> = row.try_get("role")?;
        let phase_str: Option<String> = row.try_get("phase")?;
        let salience: Option<f64> = row.try_get("salience_score")?;
        let depth: Option<i32> = row.try_get("trajectory_depth")?;
        let sibling_order: Option<i32> = row.try_get("trajectory_sibling_order")?;
        let homogeneity: Option<f64> = row.try_get("trajectory_homogeneity")?;
        let temporal: Option<f64> = row.try_get("trajectory_temporal")?;
        let complexity: Option<i32> = row.try_get("trajectory_complexity")?;
        let created_at_str: String = row.try_get("created_at")?;
        let content_hash: Option<String> = row.try_get("content_hash")?;

        // Parse created_at as unix timestamp (try integer first, then ISO string)
        let created_at_ts = created_at_str.parse::<i64>().unwrap_or_else(|_| {
            chrono::DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.timestamp())
                .unwrap_or(0)
        });

        Ok(TurnSnapshot::new(
            TurnId::new(id),
            conversation_id.unwrap_or_default(),
            role_str.and_then(|s| Role::from_str(&s)).unwrap_or_default(),
            phase_str.and_then(|s| Phase::from_str(&s)).unwrap_or_default(),
            salience.unwrap_or(0.5) as f32,
            depth.unwrap_or(0) as u32,
            sibling_order.unwrap_or(0) as u32,
            homogeneity.unwrap_or(0.5) as f32,
            temporal.unwrap_or(0.5) as f32,
            complexity.unwrap_or(1) as f32,
            created_at_ts,
        ).with_content_hash(content_hash))
    }
}

/// Pool statistics for monitoring.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SqlitePoolStats {
    /// Current pool size.
    pub size: u32,
    /// Number of idle connections.
    pub idle: usize,
    /// Maximum pool size.
    pub max: u32,
}

/// Error type for SQLite store.
#[derive(Debug, thiserror::Error)]
pub enum SqliteError {
    /// Database error.
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    /// Content hash verification failed.
    #[error("Content hash verification failed: {0}")]
    ContentHashMismatch(#[from] crate::types::ContentHashError),
}

impl GraphStore for SqliteGraphStore {
    type Error = SqliteError;

    async fn get_turn(&self, id: &TurnId) -> Result<Option<TurnSnapshot>, Self::Error> {
        let row = sqlx::query(
            "SELECT id, conversation_id, role, phase, salience_score,
                    trajectory_depth, trajectory_sibling_order, trajectory_homogeneity,
                    trajectory_temporal, trajectory_complexity, created_at, content_hash
             FROM memory_turns
             WHERE id = ?1"
        )
        .bind(id.as_uuid().to_string())
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(ref r) => Ok(Some(Self::parse_turn_row(r)?)),
            None => Ok(None),
        }
    }

    async fn get_turns(&self, ids: &[TurnId]) -> Result<Vec<TurnSnapshot>, Self::Error> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        // SQLite doesn't support ANY($1), so build a WHERE IN clause
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{}", i)).collect();
        let query_str = format!(
            "SELECT id, conversation_id, role, phase, salience_score,
                    trajectory_depth, trajectory_sibling_order, trajectory_homogeneity,
                    trajectory_temporal, trajectory_complexity, created_at, content_hash
             FROM memory_turns
             WHERE id IN ({})
             ORDER BY id",
            placeholders.join(", ")
        );

        let mut query = sqlx::query(&query_str);
        for id in ids {
            query = query.bind(id.as_uuid().to_string());
        }

        let rows = query.fetch_all(&self.pool).await?;

        rows.iter()
            .map(Self::parse_turn_row)
            .collect::<Result<Vec<_>, _>>()
            .map_err(SqliteError::from)
    }

    async fn get_parents(&self, id: &TurnId) -> Result<Vec<TurnId>, Self::Error> {
        let rows = sqlx::query(
            "SELECT parent_turn_id
             FROM memory_turn_edges
             WHERE child_turn_id = ?1
             ORDER BY parent_turn_id"
        )
        .bind(id.as_uuid().to_string())
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter()
            .filter_map(|row| {
                let id_str: String = row.get("parent_turn_id");
                Uuid::parse_str(&id_str).ok().map(TurnId::new)
            })
            .collect())
    }

    async fn get_children(&self, id: &TurnId) -> Result<Vec<TurnId>, Self::Error> {
        let rows = sqlx::query(
            "SELECT child_turn_id
             FROM memory_turn_edges
             WHERE parent_turn_id = ?1
             ORDER BY child_turn_id"
        )
        .bind(id.as_uuid().to_string())
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter()
            .filter_map(|row| {
                let id_str: String = row.get("child_turn_id");
                Uuid::parse_str(&id_str).ok().map(TurnId::new)
            })
            .collect())
    }

    async fn get_siblings(&self, id: &TurnId, limit: usize) -> Result<Vec<TurnId>, Self::Error> {
        let rows = sqlx::query(
            "SELECT mt.id, mt.salience_score
             FROM memory_turns mt
             JOIN memory_turn_edges e ON e.child_turn_id = mt.id
             WHERE e.parent_turn_id IN (
                 SELECT parent_turn_id FROM memory_turn_edges WHERE child_turn_id = ?1
             )
             AND mt.id != ?1
             ORDER BY mt.salience_score DESC, mt.id
             LIMIT ?2"
        )
        .bind(id.as_uuid().to_string())
        .bind(limit as i32)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.iter()
            .filter_map(|row| {
                let id_str: String = row.get("id");
                Uuid::parse_str(&id_str).ok().map(TurnId::new)
            })
            .collect())
    }

    async fn get_edges(&self, turn_ids: &[TurnId]) -> Result<Vec<Edge>, Self::Error> {
        if turn_ids.is_empty() {
            return Ok(vec![]);
        }

        // Build WHERE IN clauses for both parent and child
        let placeholders: Vec<String> = (1..=turn_ids.len()).map(|i| format!("?{}", i)).collect();
        let in_clause = placeholders.join(", ");

        // We need the same IDs bound twice (for parent_turn_id and child_turn_id)
        let offset = turn_ids.len();
        let placeholders2: Vec<String> = (1..=turn_ids.len()).map(|i| format!("?{}", offset + i)).collect();
        let in_clause2 = placeholders2.join(", ");

        let query_str = format!(
            "SELECT parent_turn_id, child_turn_id, edge_type
             FROM memory_turn_edges
             WHERE parent_turn_id IN ({}) AND child_turn_id IN ({})
             ORDER BY parent_turn_id, child_turn_id",
            in_clause, in_clause2
        );

        let mut query = sqlx::query(&query_str);
        // Bind IDs twice — once for each IN clause
        for id in turn_ids {
            query = query.bind(id.as_uuid().to_string());
        }
        for id in turn_ids {
            query = query.bind(id.as_uuid().to_string());
        }

        let rows = query.fetch_all(&self.pool).await?;

        Ok(rows.iter()
            .filter_map(|row| {
                let parent_str: String = row.get("parent_turn_id");
                let child_str: String = row.get("child_turn_id");
                let edge_type_str: Option<String> = row.get("edge_type");

                let parent = Uuid::parse_str(&parent_str).ok()?;
                let child = Uuid::parse_str(&child_str).ok()?;

                Some(Edge::new(
                    TurnId::new(parent),
                    TurnId::new(child),
                    edge_type_str
                        .and_then(|s| EdgeType::from_str(&s))
                        .unwrap_or_default(),
                ))
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Role, Phase, EdgeType};

    fn make_turn(id: u128, salience: f32) -> TurnSnapshot {
        TurnSnapshot::new(
            TurnId::new(Uuid::from_u128(id)),
            "session_1".to_string(),
            Role::User,
            Phase::Consolidation,
            salience,
            1,
            0,
            0.5,
            0.5,
            1.0,
            1000,
        )
    }

    #[tokio::test]
    async fn test_sqlite_migrations() {
        let store = SqliteGraphStore::in_memory().await.unwrap();
        assert!(store.is_healthy().await);
    }

    #[tokio::test]
    async fn test_sqlite_knowledge_graph_crud() {
        let store = SqliteGraphStore::in_memory().await.unwrap();
        let pool = store.pool();

        // Insert a triple
        sqlx::query(
            "INSERT INTO knowledge_graph (subject, predicate, object, confidence, source)
             VALUES (?1, ?2, ?3, ?4, ?5)"
        )
        .bind("clawdbot")
        .bind("uses")
        .bind("graph-kernel")
        .bind(0.95)
        .bind("test")
        .execute(pool)
        .await
        .unwrap();

        // Query it back
        let row: (i64, String, String, String, f64) = sqlx::query_as(
            "SELECT id, subject, predicate, object, confidence FROM knowledge_graph WHERE subject = ?1"
        )
        .bind("clawdbot")
        .fetch_one(pool)
        .await
        .unwrap();

        assert_eq!(row.1, "clawdbot");
        assert_eq!(row.2, "uses");
        assert_eq!(row.3, "graph-kernel");
        assert!((row.4 - 0.95).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_sqlite_graphstore_turns() {
        let store = SqliteGraphStore::in_memory().await.unwrap();
        let pool = store.pool();

        let turn = make_turn(1, 0.8);
        let id = turn.id;

        // Insert turn directly
        sqlx::query(
            "INSERT INTO memory_turns (id, conversation_id, role, phase, salience_score,
             trajectory_depth, trajectory_sibling_order, trajectory_homogeneity,
             trajectory_temporal, trajectory_complexity, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)"
        )
        .bind(id.as_uuid().to_string())
        .bind("session_1")
        .bind("user")
        .bind("consolidation")
        .bind(0.8f64)
        .bind(1i32)
        .bind(0i32)
        .bind(0.5f64)
        .bind(0.5f64)
        .bind(1i32)
        .bind("1000")
        .execute(pool)
        .await
        .unwrap();

        // Retrieve via GraphStore trait
        let retrieved = store.get_turn(&id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, id);
    }

    #[tokio::test]
    async fn test_sqlite_graphstore_edges() {
        let store = SqliteGraphStore::in_memory().await.unwrap();
        let pool = store.pool();

        let t1 = make_turn(1, 0.5);
        let t2 = make_turn(2, 0.5);
        let id1 = t1.id;
        let id2 = t2.id;

        // Insert turns
        for (id, turn) in [(id1, &t1), (id2, &t2)] {
            sqlx::query(
                "INSERT INTO memory_turns (id, conversation_id, role, phase, salience_score,
                 trajectory_depth, trajectory_sibling_order, trajectory_homogeneity,
                 trajectory_temporal, trajectory_complexity, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)"
            )
            .bind(id.as_uuid().to_string())
            .bind("session_1")
            .bind("user")
            .bind("consolidation")
            .bind(turn.salience as f64)
            .bind(1i32)
            .bind(0i32)
            .bind(0.5f64)
            .bind(0.5f64)
            .bind(1i32)
            .bind("1000")
            .execute(pool)
            .await
            .unwrap();
        }

        // Insert edge
        sqlx::query(
            "INSERT INTO memory_turn_edges (parent_turn_id, child_turn_id, edge_type)
             VALUES (?1, ?2, ?3)"
        )
        .bind(id1.as_uuid().to_string())
        .bind(id2.as_uuid().to_string())
        .bind("reply")
        .execute(pool)
        .await
        .unwrap();

        // Test get_children
        let children = store.get_children(&id1).await.unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0], id2);

        // Test get_parents
        let parents = store.get_parents(&id2).await.unwrap();
        assert_eq!(parents.len(), 1);
        assert_eq!(parents[0], id1);

        // Test get_edges
        let edges = store.get_edges(&[id1, id2]).await.unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].parent, id1);
        assert_eq!(edges[0].child, id2);
    }

    #[tokio::test]
    async fn test_sqlite_upsert() {
        let store = SqliteGraphStore::in_memory().await.unwrap();
        let pool = store.pool();

        // Insert
        sqlx::query(
            "INSERT INTO knowledge_graph (subject, predicate, object, confidence, source)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT (subject, predicate, object) DO UPDATE SET
             confidence = MAX(knowledge_graph.confidence, excluded.confidence),
             source = excluded.source"
        )
        .bind("a")
        .bind("b")
        .bind("c")
        .bind(0.5)
        .bind("test1")
        .execute(pool)
        .await
        .unwrap();

        // Upsert with higher confidence
        sqlx::query(
            "INSERT INTO knowledge_graph (subject, predicate, object, confidence, source)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT (subject, predicate, object) DO UPDATE SET
             confidence = MAX(knowledge_graph.confidence, excluded.confidence),
             source = excluded.source"
        )
        .bind("a")
        .bind("b")
        .bind("c")
        .bind(0.9)
        .bind("test2")
        .execute(pool)
        .await
        .unwrap();

        let row: (f64, String) = sqlx::query_as(
            "SELECT confidence, source FROM knowledge_graph WHERE subject = ?1"
        )
        .bind("a")
        .fetch_one(pool)
        .await
        .unwrap();

        assert!((row.0 - 0.9).abs() < 0.001);
        assert_eq!(row.1, "test2");
    }
}
