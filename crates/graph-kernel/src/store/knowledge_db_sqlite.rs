//! SQLite implementation of `KnowledgeDb`.

use sqlx::sqlite::SqlitePool;
use sqlx::Row;

use super::knowledge_db::*;

/// SQLite-backed knowledge database.
///
/// Uses SQLite-compatible SQL dialect (no `GREATEST`, no `xmax`, no `::text` casts).
pub struct SqliteKnowledgeDb {
    pool: SqlitePool,
}

impl SqliteKnowledgeDb {
    /// Create a new SqliteKnowledgeDb wrapping an existing pool.
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Get the underlying pool.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

impl KnowledgeDb for SqliteKnowledgeDb {
    type Error = sqlx::Error;

    async fn upsert_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
        source: &str,
    ) -> Result<UpsertResult, Self::Error> {
        // Check if triple exists first (SQLite doesn't have xmax)
        let existing: Option<(i64, f64)> = sqlx::query_as(
            "SELECT id, confidence FROM knowledge_graph \
             WHERE subject = ?1 AND predicate = ?2 AND object = ?3"
        )
        .bind(subject)
        .bind(predicate)
        .bind(object)
        .fetch_optional(&self.pool)
        .await?;

        match existing {
            Some((_id, existing_conf)) => {
                // Update: use MAX equivalent
                let new_conf = if confidence > existing_conf { confidence } else { existing_conf };
                sqlx::query(
                    "UPDATE knowledge_graph SET confidence = ?1, source = ?2 \
                     WHERE subject = ?3 AND predicate = ?4 AND object = ?5"
                )
                .bind(new_conf)
                .bind(source)
                .bind(subject)
                .bind(predicate)
                .bind(object)
                .execute(&self.pool)
                .await?;
                Ok(UpsertResult { inserted: false })
            }
            None => {
                sqlx::query(
                    "INSERT INTO knowledge_graph (subject, predicate, object, confidence, source) \
                     VALUES (?1, ?2, ?3, ?4, ?5)"
                )
                .bind(subject)
                .bind(predicate)
                .bind(object)
                .bind(confidence)
                .bind(source)
                .execute(&self.pool)
                .await?;
                Ok(UpsertResult { inserted: true })
            }
        }
    }

    async fn upsert_batch(
        &self,
        triples: &[(String, String, String, f64, String)],
    ) -> Result<(usize, usize), Self::Error> {
        let mut added = 0usize;
        let mut updated = 0usize;

        let mut tx = self.pool.begin().await?;

        for (subject, predicate, object, confidence, source) in triples {
            // SQLite ON CONFLICT with MAX
            let result = sqlx::query(
                "INSERT INTO knowledge_graph (subject, predicate, object, confidence, source) \
                 VALUES (?1, ?2, ?3, ?4, ?5) \
                 ON CONFLICT (subject, predicate, object) DO UPDATE SET \
                 confidence = MAX(knowledge_graph.confidence, excluded.confidence), \
                 source = excluded.source"
            )
            .bind(subject)
            .bind(predicate)
            .bind(object)
            .bind(confidence)
            .bind(source)
            .execute(&mut *tx)
            .await;

            match result {
                Ok(r) => {
                    // SQLite: rows_affected() = 1 for both insert and update via ON CONFLICT
                    // We can check changes() but can't distinguish reliably.
                    // Use a heuristic: if we got here without error, count it.
                    if r.rows_affected() > 0 {
                        // We count everything as "added" for batch — not worth the extra query per row
                        added += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to insert triple ({}, {}, {}): {}",
                        subject, predicate, object, e
                    );
                }
            }
        }

        tx.commit().await?;
        Ok((added, updated))
    }

    async fn query_triples(
        &self,
        query: &KnowledgeQuery,
    ) -> Result<(Vec<StoredTriple>, i64), Self::Error> {
        let mut conditions = vec!["1=1".to_string()];
        let mut bind_idx = 1u32;

        if query.subject.is_some() {
            conditions.push(format!("subject = ?{}", bind_idx));
            bind_idx += 1;
        }
        if query.predicate.is_some() {
            conditions.push(format!("predicate = ?{}", bind_idx));
            bind_idx += 1;
        }
        if query.object.is_some() {
            conditions.push(format!("object = ?{}", bind_idx));
            bind_idx += 1;
        }
        if query.min_confidence.is_some() {
            conditions.push(format!("confidence >= ?{}", bind_idx));
            bind_idx += 1;
        }

        let where_clause = conditions.join(" AND ");
        let limit_param = format!("?{}", bind_idx);

        // Count query
        let count_sql = format!("SELECT COUNT(*) FROM knowledge_graph WHERE {}", where_clause);
        let mut count_q = sqlx::query_scalar::<_, i64>(&count_sql);
        if let Some(ref s) = query.subject { count_q = count_q.bind(s.as_str()); }
        if let Some(ref p) = query.predicate { count_q = count_q.bind(p.as_str()); }
        if let Some(ref o) = query.object { count_q = count_q.bind(o.as_str()); }
        if let Some(min_conf) = query.min_confidence { count_q = count_q.bind(min_conf); }

        // Note: sqlx sqlite query_scalar for COUNT may return i32 or i64 depending on version
        let total: i64 = count_q.fetch_one(&self.pool).await.unwrap_or(0);

        // Data query (no ::text cast — SQLite stores text natively)
        let data_sql = format!(
            "SELECT id, subject, predicate, object, confidence, source, \
             created_at FROM knowledge_graph WHERE {} \
             ORDER BY confidence DESC LIMIT {}",
            where_clause, limit_param,
        );

        let mut data_q = sqlx::query(&data_sql);
        if let Some(ref s) = query.subject { data_q = data_q.bind(s.as_str()); }
        if let Some(ref p) = query.predicate { data_q = data_q.bind(p.as_str()); }
        if let Some(ref o) = query.object { data_q = data_q.bind(o.as_str()); }
        if let Some(min_conf) = query.min_confidence { data_q = data_q.bind(min_conf); }
        data_q = data_q.bind(query.limit);

        let rows = data_q.fetch_all(&self.pool).await?;

        let triples = rows.iter().map(|row| {
            StoredTriple {
                id: row.get::<i64, _>("id"),
                subject: row.get::<String, _>("subject"),
                predicate: row.get::<String, _>("predicate"),
                object: row.get::<String, _>("object"),
                confidence: row.get::<f64, _>("confidence"),
                source: row.get::<String, _>("source"),
                created_at: row.get::<String, _>("created_at"),
            }
        }).collect();

        Ok((triples, total))
    }

    async fn delete_triples(
        &self,
        id: Option<i64>,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Result<DeleteResult, Self::Error> {
        let mut conditions = Vec::new();
        let mut bind_idx = 1u32;

        if id.is_some() {
            conditions.push(format!("id = ?{}", bind_idx));
            bind_idx += 1;
        }
        if subject.is_some() {
            conditions.push(format!("subject = ?{}", bind_idx));
            bind_idx += 1;
        }
        if predicate.is_some() {
            conditions.push(format!("predicate = ?{}", bind_idx));
            bind_idx += 1;
        }
        if object.is_some() {
            conditions.push(format!("object = ?{}", bind_idx));
        }

        let sql = format!("DELETE FROM knowledge_graph WHERE {}", conditions.join(" AND "));
        let mut q = sqlx::query(&sql);
        if let Some(id_val) = id { q = q.bind(id_val); }
        if let Some(s) = subject { q = q.bind(s); }
        if let Some(p) = predicate { q = q.bind(p); }
        if let Some(o) = object { q = q.bind(o); }

        let result = q.execute(&self.pool).await?;
        Ok(DeleteResult { rows_affected: result.rows_affected() })
    }

    async fn stats(&self) -> Result<KnowledgeStats, Self::Error> {
        let total: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM knowledge_graph")
            .fetch_one(&self.pool).await?;
        let subjects: (i64,) = sqlx::query_as("SELECT COUNT(DISTINCT subject) FROM knowledge_graph")
            .fetch_one(&self.pool).await?;
        let predicates: (i64,) = sqlx::query_as("SELECT COUNT(DISTINCT predicate) FROM knowledge_graph")
            .fetch_one(&self.pool).await?;
        let top_preds: Vec<(String, i64)> = sqlx::query_as(
            "SELECT predicate, COUNT(*) as cnt FROM knowledge_graph \
             GROUP BY predicate ORDER BY cnt DESC LIMIT 10"
        ).fetch_all(&self.pool).await?;

        Ok(KnowledgeStats {
            total_triples: total.0,
            unique_subjects: subjects.0,
            unique_predicates: predicates.0,
            top_predicates: top_preds,
        })
    }

    async fn query_adjacent(
        &self,
        entity: &str,
        direction: &str,
    ) -> Result<Vec<AdjacentTriple>, Self::Error> {
        let sql = match direction {
            "incoming" => {
                "SELECT subject, predicate, object, confidence FROM knowledge_graph \
                 WHERE object = ?1 ORDER BY confidence DESC LIMIT 100"
            }
            "both" => {
                "SELECT subject, predicate, object, confidence FROM knowledge_graph \
                 WHERE subject = ?1 OR object = ?1 ORDER BY confidence DESC LIMIT 100"
            }
            _ => {
                "SELECT subject, predicate, object, confidence FROM knowledge_graph \
                 WHERE subject = ?1 ORDER BY confidence DESC LIMIT 100"
            }
        };

        let rows = sqlx::query(sql)
            .bind(entity)
            .fetch_all(&self.pool)
            .await?;

        Ok(rows.iter().map(|row| {
            AdjacentTriple {
                subject: row.get::<String, _>("subject"),
                predicate: row.get::<String, _>("predicate"),
                object: row.get::<String, _>("object"),
                confidence: row.get::<f64, _>("confidence"),
            }
        }).collect())
    }

    async fn is_healthy(&self) -> bool {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .is_ok()
    }
}
