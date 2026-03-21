//! PostgreSQL implementation of `KnowledgeDb`.

use sqlx::postgres::PgPool;
use sqlx::Row;

use super::knowledge_db::*;

/// PostgreSQL-backed knowledge database.
pub struct PgKnowledgeDb {
    pool: PgPool,
}

impl PgKnowledgeDb {
    /// Create a new PgKnowledgeDb wrapping an existing pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Get the underlying pool (for health checks, etc.)
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

impl KnowledgeDb for PgKnowledgeDb {
    type Error = sqlx::Error;

    async fn upsert_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
        source: &str,
    ) -> Result<UpsertResult, Self::Error> {
        let result = sqlx::query(
            r#"
            INSERT INTO knowledge_graph (subject, predicate, object, confidence, source)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (subject, predicate, object) DO UPDATE SET
                confidence = GREATEST(knowledge_graph.confidence, EXCLUDED.confidence),
                source = EXCLUDED.source
            RETURNING (xmax = 0) AS inserted
            "#,
        )
        .bind(subject)
        .bind(predicate)
        .bind(object)
        .bind(confidence)
        .bind(source)
        .fetch_one(&self.pool)
        .await?;

        let inserted: bool = result.get("inserted");
        Ok(UpsertResult { inserted })
    }

    async fn upsert_batch(
        &self,
        triples: &[(String, String, String, f64, String)],
    ) -> Result<(usize, usize), Self::Error> {
        let mut added = 0usize;
        let mut updated = 0usize;

        let mut tx = self.pool.begin().await?;

        for (subject, predicate, object, confidence, source) in triples {
            let result = sqlx::query(
                r#"
                INSERT INTO knowledge_graph (subject, predicate, object, confidence, source)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (subject, predicate, object) DO UPDATE SET
                    confidence = GREATEST(knowledge_graph.confidence, EXCLUDED.confidence),
                    source = EXCLUDED.source
                RETURNING (xmax = 0) AS inserted
                "#,
            )
            .bind(subject)
            .bind(predicate)
            .bind(object)
            .bind(confidence)
            .bind(source)
            .fetch_one(&mut *tx)
            .await;

            match result {
                Ok(row) => {
                    let inserted: bool = row.get("inserted");
                    if inserted { added += 1; } else { updated += 1; }
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
            conditions.push(format!("subject = ${}", bind_idx));
            bind_idx += 1;
        }
        if query.predicate.is_some() {
            conditions.push(format!("predicate = ${}", bind_idx));
            bind_idx += 1;
        }
        if query.object.is_some() {
            conditions.push(format!("object = ${}", bind_idx));
            bind_idx += 1;
        }
        if query.min_confidence.is_some() {
            conditions.push(format!("confidence >= ${}", bind_idx));
            bind_idx += 1;
        }

        let where_clause = conditions.join(" AND ");
        let limit_param = format!("${}", bind_idx);

        // Count query
        let count_sql = format!("SELECT COUNT(*) as cnt FROM knowledge_graph WHERE {}", where_clause);
        let mut count_q = sqlx::query_scalar::<_, i64>(&count_sql);
        if let Some(ref s) = query.subject { count_q = count_q.bind(s.as_str()); }
        if let Some(ref p) = query.predicate { count_q = count_q.bind(p.as_str()); }
        if let Some(ref o) = query.object { count_q = count_q.bind(o.as_str()); }
        if let Some(min_conf) = query.min_confidence { count_q = count_q.bind(min_conf); }

        let total = count_q.fetch_one(&self.pool).await?;

        // Data query
        let data_sql = format!(
            "SELECT id, subject, predicate, object, confidence, source, \
             created_at::text as created_at FROM knowledge_graph WHERE {} \
             ORDER BY confidence DESC LIMIT {}",
            where_clause, limit_param,
        );

        let mut data_q = sqlx::query_as::<_, (i64, String, String, String, f64, String, String)>(&data_sql);
        if let Some(ref s) = query.subject { data_q = data_q.bind(s.as_str()); }
        if let Some(ref p) = query.predicate { data_q = data_q.bind(p.as_str()); }
        if let Some(ref o) = query.object { data_q = data_q.bind(o.as_str()); }
        if let Some(min_conf) = query.min_confidence { data_q = data_q.bind(min_conf); }
        data_q = data_q.bind(query.limit);

        let rows = data_q.fetch_all(&self.pool).await?;

        let triples = rows.into_iter().map(|(id, subject, predicate, object, confidence, source, created_at)| {
            StoredTriple { id, subject, predicate, object, confidence, source, created_at }
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
            conditions.push(format!("id = ${}", bind_idx));
            bind_idx += 1;
        }
        if subject.is_some() {
            conditions.push(format!("subject = ${}", bind_idx));
            bind_idx += 1;
        }
        if predicate.is_some() {
            conditions.push(format!("predicate = ${}", bind_idx));
            bind_idx += 1;
        }
        if object.is_some() {
            conditions.push(format!("object = ${}", bind_idx));
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
                 WHERE object = $1 ORDER BY confidence DESC LIMIT 100"
            }
            "both" => {
                "SELECT subject, predicate, object, confidence FROM knowledge_graph \
                 WHERE subject = $1 OR object = $1 ORDER BY confidence DESC LIMIT 100"
            }
            _ => {
                "SELECT subject, predicate, object, confidence FROM knowledge_graph \
                 WHERE subject = $1 ORDER BY confidence DESC LIMIT 100"
            }
        };

        let rows = sqlx::query_as::<_, (String, String, String, f64)>(sql)
            .bind(entity)
            .fetch_all(&self.pool)
            .await?;

        Ok(rows.into_iter().map(|(subject, predicate, object, confidence)| {
            AdjacentTriple { subject, predicate, object, confidence }
        }).collect())
    }

    async fn is_healthy(&self) -> bool {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .is_ok()
    }
}
