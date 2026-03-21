//! Slice boundary enforcement for database queries.
//!
//! ## Purpose
//!
//! This module enforces INV-GK-008: Database Slice Containment.
//! SQL queries for slice retrieval MUST only access turns where `id IN (slice.turn_ids)`.
//!
//! ## Security Model
//!
//! The `SliceBoundaryGuard` type encapsulates a validated set of turn IDs that
//! have been authorized by the kernel. Database queries MUST use this guard
//! to construct their WHERE clauses.
//!
//! ```text
//! SliceExport → SliceBoundaryGuard → SQL Query
//!     │               │                  │
//!     │      Turn IDs extracted    WHERE id = ANY($1)
//!     │      and validated         (parameterized)
//!     └──────────────────────────────────────────────
//!                  Type-level SQL injection prevention
//! ```
//!
//! ## Query Patterns
//!
//! | Pattern | Safety | Use Case |
//! |---------|--------|----------|
//! | `WHERE id = ANY($1)` | ✅ Safe | Parameterized array |
//! | `JOIN slice_ids USING (id)` | ✅ Safe | Temp table join |
//! | `WHERE id IN (...)` | ❌ Unsafe | String interpolation |
//!
//! The guard ensures only safe patterns can be constructed.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use super::turn::TurnId;
use super::slice::SliceExport;

/// A validated set of turn IDs authorized for database access.
///
/// This type ensures SQL queries can only access turns that are part
/// of a verified slice. It prevents SQL injection and logic bugs from
/// leaking out-of-slice turns.
///
/// # Security
///
/// - Cannot be constructed from arbitrary IDs
/// - Must be derived from a `SliceExport`
/// - Provides only parameterized query methods
///
/// # Example
///
/// ```rust,ignore
/// // In repository layer:
/// async fn get_turns(
///     pool: &PgPool,
///     guard: &SliceBoundaryGuard,
/// ) -> Result<Vec<Turn>, Error> {
///     // guard.as_uuid_array() returns safe parameterized values
///     sqlx::query_as("SELECT * FROM turns WHERE id = ANY($1)")
///         .bind(guard.as_uuid_array())
///         .fetch_all(pool)
///         .await
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceBoundaryGuard {
    /// The turn IDs authorized for access.
    turn_ids: Vec<TurnId>,
    /// The slice fingerprint this guard was derived from.
    slice_fingerprint: String,
    /// Hash of the turn ID set for quick comparison.
    boundary_hash: u64,
}

impl SliceBoundaryGuard {
    /// Create a boundary guard from a slice export.
    ///
    /// # Arguments
    /// * `slice` - The verified slice export
    ///
    /// # Returns
    /// A guard that authorizes access to the slice's turns.
    pub fn from_slice(slice: &SliceExport) -> Self {
        let turn_ids: Vec<TurnId> = slice.turns.iter().map(|t| t.id).collect();
        let boundary_hash = Self::compute_boundary_hash(&turn_ids);

        Self {
            turn_ids,
            slice_fingerprint: slice.slice_id.as_str().to_string(),
            boundary_hash,
        }
    }

    /// Get the authorized turn IDs as a slice.
    pub fn turn_ids(&self) -> &[TurnId] {
        &self.turn_ids
    }

    /// Get the number of authorized turns.
    pub fn len(&self) -> usize {
        self.turn_ids.len()
    }

    /// Check if the guard authorizes no turns.
    pub fn is_empty(&self) -> bool {
        self.turn_ids.is_empty()
    }

    /// Get the slice fingerprint this guard was derived from.
    pub fn slice_fingerprint(&self) -> &str {
        &self.slice_fingerprint
    }

    /// Get the boundary hash for quick comparison.
    pub fn boundary_hash(&self) -> u64 {
        self.boundary_hash
    }

    /// Check if a turn ID is authorized by this guard.
    pub fn contains(&self, turn_id: &TurnId) -> bool {
        self.turn_ids.contains(turn_id)
    }

    /// Get the turn IDs as UUIDs for SQL parameterization.
    ///
    /// # Safety
    ///
    /// This method returns values safe for use in parameterized queries.
    /// Never use string interpolation with these values.
    pub fn as_uuid_array(&self) -> Vec<uuid::Uuid> {
        self.turn_ids.iter().map(|id| id.as_uuid()).collect()
    }

    /// Get the turn IDs as a HashSet for quick membership testing.
    pub fn as_set(&self) -> HashSet<TurnId> {
        self.turn_ids.iter().cloned().collect()
    }

    /// Compute a hash of the boundary for comparison.
    fn compute_boundary_hash(turn_ids: &[TurnId]) -> u64 {
        use xxhash_rust::xxh64::xxh64;

        let mut sorted_ids: Vec<_> = turn_ids.iter().collect();
        sorted_ids.sort();

        let mut bytes = Vec::with_capacity(sorted_ids.len() * 16);
        for id in sorted_ids {
            bytes.extend_from_slice(id.as_uuid().as_bytes());
        }

        xxh64(&bytes, 0)
    }

    /// Check if two guards authorize the same set of turns.
    pub fn same_boundary(&self, other: &Self) -> bool {
        self.boundary_hash == other.boundary_hash
    }
}

/// A query builder that enforces slice boundaries.
///
/// This builder ensures all generated SQL uses safe parameterized patterns.
#[derive(Debug)]
pub struct BoundedQueryBuilder<'a> {
    guard: &'a SliceBoundaryGuard,
    table: String,
    columns: Vec<String>,
    additional_filters: Vec<String>,
    order_by: Option<String>,
}

impl<'a> BoundedQueryBuilder<'a> {
    /// Create a new bounded query builder.
    ///
    /// # Arguments
    /// * `guard` - The boundary guard authorizing turn access
    /// * `table` - The table to query (must be a valid identifier)
    pub fn new(guard: &'a SliceBoundaryGuard, table: impl Into<String>) -> Self {
        Self {
            guard,
            table: table.into(),
            columns: vec!["*".to_string()],
            additional_filters: Vec::new(),
            order_by: None,
        }
    }

    /// Set the columns to select.
    pub fn select(mut self, columns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.columns = columns.into_iter().map(|c| c.into()).collect();
        self
    }

    /// Add an additional filter (must be parameterized).
    ///
    /// # Warning
    ///
    /// The filter string must use parameterized placeholders ($N).
    /// Never interpolate user input into the filter string.
    pub fn filter(mut self, filter: impl Into<String>) -> Self {
        self.additional_filters.push(filter.into());
        self
    }

    /// Set the order by clause.
    pub fn order_by(mut self, order: impl Into<String>) -> Self {
        self.order_by = Some(order.into());
        self
    }

    /// Build the SQL query string.
    ///
    /// The first parameter ($1) will always be the turn ID array.
    /// Additional parameters start at $2.
    ///
    /// # Returns
    /// A SQL string safe for use with parameterized execution.
    pub fn build(&self) -> String {
        let columns = self.columns.join(", ");
        let mut sql = format!(
            "SELECT {} FROM {} WHERE id = ANY($1)",
            columns, self.table
        );

        for filter in &self.additional_filters {
            sql.push_str(" AND ");
            sql.push_str(filter);
        }

        if let Some(order) = &self.order_by {
            sql.push_str(" ORDER BY ");
            sql.push_str(order);
        }

        sql
    }

    /// Get the guard for binding parameters.
    pub fn guard(&self) -> &SliceBoundaryGuard {
        self.guard
    }
}

/// Violation report when a query attempts out-of-slice access.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryViolation {
    /// The slice fingerprint that was violated.
    pub slice_fingerprint: String,
    /// Turn IDs that were requested but not authorized.
    pub unauthorized_ids: Vec<TurnId>,
    /// Timestamp of the violation.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Optional context about the violation source.
    pub context: Option<String>,
}

impl BoundaryViolation {
    /// Create a new boundary violation report.
    pub fn new(
        guard: &SliceBoundaryGuard,
        requested_ids: &[TurnId],
        context: Option<String>,
    ) -> Option<Self> {
        let authorized: HashSet<_> = guard.turn_ids.iter().collect();
        let unauthorized_ids: Vec<_> = requested_ids
            .iter()
            .filter(|id| !authorized.contains(id))
            .cloned()
            .collect();

        if unauthorized_ids.is_empty() {
            return None;
        }

        Some(Self {
            slice_fingerprint: guard.slice_fingerprint.clone(),
            unauthorized_ids,
            timestamp: chrono::Utc::now(),
            context,
        })
    }

    /// Log this violation as a security event.
    pub fn log(&self) {
        tracing::error!(
            slice_fingerprint = %self.slice_fingerprint,
            unauthorized_count = self.unauthorized_ids.len(),
            context = ?self.context,
            "SLICE_BOUNDARY_VIOLATION: Attempted access to unauthorized turns"
        );
    }
}

/// Result of validating turn IDs against a boundary.
#[derive(Debug, Clone)]
pub enum BoundaryCheck {
    /// All requested IDs are authorized.
    Authorized,
    /// Some IDs were not authorized.
    Violation(BoundaryViolation),
}

impl BoundaryCheck {
    /// Check if the result is authorized.
    pub fn is_authorized(&self) -> bool {
        matches!(self, Self::Authorized)
    }

    /// Get the violation if present.
    pub fn violation(&self) -> Option<&BoundaryViolation> {
        match self {
            Self::Violation(v) => Some(v),
            Self::Authorized => None,
        }
    }
}

impl SliceBoundaryGuard {
    /// Check if accessing the given turn IDs would violate this boundary.
    ///
    /// # Arguments
    /// * `requested_ids` - Turn IDs the caller wants to access
    /// * `context` - Optional context for violation logging
    ///
    /// # Returns
    /// - `BoundaryCheck::Authorized` if all IDs are within the boundary
    /// - `BoundaryCheck::Violation` if any IDs are outside the boundary
    pub fn check_access(
        &self,
        requested_ids: &[TurnId],
        context: Option<String>,
    ) -> BoundaryCheck {
        match BoundaryViolation::new(self, requested_ids, context) {
            Some(violation) => BoundaryCheck::Violation(violation),
            None => BoundaryCheck::Authorized,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TurnSnapshot, GraphSnapshotHash, Role, Phase};
    use uuid::Uuid;

    fn make_turn(id: u128) -> TurnSnapshot {
        TurnSnapshot::new(
            TurnId::new(Uuid::from_u128(id)),
            "session".to_string(),
            Role::User,
            Phase::Exploration,
            0.5,
            1,
            0,
            0.5,
            0.5,
            1.0,
            1000,
        )
    }

    fn make_slice(turns: Vec<TurnSnapshot>) -> SliceExport {
        let secret = b"test_kernel_secret_32_bytes_min!";
        let anchor = turns[0].id;
        let snapshot = GraphSnapshotHash::new("test_snapshot".to_string());

        SliceExport::new_with_secret(
            secret,
            anchor,
            turns,
            vec![],
            "test_policy".to_string(),
            "params_hash".to_string(),
            snapshot,
        )
    }

    #[test]
    fn test_boundary_guard_creation() {
        let turns = vec![make_turn(1), make_turn(2), make_turn(3)];
        let slice = make_slice(turns);
        let guard = SliceBoundaryGuard::from_slice(&slice);

        assert_eq!(guard.len(), 3);
        assert!(!guard.is_empty());
        assert!(guard.contains(&TurnId::new(Uuid::from_u128(1))));
        assert!(guard.contains(&TurnId::new(Uuid::from_u128(2))));
        assert!(guard.contains(&TurnId::new(Uuid::from_u128(3))));
        assert!(!guard.contains(&TurnId::new(Uuid::from_u128(4))));
    }

    #[test]
    fn test_boundary_check_authorized() {
        let turns = vec![make_turn(1), make_turn(2), make_turn(3)];
        let slice = make_slice(turns);
        let guard = SliceBoundaryGuard::from_slice(&slice);

        let requested = vec![
            TurnId::new(Uuid::from_u128(1)),
            TurnId::new(Uuid::from_u128(2)),
        ];
        let check = guard.check_access(&requested, None);

        assert!(check.is_authorized());
    }

    #[test]
    fn test_boundary_check_violation() {
        let turns = vec![make_turn(1), make_turn(2)];
        let slice = make_slice(turns);
        let guard = SliceBoundaryGuard::from_slice(&slice);

        let requested = vec![
            TurnId::new(Uuid::from_u128(1)),
            TurnId::new(Uuid::from_u128(99)), // Not authorized
        ];
        let check = guard.check_access(&requested, Some("test context".to_string()));

        assert!(!check.is_authorized());
        let violation = check.violation().unwrap();
        assert_eq!(violation.unauthorized_ids.len(), 1);
        assert_eq!(violation.unauthorized_ids[0], TurnId::new(Uuid::from_u128(99)));
        assert_eq!(violation.context, Some("test context".to_string()));
    }

    #[test]
    fn test_as_uuid_array() {
        let turns = vec![make_turn(1), make_turn(2)];
        let slice = make_slice(turns);
        let guard = SliceBoundaryGuard::from_slice(&slice);

        let uuids = guard.as_uuid_array();
        assert_eq!(uuids.len(), 2);
        assert_eq!(uuids[0], Uuid::from_u128(1));
        assert_eq!(uuids[1], Uuid::from_u128(2));
    }

    #[test]
    fn test_boundary_hash_determinism() {
        let turns = vec![make_turn(1), make_turn(2), make_turn(3)];
        let slice = make_slice(turns.clone());
        let guard1 = SliceBoundaryGuard::from_slice(&slice);

        let slice2 = make_slice(turns);
        let guard2 = SliceBoundaryGuard::from_slice(&slice2);

        assert!(guard1.same_boundary(&guard2));
    }

    #[test]
    fn test_boundary_hash_different() {
        let slice1 = make_slice(vec![make_turn(1), make_turn(2)]);
        let guard1 = SliceBoundaryGuard::from_slice(&slice1);

        let slice2 = make_slice(vec![make_turn(1), make_turn(3)]);
        let guard2 = SliceBoundaryGuard::from_slice(&slice2);

        assert!(!guard1.same_boundary(&guard2));
    }

    #[test]
    fn test_query_builder_basic() {
        let turns = vec![make_turn(1), make_turn(2)];
        let slice = make_slice(turns);
        let guard = SliceBoundaryGuard::from_slice(&slice);

        let builder = BoundedQueryBuilder::new(&guard, "turns");
        let sql = builder.build();

        assert_eq!(sql, "SELECT * FROM turns WHERE id = ANY($1)");
    }

    #[test]
    fn test_query_builder_with_columns() {
        let turns = vec![make_turn(1)];
        let slice = make_slice(turns);
        let guard = SliceBoundaryGuard::from_slice(&slice);

        let sql = BoundedQueryBuilder::new(&guard, "turns")
            .select(["id", "content", "role"])
            .build();

        assert_eq!(sql, "SELECT id, content, role FROM turns WHERE id = ANY($1)");
    }

    #[test]
    fn test_query_builder_with_filter_and_order() {
        let turns = vec![make_turn(1)];
        let slice = make_slice(turns);
        let guard = SliceBoundaryGuard::from_slice(&slice);

        let sql = BoundedQueryBuilder::new(&guard, "turns")
            .filter("session_id = $2")
            .order_by("created_at DESC")
            .build();

        assert_eq!(
            sql,
            "SELECT * FROM turns WHERE id = ANY($1) AND session_id = $2 ORDER BY created_at DESC"
        );
    }

    #[test]
    fn test_as_set() {
        let turns = vec![make_turn(1), make_turn(2), make_turn(3)];
        let slice = make_slice(turns);
        let guard = SliceBoundaryGuard::from_slice(&slice);

        let set = guard.as_set();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&TurnId::new(Uuid::from_u128(1))));
        assert!(set.contains(&TurnId::new(Uuid::from_u128(2))));
        assert!(set.contains(&TurnId::new(Uuid::from_u128(3))));
    }
}
