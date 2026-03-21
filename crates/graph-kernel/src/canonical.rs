//! Canonical serialization for deterministic hashing.
//!
//! This module provides functions to serialize data in a canonical, deterministic format
//! suitable for hashing and replay verification.
//!
//! ## Determinism Guarantees
//!
//! - Stable field order: Struct fields serialize in declaration order
//! - Stable Vec order: Vectors serialize in index order
//! - No HashMap allowed: Use BTreeMap for maps in hashed data
//! - Stable float format: f32/f64 serialize consistently

use serde::Serialize;
use xxhash_rust::xxh64::xxh64;

/// Serialize a value to canonical JSON bytes for hashing.
///
/// This function produces deterministic output for the same input,
/// suitable for hash computation and replay verification.
pub fn to_canonical_bytes<T: Serialize>(value: &T) -> Vec<u8> {
    serde_json::to_vec(value).expect("Canonical serialization failed")
}

/// Compute canonical hash of a serializable value.
pub fn canonical_hash<T: Serialize>(value: &T) -> u64 {
    let bytes = to_canonical_bytes(value);
    xxh64(&bytes, 0)
}

/// Compute canonical hash and return as hex string.
pub fn canonical_hash_hex<T: Serialize>(value: &T) -> String {
    format!("{:016x}", canonical_hash(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct TestStruct {
        name: String,
        value: i32,
    }

    #[test]
    fn test_determinism() {
        let s = TestStruct {
            name: "test".to_string(),
            value: 42,
        };

        let h1 = canonical_hash(&s);
        let h2 = canonical_hash(&s);
        assert_eq!(h1, h2);
    }
}

