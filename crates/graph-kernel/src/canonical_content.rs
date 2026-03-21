//! Canonical content transformation for deterministic hashing.
//!
//! ## Purpose
//!
//! This module defines the **canonical content specification** for turn content hashing.
//! Canonical content ensures that:
//!
//! 1. **Determinism**: Same text → same hash, regardless of encoding artifacts
//! 2. **Immutability**: Content changes → hash changes (INV-GK-004)
//! 3. **Replay**: Hashes computed at any time are comparable
//!
//! ## Canonical Content Specification
//!
//! The canonical form of turn content is computed as:
//!
//! ```text
//! canonical_content(text) = UTF-8(trim(normalize_newlines(text)))
//! ```
//!
//! Where:
//! - `normalize_newlines`: CRLF → LF, CR → LF
//! - `trim`: Remove leading and trailing whitespace
//! - `UTF-8`: Encode as UTF-8 bytes
//!
//! ## What Is NOT Included
//!
//! The following are **excluded** from canonical content:
//! - Role (user/assistant/system)
//! - Metadata (timestamps, session IDs)
//! - Turn ID
//! - Phase or salience scores
//!
//! Only the raw text content is hashed. This ensures content hashes are stable
//! across schema changes to metadata fields.
//!
//! ## Security Invariant
//!
//! This module enforces **INV-GK-004: Content Immutability**.
//! If `content_hash` exists, it MUST match `SHA256(canonical_content(content_text))`.

use sha2::{Sha256, Digest};

/// Version of the canonical content specification.
///
/// Increment this when the canonicalization algorithm changes.
/// Changes to this version invalidate all existing content hashes.
pub const CANONICAL_CONTENT_VERSION: &str = "1.0.0";

/// Normalize text to canonical form.
///
/// Transformations applied:
/// 1. Normalize newlines: CRLF → LF, isolated CR → LF
/// 2. Trim leading and trailing whitespace
///
/// # Arguments
/// * `text` - Raw content text
///
/// # Returns
/// Normalized text as a `String`
///
/// # Determinism
/// This function is deterministic: same input → same output.
///
/// # Example
///
/// ```rust
/// use cc_graph_kernel::canonical_content::normalize_text;
///
/// let text = "  Hello\r\nWorld  ";
/// let normalized = normalize_text(text);
/// assert_eq!(normalized, "Hello\nWorld");
/// ```
pub fn normalize_text(text: &str) -> String {
    // Step 1: Normalize newlines
    // CRLF → LF, isolated CR → LF
    let normalized = text
        .replace("\r\n", "\n")
        .replace('\r', "\n");

    // Step 2: Trim leading and trailing whitespace
    normalized.trim().to_string()
}

/// Convert text to canonical bytes for hashing.
///
/// This is the byte representation used for SHA-256 hashing.
///
/// # Arguments
/// * `text` - Raw content text
///
/// # Returns
/// Canonical UTF-8 bytes
///
/// # Determinism
/// This function is deterministic: same input → same output.
pub fn canonical_content(text: &str) -> Vec<u8> {
    normalize_text(text).into_bytes()
}

/// Compute SHA-256 content hash of canonical content.
///
/// This is the production method for computing content hashes.
/// The hash is returned as a lowercase hex string.
///
/// # Arguments
/// * `text` - Raw content text
///
/// # Returns
/// SHA-256 hash as 64-character lowercase hex string
///
/// # Security
///
/// This function enforces **INV-GK-004: Content Immutability**.
/// The hash uniquely identifies the canonical content.
///
/// # Example
///
/// ```rust
/// use cc_graph_kernel::canonical_content::compute_content_hash;
///
/// let hash = compute_content_hash("Hello World");
/// assert_eq!(hash.len(), 64); // SHA-256 = 32 bytes = 64 hex chars
/// ```
pub fn compute_content_hash(text: &str) -> String {
    let canonical = canonical_content(text);
    let mut hasher = Sha256::new();
    hasher.update(&canonical);
    let result = hasher.finalize();
    hex::encode(result)
}

/// Verify that a content hash matches the expected hash for given text.
///
/// Uses constant-time comparison to prevent timing attacks.
///
/// # Arguments
/// * `text` - Raw content text
/// * `expected_hash` - Expected SHA-256 hash (hex string)
///
/// # Returns
/// `true` if the computed hash matches the expected hash
pub fn verify_content_hash(text: &str, expected_hash: &str) -> bool {
    let computed = compute_content_hash(text);

    // Constant-time comparison
    if computed.len() != expected_hash.len() {
        return false;
    }

    computed.bytes()
        .zip(expected_hash.bytes())
        .fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
}

/// Content hash validation result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HashValidation {
    /// Hash matches canonical content.
    Valid,
    /// Hash does not match canonical content.
    Mismatch {
        /// The expected hash that was stored.
        expected: String,
        /// The hash computed from the current content.
        computed: String,
    },
    /// No hash was stored (backwards compatibility).
    Missing,
}

/// Validate a stored content hash against the canonical content.
///
/// # Arguments
/// * `text` - Raw content text
/// * `stored_hash` - Stored hash (may be `None` for old turns)
///
/// # Returns
/// Validation result indicating match, mismatch, or missing
pub fn validate_content_hash(text: &str, stored_hash: Option<&str>) -> HashValidation {
    match stored_hash {
        None => HashValidation::Missing,
        Some(expected) => {
            let computed = compute_content_hash(text);
            if verify_content_hash(text, expected) {
                HashValidation::Valid
            } else {
                HashValidation::Mismatch {
                    expected: expected.to_string(),
                    computed,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text_crlf() {
        let text = "Hello\r\nWorld";
        assert_eq!(normalize_text(text), "Hello\nWorld");
    }

    #[test]
    fn test_normalize_text_cr() {
        let text = "Hello\rWorld";
        assert_eq!(normalize_text(text), "Hello\nWorld");
    }

    #[test]
    fn test_normalize_text_trim() {
        let text = "  Hello World  \n";
        assert_eq!(normalize_text(text), "Hello World");
    }

    #[test]
    fn test_normalize_text_combined() {
        let text = "  Hello\r\nWorld\r  ";
        assert_eq!(normalize_text(text), "Hello\nWorld");
    }

    #[test]
    fn test_normalize_text_empty() {
        assert_eq!(normalize_text(""), "");
        assert_eq!(normalize_text("   "), "");
    }

    #[test]
    fn test_canonical_content_determinism() {
        let text = "Hello World";
        let bytes1 = canonical_content(text);
        let bytes2 = canonical_content(text);
        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn test_canonical_content_utf8() {
        let text = "Hello 世界 🌍";
        let bytes = canonical_content(text);
        assert_eq!(String::from_utf8(bytes.clone()).unwrap(), "Hello 世界 🌍");
    }

    #[test]
    fn test_compute_content_hash_determinism() {
        let text = "Hello World";
        let hash1 = compute_content_hash(text);
        let hash2 = compute_content_hash(text);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_compute_content_hash_format() {
        let hash = compute_content_hash("test");
        assert_eq!(hash.len(), 64); // SHA-256 = 64 hex chars
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_compute_content_hash_known_value() {
        // SHA-256("Hello World") after normalization
        let hash = compute_content_hash("Hello World");
        // Verify it's a valid SHA-256 hash
        assert_eq!(hash.len(), 64);

        // Verify consistency with expected SHA-256
        let expected = "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e";
        assert_eq!(hash, expected);
    }

    #[test]
    fn test_content_hash_newline_normalization() {
        // Same content with different newline styles should hash the same
        let text_lf = "Hello\nWorld";
        let text_crlf = "Hello\r\nWorld";
        let text_cr = "Hello\rWorld";

        let hash_lf = compute_content_hash(text_lf);
        let hash_crlf = compute_content_hash(text_crlf);
        let hash_cr = compute_content_hash(text_cr);

        assert_eq!(hash_lf, hash_crlf);
        assert_eq!(hash_lf, hash_cr);
    }

    #[test]
    fn test_content_hash_whitespace_normalization() {
        // Same content with different leading/trailing whitespace should hash the same
        let text1 = "Hello World";
        let text2 = "  Hello World  ";
        let text3 = "\n\nHello World\n\n";

        let hash1 = compute_content_hash(text1);
        let hash2 = compute_content_hash(text2);
        let hash3 = compute_content_hash(text3);

        assert_eq!(hash1, hash2);
        assert_eq!(hash1, hash3);
    }

    #[test]
    fn test_content_hash_different_content() {
        // Different content should have different hashes
        let hash1 = compute_content_hash("Hello");
        let hash2 = compute_content_hash("World");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_verify_content_hash_valid() {
        let text = "Hello World";
        let hash = compute_content_hash(text);
        assert!(verify_content_hash(text, &hash));
    }

    #[test]
    fn test_verify_content_hash_invalid() {
        let text = "Hello World";
        let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        assert!(!verify_content_hash(text, wrong_hash));
    }

    #[test]
    fn test_validate_content_hash_valid() {
        let text = "Hello World";
        let hash = compute_content_hash(text);
        assert_eq!(
            validate_content_hash(text, Some(&hash)),
            HashValidation::Valid
        );
    }

    #[test]
    fn test_validate_content_hash_mismatch() {
        let text = "Hello World";
        let wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000";
        match validate_content_hash(text, Some(wrong_hash)) {
            HashValidation::Mismatch { expected, computed } => {
                assert_eq!(expected, wrong_hash);
                assert_ne!(computed, wrong_hash);
            }
            _ => panic!("Expected Mismatch"),
        }
    }

    #[test]
    fn test_validate_content_hash_missing() {
        let text = "Hello World";
        assert_eq!(
            validate_content_hash(text, None),
            HashValidation::Missing
        );
    }

    #[test]
    fn test_unicode_content_hash() {
        // Test various Unicode content
        let texts = vec![
            "Hello 世界",
            "مرحبا بالعالم",
            "🎉🎊🎈",
            "Ñoño",
            "α + β = γ",
        ];

        for text in texts {
            let hash = compute_content_hash(text);
            assert_eq!(hash.len(), 64);
            assert!(verify_content_hash(text, &hash));
        }
    }

    #[test]
    fn test_empty_content_hash() {
        let hash = compute_content_hash("");
        // SHA-256 of empty string
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_whitespace_only_content_hash() {
        // Whitespace-only content normalizes to empty string
        let hash1 = compute_content_hash("   ");
        let hash2 = compute_content_hash("\n\n\n");
        let hash3 = compute_content_hash("\t\t");
        let empty_hash = compute_content_hash("");

        assert_eq!(hash1, empty_hash);
        assert_eq!(hash2, empty_hash);
        assert_eq!(hash3, empty_hash);
    }
}
