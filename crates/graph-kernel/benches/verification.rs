//! Performance benchmarks for token verification.
//!
//! Run with: `cargo bench --bench verification`
//!
//! ## Performance Targets
//!
//! | Operation | Target | Notes |
//! |-----------|--------|-------|
//! | Cold verification | <50ms p99 | Full HMAC computation |
//! | Cached verification | <5ms p99 | LRU cache hit |
//! | Cache contention | Linear scaling | Multi-threaded access |

use criterion::{
    black_box, criterion_group, criterion_main,
    BenchmarkId, Criterion, Throughput,
};
use std::sync::Arc;
use std::thread;

use cc_graph_kernel::{
    TokenVerifier, VerificationMode, CacheConfig,
    SliceExport, TurnSnapshot, TurnId, Role, Phase, GraphSnapshotHash,
};
use uuid::Uuid;

/// Create a test turn snapshot.
fn make_turn(id: u128) -> TurnSnapshot {
    TurnSnapshot::new(
        TurnId::new(Uuid::from_u128(id)),
        "session_bench".to_string(),
        Role::User,
        Phase::Synthesis,
        0.8,
        1,
        0,
        0.5,
        0.5,
        1.0,
        1000,
    )
}

/// Create a test slice with valid HMAC token.
fn make_slice(secret: &[u8], turn_count: usize) -> SliceExport {
    let anchor = TurnId::new(Uuid::from_u128(1));
    let turns: Vec<_> = (1..=turn_count as u128).map(make_turn).collect();
    let snapshot = GraphSnapshotHash::new("bench_snapshot".to_string());

    SliceExport::new_with_secret(
        secret,
        anchor,
        turns,
        vec![],
        "bench_policy".to_string(),
        "bench_params_hash".to_string(),
        snapshot,
    )
}

/// Benchmark cold verification (no cache).
fn bench_cold_verification(c: &mut Criterion) {
    let secret = b"benchmark_secret_32_bytes_min___";
    let verifier = TokenVerifier::new(VerificationMode::local_secret(secret.to_vec()));

    let mut group = c.benchmark_group("cold_verification");

    for turn_count in [1, 10, 50, 100] {
        let slice = make_slice(secret, turn_count);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("turns", turn_count),
            &slice,
            |b, slice| {
                b.iter(|| {
                    let result = verifier.verify_slice(black_box(slice));
                    assert!(result.is_valid);
                    result
                })
            },
        );
    }

    group.finish();
}

/// Benchmark cached verification (cache hit).
fn bench_cached_verification(c: &mut Criterion) {
    let secret = b"benchmark_secret_32_bytes_min___";
    let config = CacheConfig {
        max_entries: 10_000,
        enabled: true,
    };
    let verifier = TokenVerifier::new(VerificationMode::cached_with_config(
        secret.to_vec(),
        config,
    ));

    let mut group = c.benchmark_group("cached_verification");

    for turn_count in [1, 10, 50, 100] {
        let slice = make_slice(secret, turn_count);

        // Warm the cache
        let warmup = verifier.verify_slice(&slice);
        assert!(warmup.is_valid);
        assert!(!warmup.cache_hit);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("turns", turn_count),
            &slice,
            |b, slice| {
                b.iter(|| {
                    let result = verifier.verify_slice(black_box(slice));
                    assert!(result.is_valid);
                    assert!(result.cache_hit);
                    result
                })
            },
        );
    }

    group.finish();
}

/// Benchmark cache miss scenario (new entries).
fn bench_cache_miss(c: &mut Criterion) {
    let secret = b"benchmark_secret_32_bytes_min___";
    let config = CacheConfig {
        max_entries: 100_000,
        enabled: true,
    };
    let verifier = TokenVerifier::new(VerificationMode::cached_with_config(
        secret.to_vec(),
        config,
    ));

    // Pre-generate many unique slices
    let slices: Vec<_> = (0..1000)
        .map(|i| {
            let anchor = TurnId::new(Uuid::from_u128(i + 1000));
            let turns = vec![TurnSnapshot::new(
                TurnId::new(Uuid::from_u128(i + 1000)),
                format!("session_{}", i),
                Role::User,
                Phase::Exploration,
                0.5,
                1,
                0,
                0.5,
                0.5,
                1.0,
                1000,
            )];
            let snapshot = GraphSnapshotHash::new(format!("snapshot_{}", i));
            SliceExport::new_with_secret(
                secret,
                anchor,
                turns,
                vec![],
                "bench_policy".to_string(),
                format!("params_{}", i),
                snapshot,
            )
        })
        .collect();

    c.bench_function("cache_miss", |b| {
        let mut idx = 0;
        b.iter(|| {
            let slice = &slices[idx % slices.len()];
            idx += 1;
            let result = verifier.verify_slice(black_box(slice));
            assert!(result.is_valid);
            result
        })
    });
}

/// Benchmark multi-threaded cache access.
fn bench_cache_contention(c: &mut Criterion) {
    let secret = b"benchmark_secret_32_bytes_min___";

    let mut group = c.benchmark_group("cache_contention");

    for num_threads in [1, 2, 4, 8] {
        let config = CacheConfig {
            max_entries: 10_000,
            enabled: true,
        };
        let verifier = Arc::new(TokenVerifier::new(VerificationMode::cached_with_config(
            secret.to_vec(),
            config,
        )));

        // Create slices for each thread
        let slices: Vec<_> = (0..num_threads)
            .map(|i| make_slice(secret, 10 + i))
            .collect();

        // Warm the cache
        for slice in &slices {
            verifier.verify_slice(slice);
        }

        group.throughput(Throughput::Elements(num_threads as u64));
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &num_threads,
            |b, &n| {
                b.iter(|| {
                    let handles: Vec<_> = (0..n)
                        .map(|i| {
                            let v = Arc::clone(&verifier);
                            let s = slices[i].clone();
                            thread::spawn(move || {
                                for _ in 0..100 {
                                    let result = v.verify_slice(black_box(&s));
                                    assert!(result.is_valid);
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark HMAC computation overhead (without cache).
fn bench_hmac_computation(c: &mut Criterion) {
    let secret = b"benchmark_secret_32_bytes_min___";

    let mut group = c.benchmark_group("hmac_computation");

    // Test with increasing payload sizes
    for payload_multiplier in [1, 10, 100] {
        let turn_count = payload_multiplier;
        let slice = make_slice(secret, turn_count);

        // Use local secret mode (no caching)
        let verifier = TokenVerifier::new(VerificationMode::local_secret(secret.to_vec()));

        group.throughput(Throughput::Bytes(
            (std::mem::size_of::<TurnSnapshot>() * turn_count) as u64
        ));
        group.bench_with_input(
            BenchmarkId::new("turns", turn_count),
            &slice,
            |b, slice| {
                b.iter(|| {
                    verifier.verify_slice(black_box(slice))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cold_verification,
    bench_cached_verification,
    bench_cache_miss,
    bench_cache_contention,
    bench_hmac_computation,
);
criterion_main!(benches);
