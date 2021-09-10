use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use rayon::prelude::*;
use std::time::{Duration, Instant};

fn random_bytes<const BYTES: usize>() -> Vec<u8> {
    let mut bytes = vec![0u8; BYTES];
    rand::thread_rng().fill(&mut bytes[..]);
    bytes
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sloth256-189-cuda");
    group.sample_size(500);
    group.measurement_time(Duration::from_secs(30));

    let genesis_piece = random_bytes::<4096>();
    let big_piece = random_bytes::<4194304>();
    let expanded_iv = random_bytes::<32>();
    let expanded_ivs = random_bytes::<32>();

    group.bench_with_input("Encode-single", &genesis_piece, |b, &input| {
        b.iter(|| {
            let mut piece = input;
            sloth256_189::cuda::cuda_test_single_piece(&mut piece, &expanded_iv, 1);
        })
    });

    group.bench_with_input("Encode-parallel", &big_piece, |b, &input| {
        b.iter(|| {
            let mut piece = input;
            sloth256_189::cuda::cuda_encode(&mut piece, &expanded_iv, 1);
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
