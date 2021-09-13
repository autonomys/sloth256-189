use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use rayon::prelude::*;
use std::time::{Duration, Instant};

fn random_bytes<const BYTES: usize>() -> [u8; BYTES] {
    let mut bytes = [0u8; BYTES];
    rand::thread_rng().fill(&mut bytes[..]);
    bytes
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sloth256-189");
    group.sample_size(500);
    group.measurement_time(Duration::from_secs(30));

    let genesis_piece = random_bytes::<4096>();
    let expanded_iv = random_bytes::<32>();

    group.bench_with_input("Encode-single", &genesis_piece, |b, &input| {
        b.iter(|| {
            let mut piece = input;
            sloth256_189::encode(&mut piece, expanded_iv, 1).unwrap();
        })
    });

    group.bench_with_input("Encode-parallel", &genesis_piece, |b, &input| {
        b.iter_custom(|iters| {
            let start = Instant::now();

            black_box((0..iters).into_par_iter().for_each(|_i| {
                let mut piece = input;
                sloth256_189::encode(&mut piece, expanded_iv, 1).unwrap();
            }));

            start.elapsed()
        })
    });

    let mut encoding = genesis_piece;
    sloth256_189::encode(&mut encoding, expanded_iv, 1).unwrap();

    group.bench_with_input("Decode", &encoding, |b, &input| {
        b.iter(|| {
            let mut piece = input;
            sloth256_189::decode(&mut piece, expanded_iv, 1);
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
