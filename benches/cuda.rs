use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use std::time::Duration;

fn random_bytes<const BYTES: usize>() -> Vec<u8> {
    let mut bytes = vec![0u8; BYTES];
    rand::thread_rng().fill(&mut bytes[..]);
    bytes
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");
    group.sample_size(500);
    group.measurement_time(Duration::from_secs(30));

    let big_piece = random_bytes::<4194304>(); // 1024 * 4096
    let expanded_ivs = random_bytes::<32768>(); // 1024 * 32

    group.bench_with_input("Encode-parallel", &big_piece, |b, input| {
        b.iter(|| {
            let mut piece = input.clone();
            sloth256_189::cuda::encode(&mut piece, &expanded_ivs, 1).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
