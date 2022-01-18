use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::Rng;
use std::time::Duration;
use num_format::{Locale, ToFormattedString};

fn random_bytes<const BYTES: usize>() -> Vec<u8> {
    let mut bytes = vec![0u8; BYTES];
    rand::thread_rng().fill(&mut bytes[..]);
    bytes
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(30));

    const NUM_PIECES: usize = 1024 * 1024;
    let size = NUM_PIECES * 4096 / 1073741824;
    let big_piece = random_bytes::<{4096 * NUM_PIECES}>();
    let expanded_ivs = random_bytes::<{32 * NUM_PIECES}>();

    /*group.bench_with_input("Encode-parallel", &big_piece, |b, input| {
        b.iter(|| {
            let mut piece = input.clone();
            sloth256_189::cuda::encode(&mut piece, &expanded_ivs, 1).unwrap();
        })
    });*/

    group.bench_function(format!("Encode-parallel/{} GB/{} pieces", size, NUM_PIECES.to_formatted_string(&Locale::en)), |b| {
        b.iter_batched_ref(
            || big_piece.clone(),
            |mut piece| sloth256_189::cuda::encode(&mut piece, &expanded_ivs, 1),
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
