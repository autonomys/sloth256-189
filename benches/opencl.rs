use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use num_format::{Locale, ToFormattedString};
use rand::Rng;
use sloth256_189::opencl::{OpenClBatch, OpenClEncoder};
use std::time::Duration;

fn random_bytes<const BYTES: usize>() -> Vec<u8> {
    let mut bytes = vec![0u8; BYTES];
    rand::thread_rng().fill(&mut bytes[..]);
    bytes
}

// TODO: Pinned memory allocation is not exposed; check performance implications before adding back
// fn random_bytes_inplace<const BYTES: usize>(vec: &mut Vec<u8>) {
//     rand::thread_rng().fill(&mut vec[..]);
// }

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("opencl");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(30));

    const NUM_PIECES: usize = 1024 * 256;
    const MAX_LAYERS: usize = 1;
    let size: f64 = NUM_PIECES as f64 * 4096 as f64 / 1073741824 as f64;
    let expanded_ivs = random_bytes::<{ 32 * NUM_PIECES }>();

    let mut layers: usize = 1;
    while layers <= MAX_LAYERS {
        let batch = OpenClBatch {
            size: 4096 * NUM_PIECES,
            layers,
        };
        let mut encoder = OpenClEncoder::new(Some(batch)).unwrap();

        group.bench_function(
            format!(
                "Encode-parallel/{} GB/{} pieces/{} layer(s)",
                size,
                NUM_PIECES.to_formatted_string(&Locale::en),
                layers
            ),
            |b| {
                let piece = random_bytes::<{ 4096 * NUM_PIECES }>();

                b.iter_batched_ref(
                    || piece.clone(),
                    |mut input| encoder.encode(&mut input, &expanded_ivs, layers).unwrap(),
                    BatchSize::LargeInput,
                );
            },
        );

        // TODO: Pinned memory allocation is not exposed; check performance implications before adding back
        // group.bench_function(
        //     format!(
        //         "Encode-parallel-pinned/{} GB/{} pieces/{} layer(s)",
        //         size,
        //         NUM_PIECES.to_formatted_string(&Locale::en),
        //         layers
        //     ),
        //     |b| {
        //         b.iter_custom(|iters| {
        //             let mut piece =
        //                 sloth256_189::opencl::pinned_memory_alloc(instances, 4096 * NUM_PIECES)
        //                     .unwrap();
        //             random_bytes_inplace::<{ 4096 * NUM_PIECES }>(&mut piece);
        //
        //             let start = Instant::now();
        //             for _ in 0..iters {
        //                 sloth256_189::opencl::encode(&mut piece, &expanded_ivs, layers, instances)
        //                     .unwrap();
        //             }
        //             let elapsed = start.elapsed();
        //
        //             sloth256_189::opencl::pinned_memory_free(instances).unwrap();
        //             std::mem::forget(piece);
        //
        //             elapsed
        //         })
        //     },
        // );

        layers = layers * 2;
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
