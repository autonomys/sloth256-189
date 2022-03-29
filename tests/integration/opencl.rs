use crate::test_vectors::{CORRECT_ENCODING, EXPANDED_IV, LAYERS, PIECE};
use crate::utils;
use rand::Rng;
use sloth256_189::cpu;
use sloth256_189::opencl::{OpenClBatch, OpenClEncoder};
use std::convert::TryInto;

fn random_bytes_vec<const BYTES: usize>() -> Vec<u8> {
    let mut bytes = vec![0u8; BYTES];
    rand::thread_rng().fill(&mut bytes[..]);
    bytes
}

#[test]
fn test_random_piece() {
    let expanded_iv = utils::random_bytes::<32>();
    let piece = utils::random_bytes::<4096>();
    let layers = 8;

    let mut encodings = Vec::with_capacity(1024 * 4096);
    for _ in 0..1024 {
        encodings.extend_from_slice(&piece);
    }
    let mut ivs = Vec::with_capacity(32 * 4096);
    for _ in 0..1024 {
        ivs.extend_from_slice(&expanded_iv);
    }

    let batch = OpenClBatch {
        size: 4096 * 1024,
        layers,
    };
    let mut encoder = OpenClEncoder::new(Some(batch)).unwrap();

    encoder.encode(&mut encodings, &ivs, layers).unwrap();

    encoder.destroy().unwrap();

    // Verify wth CPU implementation as we don't have GPU-based decoding
    for encoding in encodings.chunks_exact(4096) {
        let mut decoding: [u8; 4096] = encoding.try_into().unwrap();
        cpu::decode(&mut decoding, &expanded_iv, layers).unwrap();

        assert_eq!(piece, decoding);
    }
}

#[test]
fn test_known_piece() {
    let mut encodings = Vec::with_capacity(1024 * 4096);
    for _ in 0..1024 {
        encodings.extend_from_slice(&PIECE);
    }
    let mut ivs = Vec::with_capacity(32 * 4096);
    for _ in 0..1024 {
        ivs.extend_from_slice(&EXPANDED_IV);
    }

    let mut encoder = OpenClEncoder::new(None).unwrap();

    encoder.encode(&mut encodings, &ivs, LAYERS).unwrap();

    encoder.destroy().unwrap();

    let mut correct_encodings = Vec::with_capacity(1024 * 4096);
    for _ in 0..1024 {
        correct_encodings.extend_from_slice(&CORRECT_ENCODING);
    }
    assert_eq!(encodings, correct_encodings);
}

#[test]
#[ignore]
fn test_big_random_piece() {
    const NUM_PIECES: usize = 1024 * 256;
    let layers = 2;

    let correct_encodings = random_bytes_vec::<{ 4096 * NUM_PIECES }>();
    let ivs = random_bytes_vec::<{ 32 * NUM_PIECES }>();

    let mut encodings = correct_encodings.clone();

    let mut encoder = OpenClEncoder::new(None).unwrap();

    encoder.encode(&mut encodings, &ivs, layers).unwrap();

    encoder.destroy().unwrap();

    // Verify wth CPU implementation as we don't have GPU-based decoding
    for (encoding, (correct_encoding, iv)) in encodings.chunks_exact(4096).zip(
        correct_encodings
            .chunks_exact(4096)
            .zip(ivs.chunks_exact(32)),
    ) {
        let mut decoding: [u8; 4096] = encoding.try_into().unwrap();
        cpu::decode(&mut decoding, &iv, layers).unwrap();

        assert_eq!(correct_encoding, decoding);
    }
}
