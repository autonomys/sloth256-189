use crate::test_vectors::{CORRECT_ENCODING, EXPANDED_IV, LAYERS, PIECE};
use crate::utils;
use rand::Rng;
use serial_test::serial;
use sloth256_189::{cpu, opencl};
use std::convert::TryInto;

fn random_bytes_vec<const BYTES: usize>() -> Vec<u8> {
    let mut bytes = vec![0u8; BYTES];
    rand::thread_rng().fill(&mut bytes[..]);
    bytes
}

fn random_bytes_vec_inplace<const BYTES: usize>(vec: &mut Vec<u8>) {
    rand::thread_rng().fill(&mut vec[..]);
}

#[test]
#[serial]
fn test_random_piece() {
    let expanded_iv = utils::random_bytes::<32>();
    let piece = utils::random_bytes::<4096>();
    let layers = 4096 / 32;

    let mut encodings = Vec::with_capacity(1024 * 4096);
    for _ in 0..1024 {
        encodings.extend_from_slice(&piece);
    }
    let mut ivs = Vec::with_capacity(32 * 4096);
    for _ in 0..1024 {
        ivs.extend_from_slice(&expanded_iv);
    }
    let instances = opencl::initialize().unwrap();
    opencl::encode(&mut encodings, &ivs, layers, instances).unwrap();
    opencl::cleanup(instances).unwrap();

    // Verify wth CPU implementation as we don't have GPU-based decoding
    for encoding in encodings.chunks_exact(4096) {
        let mut decoding: [u8; 4096] = encoding.try_into().unwrap();
        cpu::decode(&mut decoding, &expanded_iv, layers).unwrap();

        assert_eq!(piece, decoding);
    }
}

#[test]
#[serial]
fn test_known_piece() {
    let mut encodings = Vec::with_capacity(1024 * 4096);
    for _ in 0..1024 {
        encodings.extend_from_slice(&PIECE);
    }
    let mut ivs = Vec::with_capacity(32 * 4096);
    for _ in 0..1024 {
        ivs.extend_from_slice(&EXPANDED_IV);
    }
    let instances = opencl::initialize().unwrap();
    opencl::encode(&mut encodings, &ivs, LAYERS, instances).unwrap();
    opencl::cleanup(instances).unwrap();

    let mut correct_encodings = Vec::with_capacity(1024 * 4096);
    for _ in 0..1024 {
        correct_encodings.extend_from_slice(&CORRECT_ENCODING);
    }
    assert_eq!(encodings, correct_encodings);
}

#[test]
#[serial]
fn test_big_random_piece() {
    const NUM_PIECES: usize = 1024 * 999 * 7;
    //let layers = 4096 / 32;
    let layers = 2;

    let correct_encodings = random_bytes_vec::<{ 4096 * NUM_PIECES }>();
    let ivs = random_bytes_vec::<{ 32 * NUM_PIECES }>();

    let mut encodings = correct_encodings.clone();
    let instances = opencl::initialize().unwrap();
    opencl::encode(&mut encodings, &ivs, layers, instances).unwrap();
    opencl::cleanup(instances).unwrap();

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

#[test]
#[serial]
fn test_big_random_piece_pinned() {
    const NUM_PIECES: usize = 1024 * 1024;
    //let layers = 4096 / 32;
    let layers = 2;

    let instances = opencl::initialize().unwrap();

    let mut encodings = opencl::pinned_memory_alloc(instances, 4096 * NUM_PIECES).unwrap();
    random_bytes_vec_inplace::<{ 4096 * NUM_PIECES }>(&mut encodings);
    let ivs = random_bytes_vec::<{ 32 * NUM_PIECES }>();

    let correct_encodings = encodings.clone();

    opencl::encode(&mut encodings, &ivs, layers, instances).unwrap();

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

    opencl::pinned_memory_free(instances).unwrap();
    opencl::cleanup(instances).unwrap();
    std::mem::forget(encodings);
}
