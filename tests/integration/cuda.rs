use crate::test_vectors::{CORRECT_ENCODING, EXPANDED_IV, LAYERS, PIECE};
use crate::utils;
use sloth256_189::{cpu, cuda};
use std::convert::TryInto;

#[test]
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
    cuda::encode(&mut encodings, &ivs, layers).unwrap();

    // Verify with CPU implementation as we don't have GPU-based decoding
    for encoding in encodings.chunks_exact(4096) {
        let mut decoding: [u8; 4096] = encoding.try_into().unwrap();
        cpu::decode(&mut decoding, &expanded_iv, layers);

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
    cuda::encode(&mut encodings, &ivs, LAYERS).unwrap();

    let mut correct_encodings = Vec::with_capacity(1024 * 4096);
    for _ in 0..1024 {
        correct_encodings.extend_from_slice(&CORRECT_ENCODING);
    }
    assert_eq!(encodings, correct_encodings);
}
