use crate::test_vectors::{CORRECT_ENCODING, EXPANDED_IV, LAYERS, PIECE};
use crate::utils;
use sloth256_189::cpu;

#[test]
fn test_random_piece() {
    let expanded_iv = utils::random_bytes();
    let piece = utils::random_bytes();

    let layers = 4096 / 32;
    let mut encoding = piece.clone();
    cpu::encode(&mut encoding, expanded_iv, layers).unwrap();

    let mut decoding = encoding.clone();
    cpu::decode(&mut decoding, expanded_iv, layers);

    assert_eq!(piece.to_vec(), decoding.to_vec());
}

#[test]
fn test_known_piece() {
    let mut encoding = PIECE;
    cpu::encode(&mut encoding, EXPANDED_IV, LAYERS).unwrap();
    assert_eq!(encoding, CORRECT_ENCODING);

    let mut decoding = encoding.clone();
    cpu::decode(&mut decoding, EXPANDED_IV, LAYERS);

    assert_eq!(PIECE, decoding);
}
