use crate::a_piece::CORRECT_ENCODING;
use crate::cpu::*;
use rand::prelude::*;

fn random_bytes<const BYTES: usize>() -> [u8; BYTES] {
    let mut bytes = [0u8; BYTES];
    rand::thread_rng().fill(&mut bytes[..]);
    bytes
}

// 256 bits
#[test]
fn test_random_piece_256_bits() {
    let expanded_iv = random_bytes();
    let piece = random_bytes();

    let layers = 4096 / 32;
    let mut encoding = piece.clone();
    encode(&mut encoding, expanded_iv, layers).unwrap();
    let mut decoding = encoding.clone();
    decode(&mut decoding, expanded_iv, layers);

    // println!("\nPiece is {:?}\n", piece.to_vec());
    // println!("\nDecoding is {:?}\n", decoding.to_vec());
    // println!("\nEncoding is {:?}\n", encoding.to_vec());

    assert_eq!(piece.to_vec(), decoding.to_vec());
}

#[test]
fn test_known_piece() {
    let expanded_iv = [3u8; 32];
    let piece = [5u8; 4096];

    let layers = 1;
    let mut encoding = piece.clone();
    encode(&mut encoding, expanded_iv, layers).unwrap();
    assert_eq!(encoding, CORRECT_ENCODING);
    let mut decoding = encoding.clone();
    decode(&mut decoding, expanded_iv, layers);

    assert_eq!(piece.to_vec(), decoding.to_vec());
}
