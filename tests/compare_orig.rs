use rand::Rng;

const PRIME: &str =
    "115792089237316195423570985008687907853269984665640564039457584007913129639747";

fn random_bytes<const BYTES: usize>() -> [u8; BYTES] {
    let mut bytes = [0u8; BYTES];
    rand::thread_rng().fill(&mut bytes[..]);
    bytes
}

#[test]
fn spartan_orig_vs_asm() {
    let sloth_orig: spartan_sloth::software::Sloth<32, 4096> =
        spartan_sloth::software::Sloth::with_prime(PRIME.parse().unwrap());
    let sloth_asm = sloth256_189::Sloth::<32, 4096> {};

    for _ in 0..1000 {
        let expanded_iv = random_bytes();
        let piece = random_bytes();

        let mut encoding_orig = piece.clone();
        sloth_orig
            .encode(&mut encoding_orig, expanded_iv, 1)
            .unwrap();
        let mut decoding_orig = encoding_orig.clone();
        sloth_orig.decode(&mut decoding_orig, expanded_iv, 1);

        let mut encoding_asm = piece.clone();
        sloth_asm.encode(&mut encoding_asm, expanded_iv, 1).unwrap();
        let mut decoding_asm = encoding_asm.clone();
        sloth_asm.decode(&mut decoding_asm, expanded_iv, 1);

        //println!("\nPiece is {:?}\n", piece.to_vec());
        //println!("\nDecoding is {:?}\n", decoding_orig.to_vec());
        //println!("\nEncoding is {:?}\n", encoding_orig.to_vec());

        assert_eq!(piece.to_vec(), decoding_orig.to_vec());
        assert_eq!(piece.to_vec(), decoding_asm.to_vec());
        assert_eq!(encoding_orig.to_vec(), encoding_asm.to_vec());
    }
}
