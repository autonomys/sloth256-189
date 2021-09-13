use crate::a_piece::CORRECT_ENCODING;
use crate::cuda::*;

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_batch() {
    if check_cuda() {
        let expanded_ivs = vec![3u8; 1024 * 32]; // 1024 expanded_ivs
        let mut pieces = vec![5u8; 1024 * 4096]; // 1024 pieces

        cuda_encode(&mut pieces, &expanded_ivs, 1).unwrap();
        for i in 0..1024 {
            assert_eq!(pieces[i * 4096..(i + 1) * 4096], CORRECT_ENCODING);
        }
    } else {
        panic!("no Nvidia card detected, skip test_cuda_single_piece");
    }
}
