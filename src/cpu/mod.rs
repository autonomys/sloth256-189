//! Universal CPU implementation

use thiserror::Error;

mod ffi {
    extern "C" {
        pub(super) fn sloth256_189_encode(
            inout: *mut u8,
            len: usize,
            iv_: *const u8,
            layers: usize,
        ) -> bool;
        pub(super) fn sloth256_189_decode(
            inout: *mut u8,
            len: usize,
            iv_: *const u8,
            layers: usize,
        );
    }
}

/// CPU encoding errors
#[derive(Debug, Error)]
pub enum EncodeError {
    /// Piece argument is invalid, must be 4096-bytes piece
    #[error("Piece argument is invalid, must be 4096-bytes piece, {0} bytes given")]
    InvalidPiece(usize),
    /// IV argument is invalid, must be 32-bytes IV
    #[error("IV argument is invalid, must be 32-bytes IV, {0} bytes given")]
    InvalidIV(usize),
    /// Data bigger than the prime, this is not supported
    #[error("Data bigger than the prime, this is not supported")]
    DataBiggerThanPrime,
}

/// CPU decoding errors
#[derive(Debug, Error)]
pub enum DecodeError {
    /// Piece argument is invalid, must be 4096-bytes piece
    #[error("Piece argument is invalid, must be 4096-bytes piece, {0} bytes given")]
    InvalidPiece(usize),
    /// IV argument is invalid, must be 32-bytes IV
    #[error("IV argument is invalid, must be 32-bytes IV, {0} bytes given")]
    InvalidIV(usize),
}

/// Sequentially encodes a 4096 byte piece s.t. a minimum amount of wall clock time elapses
pub fn encode(piece: &mut [u8], iv: &[u8], layers: usize) -> Result<(), EncodeError> {
    // if piece.len() != 4096 {
    //     return Err(EncodeError::InvalidPiece(piece.len()));
    // }
    // if iv.len() != 32 {
    //     return Err(EncodeError::InvalidPiece(iv.len()));
    // }

    if unsafe { ffi::sloth256_189_encode(piece.as_mut_ptr(), piece.len(), iv.as_ptr(), layers) } {
        return Err(EncodeError::DataBiggerThanPrime);
    }

    Ok(())
}

/// Sequentially decodes a 4096 byte encoding in time << encode time
pub fn decode(piece: &mut [u8; 4096], iv: [u8; 32], layers: usize) -> Result<(), DecodeError> {
    if piece.len() != 4096 {
        return Err(DecodeError::InvalidPiece(piece.len()));
    }
    if iv.len() != 32 {
        return Err(DecodeError::InvalidPiece(iv.len()));
    }

    unsafe { ffi::sloth256_189_decode(piece.as_mut_ptr(), piece.len(), iv.as_ptr(), layers) };

    Ok(())
}
