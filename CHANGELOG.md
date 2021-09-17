# 0.2.1
* ffi functions were declared as `pub(super)` but causing `unresolved external symbol` error when this crate was included as a dependency. 
Now these functions are declared as `pub(crate)` to fix this issue.

# 0.2.0
* CPU implementation moved into `cpu` module
* CUDA (PTX) implementation added in `cuda` module (requires `cuda` feature flag, disabled by default)

# 0.1.0
* Initial release
