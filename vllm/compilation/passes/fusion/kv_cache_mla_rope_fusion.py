# fused_concat_and_cache_mla_rope returns a fresh contiguous post-rope k_pe
# tensor (shape [num_tokens, 1, rot_dim]) rather than mutating the strided
# `k_pe` input view. This lets downstream ops (in particular the opaque
# unified_mla_attention_with_output, which lives in a separate Inductor
# partition) read k_pe directly across the partition boundary, avoiding an
# extra Triton copy kernel that would otherwise be inserted to materialise
# the strided view. The op still mutates `q_pe` in place (via the original
# strided slice), and still writes into `kv_cache`; only the k_pe output
# path changed.


    # Allocate a fresh contiguous output for the post-rope k_pe (see module
    # comment above the op for why this is not done in-place on `k_pe`).
    k_pe_out = torch.empty_like(k_pe, memory_format=torch.contiguous_format)
        k_pe_out,
    return k_pe_out
    return torch.empty_like(k_pe, memory_format=torch.contiguous_format)
    mutates_args=["q_pe"],
            # auto_functionalized layout (mutates_args=["q_pe"]):
            #   res[0] -> actual op return (k_pe_out, contiguous)
            #   res[1] -> mutated q_pe
            # The pattern's first slot must stay rank-1 (it feeds
            # `kv_cache_dummy_dep` of unified_mla_attention_with_output, which
            # the original pattern produced via
            # unified_mla_kv_cache_update -> torch.empty(0)). Mint a fresh
            # rank-1 dummy with the same dtype/device.
            return kv_c_normed.new_empty(0), res[1], res[0]
            # See _mk_pattern_with_layer_name_input for slot semantics.
            return kv_c_normed.new_empty(0), res[1], res[0]
            k_pe_out = self.FUSED_OP(
            # k_pe_out is the contiguous post-rope key returned by the op.
            # The first slot must stay rank-1 (it feeds `kv_cache_dummy_dep`
            # of unified_mla_attention_with_output, originally produced via
            # unified_mla_kv_cache_update -> torch.empty(0)).
            return kv_c_normed.new_empty(0), v2, k_pe_out
            k_pe_out = self.FUSED_OP(
            # See _mk_pattern_with_layer_name_input for slot semantics.
            return kv_c_normed.new_empty(0), v2, k_pe_out
