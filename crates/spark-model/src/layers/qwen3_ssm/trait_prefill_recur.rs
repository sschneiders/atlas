// SPDX-License-Identifier: AGPL-3.0-only

//! GDN recurrence kernel dispatch for `Qwen3SsmLayer::prefill_inner`.
//!
//! Hoisted from `trait_prefill.rs` to keep that file under the 500 LoC
//! cap. [`Qwen3SsmLayer::prefill_gdn_recurrence`] mirrors the original
//! step 8 block 1:1 — same WY4-persistent / single-token persistent /
//! split4 dispatch, same env overrides, same kernel launches.

use super::*;

impl Qwen3SsmLayer {
    /// GDN prefill recurrence via the WY4-persistent kernel.
    ///
    /// Processes 4 tokens per iteration with WY algebraic correction,
    /// keeping H state in shared memory for the entire sequence. Falls
    /// back to single-token persistent (256..=4096), then split4 for
    /// unsupported configurations.
    ///
    /// Env overrides:
    /// - `ATLAS_DISABLE_WY4=1` — skip WY4-persistent.
    /// - `ATLAS_FORCE_PERSISTENT=1` — force single-token persistent at any `k`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_gdn_recurrence(
        &self,
        h_state: DevicePtr,
        q_ptr: DevicePtr,
        k_ptr: DevicePtr,
        v_ptr: DevicePtr,
        gates_buf: DevicePtr,
        gdn_out_buf: DevicePtr,
        k: u32,
        nk: usize,
        nv: usize,
        kd: usize,
        vd: usize,
        conv_dim: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let fp32 = 4usize;
        let gb_stride = (nv * 2) as u32;

        // Env overrides for kernel investigation:
        //   ATLAS_DISABLE_WY4=1       — skip WY4-persistent, fall through to
        //                               single-token persistent (256..=4096)
        //                               or split4.
        //   ATLAS_FORCE_PERSISTENT=1  — force the single-token persistent
        //                               kernel at any k (lifts the 4096 cap).
        //                               Mathematically correct per-token
        //                               sequential recurrence with FP32 SMEM
        //                               H state — useful for isolating WY
        //                               chunkwise reduction noise.
        let wy4_disabled = matches!(
            std::env::var("ATLAS_DISABLE_WY4").ok().as_deref(),
            Some("1")
        );
        let force_persistent = matches!(
            std::env::var("ATLAS_FORCE_PERSISTENT").ok().as_deref(),
            Some("1")
        );
        // ATLAS_GDN_CHUNK64=1 — chunked-scan prefill (GATE-C precision experiment):
        // kk/kq Gram matmuls on tensor cores; H-recurrence scalar (bf16-H). Takes
        // priority over wy4 when set. smem = sk_bf+sq_bf(bf16) + kk+kq(f32) + gc.
        let chunk64_on = matches!(
            std::env::var("ATLAS_GDN_CHUNK64").ok().as_deref(),
            Some("1")
        );
        // ATLAS_GDN_FLA=1 — FLA multi-kernel chunked prefill (recompute_wu →
        // chunk_delta_h_ksplit → chunk_fwd_o). 1.75x vs wy4 @16k, token-equal
        // (cos=1.0 vs scalar). HIGHEST priority when set + 128-dim linear heads +
        // kernels & scratch present. Scratch = BufferArena.gdn_fla_scratch, carved
        // W|U|S|uc by the runtime chunk count (≤ the max_batch_tokens it was sized for).
        let fla_on = matches!(
            std::env::var("ATLAS_GDN_FLA").ok().as_deref(),
            Some("1")
        );
        let fla_scratch = ctx.buffers.gdn_fla_scratch();
        if fla_on
            && kd == 128
            && vd == 128
            && fla_scratch.0 != 0
            && self.gdn_prefill_fla_recompute_wu_k.0 != 0
            && self.gdn_prefill_fla_chunk_delta_h_k.0 != 0
            && self.gdn_prefill_fla_chunk_fwd_o_k.0 != 0
        {
            // One-time positive signal that the FLA path is live (vs silently
            // falling through to wy4 on a guard miss) — greppable in the server log.
            static FLA_LOG: std::sync::Once = std::sync::Once::new();
            FLA_LOG.call_once(|| {
                tracing::info!(
                    "GDN prefill: ATLAS_GDN_FLA path ACTIVE (recompute_wu → chunk_delta_h_ksplit → chunk_fwd_o)"
                );
            });
            let num_chunks = k.div_ceil(64);
            let nt = num_chunks as usize;
            let w_out = fla_scratch;
            let u_out = w_out.offset(nt * nv * 64 * kd * 2);
            let s_out = u_out.offset(nt * nv * 64 * vd * 2);
            let uc_out = s_out.offset(nt * nv * kd * vd * 4);
            ops::gdn_prefill_fla(
                ctx.gpu,
                self.gdn_prefill_fla_recompute_wu_k,
                self.gdn_prefill_fla_chunk_delta_h_k,
                self.gdn_prefill_fla_chunk_fwd_o_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                w_out,
                u_out,
                s_out,
                uc_out,
                1,
                k,
                num_chunks,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else if chunk64_on && self.gdn_prefill_chunk64_k.0 != 0 {
            let smem64 = (3 * 64 * kd * 2 + 2 * 64 * 64 * 4 + 2 * 64 * 64 * 2 + 64 * 4) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_chunk64_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem64,
                stream,
            )?;
        } else if force_persistent && self.gdn_prefill_persistent_k.0 != 0 {
            // Forced per-token persistent at ANY k.
            ops::gdn_prefill_persistent(
                ctx.gpu,
                self.gdn_prefill_persistent_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else if !wy4_disabled && self.gdn_prefill_persistent_wy4_k.0 != 0 {
            // WY4-persistent: H in shared memory, 4 tokens per iteration
            // smem = H[K_DIM*V_DIM] + 8*k/q buffers + warp sums + WY scalars
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if (256..=4096).contains(&k) && self.gdn_prefill_persistent_k.0 != 0 {
            ops::gdn_prefill_persistent(
                ctx.gpu,
                self.gdn_prefill_persistent_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else {
            ops::gdn_prefill_split4(
                ctx.gpu,
                self.gdn_prefill_split4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        }
        Ok(())
    }
}
