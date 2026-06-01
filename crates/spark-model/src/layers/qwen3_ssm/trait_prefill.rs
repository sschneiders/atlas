// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::prefill.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize, // SSM layers ignore — recurrent state requires all tokens
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd; // 2048
        let value_dim = nv * vd; // 4096
        let conv_dim = key_dim * 2 + value_dim; // 8192
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size(); // 12288

        // Profiling helper: sync + timestamp when ATLAS_PROFILE=1
        macro_rules! prof {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    if let Some(t0) = $t0 {
                        ctx.gpu.synchronize(stream)?;
                        let elapsed = t0.elapsed().as_micros();
                        tracing::info!("  SSM prefill [{}] N={}: {}µs", $label, k, elapsed);
                    }
                }
            };
        }
        let mut t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Diagnostic: sync at entry to catch prior-layer errors
        if k > 4096 {
            tracing::info!("SSM prefill ENTRY: k={k} h={h}");
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("SSM prefill ENTRY: stream broken (k={k}): {e}"))?;
        }

        // ── 1. RMS norm + residual for N tokens ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            k,
            h as u32,
            eps,
            stream,
        )?;
        if k > 4096 {
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("SSM prefill: SYNC after rms_norm (k={k}): {e}"))?;
        }

        prof!("rms_norm_residual", t0);
        if std::env::var("ATLAS_DUMP_GDN").is_ok() {
            let _ = ctx.gpu.synchronize(stream);
            let dmp = |tag: &str, p: spark_runtime::gpu::DevicePtr| {
                let mut b = vec![0u8; 64 * 2];
                let _ = ctx.gpu.copy_d2h(p, &mut b);
                let mut ss = 0f64;
                for c in b.chunks_exact(2) { let x = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16); ss += (x as f64) * (x as f64); }
                eprintln!("[gdn] {} norm={:.3}", tag, ss.sqrt());
            };
            dmp("hidden_in", hidden);
            dmp("normed_in", normed);
        }
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 2+3. QKVZ GEMM (+ deinterleave if needed) ──
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        if let Some(fp8) = self.qkvz_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                normed,
                fp8,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm prefill: QKVZ FP8 GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else if let Some(ref nvfp4_t) = self.qkvz_nvfp4_t {
            if k > 128 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "ssm prefill: QKVZ m128 GEMM failed (M={k}, N={qkvz_size}): {e}"
                    )
                })?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("ssm prefill: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
                })?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                nvfp4,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm prefill: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &self.ssm.in_proj_qkvz,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        if !self.sequential_qkvz {
            ops::deinterleave_qkvz(
                ctx.gpu,
                self.deinterleave_k,
                proj_dst,
                deinterleaved,
                k,
                nk as u32,
                kd as u32,
                vpg as u32,
                vd as u32,
                stream,
            )?;
        }

        prof!("qkvz_gemm", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 4+5. Fused BA GEMM + GDN gates (token-parallel) ──
        // Replaces dense_gemm([M,K]×[N,K]) + compute_gdn_gates.
        // Vectorized uint4 loads, warp shuffle reduction, inline sigmoid/exp.
        // gate_out layout: [gate(nv), beta(nv)] per token, gate_stride = 2*nv FP32.
        let ba_size = ctx.config.ssm_ba_size(); // 64
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_stride = nv * 2; // FP32 elements per token
        ops::dense_gemm_ba_gates_prefill(
            ctx.gpu,
            self.ba_gates_prefill_k,
            normed,
            &self.ssm.in_proj_ba,
            self.ssm.a_log.weight,
            self.ssm.dt_bias.weight,
            gates_buf,
            k,
            ba_size as u32,
            h as u32,
            h as u32,
            gate_stride as u32,
            nv as u32,
            vpg as u32,
            stream,
        )?;
        prof!("ba+gates", t0);
        if std::env::var("ATLAS_DUMP_GDN").is_ok() { let _=ctx.gpu.synchronize(stream); let n = nv as usize; let mut bd=vec![0u8; n*4]; let _=ctx.gpu.copy_d2h(gates_buf, &mut bd); let vd:Vec<f32>=bd.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(); let mut bb=vec![0u8; n*4]; let _=ctx.gpu.copy_d2h(gates_buf.offset(nv * fp32), &mut bb); let vb:Vec<f32>=bb.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect(); eprintln!("[gdn] decay[0..4]={:?} beta[0..4]={:?}", &vd[..4.min(vd.len())], &vb[..4.min(vb.len())]); }
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 6. Batched conv1d for all N tokens (sequential per-channel in registers) ──
        // Reuse ssm_qkvz buffer for conv output (safe: deinterleave is done)
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        let gdn_out_buf = ctx.buffers.attn_output();

        // Input: deinterleaved [N, qkvz_size], output: conv_out [N, conv_dim]
        // Conv1d processes QKV channels (first conv_dim of each token's qkvz_size)
        ops::conv1d_update_prefill(
            ctx.gpu,
            self.conv1d_prefill_k,
            ssm_state.conv_state,
            deinterleaved,
            &self.ssm.conv1d,
            DevicePtr::NULL,
            conv_out_buf,
            conv_dim as u32,
            d_conv as u32,
            k,
            qkvz_size as u32,
            conv_dim as u32,
            stream,
        )?;
        prof!("conv1d", t0);
        if std::env::var("ATLAS_DUMP_GDN").is_ok() { let _=ctx.gpu.synchronize(stream); let mut b=vec![0u8;128]; let _=ctx.gpu.copy_d2h(conv_out_buf,&mut b); let mut ss=0f64; for c in b.chunks_exact(2){ let x=f32::from_bits((u16::from_le_bytes([c[0],c[1]]) as u32)<<16); ss+=(x as f64)*(x as f64); } eprintln!("[gdn] conv_out norm={:.3}", ss.sqrt()); }
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 7. Batched L2 norm on Q,K for all N tokens ──
        // Q,K are the first 2*key_dim elements of each token's conv_out.
        // Stride between tokens in conv_out = conv_dim.
        ops::l2_norm(
            ctx.gpu,
            self.l2_norm_k,
            conv_out_buf,
            (nk * 2) as u32,
            kd as u32,
            1e-6,
            k,
            conv_dim as u32,
            stream,
        )?;
        prof!("l2_norm", t0);
        if std::env::var("ATLAS_DUMP_GDN").is_ok() { let _=ctx.gpu.synchronize(stream); let mut b=vec![0u8;128]; let _=ctx.gpu.copy_d2h(conv_out_buf,&mut b); let mut ss=0f64; for c in b.chunks_exact(2){ let x=f32::from_bits((u16::from_le_bytes([c[0],c[1]]) as u32)<<16); ss+=(x as f64)*(x as f64); } eprintln!("[gdn] post_l2norm_q norm={:.3}", ss.sqrt()); }
        if std::env::var("ATLAS_DUMP_GDN").is_ok() { let _=ctx.gpu.synchronize(stream); let mut b=vec![0u8;128]; let _=ctx.gpu.copy_d2h(conv_out_buf.offset(key_dim * 2 * bf16),&mut b); let mut ss=0f64; for c in b.chunks_exact(2){ let x=f32::from_bits((u16::from_le_bytes([c[0],c[1]]) as u32)<<16); ss+=(x as f64)*(x as f64); } eprintln!("[gdn] v_in norm={:.3}", ss.sqrt()); }
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 8. GDN prefill via WY4-persistent kernel ──
        // Processes 4 tokens per iteration with WY algebraic correction, keeping
        // H state in shared memory for the entire sequence. 4× fewer sequential
        // state multiplications vs single-token kernel, preventing precision
        // drift at long context (28K+). Falls back to single-token persistent,
        // then split4 for unsupported configurations.
        let q_ptr = conv_out_buf;
        let k_ptr = conv_out_buf.offset(key_dim * bf16);
        let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);
        let gb_stride = (nv * 2) as u32;

        if self.gdn_prefill_persistent_wy4_k.0 != 0 {
            // WY4-persistent: H in shared memory, 4 tokens per iteration
            // smem = H[K_DIM*V_DIM] + 8*k/q buffers + warp sums + WY scalars
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_k,
                ssm_state.h_state,
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
                ssm_state.h_state,
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
                ssm_state.h_state,
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

        prof!("gdn_prefill", t0);
        macro_rules! gdnmag {
            ($tag:expr, $ptr:expr, $n:expr) => {
                if std::env::var("ATLAS_DUMP_GDN").is_ok() {
                    let _ = ctx.gpu.synchronize(stream);
                    let mut b = vec![0u8; ($n) * 2];
                    let _ = ctx.gpu.copy_d2h($ptr, &mut b);
                    let mut ss = 0f64;
                    let mut nf = false;
                    for c in b.chunks_exact(2) {
                        let x = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                        if !x.is_finite() { nf = true; }
                        ss += (x as f64) * (x as f64);
                    }
                    let mut first=[0f32;6]; for ii in 0..6.min(b.len()/2){ first[ii]=f32::from_bits((u16::from_le_bytes([b[ii*2],b[ii*2+1]]) as u32)<<16);} eprintln!("[gdn] {} norm={:.6} nonfinite={} first={:?}", $tag, ss.sqrt(), nf, first);
                }
            };
        }
        gdnmag!("gdn_out", gdn_out_buf, 64);
        gdnmag!("qkvz_q", deinterleaved, 64);
        if k > 1 { gdnmag!("qkvz_qLAST", deinterleaved.offset((k as usize - 1) * (qkvz_size as usize) * bf16), 64); }
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 9. Gated RMS norm (batched: all tokens × heads in one launch) ──
        let normed_out_buf = conv_out_buf;
        let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
        gdnmag!("gate_z", z_base, 64);
        gdnmag!("norm_w", self.ssm.norm.weight, 64);
        ops::gated_rms_norm_prefill(
            ctx.gpu,
            self.gated_rms_norm_prefill_k,
            gdn_out_buf,
            z_base,
            &self.ssm.norm,
            normed_out_buf,
            nv as u32,
            vd as u32,
            eps,
            k,
            value_dim as u32,
            qkvz_size as u32,
            stream,
        )?;
        prof!("gated_rms_norm", t0);
        gdnmag!("post_norm", normed_out_buf, 64);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 10. Output projection GEMM: [N, 4096] × [4096, 2048] → [N, 2048] ──
        let out_proj_buf = ctx.buffers.moe_output();
        self.prefill_out_proj_dispatch(ctx, normed_out_buf, out_proj_buf, k, h, value_dim, stream)?;

        prof!("out_proj", t0);
        gdnmag!("out_proj", out_proj_buf, 64);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 11. Batched residual + post-norm + MoE ──
        // residual_add_rms_norm already supports num_tokens via grid.x
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            out_proj_buf,
            &self.post_attn_norm,
            ctx.buffers.norm_output(),
            residual,
            num_tokens as u32,
            h as u32,
            eps,
            stream,
        )?;
        // Batched MoE: 5 kernel launches for all N tokens
        self.ffn
            .forward_prefill(ctx.buffers.norm_output(), num_tokens, ctx, stream)?;
        // Batch residual_add: moe_output[N*H] → hidden[N*H]
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            ctx.buffers.moe_output(),
            (num_tokens * h) as u32,
            stream,
        )?;

        prof!("moe_ffn", t0);

        Ok(())
    }
}
