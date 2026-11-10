// OCC-SILICON: the credited-mode lock and its counters, one SM, one clock, as combinational steps.
// State is registered outside and priced at a DFF a bit, as SI-ALG's synth.py prices accumulators.
//
// Issue word (illustrative encoding): [23:12] opcode, [11:0] modifiers (type, shape, sparsity, scale format, mode).
// The class ROM is a standalone masked-comparator bank on each of the 4 issue ports: an upper bound on a class bit
// the decoder would otherwise carry per opcode.

module lock_class(input [23:0] w, output c, output s, output m0, output m1, output seal, output enter, output tma);
  // credited MMA descriptors: the whole descriptor, every modifier bit
  assign m0 = (w == 24'h5A10C3);                       // e.g. wgmma s8.s8.s32 m64n256k32 dense
  assign m1 = (w == 24'h5A10C7);                       // e.g. a second credited shape
  wire tma_ld = (w[23:8] == 16'h3B01);                 // TMA tile load global -> own smem (not reduce, not L2 prefetch)
  wire tma_st = (w[23:8] == 16'h3B02);                 // TMA tile store own smem -> global (not reduce)
  wire stsm   = (w[23:12] == 12'h2E4);                 // accumulator registers -> smem
  wire ldsm   = (w[23:12] == 12'h2E5);                 // smem -> registers (register-A operand)
  wire mbar   = (w[23:12] == 12'h3C0);                 // mbarrier arrive / expect_tx / try_wait
  wire wgf    = (w[23:12] == 12'h5A2);                 // wgmma fence / commit_group / wait_group
  wire bar    = (w[23:12] == 12'h3D0);                 // named barrier
  wire nop    = (w == 24'h000000);
  assign seal  = (w == 24'h7F0001);
  assign enter = (w == 24'h7F0000);
  // support class S: uniform datapath and branch, counted against the declared budget
  wire u0 = (w[23:12] == 12'h1A0), u1 = (w[23:12] == 12'h1A5), u2 = (w[23:12] == 12'h1B2),
       u3 = (w[23:12] == 12'h1C7), br = (w[23:12] == 12'h0B0);
  assign s = u0 | u1 | u2 | u3 | br;
  assign tma = tma_ld | tma_st;
  assign c = m0 | m1 | tma_ld | tma_st | stsm | ldsm | mbar | wgf | bar | nop | seal;
endmodule

module lock_step(
    input cm, input p, input [15:0] tag,
    input [3:0] iv, input [23:0] w0, input [23:0] w1, input [23:0] w2, input [23:0] w3,
    input [1:0] rv, input [1:0] rcm, input [15:0] rtag0, input [15:0] rtag1,
    input launch, input trap, input preempt, input dbg, input scan,
    input [15:0] new_tag,
    output cm_n, output p_n, output [15:0] tag_n,
    output [2:0] n_m0, output [2:0] n_m1, output [2:0] n_s, output [2:0] n_tma, output any_issue,
    output seal_fire, output launch_ok);
  wire [3:0] c, s, m0, m1, se, en, tm;
  lock_class k0(w0, c[0], s[0], m0[0], m1[0], se[0], en[0], tm[0]);
  lock_class k1(w1, c[1], s[1], m0[1], m1[1], se[1], en[1], tm[1]);
  lock_class k2(w2, c[2], s[2], m0[2], m1[2], se[2], en[2], tm[2]);
  lock_class k3(w3, c[3], s[3], m0[3], m1[3], se[3], en[3], tm[3]);
  wire [3:0] x = iv & ~c & ~s;                          // an issue of neither class
  wire [1:0] rbad = rv & (~rcm | {rtag1 != tag, rtag0 != tag});
  wire enter_req = |(iv & en);
  wire enter_ok = enter_req & ~cm;
  wire bad = |x | |rbad | trap | preempt | dbg | scan | enter_req;     // a refused launch needs no poison
  assign seal_fire = cm & |(iv & se);
  assign cm_n = enter_ok | (cm & ~seal_fire);
  assign p_n = enter_ok ? 1'b0 : (p | (cm & bad));
  assign tag_n = enter_ok ? new_tag : tag;
  assign launch_ok = ~cm;
  wire [3:0] g = cm ? iv : 4'b0;
  assign n_m0 = (g[0] & m0[0]) + (g[1] & m0[1]) + (g[2] & m0[2]) + (g[3] & m0[3]);
  assign n_m1 = (g[0] & m1[0]) + (g[1] & m1[1]) + (g[2] & m1[2]) + (g[3] & m1[3]);
  assign n_s  = (g[0] & s[0]) + (g[1] & s[1]) + (g[2] & s[2]) + (g[3] & s[3]);
  assign n_tma = (g[0] & tm[0]) + (g[1] & tm[1]) + (g[2] & tm[2]) + (g[3] & tm[3]);
  assign any_issue = |g;
endmodule

// The one counter the lock needs: support issues in the session, checked by the proof against budget(shape).
module ns_step(input enter_ok, input [2:0] n_s, input [47:0] ns, output [47:0] ns_n);
  assign ns_n = enter_ok ? 48'd0 : ns + n_s;
endmodule

// Diagnostic counters (believed, not checkable): credited cycles, stall cycles, MMA issues by descriptor
// (derivable from the proven chain), TMA bytes, global-timer snapshots at enter and seal.
module diag_step(
    input cm, input enter_ok, input seal_fire, input any_issue,
    input [2:0] n_m0, input [2:0] n_m1, input tma_v, input [16:0] tma_bytes, input [63:0] gtimer,
    input [47:0] c_cm, input [47:0] c_st, input [47:0] q0, input [47:0] q1, input [47:0] tb,
    input [63:0] t_en, input [63:0] t_se,
    output [47:0] c_cm_n, output [47:0] c_st_n, output [47:0] q0_n, output [47:0] q1_n, output [47:0] tb_n,
    output [63:0] t_en_n, output [63:0] t_se_n);
  assign c_cm_n = enter_ok ? 48'd0 : c_cm + cm;
  assign c_st_n = enter_ok ? 48'd0 : c_st + (cm & ~any_issue);
  assign q0_n = enter_ok ? 48'd0 : q0 + n_m0;
  assign q1_n = enter_ok ? 48'd0 : q1 + n_m1;
  assign tb_n = enter_ok ? 48'd0 : tb + (tma_v ? tma_bytes : 17'd0);
  assign t_en_n = enter_ok ? gtimer : t_en;
  assign t_se_n = seal_fire ? gtimer : t_se;
endmodule

// Seal absorb: at CM.SEAL the Keccak unit's word input takes (p, tag, counters) instead of the write-back words.
module seal_mux #(parameter W = 64) (input sel, input [W-1:0] wb, input [W-1:0] sb, output [W-1:0] y);
  assign y = sel ? sb : wb;
endmodule

module seal_mux_min(input sel, input [64:0] wb, input [64:0] sb, output [64:0] y);
  seal_mux #(65) m(sel, wb, sb, y);       // p, tag, n_S: 1 + 16 + 48 = 65 bits
endmodule

module seal_mux_diag(input sel, input [367:0] wb, input [367:0] sb, output [367:0] y);
  seal_mux #(368) m(sel, wb, sb, y);      // the diagnostic counters: 5x48 + 2x64 = 368 bits
endmodule
