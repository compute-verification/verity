// Native acquisition at the EXISTING hook boundaries.
//
// One `Collector` per committer.  Per step the serving thread calls `begin_step(plan_key, arena_ptr, arena_bytes)` once, then
// `collect(name_id, tensor)` once per acquired tensor IN HOOK ORDER, then `end_step()`.  The collector keeps one PLAN per step-shape class
// (`plan_key`): hook sequence number -> (name_id, dtype, sizes, nbytes, arena offset, zero-gap before it), for the plans of the current run
// and the previous one (`end_run`, native_plans.h).  On a plan hit the hook does
// exactly: compare (name_id, nbytes, dtype, sizes) against the entry, then ONE cudaMemcpyAsync(device->arena, D2D) enqueued on the
// PRODUCER's current stream (c10 current CUDA stream of the calling thread) -- so the snapshot is stream-ordered after the producer kernel
// and before any later kernel that overwrites the value in place (the known-safe placement; no collector stream, no event per tensor).
// On a miss (first step of a shape class, or a changed shape/name) the collector switches this step to LEARN mode: the plan is truncated at
// the miss and rebuilt from the tensors actually seen, with the SAME offset rule as the Python fast path (4-byte alignment, gaps zeroed)
// so the packed arena bytes -- and hence the 256-B chunk tree root -- are bit-identical to `native_host_fast+hidden_m1_gputree`.
//
// Nothing here changes what is committed or how it is laid out: same values, same byte offsets, same zero gaps, same names/dtypes/shapes.
#include <torch/extension.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>
#include <unistd.h>
#include <condition_variable>
#include <mutex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

#include "native_plans.h"

extern "C" int verity_gather_launch(const void* d_desc, int n, int blocks_per_desc, cudaStream_t stream);   // native_gather.cu
extern "C" int verity_chunk_leaves_launch(const uint32_t* stream, uint32_t* out, long n_chunks, int chunk_words, uint32_t launch_tag,
                                          uint32_t src_mask, uint32_t HB, uint32_t M, uint32_t NB, cudaStream_t st);   // native_tree.cu
extern "C" int verity_chunk_leaves_launch_base(const uint32_t* stream, uint32_t* out, long n_chunks, int chunk_words, uint32_t launch_tag,
                                               uint32_t src_mask, uint32_t HB, uint32_t M, uint32_t NB, long ci_base, cudaStream_t st);
extern "C" int verity_tree_level_launch(const uint32_t* in, uint32_t* out, long n_in, cudaStream_t st);                // native_tree.cu
extern "C" int verity_tree_levels_launch(const uint32_t* in, uint32_t* out, long n_in, int n_levels, cudaStream_t st);  // native_tree.cu
extern "C" int verity_rows_params_bytes();                                                                              // native_rows.cu
extern "C" int verity_rows_build(const uint8_t* domain_id, const uint8_t* ks, const uint8_t* row_prefix, const uint8_t* salt_prefix,
                                 const uint8_t* leaf_prefix, const char* schema, int schema_len, uint32_t lb, void* out);
extern "C" int verity_rows_leaves_launch(const void* params, const uint8_t* rows, uint64_t stride, uint64_t n_rows, const uint8_t* salts,
                                         const uint8_t* tn, uint64_t first, uint64_t m, uint64_t n_total, uint8_t* out, uint8_t* bc,
                                         uint8_t* hs, uint64_t max_blocks, cudaStream_t st);
extern "C" int verity_rows_leaves_form();
extern "C" int verity_rows_leaves_sf_launch(const void* params, const uint8_t* rows, uint64_t stride, uint64_t n_rows, const uint8_t* salts,
                                            const uint8_t* tn, uint64_t first, uint64_t m, uint64_t n_total, uint8_t* out, uint8_t* bc,
                                            uint8_t* hs, uint8_t* scratch, uint64_t max_blocks, cudaStream_t st);
extern "C" int verity_rows_h_launch(const void* params, const uint8_t* rows, uint64_t stride, uint64_t n_rows, const uint8_t* salts,
                                    const uint8_t* tn, uint64_t first, uint8_t* hs, uint8_t* bc, uint8_t* scratch, uint64_t max_blocks,
                                    cudaStream_t st);
extern "C" int verity_rows_frame_launch(const void* params, const uint8_t* hs, uint64_t n_h, uint64_t first, uint64_t m, uint64_t n_total,
                                        uint8_t* out, uint64_t max_blocks, cudaStream_t st);
extern "C" int verity_rows_level_launch(const void* params, const uint8_t* in, uint8_t* out, uint64_t n_out, int level, uint64_t first_index,
                                        uint64_t max_blocks, cudaStream_t st);
extern "C" int verity_rows_levels_launch(const void* params, const uint8_t* in, uint8_t* out, uint64_t n_in, int level, uint64_t base,
                                         int n_levels, cudaStream_t st);

namespace {

using native_plans::Entry;
using native_plans::Plan;

inline double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A level of more than kWideLevel nodes is built by one launch of a thread a parent (verity_tree_level_launch, verity_rows_level_launch;
// rows-sha512 capped: launches of at most RowStep::max_blocks blocks);
// the narrower ones kFusedLevels at a time, one launch each (verity_tree_levels_launch, verity_rows_levels_launch: a block folds its own
// aligned slice of 2^kFusedLevels nodes).  A block's threads idle as its slice narrows, which on a wide level costs more than the
// launches saved (a 2^22-leaf tree a third slower on H200 folded from its leaves), while below kWideLevel a level's launch is its cost.
constexpr int64_t kWideLevel = int64_t{1} << 16;
constexpr int64_t kFusedLevels = 10;

int64_t levels_to_root(int64_t n) {
  int64_t k = 0;
  for (; n > 1; n = (n + 1) / 2) ++k;
  return k;
}

// vllm-v1: the n_up levels above `base` (int32 [n, 8] on the current stream `side`), lowest first; each launch's levels are views of
// one tensor.
std::vector<at::Tensor> tree_levels_above(const at::Tensor& base, int64_t n_up, cudaStream_t side) {
  std::vector<at::Tensor> up;
  at::Tensor cur = base;
  while (n_up > 0) {
    if (cur.size(0) > kWideLevel) {
      at::Tensor out = at::empty({(cur.size(0) + 1) / 2, 8}, base.options());
      int lrc = verity_tree_level_launch(reinterpret_cast<const uint32_t*>(cur.data_ptr()), reinterpret_cast<uint32_t*>(out.data_ptr()),
                                         cur.size(0), side);
      if (lrc != 0) throw std::runtime_error(std::string("tree_level: ") + cudaGetErrorString(static_cast<cudaError_t>(lrc)));
      up.push_back(out);
      cur = out;
      --n_up;
      continue;
    }
    const int64_t k = std::min(n_up, kFusedLevels);
    std::vector<int64_t> widths;
    int64_t total = 0;
    for (int64_t i = 0, n = cur.size(0); i < k; ++i) {
      n = (n + 1) / 2;
      widths.push_back(n);
      total += n;
    }
    at::Tensor buf = at::empty({total, 8}, base.options());
    int lrc = verity_tree_levels_launch(reinterpret_cast<const uint32_t*>(cur.data_ptr()), reinterpret_cast<uint32_t*>(buf.data_ptr()),
                                        cur.size(0), static_cast<int>(k), side);
    if (lrc != 0) throw std::runtime_error(std::string("tree_levels: ") + cudaGetErrorString(static_cast<cudaError_t>(lrc)));
    int64_t off = 0;
    for (int64_t w : widths) {
      up.push_back(buf.narrow(0, off, w));
      off += w;
    }
    cur = up.back();
    n_up -= k;
  }
  return up;
}

// rows-sha512: the n_up levels above `base` (uint8 [n, 64], n a nonzero multiple of 2^n_up, of level `level`, the first node at index
// `first_index`) on `side`, lowest first; each launch's levels are views of one tensor.  A wide level in launches of at most max_blocks
// blocks when max_blocks > 0 (the narrow ones are at most kWideLevel >> kFusedLevels blocks a launch).
std::vector<at::Tensor> rows_levels_above(const void* params, const at::Tensor& base, int64_t level, int64_t first_index, int64_t n_up,
                                          cudaStream_t side, int64_t max_blocks = 0) {
  std::vector<at::Tensor> up;
  at::Tensor cur = base;
  while (n_up > 0) {
    if (cur.size(0) > kWideLevel) {
      const int64_t n_out = cur.size(0) / 2;
      at::Tensor out = at::empty({n_out, 64}, base.options());
      int lrc = verity_rows_level_launch(params, reinterpret_cast<const uint8_t*>(cur.data_ptr()), reinterpret_cast<uint8_t*>(out.data_ptr()),
                                         static_cast<uint64_t>(n_out), static_cast<int>(level), static_cast<uint64_t>(first_index >> 1),
                                         static_cast<uint64_t>(max_blocks), side);
      if (lrc != 0) throw std::runtime_error(std::string("rows_level: ") + cudaGetErrorString(static_cast<cudaError_t>(lrc)));
      up.push_back(out);
      cur = out;
      --n_up;
      ++level;
      first_index >>= 1;
      continue;
    }
    const int64_t k = std::min(n_up, kFusedLevels), n_in = cur.size(0);
    at::Tensor buf = at::empty({n_in - (n_in >> k), 64}, base.options());
    int lrc = verity_rows_levels_launch(params, reinterpret_cast<const uint8_t*>(cur.data_ptr()), reinterpret_cast<uint8_t*>(buf.data_ptr()),
                                        static_cast<uint64_t>(n_in), static_cast<int>(level), static_cast<uint64_t>(first_index),
                                        static_cast<int>(k), side);
    if (lrc != 0) throw std::runtime_error(std::string("rows_levels: ") + cudaGetErrorString(static_cast<cudaError_t>(lrc)));
    int64_t off = 0;
    for (int64_t i = 0, n = n_in / 2; i < k; ++i, n /= 2) {
      up.push_back(buf.narrow(0, off, n));
      off += n;
    }
    cur = up.back();
    n_up -= k;
    level += k;
    first_index >>= k;
  }
  return up;
}

struct Collector {
  native_plans::Plans plans;   // this run's and the previous run's (native_plans.h)
  Plan* plan = nullptr;
  int64_t plan_key = 0;
  size_t cursor = 0;           // hook sequence number within the step
  int64_t byte_cursor = 0;     // end of the last collected tensor (learn mode: running; hit mode: from the entry)
  int64_t pushed_end = 0;      // windowed: logical end of the bytes whose copy descriptors have been pushed (segments of the current tensor included)
  char* arena = nullptr;
  int64_t arena_bytes = 0;
  bool learning = false;
  bool changed = false;
  bool active = false;
  bool check_sizes = true;
  // diagnostics (per step, reset in begin_step) -- measured with steady_clock around the driver calls only
  int64_t step_copies = 0, step_bytes = 0, step_memsets = 0, step_misses = 0, step_strided = 0, step_fallback = 0;
  double step_issue_s = 0.0;
  // run totals
  int64_t tot_copies = 0, tot_bytes = 0, tot_memsets = 0, tot_misses = 0, tot_learn_steps = 0, tot_strided = 0, tot_fallback = 0;
  // reference integrity: acquisitions the required-value manifest ATTEMPTED (counted by the callers before the put) vs acquisitions
  // that were actually bound (counted after a successful put); a refusal mid-put leaves expected > committed
  int64_t tot_expected = 0, tot_committed = 0;
  void note_expected(int64_t n) { tot_expected += n; }
  double tot_issue_s = 0.0;
  // stress test: an artificial extra delay (ms) before every memcpy issue -- NOT used in headline runs; see native_collect.py
  int64_t fault_skip_copy_at = -1;   // deliberate ordering fault for the negative test: skip the memcpy of this hook sequence number (bytes stale)

  // ------------------------------------------------------------------ WINDOWED staging ring
  // The packed step stream (same logical offsets as the single-block layout -- the plan is unchanged) is laid over fixed windows of W bytes
  // (W a multiple of 256, the L1 chunk) drawn from a ring of K device slots.  A tensor may straddle windows (its copy is split at the
  // boundary; logical bytes stay contiguous).  When the serving thread finishes writing a window (its bytes are all in flushed copies) it
  // records a CUDA event on the compute stream; the worker waits on that event, absorbs the window (D2H to host / leaves) and hands the
  // slot back with its own event; the next taker of the slot makes the compute stream wait on it.  When no slot is free the serving thread
  // flushes what it has, records the events of the completed windows and BLOCKS (GIL released) until the worker releases one: the engine
  // stalls instead of the collector growing.  Device staging is therefore K x W regardless of the step size.
  struct StepWin { std::vector<int> slots; std::vector<cudaEvent_t> events; int64_t used = 0; bool ended = false; int64_t step_id = -1; };
  bool windowed = false;
  std::vector<char*> win_ptrs;                 // slot -> device pointer (the tensors are owned by Python)
  int64_t win_bytes = 0;                       // W
  std::vector<int> win_free;                   // free slots (LIFO)
  std::vector<cudaEvent_t> win_ready;          // slot -> event recorded by the worker when it stopped reading the slot (nullptr = never used)
  // slot -> how many times the serving thread took it, and the (step, k) it currently holds: the lifetime negative
  // (VERITY_FAULT=release_before_keep on the windowed path) waits for the slot to be REUSED and refilled before taking its "retained" copy
  std::vector<int64_t> win_gen, win_occ_step, win_occ_k;
  std::vector<cudaEvent_t> ev_pool;            // recycled window-complete events
  std::mutex win_mu;
  std::condition_variable win_cv;
  int64_t win_waits = 0, win_blocked_steps = 0, step_win_waits = 0;
  double win_wait_s = 0.0, win_wait_timeout_s = 600.0, step_win_wait_s = 0.0;
  bool win_dead = false;                        // a slot wait timed out: every later stage() fails immediately (no silent 600 s re-waits)
  // fail-fast after "collector dead": the named throw below is the fail-closed path, but it needs the GIL
  // back and a Python frame willing to fail the run; when a worker holds the GIL inside a CUDA wait (or vLLM swallows the hook's exception)
  // the process can sit for minutes at 0 % GPU with every thread in futex_wait until a SIGTERM ends it.  So the moment
  // win_dead is set, a plain std::thread (no GIL, no Python, no CUDA) starts a grace timer: the Python side acknowledges the dead collector
  // (ack_dead) once it has recorded the failure -- a run that fails closed normally is never touched -- and if no ack arrives within
  // dead_grace_s the thread writes a marker file (JSON, plain stdio), says why on stderr and _exit(3)s the process: rc 3 = the same
  // "commit_delta refused / failed closed" code row_pod already reads as COMMIT FAIL.  Never a silent hang, never a silent PASS.
  // Liveness, not a one-shot timer: the grace is measured from the LAST sign of progress (progress(): a step ended on the serving thread or
  // the worker finished/failed a step), so a slow-but-progressing fail-closed run (a starved worker draining its backlog) is never killed,
  // while a process that stops progressing -- the serving thread blocked in the GIL futex, the worker spinning in wait_window, a join in
  // finalize on a step future that never completes -- dies within one grace of its last step.  ack_dead() (finalize, AFTER the worker join
  // returned and the record was built) stands the watchdog down for good.
  std::atomic<bool> dead_acked{false};
  std::atomic<bool> watchdog_started{false};
  std::atomic<int64_t> last_progress_ms{0};
  double dead_grace_s = 120.0;                  // VERITY_COLLECT_DEAD_GRACE_S (set from Python); the forced-hang negative uses seconds
  std::string watchdog_marker;                  // <out>/COLLECTOR_DEAD_HUNG.json (set from Python; empty = stderr only)
  static int64_t now_ms() { return static_cast<int64_t>(now_s() * 1000.0); }
  void set_watchdog(const std::string& marker, double grace_s) { watchdog_marker = marker; if (grace_s > 0) dead_grace_s = grace_s; }
  void ack_dead() { dead_acked = true; }
  void progress() { last_progress_ms = now_ms(); }
  void start_dead_watchdog(const char* why) {
    if (watchdog_started.exchange(true)) return;
    last_progress_ms = now_ms();
    const double grace = dead_grace_s; const std::string marker = watchdog_marker; const std::string reason = why ? why : "";
    std::thread([grace, marker, reason, this]() {
      for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        if (dead_acked.load()) return;                              // the run failed closed on its own; nothing to do
        if (now_ms() - last_progress_ms.load() > static_cast<int64_t>(grace * 1000.0)) break;
      }
      if (dead_acked.load()) return;
      if (!marker.empty()) {
        FILE* f = std::fopen(marker.c_str(), "w");
        if (f) {
          std::fprintf(f, "{\"schema\":\"verity-collect/dead-watchdog/v1\",\"reason\":\"collector dead and the process did not fail closed: no step "
                          "progress and no acknowledgement from the Python side for %d s (hung after the slot-wait timeout)\",\"cause\":\"%s\","
                          "\"grace_s\":%d,\"rc\":3}\n",
                       static_cast<int>(grace), reason.c_str(), static_cast<int>(grace));
          std::fclose(f);
        }
      }
      std::fprintf(stderr, "[collector] DEAD-WATCHDOG: collector dead (%s) and the process did not fail closed: no step progress for %d s -- exiting rc 3 "
                           "(COMMIT FAIL by name; marker %s)\n", reason.c_str(), static_cast<int>(grace), marker.empty() ? "-" : marker.c_str());
      std::fflush(stderr);
      _exit(3);
    }).detach();
  }
  std::unordered_map<int64_t, StepWin> step_wins;   // step id -> its windows (erased by the worker's finish_step)
  StepWin* sw = nullptr;                       // current step's windows (serving thread)
  int64_t win_marked = 0;                      // windows [0, win_marked) of the current step have their complete-event recorded
  // The run failed by name: the worker abandoned a step (abandon_step; the first reason stands).  Every later window request refuses with
  // the reason (refuse_windows) rather than read the abandoned step's erased windows, a step's end leaves the plans as they were, and the
  // committer's finalize reports it; end_run clears it.  Written with the GIL and win_mu held; read under either.
  bool failed = false;
  std::string fail_reason;
  int64_t tot_refused = 0;                     // window requests refused since (stats "refused_after_failure")
  static constexpr const char* kRefused = "windowed staging: the collector failed earlier in this run: ";
  [[noreturn]] void refuse_windows() {
    if (failed) { ++tot_refused; throw std::runtime_error(kRefused + fail_reason); }
    throw std::runtime_error("windowed staging: the current step holds no windows (it began before enable_windows)");
  }

  void enable_windows(const std::vector<int64_t>& ptrs, int64_t W) {
    if (W <= 0 || W % 256 != 0) throw std::runtime_error("enable_windows: W must be a positive multiple of 256");
    std::lock_guard<std::mutex> g(win_mu);
    win_ptrs.clear(); win_free.clear();
    for (auto p : ptrs) win_ptrs.push_back(reinterpret_cast<char*>(p));
    for (cudaEvent_t e : win_ready) if (e) cudaEventDestroy(e);
    win_ready.assign(win_ptrs.size(), nullptr);
    win_gen.assign(win_ptrs.size(), 0); win_occ_step.assign(win_ptrs.size(), -1); win_occ_k.assign(win_ptrs.size(), -1);
    for (int i = static_cast<int>(win_ptrs.size()) - 1; i >= 0; --i) win_free.push_back(i);
    win_bytes = W;
    windowed = true;
  }
  void disable_windows() { std::lock_guard<std::mutex> g(win_mu); windowed = false; }

  // CALLER HOLDS win_mu: ev_pool is pushed by the worker's finish_step (under win_mu) and popped here by the serving thread; an unlocked
  // pop racing a push+sort handed the SAME event out twice -> the later record moved the worker's wait to a later point of the compute
  // stream, behind that stream's wait on a slot the worker had yet to release -> GPU-side cycle (B=32 v2 / B=64 hangs, 07:3xZ)
  cudaEvent_t take_event_locked() {
    if (!ev_pool.empty()) { cudaEvent_t e = ev_pool.back(); ev_pool.pop_back(); return e; }
    cudaEvent_t e = nullptr;
    cudaError_t rc = cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaEventCreate: ") + cudaGetErrorString(rc));
    return e;
  }

  inline char* dst_ptr(int64_t off) const {
    if (!windowed) return arena + off;
    return win_ptrs[sw->slots[static_cast<size_t>(off / win_bytes)]] + (off % win_bytes);
  }

  // record the complete-events of the windows whose bytes are all covered by ISSUED copies (logical end `upto`); serving thread, compute
  // stream (or the side stream, from a side flush)
  void mark_windows(int64_t upto, cudaStream_t side = nullptr) {
    if (!windowed || !sw) return;
    if (!side) join_side();
    cudaStream_t s = side ? side : c10::cuda::getCurrentCUDAStream().stream();
    const int64_t done_w = upto / win_bytes;                          // windows [0, done_w) are fully written
    if (win_marked >= done_w || win_marked >= static_cast<int64_t>(sw->slots.size())) return;
    std::lock_guard<std::mutex> g(win_mu);
    cudaEvent_t e = take_event_locked();                              // ONE event (stream order) serves every window completed at this point
    cudaEventRecord(e, s);
    while (win_marked < done_w && win_marked < static_cast<int64_t>(sw->slots.size())) sw->events[static_cast<size_t>(win_marked++)] = e;
    win_cv.notify_all();
  }

  // make sure logical bytes [0, end_off) have a slot; may block for the worker (GIL released); called with the GIL held from stage()/collect()
  void ensure_windows(int64_t end_off) {
    if (!windowed) return;
    if (win_dead) throw std::runtime_error("windowed staging: collector dead after a slot-wait timeout earlier in this run");
    if (failed || !sw) refuse_windows();
    const int64_t k_last = (end_off - 1) / win_bytes;
    while (static_cast<int64_t>(sw->slots.size()) <= k_last) {
      int slot = -1;
      {
        std::lock_guard<std::mutex> g(win_mu);
        if (!win_free.empty()) { slot = win_free.back(); win_free.pop_back(); }
      }
      if (slot < 0) {
        if (ring) flush();                                          // let the worker absorb what is already staged (earlier copies are always legal)
        mark_windows(pushed_end);
        const double t0 = now_s();
        ++win_waits; ++step_win_waits;
        {
          // LOCK ORDER: the GIL is dropped BEFORE win_mu is taken and win_mu is dropped BEFORE the GIL is re-taken (reverse destruction),
          // otherwise the worker (GIL held, wants win_mu in release_slot) and this thread (win_mu held, wants the GIL) deadlock
          py::gil_scoped_release nogil;
          std::unique_lock<std::mutex> lk(win_mu);
          while (win_free.empty() && !failed) {
            win_cv.wait_for(lk, std::chrono::milliseconds(200));
            if (win_free.empty() && now_s() - t0 > win_wait_timeout_s) {
              win_dead = true;
              fprintf(stderr, "[collector] windowed staging: waited %d s for a free window slot -- the worker is not releasing windows; collector dead for this run\n", static_cast<int>(win_wait_timeout_s));
              fflush(stderr);
              start_dead_watchdog("windowed staging: slot-wait timeout, the worker is not releasing windows");
            }
            if (win_dead)
              throw std::runtime_error("windowed staging: waited " + std::to_string(static_cast<int>(win_wait_timeout_s)) + " s for a free window slot (ring of "
                                       + std::to_string(win_ptrs.size()) + " x " + std::to_string(win_bytes >> 20) + " MiB, all held): the worker is not releasing windows");
          }
          if (!failed) { slot = win_free.back(); win_free.pop_back(); }
        }
        win_wait_s += now_s() - t0;
        step_win_wait_s += now_s() - t0;
        // the worker may abandon the step while this thread waits for a slot or for the GIL back (abandon_step: `sw` erased).  A slot taken
        // here whose last occupant was one of the step's windows went back with the step's: it must not go back twice (two windows of a
        // later step would share it)
        if (failed || !sw) {
          if (slot >= 0) {
            std::lock_guard<std::mutex> g(win_mu);
            if (std::find(win_free.begin(), win_free.end(), slot) == win_free.end()) win_free.push_back(slot);
            win_cv.notify_all();
          }
          refuse_windows();
        }
      }
      if (win_ready[slot]) {                                        // the worker's reads of the previous occupant precede our writes
        cudaError_t rc = cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream().stream(), win_ready[slot], 0);
        if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaStreamWaitEvent(slot): ") + cudaGetErrorString(rc));
      }
      {
        std::lock_guard<std::mutex> g(win_mu);                      // the worker reads these vectors under the same lock
        sw->slots.push_back(slot);
        sw->events.push_back(nullptr);
        ++win_gen[slot]; win_occ_step[slot] = sw->step_id; win_occ_k[slot] = static_cast<int64_t>(sw->slots.size()) - 1;
      }
    }
  }
  int64_t slot_generation(int64_t slot) { std::lock_guard<std::mutex> g(win_mu); return win_gen[static_cast<size_t>(slot)]; }
  // NEGATIVE-TEST ONLY (VERITY_FAULT=release_before_keep, windowed path): block (GIL released) until `slot` has been taken
  // again since generation `gen` AND its new occupant window is complete (its bytes are all written), then make `stream` wait on that
  // completion so a copy enqueued next reads the NEW occupant's bytes.  Returns false on timeout (the run ended / no further window).
  bool wait_slot_reused(int64_t slot, int64_t gen, double timeout_s, int64_t stream_ptr) {
    cudaEvent_t e = nullptr;
    {
      py::gil_scoped_release nogil;
      std::unique_lock<std::mutex> lk(win_mu);
      const double t0 = now_s();
      for (;;) {
        if (win_gen[static_cast<size_t>(slot)] > gen) {
          auto it = step_wins.find(win_occ_step[static_cast<size_t>(slot)]);
          if (it != step_wins.end()) {
            const size_t k = static_cast<size_t>(win_occ_k[static_cast<size_t>(slot)]);
            if (k < it->second.events.size() && it->second.events[k]) { e = it->second.events[k]; break; }
          }
        }
        if (now_s() - t0 > timeout_s) break;
        win_cv.wait_for(lk, std::chrono::milliseconds(2));
      }
    }
    if (!e) return false;
    cudaError_t rc = cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream_ptr), e, 0);
    if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaStreamWaitEvent(reused slot): ") + cudaGetErrorString(rc));
    return true;
  }

  // ---- worker side
  // blocks (GIL released) until window k of `step_id` is complete (returns (true, used_bytes, slot) after making `stream` wait on its
  // event) or until the step ended with fewer windows (returns (false, 0, -1))
  py::tuple wait_window(int64_t step_id, int64_t k, int64_t stream_ptr) {
    cudaEvent_t e = nullptr; int slot = -1; int64_t used = 0;
    {
      py::gil_scoped_release nogil;
      std::unique_lock<std::mutex> lk(win_mu);
      for (;;) {
        auto it = step_wins.find(step_id);
        if (it != step_wins.end()) {
          StepWin& w = it->second;
          if (k < static_cast<int64_t>(w.events.size()) && w.events[static_cast<size_t>(k)]) {
            e = w.events[static_cast<size_t>(k)]; slot = w.slots[static_cast<size_t>(k)];
            const int64_t start = k * win_bytes;
            used = w.ended ? std::min(win_bytes, w.used - start) : win_bytes;
            if (used < 0) used = 0;
            break;
          }
          if (w.ended && k >= static_cast<int64_t>(w.slots.size())) break;
        }
        // the serving thread declared the collector dead: it stopped staging, so window k's event will
        // never be recorded and this step never "ends" -- return (false, -1, -1) BY NAME instead of spinning until COMMIT_TIMEOUT (the
        // Python k-loop raises "collector dead" for the step; the run then fails closed within a step, not at the workload timeout).
        if (win_dead) { used = -1; break; }
        win_cv.wait_for(lk, std::chrono::milliseconds(2));
      }
    }
    if (!e) return py::make_tuple(false, used, static_cast<int64_t>(-1));
    cudaError_t rc = cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream_ptr), e, 0);
    if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaStreamWaitEvent(window): ") + cudaGetErrorString(rc));
    return py::make_tuple(true, used, static_cast<int64_t>(slot));
  }
  // the worker gave up on `step_id` (exception): the run fails by name (`why`; the first stands, printed once), and every slot the step
  // still holds goes back to the ring, once, after the worker's reads queued on `stream_ptr` (0: none), so the serving thread stops waiting
  // for a release that would never come.  A slot the worker already released, or that a later step has taken since, stays where it is;
  // one the serving thread took off the free list and has not recorded yet goes back here if its last occupant was one of this step's
  // windows (ensure_windows then leaves it).
  void abandon_step(int64_t step_id, const std::string& why, int64_t stream_ptr) {
    std::lock_guard<std::mutex> g(win_mu);
    if (!failed) {
      fail_reason = why.empty() ? "the worker abandoned step " + std::to_string(step_id) : why;
      failed = true;
      std::fprintf(stderr, "[collector] %s\n", fail_reason.c_str());
    }
    auto it = step_wins.find(step_id);
    if (it != step_wins.end()) {
      for (int slot : it->second.slots) {
        if (win_occ_step[slot] != step_id || std::find(win_free.begin(), win_free.end(), slot) != win_free.end()) continue;
        if (stream_ptr) {
          if (!win_ready[slot]) cudaEventCreateWithFlags(&win_ready[slot], cudaEventDisableTiming);
          if (win_ready[slot]) cudaEventRecord(win_ready[slot], reinterpret_cast<cudaStream_t>(stream_ptr));
        }
        win_free.push_back(slot);
      }
      if (sw == &it->second) sw = nullptr;
      step_wins.erase(it);
    }
    win_cv.notify_all();
  }
  // the worker is done reading `slot` once everything it enqueued on `stream` so far has run
  void release_slot(int64_t slot, int64_t stream_ptr) {
    std::lock_guard<std::mutex> g(win_mu);
    if (!win_ready[slot]) {
      cudaError_t rc = cudaEventCreateWithFlags(&win_ready[slot], cudaEventDisableTiming);
      if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaEventCreate(ready): ") + cudaGetErrorString(rc));
    }
    cudaEventRecord(win_ready[slot], reinterpret_cast<cudaStream_t>(stream_ptr));
    win_free.push_back(static_cast<int>(slot));
    win_cv.notify_all();
  }
  py::tuple step_windows(int64_t step_id) {         // (ended, n_windows, used_bytes)
    std::lock_guard<std::mutex> g(win_mu);
    auto it = step_wins.find(step_id);
    if (it == step_wins.end()) return py::make_tuple(false, static_cast<int64_t>(0), static_cast<int64_t>(0));
    return py::make_tuple(it->second.ended, static_cast<int64_t>(it->second.slots.size()), it->second.used);
  }
  void finish_step(int64_t step_id) {
    std::lock_guard<std::mutex> g(win_mu);
    auto it = step_wins.find(step_id);
    if (it == step_wins.end()) return;
    for (cudaEvent_t e : it->second.events) if (e) ev_pool.push_back(e);   // (an event shared by several windows is pushed once per window: harmless duplicates avoided below)
    std::sort(ev_pool.begin(), ev_pool.end()); ev_pool.erase(std::unique(ev_pool.begin(), ev_pool.end()), ev_pool.end());
    step_wins.erase(it);
  }
  int64_t plan_used(int64_t key) const {                 // the step length a HIT plan predicts (-1: no plan yet -> learning step)
    const Plan* p = plans.find(key);
    if (!p || p->entries.empty()) return -1;
    return p->used;
  }
  int64_t plan_version(int64_t key) const {              // the version of the plan a hit step follows (the worker's metas cache key)
    const Plan* p = plans.find(key);
    return p ? p->version : -1;
  }
  // the ring's free slots now (the worker's: under win_mu with the GIL held, as release_slot)
  int64_t free_slots() {
    std::lock_guard<std::mutex> g(win_mu);
    return static_cast<int64_t>(win_free.size());
  }

  py::dict window_stats() const {
    py::dict d;
    d["windowed"] = windowed; d["slots"] = static_cast<int64_t>(win_ptrs.size()); d["window_bytes"] = win_bytes;
    d["waits"] = win_waits; d["wait_s"] = win_wait_s; d["steps_pending"] = static_cast<int64_t>(step_wins.size());
    return d;
  }

  void begin_step(int64_t key, int64_t arena_ptr, int64_t nbytes, bool use_plan, int64_t step_id = -1) {
    if (active) throw std::runtime_error("begin_step while a step is active");
    plan_key = key;
    plan = &plans.use(key);
    arena = reinterpret_cast<char*>(arena_ptr);
    arena_bytes = nbytes;
    if (windowed) {
      std::lock_guard<std::mutex> g(win_mu);
      sw = &step_wins[step_id];
      *sw = StepWin();
      sw->step_id = step_id;
      arena = nullptr;
      arena_bytes = std::numeric_limits<int64_t>::max();          // no per-step size limit: backpressure bounds the device footprint
      pushed_end = 0; win_marked = 0;
    } else {
      sw = nullptr;
    }
    cursor = 0;
    byte_cursor = 0;
    learning = !use_plan || plan->entries.empty();
    changed = false;
    active = true;
    step_copies = step_bytes = step_memsets = step_misses = step_strided = step_fallback = 0;
    step_issue_s = 0.0;
    step_flushes = 0;
    step_win_waits = 0; step_win_wait_s = 0.0;
    side_next = false;
  }

  // A non-contiguous tensor whose memory is a set of equally spaced contiguous rows (e.g. RoPE's q/k views into the qkv buffer:
  // [T, H*D] with strides (QKV, 1)) is copied with ONE cudaMemcpy2DAsync (rows x width, source pitch) -- the logical (row-major) bytes
  // land contiguously in the arena, exactly what the Python path's `.contiguous()` + `copy_` produces.  Returns false if the layout is not 2-D pitched.
  static bool pitched_2d(const at::Tensor& t, int64_t& rows, int64_t& width_b, int64_t& pitch_b) {
    const int64_t is = t.element_size();
    std::vector<int64_t> sz, st;
    for (int64_t i = 0; i < t.dim(); ++i) {
      if (t.size(i) == 1) continue;                                // size-1 dims carry no layout information
      sz.push_back(t.size(i)); st.push_back(t.stride(i));
    }
    if (sz.empty()) { rows = 1; width_b = is; pitch_b = is; return true; }
    // largest contiguous suffix
    int64_t k = static_cast<int64_t>(sz.size()) - 1;
    int64_t expect = 1;
    while (k >= 0 && st[k] == expect) { expect *= sz[k]; --k; }
    // k = last non-contiguous dim (or -1 => fully contiguous)
    if (k < 0) { rows = 1; width_b = expect * is; pitch_b = width_b; return true; }
    // dims [0..k] must collapse into one pitched dim
    for (int64_t j = 0; j < k; ++j) {
      if (st[j] != st[j + 1] * sz[j + 1]) return false;
    }
    rows = 1;
    for (int64_t j = 0; j <= k; ++j) rows *= sz[j];
    width_b = expect * is;
    pitch_b = st[k] * is;
    return pitch_b >= width_b;
  }

  inline bool straddles(int64_t off, int64_t nb) const { return windowed && nb > 0 && (off / win_bytes) != ((off + nb - 1) / win_bytes); }

  // contiguous source bytes -> logical [off, off+nb), split at window boundaries
  inline cudaError_t copy_segments(int64_t off, const char* src, int64_t nb, cudaStream_t s) {
    cudaError_t rc = cudaSuccess;
    int64_t done = 0;
    while (done < nb && rc == cudaSuccess) {
      const int64_t o = off + done;
      const int64_t seg = std::min(nb - done, win_bytes - (o % win_bytes));
      rc = cudaMemcpyAsync(dst_ptr(o), src + done, seg, cudaMemcpyDeviceToDevice, s);
      done += seg;
    }
    return rc;
  }

  inline void issue(int64_t off, int64_t gap, const at::Tensor& t, int64_t nb, bool skip_copy) {
    join_side();
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();   // the PRODUCER's stream (same thread, same stream as the kernel that made `src`)
    ensure_windows(off + nb);                                        // (no-op unless windowed; may block for the worker)
    double t0 = now_s();
    if (gap > 0) {                                                   // gap <= 3 and off 4-aligned: never straddles a (256-aligned) window boundary
      cudaError_t rc = cudaMemsetAsync(dst_ptr(off - gap), 0, gap, s);
      if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemsetAsync: ") + cudaGetErrorString(rc));
      ++step_memsets;
    }
    if (!skip_copy) {
      cudaError_t rc;
      if (t.is_contiguous()) {
        rc = straddles(off, nb) ? copy_segments(off, reinterpret_cast<const char*>(t.data_ptr()), nb, s)
                                : cudaMemcpyAsync(dst_ptr(off), t.data_ptr(), nb, cudaMemcpyDeviceToDevice, s);
      } else {
        int64_t rows, width_b, pitch_b;
        if (!straddles(off, nb) && pitched_2d(t, rows, width_b, pitch_b)) {
          rc = cudaMemcpy2DAsync(dst_ptr(off), width_b, t.data_ptr(), pitch_b, width_b, rows, cudaMemcpyDeviceToDevice, s);
          ++step_strided;
        } else {
          at::Tensor c = t.contiguous();                             // general fallback (a torch op; counted) -- not hit on the reference workload
          rc = copy_segments(off, reinterpret_cast<const char*>(c.data_ptr()), nb, s);
          ++step_fallback;
        }
      }
      if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync: ") + cudaGetErrorString(rc));
    }
    step_issue_s += now_s() - t0;
    ++step_copies;
    step_bytes += nb;
    if (windowed && !ring) mark_windows(off + nb);                   // memcpy path: the copies are issued, completed windows can be absorbed
  }

  // Returns the arena offset (>= 0) of the collected tensor, or -1 when nothing was collected (0 bytes: the Python path skips those too).
  int64_t collect(int64_t name_id, const at::Tensor& t) {
    if (!active) return -1;
    if (!t.is_cuda()) return -1;                                   // as `_acquire`: non-CUDA tensors are not acquired
    const int64_t nb = static_cast<int64_t>(t.nbytes());               // numel * itemsize: the logical bytes, as the Python `.contiguous()` path
    if (nb == 0) return -1;
    const int32_t dt = static_cast<int32_t>(t.scalar_type());
    const bool skip = (static_cast<int64_t>(cursor) == fault_skip_copy_at);
    if (!learning) {
      if (cursor < plan->entries.size()) {
        const Entry& e = plan->entries[cursor];
        bool hit = (e.name_id == name_id && e.nbytes == nb && e.dtype == dt);
        if (hit && check_sizes) {
          auto sz = t.sizes();
          hit = (static_cast<size_t>(sz.size()) == e.sizes.size());
          for (size_t i = 0; hit && i < e.sizes.size(); ++i) hit = (sz[i] == e.sizes[i]);
        }
        if (hit) {
          issue(e.off, e.gap, t, nb, skip);
          byte_cursor = e.off + nb;
          ++cursor;
          return e.off;
        }
      }
      // miss: rebuild the plan from here on (this step and later steps of this shape class)
      plan->entries.resize(cursor);
      learning = true;
      changed = true;
      ++step_misses;
    }
    const int64_t off = (byte_cursor + 3) & ~static_cast<int64_t>(3);    // as `_acquire_by_cudart`: 4-byte alignment, gap zeroed
    const int64_t gap = off - byte_cursor;
    if (off + nb > arena_bytes) throw std::runtime_error("native_collect: arena block too small for this step (no overflow chaining in the candidate): needed >= "
                                                        + std::to_string(off + nb) + " B (tensor " + std::to_string(nb) + " B at offset " + std::to_string(off)
                                                        + "), block " + std::to_string(arena_bytes) + " B -- size the arena from the workload envelope (auto, "
                                                        "--arena-mb 0) or lower gpu_memory_utilization / max_num_batched_tokens");
    issue(off, gap, t, nb, skip);
    Entry e;
    e.name_id = name_id; e.nbytes = nb; e.off = off; e.gap = gap; e.dtype = dt;
    auto sz = t.sizes();
    e.sizes.assign(sz.begin(), sz.end());
    plan->entries.push_back(std::move(e));
    changed = true;                                                // a learning step always publishes its plan
    byte_cursor = off + nb;
    ++cursor;
    return off;
  }

  // -> (used_bytes, n_entries, changed, version)
  py::tuple end_step() {
    if (!active) throw std::runtime_error("end_step without begin_step");
    if (ring_pos != staged_from) throw std::runtime_error("end_step with staged copies not flushed");
    join_side();
    active = false;
    if (windowed && sw) {
      // the last (partial) window: one event after everything issued this step
      // ONE critical section for the last events AND (used, ended): the worker computes the last window's length as
      // `ended ? used - start : W`, so publishing the event before `ended` would let it read the partial window as a full one
      std::lock_guard<std::mutex> g(win_mu);
      cudaEvent_t e = nullptr;
      if (!sw->slots.empty() && win_marked < static_cast<int64_t>(sw->slots.size())) {
        e = take_event_locked();
        cudaEventRecord(e, c10::cuda::getCurrentCUDAStream().stream());
      }
      if (e) while (win_marked < static_cast<int64_t>(sw->slots.size())) sw->events[static_cast<size_t>(win_marked++)] = e;
      sw->used = byte_cursor;
      sw->ended = true;
      win_cv.notify_all();
      sw = nullptr;
    }
    if (failed) {                                            // the step's later acquisitions were refused: a plan it followed is not
      if (changed) plan->entries.clear();                    // stale, and one it was learning is incomplete (its key learns again)
    } else if (!learning && cursor < plan->entries.size()) {   // fewer acquisitions than planned: the plan is stale
      plan->entries.resize(cursor);
      changed = true;
    }
    if (changed) {
      plan->version += 1;
      plan->used = plan->entries.empty() ? 0 : (plan->entries.back().off + plan->entries.back().nbytes);
      ++tot_learn_steps;
    }
    tot_copies += step_copies; tot_bytes += step_bytes; tot_memsets += step_memsets; tot_misses += step_misses; tot_issue_s += step_issue_s;
    tot_strided += step_strided; tot_fallback += step_fallback;
    return py::make_tuple(byte_cursor, static_cast<int64_t>(cursor), changed, plan->version);
  }

  // The serving thread's step end in one call: flush (gather), end_step, the dead-watchdog's progress, the hook errors (native hooks) and
  // the step stats the committer records -> (used_bytes, n_entries, changed, version, bytes, issue_s, window_waits, window_wait_s, errors)
  py::tuple end_step_full(bool do_flush, bool drain) {
    if (do_flush) flush();
    py::tuple r = end_step();
    if (win_dead) progress();
    py::list errs = drain && !errors.empty() ? drain_errors() : py::list();
    return py::make_tuple(r[0], r[1], r[2], r[3], step_bytes, step_issue_s, step_win_waits, step_win_wait_s, errs);
  }

  // Snapshot of the plan for `key`: list of (name_id, dtype, sizes, nbytes, off, gap) -- called by the serving thread only when `changed`.
  py::list export_plan(int64_t key) {
    py::list out;
    const Plan* p = plans.find(key);
    if (!p) return out;
    for (const Entry& e : p->entries) {
      py::tuple sizes(e.sizes.size());
      for (size_t i = 0; i < e.sizes.size(); ++i) sizes[i] = e.sizes[i];
      out.append(py::make_tuple(e.name_id, static_cast<c10::ScalarType>(e.dtype), sizes, e.nbytes, e.off, e.gap));
    }
    return out;
  }

  py::dict step_stats() const {
    py::dict d;
    d["copies"] = step_copies; d["bytes"] = step_bytes; d["memsets"] = step_memsets; d["misses"] = step_misses;
    d["issue_s"] = step_issue_s; d["learning"] = learning; d["strided"] = step_strided; d["fallback"] = step_fallback; d["flushes"] = step_flushes;
    d["window_waits"] = step_win_waits; d["window_wait_s"] = step_win_wait_s;
    return d;
  }

  py::dict stats() const {
    py::dict d;
    d["copies"] = tot_copies; d["bytes"] = tot_bytes; d["memsets"] = tot_memsets; d["misses"] = tot_misses;
    d["learn_steps"] = tot_learn_steps; d["issue_s"] = tot_issue_s; d["plans"] = plans.size(); d["plan_entries"] = plans.entries();
    d["plans_pending"] = plans.pending(); d["plan_entries_pending"] = plans.pending_entries();
    d["plans_dropped"] = plans.dropped(); d["plans_pending_emptied_step"] = plans.pending_emptied_step();
    d["strided_2d"] = tot_strided; d["contiguous_fallback"] = tot_fallback; d["flushes"] = tot_flushes; d["staged"] = tot_staged;
    d["gather_blocks"] = tot_blocks; d["auto_blocks"] = auto_blocks;
    d["gather_side_flushes"] = tot_side_flushes; d["drain_side"] = side_drain; d["gather_desc_copies"] = tot_desc_copies;
    d["expected"] = tot_expected; d["committed"] = tot_committed;
    d["window_waits"] = win_waits; d["window_wait_s"] = win_wait_s; d["windowed"] = windowed;
    d["refused_after_failure"] = tot_refused;
    return d;
  }

  void reset_stats() { tot_copies = tot_bytes = tot_memsets = tot_misses = tot_learn_steps = tot_strided = tot_fallback = 0; tot_issue_s = 0.0; tot_flushes = 0; tot_staged = 0; tot_blocks = 0; tot_side_flushes = 0; tot_desc_copies = 0; tot_expected = tot_committed = 0; tot_refused = 0; }
  void clear_plans() { plans.clear(); plan = nullptr; }
  void end_run() {             // the committer's reset between runs: the run totals, and the plans the run that ended did not use (freed by
    reset_stats();             // the next run's steps, native_plans.h), and the run's failure
    plans.end_run();
    if (!active) plan = nullptr;
    std::lock_guard<std::mutex> g(win_mu);
    failed = false;
    fail_reason.clear();
  }

  // ------------------------------------------------------------------ SECONDARY bet: staged gather-copy (native_gather.cu)
  // `stage()` = the same plan logic as `collect()` but the copy is only recorded (a reference to the tensor is HELD so the allocator cannot
  // reuse its storage -- owner 2: with refs held only in-place overwrites matter); `flush()` launches ONE gather kernel for everything
  // staged since the previous flush.  Legal flush points are the caller's responsibility (the P0/P1/P2 hooks + step end).
  struct GatherDesc { const char* src; char* dst; int64_t nbytes; int64_t width; int64_t pitch; int64_t skip; };
  GatherDesc* ring = nullptr;       // host-mapped pinned ring
  GatherDesc* ring_dev = nullptr;   // its device alias
  // A device copy of the ring: a flush copies its descriptors there (one H2D on the gather's stream) and the kernel reads them from
  // device memory, not over PCIe from every block (VERITY_GATHER_DESC_DEV=0, read at enable_gather, keeps the host-mapped reads).
  GatherDesc* ring_gpu = nullptr;
  int64_t ring_cap = 0, ring_pos = 0, staged_from = 0;
  std::vector<at::Tensor> held;
  int blocks_per_desc = 8;
  int64_t tot_flushes = 0, tot_staged = 0, step_flushes = 0, tot_blocks = 0;

  void enable_gather(int64_t capacity) {
    if (ring) return;
    void* h = nullptr;
    cudaError_t rc = cudaHostAlloc(&h, capacity * sizeof(GatherDesc), cudaHostAllocMapped | cudaHostAllocPortable);
    if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaHostAlloc: ") + cudaGetErrorString(rc));
    void* d = nullptr;
    rc = cudaHostGetDevicePointer(&d, h, 0);
    if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaHostGetDevicePointer: ") + cudaGetErrorString(rc));
    ring = reinterpret_cast<GatherDesc*>(h);
    ring_dev = reinterpret_cast<GatherDesc*>(d);
    ring_cap = capacity;
    ring_pos = staged_from = 0;
    if (const char* e = std::getenv("VERITY_DRAIN_SIDE")) side_drain = std::string(e) != "0";
    const char* dd = std::getenv("VERITY_GATHER_DESC_DEV");
    if (!dd || std::string(dd) != "0") {
      rc = cudaMalloc(&d, capacity * sizeof(GatherDesc));
      if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc(gather descriptors): ") + cudaGetErrorString(rc));
      ring_gpu = reinterpret_cast<GatherDesc*>(d);
    }
  }

  // the n descriptors at ring index start, where the gather on s reads them: copied to the device ring ahead of it on s, or (no device
  // ring, or s capturing a graph, whose replay must not re-copy a later step's ring) the host-mapped ring itself
  const GatherDesc* descs_for(int64_t start, int64_t n, cudaStream_t s) {
    if (!ring_gpu) return ring_dev + start;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) return ring_dev + start;
    const cudaError_t rc = cudaMemcpyAsync(ring_gpu + start, ring + start, n * sizeof(GatherDesc), cudaMemcpyHostToDevice, s);
    if (rc != cudaSuccess) throw std::runtime_error(std::string("gather descriptors H2D: ") + cudaGetErrorString(rc));
    ++tot_desc_copies;
    return ring_gpu + start;
  }

  inline void push_one(const char* src, char* dst, int64_t nb, int64_t width, int64_t pitch, int64_t skip) {
    if (ring_pos - staged_from >= ring_cap) throw std::runtime_error("gather ring full (flush more often)");
    GatherDesc& g = ring[ring_pos % ring_cap];
    g.src = src; g.dst = dst; g.nbytes = nb; g.width = width; g.pitch = pitch; g.skip = skip;
    if (nb > win_max_nb) win_max_nb = nb;
    ++ring_pos;
  }
  inline void push_desc(const char* src, int64_t off, int64_t nb, int64_t width, int64_t pitch) {
    if (!straddles(off, nb)) {
      ensure_windows(off + nb);                                      // (no-op unless windowed; may block for the worker)
      push_one(src, dst_ptr(off), nb, width, pitch, 0);
      if (windowed && off + nb > pushed_end) pushed_end = off + nb;
      return;
    }
    // windowed: split the logical range at window boundaries; a pitched source keeps its (width, pitch) and gets the segment's logical
    // offset as `skip` (the kernel maps logical byte j = i + skip -> row j / width, col j % width); a contiguous source just advances.
    // Slots are acquired PER SEGMENT: a tensor larger than the whole ring (FA2 M1 of a long prefill) is staged window by window -- when
    // the ring is exhausted the segments pushed so far are flushed, their windows marked complete, and this thread waits for the worker
    // to hand a slot back (backpressure: the engine stalls in the producer's hook rather than the collector growing)
    int64_t done = 0;
    while (done < nb) {
      const int64_t o = off + done;
      const int64_t seg = std::min(nb - done, win_bytes - (o % win_bytes));
      ensure_windows(o + seg);
      if (src == nullptr) push_one(nullptr, dst_ptr(o), seg, seg, seg, 0);
      else if (pitch == width) push_one(src + done, dst_ptr(o), seg, seg, seg, 0);
      else push_one(src, dst_ptr(o), seg, width, pitch, done);
      done += seg;
      if (o + seg > pushed_end) pushed_end = o + seg;
    }
  }

  // Size-adaptive gather geometry.  `blocks_per_desc` is the CAP (the committer's default 1024; 32 is the Llama geometry);
  // a decode window whose largest descriptor is a few KB gets ceil(max_nbytes / auto_bytes_per_block) blocks instead (1-2), so the 49
  // launches per decode step stop occupying thousands of mostly idle thread blocks on the serving stream (SM time the worker's SHA
  // kernels and the model's own kernels then compete for).  Same descriptors, same bytes, same destination offsets.
  bool auto_blocks = false;
  int64_t auto_bytes_per_block = 32768;
  int64_t win_max_nb = 0;
  inline int blocks_for_window() {
    if (!auto_blocks) return blocks_per_desc;
    int64_t b = (win_max_nb + auto_bytes_per_block - 1) / auto_bytes_per_block;
    if (b < 1) b = 1;
    if (b > blocks_per_desc) b = blocks_per_desc;
    return static_cast<int>(b);
  }

  int64_t stage(int64_t name_id, const at::Tensor& t) {
    if (!active) return -1;
    if (!t.is_cuda()) return -1;
    const int64_t nb = static_cast<int64_t>(t.nbytes());
    if (nb == 0) return -1;
    int64_t off;
    if (t.is_contiguous()) {
      off = stage_bytes(name_id, reinterpret_cast<const char*>(t.data_ptr()), nb, static_cast<int32_t>(t.scalar_type()), t.sizes(), nb, nb);
    } else {
      int64_t rows, width_b, pitch_b;
      if (!pitched_2d(t, rows, width_b, pitch_b)) throw std::runtime_error("native_collect.stage: layout is not 2-D pitched");
      off = stage_bytes(name_id, reinterpret_cast<const char*>(t.data_ptr()), nb, static_cast<int32_t>(t.scalar_type()), t.sizes(), width_b, pitch_b);
      ++step_strided;
    }
    held.push_back(t);                  // keep the storage alive until the flush has been ENQUEUED (stream order then protects the read)
    return off;
  }

  // stage()'s plan logic and descriptors for `nb` bytes at `src` (contiguous when width_b == pitch_b == nb), whose owner keeps them alive
  int64_t stage_bytes(int64_t name_id, const char* src, int64_t nb, int32_t dt, c10::IntArrayRef sizes, int64_t width_b, int64_t pitch_b) {
    int64_t off, gap;
    if (!learning) {
      if (cursor < plan->entries.size()) {
        const Entry& e = plan->entries[cursor];
        bool hit = (e.name_id == name_id && e.nbytes == nb && e.dtype == dt);
        if (hit && check_sizes) {
          hit = (sizes.size() == e.sizes.size());
          for (size_t i = 0; hit && i < e.sizes.size(); ++i) hit = (sizes[i] == e.sizes[i]);
        }
        if (hit) { off = e.off; gap = e.gap; goto record; }
      }
      plan->entries.resize(cursor);
      learning = true;
      changed = true;
      ++step_misses;
    }
    off = (byte_cursor + 3) & ~static_cast<int64_t>(3);
    gap = off - byte_cursor;
    if (off + nb > arena_bytes) throw std::runtime_error("native_collect: arena block too small for this step: needed >= " + std::to_string(off + nb)
                                                        + " B (tensor " + std::to_string(nb) + " B at offset " + std::to_string(off) + "), block "
                                                        + std::to_string(arena_bytes) + " B -- size the arena from the workload envelope (auto, --arena-mb 0)");
    {
      Entry e;
      e.name_id = name_id; e.nbytes = nb; e.off = off; e.gap = gap; e.dtype = dt;
      e.sizes.assign(sizes.begin(), sizes.end());
      plan->entries.push_back(std::move(e));
      changed = true;
    }
  record:
    if (gap > 0) { push_desc(nullptr, off - gap, gap, gap, gap); ++step_memsets; }
    push_desc(src, off, nb, width_b, pitch_b);
    ++step_copies; step_bytes += nb; ++tot_staged;
    byte_cursor = off + nb;
    ++cursor;
    return off;
  }

  // A replayed graph's slots (`stage_slot_plan`, slots only: nothing held, whose storage the allocator could hand to the serving stream)
  // are gathered on a side stream, so the serving stream's next kernels (the logits GEMM, the sampler) need not wait for the copy. The
  // serving stream joins it (`join_side`) before its next copy, window mark or step end: a window it completes holds this gather's
  // bytes, and the next replay rewrites the slots. VERITY_DRAIN_SIDE=0, read at enable_gather, keeps every flush on the serving stream.
  bool side_drain = true, side_next = false, side_pending = false;
  cudaStream_t side_s = nullptr;
  cudaEvent_t side_fork = nullptr, side_done = nullptr;
  int64_t tot_side_flushes = 0, tot_desc_copies = 0;

  void join_side() {
    if (!side_pending) return;
    side_pending = false;
    cudaError_t rc = cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream().stream(), side_done, 0);
    if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaStreamWaitEvent(side drain): ") + cudaGetErrorString(rc));
  }

  cudaStream_t fork_side(cudaStream_t cur) {
    if (!side_s) {
      int prio = 0;                                                  // the serving stream's: under VERITY_HI_PRIO, ahead of the hash's blocks
      cudaError_t rc = cudaStreamGetPriority(cur, &prio);
      if (rc == cudaSuccess) rc = cudaStreamCreateWithPriority(&side_s, cudaStreamNonBlocking, prio);
      if (rc == cudaSuccess) rc = cudaEventCreateWithFlags(&side_fork, cudaEventDisableTiming);
      if (rc == cudaSuccess) rc = cudaEventCreateWithFlags(&side_done, cudaEventDisableTiming);
      if (rc != cudaSuccess) throw std::runtime_error(std::string("side drain stream: ") + cudaGetErrorString(rc));
    }
    cudaError_t rc = cudaEventRecord(side_fork, cur);
    if (rc == cudaSuccess) rc = cudaStreamWaitEvent(side_s, side_fork, 0);
    if (rc != cudaSuccess) throw std::runtime_error(std::string("side drain fork: ") + cudaGetErrorString(rc));
    return side_s;
  }

  void flush() {
    const int64_t n = ring_pos - staged_from;
    const bool side = side_next && windowed && held.empty();
    side_next = false;
    if (n <= 0) return;
    if (!ring) throw std::runtime_error("gather not enabled");
    if (!side) join_side();
    cudaStream_t cur = c10::cuda::getCurrentCUDAStream().stream();
    cudaStream_t s = side ? fork_side(cur) : cur;
    double t0 = now_s();
    const int64_t start = staged_from % ring_cap;
    const int bpd = blocks_for_window();
    win_max_nb = 0;
    tot_blocks += static_cast<int64_t>(bpd) * n;
    if (start + n <= ring_cap) {
      int rc = verity_gather_launch(descs_for(start, n, s), static_cast<int>(n), bpd, s);
      if (rc != 0) throw std::runtime_error(std::string("gather launch: ") + cudaGetErrorString(static_cast<cudaError_t>(rc)));
    } else {                                                         // ring wrap: two launches
      const int64_t n1 = ring_cap - start;
      int rc = verity_gather_launch(descs_for(start, n1, s), static_cast<int>(n1), bpd, s);
      if (rc == 0) rc = verity_gather_launch(descs_for(0, n - n1, s), static_cast<int>(n - n1), bpd, s);
      if (rc != 0) throw std::runtime_error(std::string("gather launch: ") + cudaGetErrorString(static_cast<cudaError_t>(rc)));
    }
    step_issue_s += now_s() - t0;
    staged_from = ring_pos;
    held.clear();                       // the reads are enqueued in stream order before any later kernel on this stream
    ++step_flushes; ++tot_flushes;
    if (windowed) mark_windows(pushed_end, side ? s : nullptr);   // every window fully behind the pushed frontier is now in flushed copies
    if (!side) return;
    const cudaError_t rc = cudaEventRecord(side_done, s);
    if (rc != cudaSuccess) throw std::runtime_error(std::string("side drain done: ") + cudaGetErrorString(rc));
    side_pending = true;
    ++tot_side_flushes;
  }

  // ------------------------------------------------------------------ collector v2: native hook bodies (no Python frames per producer)
  bool use_gather = false;                 // HookFwd / Fa2Wrap route through stage() (gather) or collect() (one memcpy per tensor)
  std::vector<std::string> errors;         // hook-body exceptions (never propagated into the model's forward; drained by the committer)
  inline int64_t put(int64_t name_id, const at::Tensor& t) { const int64_t r = use_gather ? stage(name_id, t) : collect(name_id, t); ++tot_committed; return r; }

  // A replayed CUDA graph's slots (the persistent buffers its captured copies write), registered once: `stage_slot_plan` then puts them
  // in this order in one call, each whole (kind 0), its first n rows (1) or its first pad rows (2) -- per slot the same expected count,
  // put and plan entry as one Python acquisition of that view, without a Python frame and a slice per slot
  // A contiguous CUDA slot under gather (`slot_fast`) is staged from its pointer, bytes per row and sizes, read once here: the same plan
  // entry and descriptors as staging the view, without a tensor per slot per step (the plan holds the slot, so nothing is `held`).
  struct SlotPlan {
    std::vector<int64_t> ids; std::vector<at::Tensor> slots; std::vector<int64_t> kinds;
    std::vector<char> fast; std::vector<const char*> ptrs; std::vector<int64_t> row_bytes; std::vector<int32_t> dtypes;
    std::vector<std::vector<int64_t>> sizes;
  };
  std::vector<SlotPlan> slot_plans;
  bool slot_fast = true;
  int64_t add_slot_plan(std::vector<int64_t> ids, std::vector<at::Tensor> slots, std::vector<int64_t> kinds) {
    if (ids.size() != slots.size() || kinds.size() != slots.size()) throw std::runtime_error("add_slot_plan: one name id and one kind per slot");
    SlotPlan p;
    for (size_t i = 0; i < slots.size(); ++i) {
      const at::Tensor& s = slots[i];
      if (kinds[i] < 0 || kinds[i] > 2) throw std::runtime_error("add_slot_plan: kind must be 0 (whole), 1 (first n rows) or 2 (first pad rows)");
      if (kinds[i] != 0 && s.dim() == 0) throw std::runtime_error("add_slot_plan: a row-sliced slot needs a leading dimension");
      const bool fast = s.is_cuda() && s.is_contiguous();
      p.fast.push_back(fast);
      p.ptrs.push_back(fast ? reinterpret_cast<const char*>(s.data_ptr()) : nullptr);
      p.row_bytes.push_back(fast && s.dim() > 0 && s.size(0) > 0 ? static_cast<int64_t>(s.nbytes()) / s.size(0) : 0);
      p.dtypes.push_back(static_cast<int32_t>(s.scalar_type()));
      p.sizes.emplace_back(s.sizes().begin(), s.sizes().end());
    }
    p.ids = std::move(ids); p.slots = std::move(slots); p.kinds = std::move(kinds);
    slot_plans.push_back(std::move(p));
    return static_cast<int64_t>(slot_plans.size()) - 1;
  }
  int64_t stage_slot_plan(int64_t handle, int64_t n, int64_t pad) {
    if (handle < 0 || handle >= static_cast<int64_t>(slot_plans.size())) throw std::runtime_error("stage_slot_plan: unknown handle " + std::to_string(handle));
    SlotPlan& p = slot_plans[static_cast<size_t>(handle)];
    const bool fast = slot_fast && use_gather && active;
    for (size_t i = 0; i < p.slots.size(); ++i) {
      const at::Tensor& s = p.slots[i];
      note_expected(1);
      const int64_t rows = p.kinds[i] == 0 ? 0 : std::max<int64_t>(0, std::min<int64_t>(p.kinds[i] == 1 ? n : pad, s.size(0)));
      if (fast && p.fast[i]) {
        std::vector<int64_t>& sz = p.sizes[i];
        int64_t nb = static_cast<int64_t>(s.nbytes());
        if (p.kinds[i] != 0) { sz[0] = rows; nb = rows * p.row_bytes[i]; }
        if (nb > 0) stage_bytes(p.ids[i], p.ptrs[i], nb, p.dtypes[i], sz, nb, nb);
        ++tot_committed;
      } else if (p.kinds[i] == 0) {
        put(p.ids[i], s);
      } else {
        put(p.ids[i], s.narrow(0, 0, rows));                                      // as slot[:rows]
      }
    }
    side_next = side_drain && fast;
    return static_cast<int64_t>(p.slots.size());
  }
  void clear_slot_plans() { slot_plans.clear(); }

  py::list drain_errors() {
    py::list out;
    for (auto& e : errors)
      if (e.find(kRefused) == std::string::npos) out.append(e);   // a refusal repeats the run's failure, which finalize names once
    errors.clear();
    return out;
  }
};

// The module.forward wrapper is a C++ callable.  The
// Python closure of `native_collect` (frame + `me._cur` check + isinstance chain + one pybind `collect` call per tensor) becomes ONE
// pybind dispatch per producer call: call the original forward, walk the result (Tensor | tuple/list of Tensors | object with
// `sampled_token_ids`) with CPython type checks, hand each tensor to the collector by its pre-interned name id, then flush if this module
// is a legal collection point (P0/P1/P2; the caller decides).  Same values, same order, same ids as the Python wrapper.
struct HookFwd {
  Collector* c;
  py::object keep;                          // the Collector's Python object (keeps `c` alive)
  py::object orig;                          // the module's original bound forward
  int64_t nid_out, nid_in, nid_st;
  std::array<int64_t, 8> nids_k;
  bool flush_after;

  HookFwd(py::object collector, py::object orig_, int64_t nid_out_, py::sequence nids_k_, int64_t nid_in_, int64_t nid_st_, bool flush_after_)
      : c(collector.cast<Collector*>()), keep(std::move(collector)), orig(std::move(orig_)), nid_out(nid_out_), nid_in(nid_in_), nid_st(nid_st_),
        flush_after(flush_after_) {
    if (py::len(nids_k_) != 8) throw std::runtime_error("HookFwd: 8 tuple-slot name ids expected");
    for (size_t i = 0; i < 8; ++i) nids_k[i] = nids_k_[i].cast<int64_t>();
  }

  inline void collect_out(PyObject* o) {
    if (THPVariable_Check(o)) {
      c->note_expected(1);
      c->put(nid_out, THPVariable_Unpack(o));
    } else if (PyTuple_Check(o)) {
      const Py_ssize_t n = PyTuple_GET_SIZE(o);
      for (Py_ssize_t k = 0; k < n; ++k) if (THPVariable_Check(PyTuple_GET_ITEM(o, k))) c->note_expected(1);
      for (Py_ssize_t k = 0; k < n; ++k) {
        PyObject* it = PyTuple_GET_ITEM(o, k);
        if (THPVariable_Check(it)) {
          if (k >= 8) throw std::runtime_error("HookFwd: tuple output with > 8 slots");
          c->put(nids_k[k], THPVariable_Unpack(it));
        }
      }
    } else if (PyList_Check(o)) {
      const Py_ssize_t n = PyList_GET_SIZE(o);
      for (Py_ssize_t k = 0; k < n; ++k) if (THPVariable_Check(PyList_GET_ITEM(o, k))) c->note_expected(1);
      for (Py_ssize_t k = 0; k < n; ++k) {
        PyObject* it = PyList_GET_ITEM(o, k);
        if (THPVariable_Check(it)) {
          if (k >= 8) throw std::runtime_error("HookFwd: list output with > 8 slots");
          c->put(nids_k[k], THPVariable_Unpack(it));
        }
      }
    } else {
      py::object st = py::getattr(py::handle(o), "sampled_token_ids", py::none());
      if (THPVariable_Check(st.ptr())) { c->note_expected(1); c->put(nid_st, THPVariable_Unpack(st.ptr())); }
    }
  }

  py::object call(py::args args, py::kwargs kwargs) {
    if (c->active && nid_in >= 0 && args.size() > 0) {
      PyObject* a0 = PyTuple_GET_ITEM(args.ptr(), 0);
      if (THPVariable_Check(a0)) {
        c->note_expected(1);
        try { c->put(nid_in, THPVariable_Unpack(a0)); } catch (const std::exception& e) { c->errors.emplace_back(std::string("hook/in: ") + e.what()); }
      }
    }
    py::object out = orig(*args, **kwargs);
    if (c->active) {
      try {
        collect_out(out.ptr());
        if (flush_after) c->flush();
      } catch (const std::exception& e) {
        c->errors.emplace_back(std::string("hook: ") + e.what());
      }
    }
    return out;
  }
};

// Native FA2 M1 tap glue.  Replaces `HiddenM1Source._wrapper()` (Python: 2 perf_counter, shape reads, dict lookup, two torch.ops
// calls through the Python dispatcher, f-string name, `_acquire` dict lookup, stats/spans updates -- per attention launch) with one C++
// callable: geometry from the tensor sizes, per-geometry i32 stream buffer (zeros, padded to 64-word chunks -- same allocation rule),
// `verity_set_tap` / `verity_set_tap_tag` through the C++ dispatcher (boxed call, no Python), the TAPPED `varlen_fwd` through its Python
// op object (same cost as the installed op the control pays), then the buffer goes to the collector under the same name
// `fa2.m1/L<k>/g<HB>x<M>x<NB>` (interned once per (k, geometry) through the committer's `_intern`).  Same bytes, same names, same order.
// Tap layout v8: the stream layout is arithmetic in the kernel tile (kBlockN, head_dim) -- (128, 64) for
// hd 64 (the v7 layout, unchanged names/roots), (64, 96) Phi-3, (64, 128) Qwen2.5/Mistral/OLMoE (num_splits == 1 "align" kernel; the
// rule is `hidden_source.fa2_kblock_n`, mirrored in `kblock_n` below).  Per launch the wrapper declares the tile through
// `verity_set_tap_v8`; the kernel arms only if its compile-time tile equals the declaration and otherwise increments
// `verity_tap_skipped` and writes nothing -- the wrapper turns a skipped launch, an uncovered head dim and num_splits != 1 into a
// collector error with the launch counted as expected but not committed (expected != committed -> COMMIT FAIL).
struct Fa2Wrap {
  Collector* c;
  py::object keep, orig_op, tapped_op, intern;
  std::string ns;
  c10::OperatorHandle set_tap, set_tap_tag, tap_skipped;
  std::optional<c10::OperatorHandle> set_tap_v8;             // absent on the v7 hd-64-only build
  int64_t hdims_mask = 1 << 2;                                // bit hd/32 set when the build has that head dim's split-KV kernels
  std::unordered_map<uint64_t, at::Tensor> bufs;             // (HB, M, NB, BN, D) -> stream buffer
  std::unordered_map<uint64_t, int64_t> nids;                 // (k, HB, M, NB, BN, D) -> name id
  int64_t launch = 0, launches_in_step = 0, tot_launches = 0, tot_bytes = 0, tot_refused = 0, tot_compacted = 0;
  double tot_wrap_s = 0.0;
  py::dict geometries;
  bool compact = true;                                       // per-sequence compaction of multi-sequence launches (VERITY_FA2_COMPACT)
  bool zero_after = true;                                    // zero each staged rectangle after staging (history-independent identity)
  std::unordered_map<std::string, int64_t> nids_s;          // compacted names -> name id
  std::vector<int64_t> seqlens;                              // cu_seqlens_q of the current step (host)
  const void* cu_q_ptr = nullptr;
  int64_t cu_q_n = 0;
  // BOUNDED TAP: the Python driver (native_collect.NativeHiddenM1Source) splits a multi-sequence launch whose padded
  // tap buffer (b*h*M*W words) would exceed `bound_words` into consecutive SEQUENCE GROUPS and calls this wrapper once per group with
  // `seq_base` = the group's first global sequence index and `hold_launch` = true (the launch / layer counters advance once, at
  // end_launch()).  Each group is tapped into a PREFIX VIEW of ONE bounded device buffer (`boundbuf`, allocated once at bound_words),
  // staged (compacted per-sequence rectangles named by the GLOBAL index, exactly the unsplit names and bytes), flushed and zeroed before the
  // next group reuses it -> Verity's tap residency is <= bound_words*4 B regardless of the batch (B64 llama 1024: 24.53 GiB -> <= 2 GiB).
  // The compacted identity never depended on the buffer's batch extent (rows < seqlen_q of the sequence's h slabs per plane; zeros
  // elsewhere), so the committed leaves/roots are unchanged by construction; unsplit launches (b == 1, swapped decode, small) keep the
  // per-geometry buffers bit-for-bit.  Telemetry: stats() split_* / bound_* fields; the Python driver writes the per-window jsonl.
  int64_t bound_words = 0;                                   // 0 = bounded path off (per-geometry buffers only)
  bool ms = false;                                           // the build writes the MS class (verity_tap_src() & 64): a seventh plane
  at::Tensor boundbuf;
  int64_t seq_base = 0;
  bool hold_launch = false;
  int64_t tot_split_launches = 0, tot_groups = 0, bound_used_peak_words = 0, tot_cap_exceeded = 0, cap_exceeded_peak_bytes = 0;
  int64_t bound_alloc_bytes = 0;
  std::unordered_map<uint64_t, std::vector<int64_t>> seqlens_cache;   // (cu_seqlens_q data_ptr, numel) -> per-sequence lengths (one D2H each)

  Fa2Wrap(py::object collector, py::object orig_op_, py::object tapped_op_, py::object intern_, std::string ns_)
      : c(collector.cast<Collector*>()), keep(std::move(collector)), orig_op(std::move(orig_op_)), tapped_op(std::move(tapped_op_)),
        intern(std::move(intern_)), ns(std::move(ns_)),
        set_tap(c10::Dispatcher::singleton().findSchemaOrThrow((ns + "::verity_set_tap").c_str(), "")),
        set_tap_tag(c10::Dispatcher::singleton().findSchemaOrThrow((ns + "::verity_set_tap_tag").c_str(), "")),
        tap_skipped(c10::Dispatcher::singleton().findSchemaOrThrow((ns + "::verity_tap_skipped").c_str(), "")) {
    set_tap_v8 = c10::Dispatcher::singleton().findSchema({ns + "::verity_set_tap_v8", ""});
    if (auto h = c10::Dispatcher::singleton().findSchema({ns + "::verity_tap_hdims", ""})) {
      torch::jit::Stack st; h->callBoxed(st); hdims_mask = st.back().toInt();
    }
  }

  static inline uint64_t gkey(int64_t HB, int64_t M, int64_t NB, int64_t BN, int64_t D) {
    return (static_cast<uint64_t>(HB) << 36) | (static_cast<uint64_t>(M) << 20) | (static_cast<uint64_t>(NB) << 6)
         | (static_cast<uint64_t>(BN == 128 ? 1 : 0) << 5) | static_cast<uint64_t>(D / 32);
  }
  // kBlockN of the pinned FA2 split-KV kernel for this head dim under num_splits == 1 (flash_fwd_launch_template.h:172-176); 0 = not tapped
  static inline int64_t kblock_n(int64_t D, int64_t num_splits) {
    if (num_splits != 1 || !(D == 64 || D == 96 || D == 128 || D == 256)) return 0;
    return D <= 64 ? 128 : 64;
  }
  // the MS class (a guarded-max build, verity_tap_src() & 64): one word per (slab, row, key block) after FIN (hidden_source.mat_total_words)
  int64_t mat_total_words(int64_t HB, int64_t M, int64_t NB, int64_t BN, int64_t D) const {
    const int64_t N = NB * BN;
    return HB * M * N * 2 + HB * M * (N / 2) + HB * M * NB * 8 + HB * M * NB * D + HB * M * 4 + (ms ? HB * M * NB : 0);
  }
  static inline bool legacy_tile(int64_t BN, int64_t D) { return BN == 128 && D == 64; }

  const at::Tensor& buffer(int64_t HB, int64_t M, int64_t NB, int64_t BN, int64_t D) {
    const uint64_t k = gkey(HB, M, NB, BN, D);
    auto it = bufs.find(k);
    if (it != bufs.end()) return it->second;
    const int64_t words = ((mat_total_words(HB, M, NB, BN, D) + 63) / 64) * 64;      // padded to whole 64-word chunks (hidden_source._buffer)
    at::Tensor b = at::zeros({words}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
    std::string g = "(" + std::to_string(HB) + ", " + std::to_string(M) + ", " + std::to_string(NB);
    if (!legacy_tile(BN, D)) g += ", " + std::to_string(BN) + ", " + std::to_string(D);
    geometries[py::str(g + ")")] = words * 4;
    return bufs.emplace(k, std::move(b)).first->second;
  }

  int64_t name_id(int64_t k, int64_t HB, int64_t M, int64_t NB, int64_t BN, int64_t D) {
    const uint64_t key = (static_cast<uint64_t>(k) << 56) ^ gkey(HB, M, NB, BN, D);
    auto it = nids.find(key);
    if (it != nids.end()) return it->second;
    std::string name = "fa2.m1/L" + std::to_string(k) + "/g" + std::to_string(HB) + "x" + std::to_string(M) + "x" + std::to_string(NB);
    if (!legacy_tile(BN, D)) name += "x" + std::to_string(BN) + "x" + std::to_string(D);   // hidden_source.stream_name
    const int64_t nid = intern(name).cast<int64_t>();
    nids.emplace(key, nid);
    return nid;
  }

  int64_t skipped() { torch::jit::Stack st; tap_skipped.callBoxed(st); return st.back().toInt(); }

  void refuse(const std::string& why) {
    c->note_expected(1);                                     // the manifest expects this launch's stream; it is not committed -> the identity check fails the run
    c->errors.emplace_back("fa2 launch " + std::to_string(launch) + ": " + why);
    ++tot_refused;
  }

  void on_step_begin() { launches_in_step = 0; seqlens_cache.clear(); cu_q_ptr = nullptr; cu_q_n = 0; }

  // the launch/layer counters advance once per ORIGINAL launch: inline when unsplit, from the driver's end_launch() when split
  void advance_launch() { ++launch; ++launches_in_step; }
  void end_launch() { advance_launch(); ++tot_launches; ++tot_split_launches; }

  // a prefix view of the one bounded buffer when the group fits, else the per-geometry buffer (recorded: cap exceeded)
  at::Tensor bounded_or_geometry_buffer(int64_t HB, int64_t M, int64_t NB, int64_t BN, int64_t D, bool split_path) {
    const int64_t words = ((mat_total_words(HB, M, NB, BN, D) + 63) / 64) * 64;
    if (split_path && bound_words > 0 && words <= bound_words) {
      if (!boundbuf.defined()) {
        boundbuf = at::zeros({bound_words}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
        bound_alloc_bytes = bound_words * 4;
      }
      if (words > bound_used_peak_words) bound_used_peak_words = words;
      return boundbuf.narrow(0, 0, words);
    }
    if (split_path && bound_words > 0) {                      // a single sequence (or group) larger than the cap: named on the record, not hidden
      ++tot_cap_exceeded;
      if (words * 4 > cap_exceeded_peak_bytes) cap_exceeded_peak_bytes = words * 4;
    }
    return buffer(HB, M, NB, BN, D);
  }

  py::object call(py::args args, py::kwargs kwargs) {
    if (!c->active) return orig_op(*args, **kwargs);              // not inside a committed step (warm-up, dummy run): the installed kernel
    if (args.size() < 21) throw std::runtime_error("Fa2Wrap: varlen_fwd called with < 21 positional args");
    const double t0 = now_s();
    const at::Tensor& q = THPVariable_Unpack(PyTuple_GET_ITEM(args.ptr(), 0));
    const at::Tensor& k = THPVariable_Unpack(PyTuple_GET_ITEM(args.ptr(), 1));
    const at::Tensor& cu_q = THPVariable_Unpack(PyTuple_GET_ITEM(args.ptr(), 4));
    const int64_t max_q = args[10].cast<int64_t>();
    const int64_t max_k = args[11].cast<int64_t>();
    const int64_t num_splits = args[20].cast<int64_t>();
    const int64_t b = cu_q.numel() - 1;
    const int64_t H = q.size(1), Hk = k.size(k.dim() - 2), D = q.size(q.dim() - 1);
    // mirror flash_api.cpp:646 `seqlenq_ngroups_swapped` exactly (hidden_source.fa2_swapped): a GQA decode launch is
    // swapped (rows = H/Hk groups, heads = Hk) only with NO sliding window (window_left < 0 && window_right < 0), dropout 0, D % 8 == 0
    // and no alibi.  vLLM sets window (sliding_window-1, 0) on Gemma-2 / Phi-3 sliding layers -> un-swapped decode kernel (b*H x 1).
    const double dropout_p = args[12].cast<double>();
    const int64_t window_left = args[16].cast<int64_t>(), window_right = args[17].cast<int64_t>();
    const bool has_alibi = !args[9].is_none();
    const bool swapped = (max_q == 1 && H > Hk && window_left < 0 && window_right < 0 && dropout_p == 0.0 && (D % 8) == 0 && !has_alibi);
    const int64_t h = swapped ? Hk : H;
    const int64_t M = swapped ? (H / Hk) : max_q;
    const int64_t HB = b * h;
    const int64_t BN = kblock_n(D, num_splits);
    if (BN == 0) {
      refuse("num_splits=" + std::to_string(num_splits) + " head_dim=" + std::to_string(D) + " not tapped (only the num_splits==1 kernel of hd 64/96/128/256)");
      if (!hold_launch) advance_launch();
      return orig_op(*args, **kwargs);
    }
    if (!(hdims_mask & (1 << (D / 32)))) {
      refuse("head_dim " + std::to_string(D) + " not compiled into " + ns + " (hdims mask " + std::to_string(hdims_mask) + "); rebuild with pod_fa2_tap.sh");
      if (!hold_launch) advance_launch();
      return orig_op(*args, **kwargs);
    }
    if (!set_tap_v8 && !legacy_tile(BN, D)) {
      refuse("the v7 tap build declares the hd-64 layout only; tile (" + std::to_string(BN) + ", " + std::to_string(D) + ") needs the v8 build");
      if (!hold_launch) advance_launch();
      return orig_op(*args, **kwargs);
    }
    const int64_t NB = (max_k + BN - 1) / BN;
    // The compacted multi-sequence path is the bounded one.  A GROUP of a split launch (hold_launch, set by the driver) is on
    // that path whatever its own b: a one-sequence group is NOT a B=1 launch -- it must stage the same per-sequence rectangle under the
    // same global name as the unsplit launch (in the whole-buffer b == 1 branch below, a cap of 512 MiB commits 27,376 tensors /
    // 243,529,063 leaves where caps 0 and 2048 commit 28,016 / 133,679,303).
    const bool in_split = hold_launch;
    const bool compacted_path = ((b >= 2 || in_split) && compact && !swapped);
    const bool split_path = (compacted_path && bound_words > 0);
    const at::Tensor buf = bounded_or_geometry_buffer(HB, M, NB, BN, D, split_path);
    if (hold_launch) ++tot_groups;
    const int64_t skipped0 = skipped();
    if (set_tap_v8) {
      torch::jit::Stack st;
      st.emplace_back(static_cast<int64_t>(3)); st.emplace_back(buf); st.emplace_back(HB); st.emplace_back(M); st.emplace_back(NB);
      st.emplace_back(BN); st.emplace_back(D); st.emplace_back(launch);
      set_tap_v8->callBoxed(st);
    } else {
      torch::jit::Stack st;
      st.emplace_back(static_cast<int64_t>(3)); st.emplace_back(); st.emplace_back(buf); st.emplace_back(HB); st.emplace_back(M);
      st.emplace_back(static_cast<int64_t>(0)); st.emplace_back(NB);
      set_tap.callBoxed(st);
      torch::jit::Stack st2;
      st2.emplace_back(launch);
      set_tap_tag.callBoxed(st2);
    }
    py::object res = tapped_op(*args, **kwargs);
    if (skipped() != skipped0) {                                  // the kernel refused the declared tile/caps and wrote nothing
      refuse("skipped by the kernel (declared tile BN=" + std::to_string(BN) + " D=" + std::to_string(D) + ", HB=" + std::to_string(HB)
             + " M=" + std::to_string(M) + " NB=" + std::to_string(NB) + "); stream not bound");
      if (!hold_launch) advance_launch();
      return res;
    }
    // acquire the stream (snapshot into the arena on the compute stream, ordered after the kernel, before the next launch reuses buf)
    if (compacted_path) {                                      // swapped decode launches ([b*Hk, H/Hk, W]) are fully written: nothing to compact
      // PER-SEQUENCE COMPACTION.  The tap buffer is [HB = b*h slabs][M = max_seqlen_q rows][W] for each of
      // its six planes; the kernel writes rows < actual_seqlen_q(i) of sequence i's h slabs only (verity_tap.h: slab = bidb*h + bidh) -- every
      // other row is stale padding from an earlier launch.  Staging the whole buffer costs b * h * M * N words per launch (quadratic in the
      // batch for one long prompt + short ones: B=4 mixed 1024 -> 1.53 GiB per launch, 25 GiB per prefill step; B=16 -> ~100 GiB).  Here each
      // sequence's written rows are staged as one pitched 2-D copy per plane (rows = h slabs, width = seqlen_q(i) * W words, pitch = M * W),
      // so the bound is h * (sum_i seqlen_q(i)) * N per launch = the single-sequence volume.  Position identity: name = layer / sequence /
      // (h, seqlen_q, NB) / plane; the bytes are exactly the plane rows in slab-major order.  b == 1 keeps the whole-buffer identity (B=1 roots
      // are unchanged).  VERITY_FA2_COMPACT=0 restores the padded whole-buffer staging.
      if (launches_in_step == 0 || cu_q.data_ptr() != cu_q_ptr || cu_q.numel() != cu_q_n) {
        // b+1 int32, once per step per distinct cu_seqlens_q (all layers share it; a split launch has one per group -> cached by
        // (ptr, numel) for the step so 16 layers x G groups cost G D2H copies, not 16 x G; the driver keeps the group tensors alive)
        cu_q_ptr = cu_q.data_ptr(); cu_q_n = cu_q.numel();
        const uint64_t ck = reinterpret_cast<uint64_t>(cu_q_ptr) ^ (static_cast<uint64_t>(cu_q_n) << 48);
        auto cit = seqlens_cache.find(ck);
        if (cit == seqlens_cache.end() || (launches_in_step == 0 && !hold_launch)) {
          at::Tensor cu_host = cu_q.to(at::kCPU);
          std::vector<int64_t> sl(b, 0);
          const int32_t* cu = cu_host.data_ptr<int32_t>();
          for (int64_t i = 0; i < b; ++i) sl[i] = static_cast<int64_t>(cu[i + 1] - cu[i]);
          cit = seqlens_cache.insert_or_assign(ck, std::move(sl)).first;
        }
        seqlens = cit->second;
      }
      const int64_t N = NB * BN;
      const int64_t widths[7] = {N, N, N / 2, NB * 8, NB * D, 4, NB};    // S, P, RP, ROW, O, FIN[, MS] (hidden_stream.StreamLayout.offsets; mat_total_words)
      const int np = ms ? 7 : 6;
      int64_t off = 0;
      int64_t plane_off[7];
      for (int p = 0; p < np; ++p) { plane_off[p] = off; off += HB * M * widths[p]; }
      std::vector<at::Tensor> staged;
      staged.reserve(static_cast<size_t>(b) * np);
      for (int64_t i = 0; i < b; ++i) {
        const int64_t sl = seqlens[i];
        if (sl <= 0) continue;
        for (int p = 0; p < np; ++p) {
          at::Tensor view = buf.narrow(0, plane_off[p], HB * M * widths[p]).view({HB, M, widths[p]}).narrow(0, i * h, h).narrow(1, 0, sl);
          c->note_expected(1);
          // the GLOBAL sequence index (seq_base + local i): a split group names its sequences exactly as the unsplit launch did
          try { c->put(name_id_seq(launches_in_step, seq_base + i, h, sl, NB, BN, D, p), view); } catch (const std::exception& e) { c->errors.emplace_back(std::string("fa2: ") + e.what()); }
          tot_bytes += view.numel() * 4;
          staged.push_back(view);
        }
      }
      // History independence: the causal kernel does not write every (row, key-block) position inside the sequence's rectangle (upper
      // triangle beyond the diagonal); those positions would otherwise carry whatever an EARLIER launch (previous layer / step / run)
      // left there, making the compacted identity depend on the buffer's history.  Zero the rectangles after they are staged so every
      // staged rectangle = this launch's writes + zeros.  ORDER: in gather mode `put` only RECORDS the copy -- the D2D runs at
      // the next flush() -- so the memset must come after an explicit flush here (an earlier flush is always a legal collection point);
      // memset straight after put() would zero the buffer BEFORE the gather copy reads it and bind zeros.
      // on the bounded path the zeroing is REQUIRED (the next group reuses the same buffer): it always runs there
      if ((zero_after || split_path) && !staged.empty()) {
        try { if (c->use_gather) c->flush(); } catch (const std::exception& e) { c->errors.emplace_back(std::string("fa2: flush: ") + e.what()); }
        cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
        for (auto& view : staged) {
          const int64_t sl = view.size(1), wp = view.size(2);
          const size_t pitch_b = static_cast<size_t>(M) * wp * 4, width_b = static_cast<size_t>(sl) * wp * 4;
          cudaError_t rc = cudaMemset2DAsync(view.data_ptr(), pitch_b, 0, width_b, static_cast<size_t>(h), st);
          if (rc != cudaSuccess) c->errors.emplace_back(std::string("fa2: memset2d: ") + cudaGetErrorString(rc));
        }
      }
      ++tot_compacted;
    } else {
      c->note_expected(1);
      try { c->put(name_id(launches_in_step, HB, M, NB, BN, D), buf); } catch (const std::exception& e) { c->errors.emplace_back(std::string("fa2: ") + e.what()); }
      tot_bytes += buf.numel() * 4;
      if (b >= 2 && zero_after) {
        // Whole-buffer (swapped decode) launches of a multi-sequence batch: a slab whose sequence has fewer key blocks than the buffer's NB cap
        // leaves its upper blocks unwritten, so without zeroing the staged identity carries the previous launch's values there (measured:
        // B=4 pair-0 vs pair-1 roots differed only in unprocessed key-block regions of these rows).  Zero after staging so the identity is
        // this launch's writes + zeros.  In gather mode put() only records the descriptor and the D2D runs at the next flush(), so flush
        // first (p0a's ordering fix for the compacted path) -- a memset straight after put() would bind zeros.  b == 1 unchanged.
        try { if (c->use_gather) c->flush(); } catch (const std::exception& e) { c->errors.emplace_back(std::string("fa2: flush: ") + e.what()); }
        cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
        cudaError_t rc = cudaMemsetAsync(buf.data_ptr(), 0, static_cast<size_t>(buf.numel()) * 4, st);
        if (rc != cudaSuccess) c->errors.emplace_back(std::string("fa2: memset: ") + cudaGetErrorString(rc));
      }
    }
    if (!hold_launch) { advance_launch(); ++tot_launches; }     // a split group defers to the driver's end_launch()
    tot_wrap_s += now_s() - t0;
    return res;
  }

  int64_t name_id_seq(int64_t k, int64_t i, int64_t h, int64_t sl, int64_t NB, int64_t BN, int64_t D, int plane) {
    static const char* planes[7] = {"S", "P", "RP", "ROW", "O", "FIN", "MS"};
    std::string name = "fa2.m1/L" + std::to_string(k) + "/s" + std::to_string(i) + "/g" + std::to_string(h) + "x" + std::to_string(sl) + "x" + std::to_string(NB);
    if (!legacy_tile(BN, D)) name += "x" + std::to_string(BN) + "x" + std::to_string(D);   // v8 tiles (hd 96/128): distinct identity per tile
    name += std::string("/") + planes[plane];
    auto it = nids_s.find(name);
    if (it != nids_s.end()) return it->second;
    const int64_t nid = intern(name).cast<int64_t>();
    nids_s.emplace(name, nid);
    return nid;
  }

  py::dict stats() const {
    py::dict d;
    d["launches"] = tot_launches; d["bytes"] = tot_bytes; d["wrap_s"] = tot_wrap_s; d["geometries"] = geometries;
    d["compacted_launches"] = tot_compacted; d["compact"] = compact; d["zero_after"] = zero_after;
    d["refused"] = tot_refused; d["tap_layout"] = set_tap_v8 ? 8 : 7; d["hdims_mask"] = hdims_mask; d["ms"] = ms;
    // bounded-tap telemetry (bytes): the cap, the one bounded allocation, its peak prefix in use, split counts
    d["bound_bytes"] = bound_words * 4; d["bound_alloc_bytes"] = bound_alloc_bytes; d["bound_used_peak_bytes"] = bound_used_peak_words * 4;
    d["split_launches"] = tot_split_launches; d["split_groups"] = tot_groups;
    d["cap_exceeded_launches"] = tot_cap_exceeded; d["cap_exceeded_peak_bytes"] = cap_exceeded_peak_bytes;
    int64_t geom_bytes = 0;
    for (auto& kv : bufs) geom_bytes += kv.second.numel() * 4;
    d["geometry_buffers_bytes"] = geom_bytes;                 // the per-geometry buffers still resident (unsplit launches: b == 1, decode)
    d["tap_resident_bytes"] = geom_bytes + bound_alloc_bytes;   // Verity's total FA2 tap device residency
    return d;
  }
};

// The same glue for the FA3 (sm90) M1 tap `verity_fa3_matReq.so` (layout v9 = the FA2 v8 word layout over PackGQA-packed
// rows; hidden_source.HiddenM1SourceFA3 is the Python twin).  Wraps `torch.ops._vllm_fa3_C.fwd` (37 positional args: q k v k_new v_new q_v out
// cu_seqlens_q cu_seqlens_k cu_seqlens_k_new seqused_q seqused_k max_seqlen_q max_seqlen_k page_table ... num_splits at 30).  Geometry: slab =
// (sequence, KV head) -> HB = b * h_k; M = max_seqlen_q * (h / h_k) packed rows; tile (kBlockM, kBlockN) = tile_size_fwd_sm90 under
// use_one_mma_wg (max_seqlen_q * g <= 64): hd 64 -> 64x192 / 192x128, hd 128 -> 64x128 / 128x128 (hidden_source.fa3_tile); NB = ceil(max_seqlen_k
// / kBlockN).  Names `fa3.m1/L<k>/g<HB>x<M>x<NB>x<BN>x<D>` (hidden_source.fa3_stream_name).  A kernel skip (declared tile != selected, caps,
// split path) or an uncovered launch is refused -> the identity check fails the run closed.
struct Fa3Wrap {
  Collector* c;
  py::object keep, orig_op, tapped_op, intern;
  std::string ns;
  c10::OperatorHandle set_tap_v9, tap_skipped;
  int64_t hdims_mask = 0, tap_layout = 9;
  bool zero_after = true;                                    // as Fa2Wrap: zero the staged buffer after staging when b >= 2 (history-independent identity)
  bool ms = false;                                           // the build writes the MS class (verity_tap_src() & 64), sized into every buffer
  std::unordered_map<uint64_t, at::Tensor> bufs;
  std::unordered_map<uint64_t, int64_t> nids;
  int64_t launch = 0, launches_in_step = 0, tot_launches = 0, tot_bytes = 0, tot_refused = 0;
  double tot_wrap_s = 0.0;
  py::dict geometries, tiles;

  Fa3Wrap(py::object collector, py::object orig_op_, py::object tapped_op_, py::object intern_, std::string ns_)
      : c(collector.cast<Collector*>()), keep(std::move(collector)), orig_op(std::move(orig_op_)), tapped_op(std::move(tapped_op_)),
        intern(std::move(intern_)), ns(std::move(ns_)),
        set_tap_v9(c10::Dispatcher::singleton().findSchemaOrThrow((ns + "::verity_set_tap_v9").c_str(), "")),
        tap_skipped(c10::Dispatcher::singleton().findSchemaOrThrow((ns + "::verity_tap_skipped").c_str(), "")) {
    { torch::jit::Stack st; c10::Dispatcher::singleton().findSchemaOrThrow((ns + "::verity_tap_hdims").c_str(), "").callBoxed(st); hdims_mask = st.back().toInt(); }
    { torch::jit::Stack st; c10::Dispatcher::singleton().findSchemaOrThrow((ns + "::verity_tap_layout_version").c_str(), "").callBoxed(st); tap_layout = st.back().toInt(); }
  }

  static inline uint64_t gkey(int64_t HB, int64_t M, int64_t NB, int64_t BN, int64_t D) {
    return (static_cast<uint64_t>(HB) << 36) | (static_cast<uint64_t>(M) << 20) | (static_cast<uint64_t>(NB) << 6)
         | (static_cast<uint64_t>(BN == 192 ? 1 : 0) << 5) | static_cast<uint64_t>(D / 32);
  }
  // (kBlockM, kBlockN) of the pinned FA3 forward (hopper/tile_size.h::tile_size_fwd_sm90 + heuristics.h::use_one_mma_wg); {0, 0} = not tapped
  static inline std::pair<int64_t, int64_t> tile(int64_t D, int64_t max_q, int64_t g, int64_t num_splits) {
    if (num_splits != 1) return {0, 0};
    const bool one_wg = max_q * g <= 64;
    if (D == 64) return one_wg ? std::make_pair<int64_t, int64_t>(64, 192) : std::make_pair<int64_t, int64_t>(192, 128);
    if (D == 128) return one_wg ? std::make_pair<int64_t, int64_t>(64, 128) : std::make_pair<int64_t, int64_t>(128, 128);
    return {0, 0};
  }
  int64_t mat_total_words(int64_t HB, int64_t M, int64_t NB, int64_t BN, int64_t D) const {
    const int64_t N = NB * BN;
    return HB * M * N * 2 + HB * M * (N / 2) + HB * M * NB * 8 + HB * M * NB * D + HB * M * 4 + (ms ? HB * M * NB : 0);
  }

  const at::Tensor& buffer(int64_t HB, int64_t M, int64_t NB, int64_t BN, int64_t D) {
    const uint64_t k = gkey(HB, M, NB, BN, D);
    auto it = bufs.find(k);
    if (it != bufs.end()) return it->second;
    const int64_t words = ((mat_total_words(HB, M, NB, BN, D) + 63) / 64) * 64;
    at::Tensor b = at::zeros({words}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
    geometries[py::str("(" + std::to_string(HB) + ", " + std::to_string(M) + ", " + std::to_string(NB) + ", " + std::to_string(BN) + ", " + std::to_string(D) + ")")] = words * 4;
    return bufs.emplace(k, std::move(b)).first->second;
  }

  int64_t name_id(int64_t k, int64_t HB, int64_t M, int64_t NB, int64_t BN, int64_t D) {
    const uint64_t key = (static_cast<uint64_t>(k) << 56) ^ gkey(HB, M, NB, BN, D);
    auto it = nids.find(key);
    if (it != nids.end()) return it->second;
    const std::string name = "fa3.m1/L" + std::to_string(k) + "/g" + std::to_string(HB) + "x" + std::to_string(M) + "x" + std::to_string(NB)
                           + "x" + std::to_string(BN) + "x" + std::to_string(D);
    const int64_t nid = intern(name).cast<int64_t>();
    nids.emplace(key, nid);
    return nid;
  }

  int64_t skipped() { torch::jit::Stack st; tap_skipped.callBoxed(st); return st.back().toInt(); }

  void refuse(const std::string& why) {
    c->note_expected(1);
    c->errors.emplace_back("fa3 launch " + std::to_string(launch) + ": " + why);
    ++tot_refused;
  }

  void on_step_begin() { launches_in_step = 0; }

  py::object call(py::args args, py::kwargs kwargs) {
    if (!c->active) return orig_op(*args, **kwargs);
    if (args.size() < 31) throw std::runtime_error("Fa3Wrap: fwd called with < 31 positional args");
    const double t0 = now_s();
    PyObject* cu_q_o = PyTuple_GET_ITEM(args.ptr(), 7);
    PyObject* pt_o = PyTuple_GET_ITEM(args.ptr(), 14);
    if (cu_q_o == Py_None || pt_o == Py_None || args[12].is_none() || args[13].is_none()) {
      refuse("not the varlen paged-KV launch the tap covers (cu_seqlens_q / page_table / max_seqlen missing)");
      ++launch; ++launches_in_step;
      return orig_op(*args, **kwargs);
    }
    const at::Tensor& q = THPVariable_Unpack(PyTuple_GET_ITEM(args.ptr(), 0));
    const at::Tensor& k = THPVariable_Unpack(PyTuple_GET_ITEM(args.ptr(), 1));
    const at::Tensor& cu_q = THPVariable_Unpack(cu_q_o);
    const int64_t max_q = args[12].cast<int64_t>();
    const int64_t max_k = args[13].cast<int64_t>();
    const int64_t num_splits = args[30].is_none() ? 1 : args[30].cast<int64_t>();
    const int64_t b = cu_q.numel() - 1;
    const int64_t H = q.size(1), Hk = k.size(k.dim() - 2), D = q.size(q.dim() - 1);
    const int64_t g = Hk > 0 ? H / Hk : 1;
    const auto [kBM, BN] = tile(D, max_q, g, num_splits);
    if (BN == 0) {
      refuse("num_splits=" + std::to_string(num_splits) + " head_dim=" + std::to_string(D) + " not tapped (only the num_splits==1 bf16 paged PackGQA kernel of hd 64/128)");
      ++launch; ++launches_in_step;
      return orig_op(*args, **kwargs);
    }
    if (!(hdims_mask & (1 << (D / 32)))) {
      refuse("head_dim " + std::to_string(D) + " not compiled into " + ns + " (hdims mask " + std::to_string(hdims_mask) + "); rebuild with pod_fa3_tap.sh");
      ++launch; ++launches_in_step;
      return orig_op(*args, **kwargs);
    }
    const int64_t HB = b * Hk, M = max_q * g;
    const int64_t NB = (max_k + BN - 1) / BN;
    const at::Tensor& buf = buffer(HB, M, NB, BN, D);
    const int64_t skipped0 = skipped();
    {
      torch::jit::Stack st;
      st.emplace_back(static_cast<int64_t>(3)); st.emplace_back(buf); st.emplace_back(HB); st.emplace_back(M); st.emplace_back(NB);
      st.emplace_back(BN); st.emplace_back(D); st.emplace_back(launch);
      set_tap_v9.callBoxed(st);
    }
    py::object res = tapped_op(*args, **kwargs);
    if (skipped() != skipped0) {
      refuse("skipped by the kernel (declared tile kBlockM=" + std::to_string(kBM) + " BN=" + std::to_string(BN) + " D=" + std::to_string(D) + ", HB=" + std::to_string(HB)
             + " M=" + std::to_string(M) + " NB=" + std::to_string(NB) + ", num_splits=" + std::to_string(num_splits) + "); stream not bound");
      ++launch; ++launches_in_step;
      return res;
    }
    c->note_expected(1);
    try { c->put(name_id(launches_in_step, HB, M, NB, BN, D), buf); } catch (const std::exception& e) { c->errors.emplace_back(std::string("fa3: ") + e.what()); }
    ++launch; ++launches_in_step; ++tot_launches;
    tot_bytes += buf.numel() * 4;
    if (b >= 2 && zero_after) {
      // Fa2Wrap's zeroing rule, for the FA3 tap: a multi-sequence launch stages the whole (HB, M, NB) buffer: a slab whose sequence has
      // fewer key blocks / query rows than the launch's NB / M cap leaves those positions unwritten, so the staged identity would carry the
      // previous launch's values there (measured on the Llama H100 b8 1024/128 workload: pair-0 root != pair-1 == pair-2 -- history, not values).
      // Zero after staging so the identity is this launch's writes + zeros; in gather mode put() only records the descriptor and the D2D
      // runs at the next flush(), so flush first -- a memset straight after put() would bind zeros.  b == 1 unchanged (roots of the b1 rows hold).
      try { if (c->use_gather) c->flush(); } catch (const std::exception& e) { c->errors.emplace_back(std::string("fa3: flush: ") + e.what()); }
      cudaStream_t st = c10::cuda::getCurrentCUDAStream().stream();
      cudaError_t rc = cudaMemsetAsync(buf.data_ptr(), 0, static_cast<size_t>(buf.numel()) * 4, st);
      if (rc != cudaSuccess) c->errors.emplace_back(std::string("fa3: memset: ") + cudaGetErrorString(rc));
    }
    {
      py::str tk(std::to_string(kBM) + "x" + std::to_string(BN));
      tiles[tk] = tiles.contains(tk) ? tiles[tk].cast<int64_t>() + 1 : 1;
    }
    tot_wrap_s += now_s() - t0;
    return res;
  }

  py::dict stats() const {
    py::dict d;
    d["launches"] = tot_launches; d["bytes"] = tot_bytes; d["wrap_s"] = tot_wrap_s; d["geometries"] = geometries; d["tiles"] = tiles;
    d["refused"] = tot_refused; d["tap_layout"] = tap_layout; d["hdims_mask"] = hdims_mask; d["zero_after"] = zero_after; d["ms"] = ms;
    return d;
  }
};

// The worker's per-step GPU commitment as ONE native call with the GIL released.  Mirrors
// `NativeHostCommitter._commit_step_gpu` exactly (same kernels -- native_tree.cu is hidden_gpu_tree.cu -- same header fields
// (chunk_words 64, launch_tag = step, src_mask, HB = step, M = padded/4, NB = 0), same zero padding to 256 B, same retained copy, same
// "root D2H + done event synchronised before the block is released" ordering).  The Python worker then only binds the root
// (hashlib, ~50 us) and files the StepCommitment; nothing Python-side runs concurrently with the serving thread's hook path.
// rows-sha512 (verity_vllm.commit.step_rows): one step's kernel parameters -- its frame domain's templates and midstates and its salt
// key -- built once on the host (native_rows.cu) and passed by value to every launch of the step.
struct RowStep {
  std::vector<uint8_t> params;
  int64_t row_bytes = 0;
  int64_t max_blocks = 0;                        // > 0: leaves and wide levels in launches of at most this many blocks (the same bytes)
  RowStep(const py::bytes& domain_id, const py::bytes& ks, const py::bytes& row_prefix, const py::bytes& salt_prefix, const py::bytes& leaf_prefix,
          const std::string& schema, int64_t row_bytes_)
      : row_bytes(row_bytes_) {
    const std::string did = domain_id, k = ks, p = row_prefix, sp = salt_prefix, lp = leaf_prefix;
    if (did.size() != 64 || k.size() != 64 || p.size() != 128 || sp.size() != 128 || lp.size() != 128)
      throw std::runtime_error("RowStep: the domain id and the salt key are 64 bytes, the row, salt and leaf prefixes 128");
    if (row_bytes <= 0 || row_bytes > (1 << 20)) throw std::runtime_error("RowStep: a row is 1 byte to 1 MiB");
    params.resize(static_cast<size_t>(verity_rows_params_bytes()));
    if (verity_rows_build(reinterpret_cast<const uint8_t*>(did.data()), reinterpret_cast<const uint8_t*>(k.data()), reinterpret_cast<const uint8_t*>(p.data()),
                          reinterpret_cast<const uint8_t*>(sp.data()), reinterpret_cast<const uint8_t*>(lp.data()), schema.data(), static_cast<int>(schema.size()),
                          static_cast<uint32_t>(row_bytes), params.data()) != 0)
      throw std::runtime_error("RowStep: the frame leaf template does not fit the schema " + schema);
  }
};

struct StepHasher {
  cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev2 = nullptr, done = nullptr, copy_from = nullptr;
  uint32_t* root_host = nullptr;                 // pinned, 32 B
  uint8_t* root64_host = nullptr;                // pinned, 64 B (rows-sha512)
  StepHasher() {
    cudaEventCreateWithFlags(&ev0, cudaEventDefault); cudaEventCreateWithFlags(&ev1, cudaEventDefault); cudaEventCreateWithFlags(&ev2, cudaEventDefault);
    cudaEventCreateWithFlags(&done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&copy_from, cudaEventDisableTiming);
    cudaHostAlloc(reinterpret_cast<void**>(&root_host), 32, cudaHostAllocDefault);
    cudaHostAlloc(reinterpret_cast<void**>(&root64_host), 64, cudaHostAllocDefault);
  }
  ~StepHasher() {
    if (ev0) cudaEventDestroy(ev0); if (ev1) cudaEventDestroy(ev1); if (ev2) cudaEventDestroy(ev2); if (done) cudaEventDestroy(done);
    if (copy_from) cudaEventDestroy(copy_from);
    if (root_host) cudaFreeHost(root_host);
    if (root64_host) cudaFreeHost(root64_host);
  }

  // windowed: leaves of ONE window of a step's packed stream.  `slot` holds the window's bytes (nbytes, a multiple of 256; the caller
  // zero-pads the tail of the last window), `first_chunk` its first global chunk index, `m_words` the step's padded words (layout v1 binds it
  // into every leaf header; v2 passes 0).  Runs on `stream_ptr` (the worker's side stream); returns the [n, 8] int32 device leaves.
  at::Tensor leaves_window(at::Tensor slot, int64_t nbytes, int64_t first_chunk, int64_t step, int64_t src_mask, int64_t chunk_bytes, int64_t m_words,
                           int64_t stream_ptr) {
    if (!slot.is_cuda() || slot.scalar_type() != at::kByte || !slot.is_contiguous()) throw std::runtime_error("leaves_window: slot must be a contiguous CUDA uint8 tensor");
    if (nbytes % chunk_bytes != 0 || nbytes > slot.numel()) throw std::runtime_error("leaves_window: nbytes must be a chunk multiple within the slot");
    const long n = nbytes / chunk_bytes;
    at::Tensor leaves;
    {
      py::gil_scoped_release nogil;
      c10::InferenceMode inf;
      cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_ptr);
      c10::cuda::CUDAStream cs = c10::cuda::getStreamFromExternal(st, slot.device().index());
      c10::cuda::CUDAStreamGuard g(cs);
      leaves = at::empty({n, 8}, at::TensorOptions().dtype(at::kInt).device(slot.device()));
      int lrc = verity_chunk_leaves_launch_base(reinterpret_cast<const uint32_t*>(slot.data_ptr()), reinterpret_cast<uint32_t*>(leaves.data_ptr()), n,
                                                static_cast<int>(chunk_bytes / 4), static_cast<uint32_t>(step), static_cast<uint32_t>(src_mask),
                                                static_cast<uint32_t>(step), static_cast<uint32_t>(m_words), 0u, first_chunk, st);
      if (lrc != 0) throw std::runtime_error(std::string("chunk_leaves(window): ") + cudaGetErrorString(static_cast<cudaError_t>(lrc)));
    }
    return leaves;
  }

  // windowed: tree levels over given device leaves (same node/lift rule), root D2H, optional levels -> pinned host; one event sync.
  // -> (levels, root: bytes(32), tree_ms)
  py::tuple tree(at::Tensor leaves, int64_t stream_ptr, bool levels_host = false, at::Tensor levels_dst = at::Tensor()) {
    if (!leaves.is_cuda() || leaves.scalar_type() != at::kInt || !leaves.is_contiguous() || leaves.dim() != 2 || leaves.size(1) != 8) throw std::runtime_error("tree: leaves must be a contiguous CUDA int32 [n,8] tensor");
    std::vector<at::Tensor> levels;
    float tree_ms = 0.f;
    {
      py::gil_scoped_release nogil;
      c10::InferenceMode inf;
      cudaStream_t side = reinterpret_cast<cudaStream_t>(stream_ptr);
      c10::cuda::CUDAStream cs = c10::cuda::getStreamFromExternal(side, leaves.device().index());
      c10::cuda::CUDAStreamGuard g(cs);
      cudaEventRecord(ev1, side);
      levels.push_back(leaves);
      for (auto& lv : tree_levels_above(leaves, levels_to_root(leaves.size(0)), side)) levels.push_back(lv);
      cudaEventRecord(ev2, side);
      cudaError_t rc = cudaMemcpyAsync(root_host, levels.back().data_ptr(), 32, cudaMemcpyDeviceToHost, side);
      if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync(root): ") + cudaGetErrorString(rc));
      if (levels_host) {
        int64_t total = 0;
        for (auto& lv : levels) total += lv.numel();
        at::Tensor hbuf;
        if (levels_dst.defined() && levels_dst.numel() >= total * 4) hbuf = levels_dst.narrow(0, 0, total * 4).view(at::kInt);
        else hbuf = at::empty({total}, at::TensorOptions().dtype(at::kInt).device(at::kCPU).pinned_memory(true));
        std::vector<at::Tensor> hl;
        int64_t hoff = 0;
        for (auto& lv : levels) {
          at::Tensor h = hbuf.narrow(0, hoff, lv.numel()).view({lv.size(0), 8});
          hoff += lv.numel();
          rc = cudaMemcpyAsync(h.data_ptr(), lv.data_ptr(), static_cast<size_t>(lv.numel()) * 4, cudaMemcpyDeviceToHost, side);
          if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync(level->host): ") + cudaGetErrorString(rc));
          hl.push_back(h);
        }
        cudaEventRecord(done, side);
        rc = cudaEventSynchronize(done);
        if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaEventSynchronize: ") + cudaGetErrorString(rc));
        levels.swap(hl);
      } else {
        cudaEventRecord(done, side);
        rc = cudaEventSynchronize(done);
        if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaEventSynchronize: ") + cudaGetErrorString(rc));
      }
      cudaEventElapsedTime(&tree_ms, ev1, ev2);
    }
    py::list lv;
    for (auto& t : levels) lv.append(t);
    return py::make_tuple(lv, py::bytes(reinterpret_cast<const char*>(root_host), 32), static_cast<double>(tree_ms));
  }

  // windowed absorb: the subtree of ONE window.  Window k covers leaves [k*2^j, (k+1)*2^j) of the step (W = chunk * 2^j), so for every
  // level L <= j the full tree's nodes over that range are a closed computation on the window's own leaves (pairs never cross the
  // 2^(j-L)-aligned boundary; the lift rule for the odd trailing node fires only in the LAST window, where the global trailing node is)
  // -> building `n_levels` levels per window (0 = leaves .. n_levels-1) and then the ordinary tree over the per-window top nodes gives
  // exactly the whole-step tree (same root, same levels).  Each level L is copied D2H into `host_dsts[L]` (pinned slices of the step's
  // level buffers); the top node stays on device (returned) for the upper tree.  Device transient: one window's leaves + levels.
  at::Tensor tree_window(at::Tensor leaves, int64_t n_levels, int64_t stream_ptr, std::vector<at::Tensor> host_dsts) {
    if (!leaves.is_cuda() || leaves.scalar_type() != at::kInt || !leaves.is_contiguous() || leaves.dim() != 2 || leaves.size(1) != 8) throw std::runtime_error("tree_window: leaves must be a contiguous CUDA int32 [n, 8]");
    if (static_cast<int64_t>(host_dsts.size()) != n_levels) throw std::runtime_error("tree_window: one host destination per level");
    if (n_levels < 1) throw std::runtime_error("tree_window: n_levels must be positive");
    py::gil_scoped_release nogil;
    c10::InferenceMode inf;
    cudaStream_t side = reinterpret_cast<cudaStream_t>(stream_ptr);
    c10::cuda::CUDAStream cs = c10::cuda::getStreamFromExternal(side, leaves.device().index());
    c10::cuda::CUDAStreamGuard g(cs);
    std::vector<at::Tensor> levels{leaves};                          // a level of one node climbs by lifts (the collapsed last window to the window level)
    for (auto& lv : tree_levels_above(leaves, n_levels - 1, side)) levels.push_back(lv);
    for (int64_t L = 0; L < n_levels; ++L) {
      const at::Tensor& h = host_dsts[static_cast<size_t>(L)];
      const at::Tensor& cur = levels[static_cast<size_t>(L)];
      if (h.numel() != cur.numel()) throw std::runtime_error("tree_window: level " + std::to_string(L) + " host slice has " + std::to_string(h.numel() / 8)
                                                             + " nodes, computed " + std::to_string(cur.size(0)));
      cudaError_t rc = cudaMemcpyAsync(h.data_ptr(), cur.data_ptr(), static_cast<size_t>(cur.numel()) * 4, cudaMemcpyDeviceToHost, side);
      if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync(window level->host): ") + cudaGetErrorString(rc));
    }
    at::Tensor cur = levels.back();
    cudaEventRecord(done, side);                                     // the host slices are complete (and `leaves` may be freed) once this fires
    cudaError_t rc = cudaEventSynchronize(done);
    if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaEventSynchronize: ") + cudaGetErrorString(rc));
    return cur;
  }

  // tree_window without the host copies (a run that keeps no tree levels: its openings take their paths from a replay): the same levels on
  // the device and the same top node, and no host sync -- every tensor here is allocated on `stream_ptr`, and the caller releases the
  // window's slot with an event recorded on that stream after these kernels.
  at::Tensor tree_window_top(at::Tensor leaves, int64_t n_levels, int64_t stream_ptr) {
    if (!leaves.is_cuda() || leaves.scalar_type() != at::kInt || !leaves.is_contiguous() || leaves.dim() != 2 || leaves.size(1) != 8) throw std::runtime_error("tree_window_top: leaves must be a contiguous CUDA int32 [n, 8]");
    if (n_levels < 1) throw std::runtime_error("tree_window_top: n_levels must be positive");
    py::gil_scoped_release nogil;
    c10::InferenceMode inf;
    cudaStream_t side = reinterpret_cast<cudaStream_t>(stream_ptr);
    c10::cuda::CUDAStream cs = c10::cuda::getStreamFromExternal(side, leaves.device().index());
    c10::cuda::CUDAStreamGuard g(cs);
    std::vector<at::Tensor> up = tree_levels_above(leaves, n_levels - 1, side);
    return up.empty() ? leaves : up.back();
  }

  // -> (levels: list[Tensor int32 [n,8]] leaves first, keep: Tensor uint8 [padded] (device | host | empty), root: bytes(32), padded, leaf_ms, tree_ms)
  // keep_mode (retention policy): 0 = compact DEVICE copy, 1 = HOST copy (D2H on the side stream; pageable), 2 = no copy
  // (openings by replay).  The leaves are always hashed from the arena block first, so the root does not depend on keep_mode.
  py::tuple commit(at::Tensor arena, int64_t used, int64_t step, int64_t src_mask, int64_t chunk_bytes, int64_t side_stream_ptr, int64_t step_event_ptr,
                   int64_t keep_mode = 0, at::Tensor keep_dst = at::Tensor(), bool levels_host = false, at::Tensor levels_dst = at::Tensor()) {
    if (!arena.is_cuda() || arena.scalar_type() != at::kByte || !arena.is_contiguous()) throw std::runtime_error("StepHasher: arena must be a contiguous CUDA uint8 tensor");
    const int64_t padded = ((used + chunk_bytes - 1) / chunk_bytes) * chunk_bytes;
    if (padded > arena.numel()) throw std::runtime_error("StepHasher: padded step larger than the arena block (overflow chaining not supported)");
    std::vector<at::Tensor> levels;
    at::Tensor keep;
    float leaf_ms = 0.f, tree_ms = 0.f;
    {
      py::gil_scoped_release nogil;
      c10::InferenceMode inf;
      const auto dev = arena.device().index();
      cudaStream_t side = reinterpret_cast<cudaStream_t>(side_stream_ptr);
      c10::cuda::CUDAStream cs = c10::cuda::getStreamFromExternal(side, dev);
      c10::cuda::CUDAStreamGuard g(cs);
      cudaError_t rc = cudaStreamWaitEvent(side, reinterpret_cast<cudaEvent_t>(step_event_ptr), 0);
      if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaStreamWaitEvent: ") + cudaGetErrorString(rc));
      char* base = reinterpret_cast<char*>(arena.data_ptr());
      if (padded > used) {
        rc = cudaMemsetAsync(base + used, 0, padded - used, side);
        if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemsetAsync(pad): ") + cudaGetErrorString(rc));
      }
      const long n = padded / chunk_bytes;
      at::Tensor leaves = at::empty({n, 8}, at::TensorOptions().dtype(at::kInt).device(arena.device()));
      cudaEventRecord(ev0, side);
      int lrc = verity_chunk_leaves_launch(reinterpret_cast<const uint32_t*>(base), reinterpret_cast<uint32_t*>(leaves.data_ptr()), n,
                                           static_cast<int>(chunk_bytes / 4), static_cast<uint32_t>(step), static_cast<uint32_t>(src_mask),
                                           static_cast<uint32_t>(step), static_cast<uint32_t>(padded / 4), 0u, side);
      if (lrc != 0) throw std::runtime_error(std::string("chunk_leaves: ") + cudaGetErrorString(static_cast<cudaError_t>(lrc)));
      cudaEventRecord(ev1, side);
      levels.push_back(leaves);
      for (auto& lv : tree_levels_above(leaves, levels_to_root(n), side)) levels.push_back(lv);
      cudaEventRecord(ev2, side);
      if (keep_mode == 0) {
        keep = at::empty({padded}, arena.options());                     // compact retained copy (lazy openings); arena block recycled by the caller
        rc = cudaMemcpyAsync(keep.data_ptr(), base, padded, cudaMemcpyDeviceToDevice, side);
        if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync(keep): ") + cudaGetErrorString(rc));
      } else if (keep_mode == 1) {
        // host copy: into the caller's PINNED buffer when given (pageable D2H runs at ~1 GB/s and serialises the side stream; pinned
        // runs at PCIe rate and stays asynchronous), else a fresh pageable tensor
        if (keep_dst.defined() && keep_dst.numel() >= padded) {
          if (keep_dst.is_cuda() || keep_dst.scalar_type() != at::kByte || !keep_dst.is_contiguous()) throw std::runtime_error("StepHasher: keep_dst must be a contiguous host uint8 tensor");
          keep = keep_dst.narrow(0, 0, padded);
        } else {
          keep = at::empty({padded}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
        }
        rc = cudaMemcpyAsync(keep.data_ptr(), base, padded, cudaMemcpyDeviceToHost, side);
        if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync(keep->host): ") + cudaGetErrorString(rc));
      } else {
        keep = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
      }
      rc = cudaMemcpyAsync(root_host, levels.back().data_ptr(), 32, cudaMemcpyDeviceToHost, side);
      if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync(root): ") + cudaGetErrorString(rc));
      if (levels_host) {
        // authentication state to PINNED host memory on the side stream (one D2H per level, one event sync below); the device
        // levels are dropped when this scope ends.  A Python-side `.cpu()` per level would synchronise the worker with the DEFAULT stream,
        // i.e. with the engine's step, which with backpressure stalls every decode step (30 ms vs 10.5 ms control).
        std::vector<at::Tensor> hl;
        hl.reserve(levels.size());
        int64_t total = 0;
        for (auto& lv : levels) total += lv.numel();
        // one pinned buffer for all levels of the step: the caller's pool block when given (recycled per run; a fresh cudaHostAlloc per
        // level per step cost ~7 ms/decode step), else one pinned allocation
        at::Tensor hbuf;
        if (levels_dst.defined() && levels_dst.numel() >= total * 4) {
          if (levels_dst.is_cuda() || levels_dst.scalar_type() != at::kByte || !levels_dst.is_contiguous()) throw std::runtime_error("StepHasher: levels_dst must be a contiguous host uint8 tensor");
          hbuf = levels_dst.narrow(0, 0, total * 4).view(at::kInt);
        } else {
          hbuf = at::empty({total}, at::TensorOptions().dtype(at::kInt).device(at::kCPU).pinned_memory(true));
        }
        int64_t hoff = 0;
        for (auto& lv : levels) {
          at::Tensor h = hbuf.narrow(0, hoff, lv.numel()).view({lv.size(0), 8});
          hoff += lv.numel();
          rc = cudaMemcpyAsync(h.data_ptr(), lv.data_ptr(), static_cast<size_t>(lv.numel()) * 4, cudaMemcpyDeviceToHost, side);
          if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync(level->host): ") + cudaGetErrorString(rc));
          hl.push_back(h);
        }
        cudaEventRecord(done, side);
        rc = cudaEventSynchronize(done);
        if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaEventSynchronize: ") + cudaGetErrorString(rc));
        levels.swap(hl);                                                  // device levels freed here (still under nogil)
      }
      cudaEventRecord(done, side);
      rc = cudaEventSynchronize(done);                                    // keep-copy + root D2H complete before the arena block can be reused
      if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaEventSynchronize: ") + cudaGetErrorString(rc));
      cudaEventElapsedTime(&leaf_ms, ev0, ev1);
      cudaEventElapsedTime(&tree_ms, ev1, ev2);
    }
    py::list lv;
    for (auto& t : levels) lv.append(t);
    return py::make_tuple(lv, keep, py::bytes(reinterpret_cast<const char*>(root_host), 32), padded, static_cast<double>(leaf_ms), static_cast<double>(tree_ms));
  }

  // rows-sha512: the leaves of positions first .. first + m - 1 of a step of n_total leaves (a pad at every position >= n_total), row i
  // at rows + i * stride for i < nbytes / stride (stride 0: the step's row bytes).  Salts are derived from the step's key, or given (192 B
  // per row: a test's).  Runs on `stream_ptr`.  -> (leaves [m, 64] uint8, b || c [m, 128] when want_bc, h [m, 64] when want_h; the last
  // two hold the data positions' values, and are empty when not asked for)
  py::tuple rows_leaves(const RowStep& rs, at::Tensor rows, int64_t nbytes, int64_t first, int64_t m, int64_t n_total, at::Tensor tn, int64_t stream_ptr,
                        int64_t stride = 0, std::optional<at::Tensor> salts_given = std::nullopt, bool want_bc = false, bool want_h = false) {
    const at::Tensor salts = salts_given.has_value() ? *salts_given : at::Tensor();
    if (stride == 0) stride = rs.row_bytes;
    if (!rows.is_cuda() || rows.scalar_type() != at::kByte || !rows.is_contiguous()) throw std::runtime_error("rows_leaves: rows must be a contiguous CUDA uint8 tensor");
    if (stride % 16 || stride < (rs.row_bytes + 15) / 16 * 16 || reinterpret_cast<uintptr_t>(rows.data_ptr()) % 16)
      throw std::runtime_error("rows_leaves: rows 16-byte aligned at a stride of whole 16-byte loads covering the row");
    if (nbytes < 0 || nbytes > rows.numel() || nbytes % stride) throw std::runtime_error("rows_leaves: nbytes must be a stride multiple within rows");
    if (!tn.is_cuda() || tn.scalar_type() != at::kByte || !tn.is_contiguous() || tn.numel() != 8192) throw std::runtime_error("rows_leaves: tn must be the 8 KB CUDA mask table");
    if (first < 0 || m < 0 || n_total < 0 || first + m > (int64_t(1) << 32)) throw std::runtime_error("rows_leaves: positions below 2^32");
    const int64_t n_rows = nbytes / stride;
    if (salts.defined() && (!salts.is_cuda() || salts.scalar_type() != at::kByte || !salts.is_contiguous() || salts.numel() < n_rows * 192))
      throw std::runtime_error("rows_leaves: salts must be a contiguous CUDA uint8 tensor of 192 bytes per row");
    at::Tensor leaves, bc, hs;
    {
      py::gil_scoped_release nogil;
      c10::InferenceMode inf;
      cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_ptr);
      c10::cuda::CUDAStream cs = c10::cuda::getStreamFromExternal(st, rows.device().index());
      c10::cuda::CUDAStreamGuard g(cs);
      const auto opt = at::TensorOptions().dtype(at::kByte).device(rows.device());
      leaves = at::empty({m, 64}, opt);
      bc = want_bc ? at::empty({m, 128}, opt) : at::empty({0}, opt);
      hs = want_h ? at::empty({m, 64}, opt) : at::empty({0}, opt);
      const auto* rows_p = reinterpret_cast<const uint8_t*>(rows.data_ptr());
      const uint8_t* salts_p = salts.defined() ? reinterpret_cast<const uint8_t*>(salts.data_ptr()) : nullptr;
      const auto* tn_p = reinterpret_cast<const uint8_t*>(tn.data_ptr());
      auto* out_p = reinterpret_cast<uint8_t*>(leaves.data_ptr());
      uint8_t* bc_p = want_bc ? reinterpret_cast<uint8_t*>(bc.data_ptr()) : nullptr;
      uint8_t* hs_p = want_h ? reinterpret_cast<uint8_t*>(hs.data_ptr()) : nullptr;
      const int form = verity_rows_leaves_form();
      if (form < 0) throw std::runtime_error("rows_leaves: VERITY_ROWS_FORM is fused, split or unset");
      at::Tensor sf = at::empty({form == 2 ? m * 192 : 0}, opt);
      int lrc = verity_rows_leaves_sf_launch(rs.params.data(), rows_p, static_cast<uint64_t>(stride), static_cast<uint64_t>(n_rows), salts_p, tn_p,
                                             static_cast<uint64_t>(first), static_cast<uint64_t>(m), static_cast<uint64_t>(n_total), out_p, bc_p, hs_p,
                                             form == 2 ? reinterpret_cast<uint8_t*>(sf.data_ptr()) : nullptr, static_cast<uint64_t>(rs.max_blocks), st);
      if (lrc != 0) throw std::runtime_error(std::string("rows_leaves: ") + cudaGetErrorString(static_cast<cudaError_t>(lrc)));
    }
    return py::make_tuple(leaves, bc, hs);
  }

  // rows_leaves in two halves, for a step whose length (and so whose frame domain) is not known while its rows are: rows_h hashes the
  // data positions first .. first + nbytes / stride - 1 to their h (and b || c when want_bc), which needs only the step's salt key (rs
  // may be any domain's RowStep), and rows_frame turns them into the leaves once the step's RowStep is known.  -> (h [n_rows, 64],
  // b || c [n_rows, 128] or empty)
  py::tuple rows_h(const RowStep& rs, at::Tensor rows, int64_t nbytes, int64_t first, at::Tensor tn, int64_t stream_ptr, int64_t stride = 0,
                   std::optional<at::Tensor> salts_given = std::nullopt, bool want_bc = false) {
    const at::Tensor salts = salts_given.has_value() ? *salts_given : at::Tensor();
    if (stride == 0) stride = rs.row_bytes;
    if (!rows.is_cuda() || rows.scalar_type() != at::kByte || !rows.is_contiguous()) throw std::runtime_error("rows_h: rows must be a contiguous CUDA uint8 tensor");
    if (stride % 16 || stride < (rs.row_bytes + 15) / 16 * 16 || reinterpret_cast<uintptr_t>(rows.data_ptr()) % 16)
      throw std::runtime_error("rows_h: rows 16-byte aligned at a stride of whole 16-byte loads covering the row");
    if (nbytes < 0 || nbytes > rows.numel() || nbytes % stride) throw std::runtime_error("rows_h: nbytes must be a stride multiple within rows");
    if (!tn.is_cuda() || tn.scalar_type() != at::kByte || !tn.is_contiguous() || tn.numel() != 8192) throw std::runtime_error("rows_h: tn must be the 8 KB CUDA mask table");
    const int64_t n_rows = nbytes / stride;
    if (first < 0 || first + n_rows > (int64_t(1) << 32)) throw std::runtime_error("rows_h: positions below 2^32");
    if (salts.defined() && (!salts.is_cuda() || salts.scalar_type() != at::kByte || !salts.is_contiguous() || salts.numel() < n_rows * 192))
      throw std::runtime_error("rows_h: salts must be a contiguous CUDA uint8 tensor of 192 bytes per row");
    at::Tensor hs, bc;
    {
      py::gil_scoped_release nogil;
      c10::InferenceMode inf;
      cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_ptr);
      c10::cuda::CUDAStream cs = c10::cuda::getStreamFromExternal(st, rows.device().index());
      c10::cuda::CUDAStreamGuard g(cs);
      const auto opt = at::TensorOptions().dtype(at::kByte).device(rows.device());
      hs = at::empty({n_rows, 64}, opt);
      bc = want_bc ? at::empty({n_rows, 128}, opt) : at::empty({0}, opt);
      at::Tensor sf = at::empty({n_rows * 192}, opt);
      const int lrc = verity_rows_h_launch(rs.params.data(), reinterpret_cast<const uint8_t*>(rows.data_ptr()), static_cast<uint64_t>(stride),
                                           static_cast<uint64_t>(n_rows), salts.defined() ? reinterpret_cast<const uint8_t*>(salts.data_ptr()) : nullptr,
                                           reinterpret_cast<const uint8_t*>(tn.data_ptr()), static_cast<uint64_t>(first), reinterpret_cast<uint8_t*>(hs.data_ptr()),
                                           want_bc ? reinterpret_cast<uint8_t*>(bc.data_ptr()) : nullptr, reinterpret_cast<uint8_t*>(sf.data_ptr()),
                                           static_cast<uint64_t>(rs.max_blocks), st);
      if (lrc != 0) throw std::runtime_error(std::string("rows_h: ") + cudaGetErrorString(static_cast<cudaError_t>(lrc)));
    }
    return py::make_tuple(hs, bc);
  }

  // The leaves of positions first .. first + m - 1 of a step of n_total leaves from rows_h's h of its positions first .. first + len(hs) - 1
  // (a pad at every position >= n_total).  -> leaves [m, 64] uint8, rows_leaves' bytes
  at::Tensor rows_frame(const RowStep& rs, at::Tensor hs, int64_t first, int64_t m, int64_t n_total, int64_t stream_ptr) {
    if (!hs.is_cuda() || hs.scalar_type() != at::kByte || !hs.is_contiguous() || (hs.numel() && (hs.dim() != 2 || hs.size(1) != 64)))
      throw std::runtime_error("rows_frame: hs must be a contiguous CUDA uint8 [n, 64]");
    if (first < 0 || m < 0 || n_total < 0 || first + m > (int64_t(1) << 32)) throw std::runtime_error("rows_frame: positions below 2^32");
    const int64_t n_h = hs.numel() ? hs.size(0) : 0;
    py::gil_scoped_release nogil;
    c10::InferenceMode inf;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_ptr);
    c10::cuda::CUDAStream cs = c10::cuda::getStreamFromExternal(st, hs.device().index());
    c10::cuda::CUDAStreamGuard g(cs);
    at::Tensor leaves = at::empty({m, 64}, at::TensorOptions().dtype(at::kByte).device(hs.device()));
    const int lrc = verity_rows_frame_launch(rs.params.data(), reinterpret_cast<const uint8_t*>(hs.data_ptr()), static_cast<uint64_t>(n_h),
                                             static_cast<uint64_t>(first), static_cast<uint64_t>(m), static_cast<uint64_t>(n_total),
                                             reinterpret_cast<uint8_t*>(leaves.data_ptr()), static_cast<uint64_t>(rs.max_blocks), st);
    if (lrc != 0) throw std::runtime_error(std::string("rows_frame: ") + cudaGetErrorString(static_cast<cudaError_t>(lrc)));
    return leaves;
  }

  // rows-sha512 absorb: levels 0 .. n_levels-1 of one window's subtree from its leaves (positions first .. first + m - 1, m a power of two
  // and first a multiple of m, so every pair is inside the window; or several consecutive windows' leaves, whose subtrees are the same
  // levels side by side), each level copied D2H into host_dsts[L] when they are given (one event sync at the end); with none, the levels
  // stay on the device and nothing syncs.  copy_stream_ptr: the copies on that stream once these kernels are done, and no sync; the levels
  // stay allocated until the copies end, and host_dsts may be read only once that stream is joined.  -> the top level (device)
  at::Tensor rows_tree_window(const RowStep& rs, at::Tensor leaves, int64_t n_levels, int64_t first, int64_t stream_ptr, std::vector<at::Tensor> host_dsts,
                              int64_t copy_stream_ptr = 0) {
    if (!leaves.is_cuda() || leaves.scalar_type() != at::kByte || !leaves.is_contiguous() || leaves.dim() != 2 || leaves.size(1) != 64)
      throw std::runtime_error("rows_tree_window: leaves must be a contiguous CUDA uint8 [m, 64]");
    if (n_levels < 1 || (!host_dsts.empty() && static_cast<int64_t>(host_dsts.size()) != n_levels))
      throw std::runtime_error("rows_tree_window: n_levels must be positive, with one host destination per level or none");
    for (int64_t L = 0, n = leaves.size(0); L + 1 < n_levels; ++L, n /= 2)
      if (n % 2) throw std::runtime_error("rows_tree_window: level " + std::to_string(L) + " has an odd " + std::to_string(n) + " nodes");
    py::gil_scoped_release nogil;
    c10::InferenceMode inf;
    cudaStream_t side = reinterpret_cast<cudaStream_t>(stream_ptr);
    c10::cuda::CUDAStream cs = c10::cuda::getStreamFromExternal(side, leaves.device().index());
    c10::cuda::CUDAStreamGuard g(cs);
    std::vector<at::Tensor> levels{leaves};
    for (auto& lv : rows_levels_above(rs.params.data(), leaves, 0, first, n_levels - 1, side, rs.max_blocks)) levels.push_back(lv);
    if (!host_dsts.empty()) {
      cudaStream_t cp = copy_stream_ptr ? reinterpret_cast<cudaStream_t>(copy_stream_ptr) : side;
      if (cp != side) {
        cudaError_t rc = cudaEventRecord(copy_from, side);
        if (rc == cudaSuccess) rc = cudaStreamWaitEvent(cp, copy_from, 0);
        if (rc != cudaSuccess) throw std::runtime_error(std::string("rows_tree_window: copy stream wait: ") + cudaGetErrorString(rc));
      }
      const c10::cuda::CUDAStream ccs = c10::cuda::getStreamFromExternal(cp, leaves.device().index());
      for (int64_t L = 0; L < n_levels; ++L) {
        const at::Tensor& h = host_dsts[static_cast<size_t>(L)];
        const at::Tensor& cur = levels[static_cast<size_t>(L)];
        if (h.is_cuda() || h.scalar_type() != at::kByte || !h.is_contiguous() || h.numel() != cur.numel())
          throw std::runtime_error("rows_tree_window: level " + std::to_string(L) + " host slice must be contiguous host uint8 of " + std::to_string(cur.size(0)) + " nodes");
        cudaError_t rc = cudaMemcpyAsync(h.data_ptr(), cur.data_ptr(), static_cast<size_t>(cur.numel()), cudaMemcpyDeviceToHost, cp);
        if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync(window level->host): ") + cudaGetErrorString(rc));
        if (cp != side) cur.record_stream(ccs);
      }
      if (!copy_stream_ptr) {
        cudaEventRecord(done, side);
        cudaError_t rc = cudaEventSynchronize(done);
        if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaEventSynchronize: ") + cudaGetErrorString(rc));
      }
    }
    return levels.back();
  }

  // rows-sha512: the tree above `nodes` (level level0, positions 0 .. n-1, n a power of two) to its root; root D2H (64 B), the levels to
  // pinned host when asked (into levels_dst when it holds them); one event sync.  -> (levels [nodes .. root], root: bytes(64), tree_ms)
  py::tuple rows_tree(const RowStep& rs, at::Tensor nodes, int64_t level0, int64_t stream_ptr, bool levels_host = false,
                      std::optional<at::Tensor> levels_into = std::nullopt) {
    const at::Tensor levels_dst = levels_into.has_value() ? *levels_into : at::Tensor();
    if (!nodes.is_cuda() || nodes.scalar_type() != at::kByte || !nodes.is_contiguous() || nodes.dim() != 2 || nodes.size(1) != 64)
      throw std::runtime_error("rows_tree: nodes must be a contiguous CUDA uint8 [n, 64]");
    const int64_t n0 = nodes.size(0);
    if (n0 < 1 || (n0 & (n0 - 1)) || level0 < 0) throw std::runtime_error("rows_tree: a power of two of nodes at a level >= 0");
    std::vector<at::Tensor> levels;
    float tree_ms = 0.f;
    {
      py::gil_scoped_release nogil;
      c10::InferenceMode inf;
      cudaStream_t side = reinterpret_cast<cudaStream_t>(stream_ptr);
      c10::cuda::CUDAStream cs = c10::cuda::getStreamFromExternal(side, nodes.device().index());
      c10::cuda::CUDAStreamGuard g(cs);
      cudaEventRecord(ev1, side);
      levels.push_back(nodes);
      for (auto& lv : rows_levels_above(rs.params.data(), nodes, level0, 0, levels_to_root(n0), side, rs.max_blocks)) levels.push_back(lv);
      cudaEventRecord(ev2, side);
      cudaError_t rc = cudaMemcpyAsync(root64_host, levels.back().data_ptr(), 64, cudaMemcpyDeviceToHost, side);
      if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync(root): ") + cudaGetErrorString(rc));
      if (levels_host) {
        int64_t total = 0;
        for (auto& lv : levels) total += lv.numel();
        at::Tensor hbuf;
        if (levels_dst.defined() && levels_dst.numel() >= total) {
          if (levels_dst.is_cuda() || levels_dst.scalar_type() != at::kByte || !levels_dst.is_contiguous()) throw std::runtime_error("rows_tree: levels_dst must be a contiguous host uint8 tensor");
          hbuf = levels_dst.narrow(0, 0, total);
        } else {
          hbuf = at::empty({total}, at::TensorOptions().dtype(at::kByte).device(at::kCPU).pinned_memory(true));
        }
        std::vector<at::Tensor> hl;
        int64_t hoff = 0;
        for (auto& lv : levels) {
          at::Tensor h = hbuf.narrow(0, hoff, lv.numel()).view({lv.size(0), 64});
          hoff += lv.numel();
          rc = cudaMemcpyAsync(h.data_ptr(), lv.data_ptr(), static_cast<size_t>(lv.numel()), cudaMemcpyDeviceToHost, side);
          if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpyAsync(level->host): ") + cudaGetErrorString(rc));
          hl.push_back(h);
        }
        levels.swap(hl);
      }
      cudaEventRecord(done, side);
      rc = cudaEventSynchronize(done);
      if (rc != cudaSuccess) throw std::runtime_error(std::string("cudaEventSynchronize: ") + cudaGetErrorString(rc));
      cudaEventElapsedTime(&tree_ms, ev1, ev2);
    }
    py::list lv;
    for (auto& t : levels) lv.append(t);
    return py::make_tuple(lv, py::bytes(reinterpret_cast<const char*>(root64_host), 64), static_cast<double>(tree_ms));
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<StepHasher>(m, "StepHasher")
      .def(py::init<>())
      .def("leaves_window", &StepHasher::leaves_window, py::arg("slot"), py::arg("nbytes"), py::arg("first_chunk"), py::arg("step"), py::arg("src_mask"),
           py::arg("chunk_bytes"), py::arg("m_words"), py::arg("stream_ptr"))
      .def("tree", &StepHasher::tree, py::arg("leaves"), py::arg("stream_ptr"), py::arg("levels_host") = false, py::arg("levels_dst") = at::Tensor())
      .def("tree_window", &StepHasher::tree_window, py::arg("leaves"), py::arg("n_levels"), py::arg("stream_ptr"), py::arg("host_dsts"))
      .def("tree_window_top", &StepHasher::tree_window_top, py::arg("leaves"), py::arg("n_levels"), py::arg("stream_ptr"))
      .def("commit", &StepHasher::commit, py::arg("arena"), py::arg("used"), py::arg("step"), py::arg("src_mask"), py::arg("chunk_bytes"),
           py::arg("side_stream_ptr"), py::arg("step_event_ptr"), py::arg("keep_mode") = 0, py::arg("keep_dst") = at::Tensor(), py::arg("levels_host") = false, py::arg("levels_dst") = at::Tensor())
      .def("rows_leaves", &StepHasher::rows_leaves, py::arg("rs"), py::arg("rows"), py::arg("nbytes"), py::arg("first"), py::arg("m"), py::arg("n_total"),
           py::arg("tn"), py::arg("stream_ptr"), py::arg("stride") = 0, py::arg("salts") = py::none(), py::arg("want_bc") = false, py::arg("want_h") = false)
      .def("rows_h", &StepHasher::rows_h, py::arg("rs"), py::arg("rows"), py::arg("nbytes"), py::arg("first"), py::arg("tn"), py::arg("stream_ptr"),
           py::arg("stride") = 0, py::arg("salts") = py::none(), py::arg("want_bc") = false)
      .def("rows_frame", &StepHasher::rows_frame, py::arg("rs"), py::arg("hs"), py::arg("first"), py::arg("m"), py::arg("n_total"), py::arg("stream_ptr"))
      .def("rows_tree_window", &StepHasher::rows_tree_window, py::arg("rs"), py::arg("leaves"), py::arg("n_levels"), py::arg("first"), py::arg("stream_ptr"),
           py::arg("host_dsts"), py::arg("copy_stream_ptr") = 0)
      .def("rows_tree", &StepHasher::rows_tree, py::arg("rs"), py::arg("nodes"), py::arg("level0"), py::arg("stream_ptr"), py::arg("levels_host") = false,
           py::arg("levels_dst") = py::none());
  py::class_<RowStep>(m, "RowStep")
      .def(py::init<const py::bytes&, const py::bytes&, const py::bytes&, const py::bytes&, const py::bytes&, const std::string&, int64_t>(), py::arg("domain_id"),
           py::arg("ks"), py::arg("row_prefix"), py::arg("salt_prefix"), py::arg("leaf_prefix"), py::arg("schema"), py::arg("row_bytes"))
      .def_readonly("row_bytes", &RowStep::row_bytes)
      .def_readwrite("max_blocks", &RowStep::max_blocks);
  py::class_<Collector>(m, "Collector")
      .def(py::init<>())
      .def("begin_step", &Collector::begin_step, py::arg("plan_key"), py::arg("arena_ptr"), py::arg("arena_bytes"), py::arg("use_plan") = true, py::arg("step_id") = -1)
      .def("enable_windows", &Collector::enable_windows, py::arg("ptrs"), py::arg("window_bytes"))
      .def("disable_windows", &Collector::disable_windows)
      .def("wait_window", &Collector::wait_window, py::arg("step_id"), py::arg("k"), py::arg("stream_ptr"))
      .def("abandon_step", &Collector::abandon_step, py::arg("step_id"), py::arg("why") = "", py::arg("stream_ptr") = 0)
      .def_readonly("failure", &Collector::fail_reason)
      .def("release_slot", &Collector::release_slot, py::arg("slot"), py::arg("stream_ptr"))
      .def("step_windows", &Collector::step_windows, py::arg("step_id"))
      .def("finish_step", &Collector::finish_step, py::arg("step_id"))
      .def("window_stats", &Collector::window_stats)
      .def("free_slots", &Collector::free_slots)
      .def("plan_used", &Collector::plan_used, py::arg("plan_key"))
      .def("plan_version", &Collector::plan_version, py::arg("plan_key"))
      .def_readwrite("win_wait_timeout_s", &Collector::win_wait_timeout_s)
      .def_readonly("win_dead", &Collector::win_dead)
      .def("set_watchdog", &Collector::set_watchdog, py::arg("marker"), py::arg("grace_s"))      // fail-fast (see start_dead_watchdog)
      .def("ack_dead", &Collector::ack_dead)
      .def("progress", &Collector::progress)
      .def("watchdog_started", [](const Collector& c) { return c.watchdog_started.load(); })
      .def("force_dead_for_test", [](Collector& c, const std::string& why) { c.win_dead = true; c.start_dead_watchdog(why.c_str()); })
      .def("slot_generation", &Collector::slot_generation, py::arg("slot"))
      .def("wait_slot_reused", &Collector::wait_slot_reused, py::arg("slot"), py::arg("gen"), py::arg("timeout_s"), py::arg("stream_ptr"))
      .def("collect", &Collector::collect, py::arg("name_id"), py::arg("tensor"))
      .def("end_step", &Collector::end_step)
      .def("end_step_full", &Collector::end_step_full, py::arg("flush"), py::arg("drain"))
      .def("export_plan", &Collector::export_plan)
      .def("step_stats", &Collector::step_stats)
      .def("note_expected", &Collector::note_expected, py::arg("n") = 1)
      .def("stats", &Collector::stats)
      .def("reset_stats", &Collector::reset_stats)
      .def("clear_plans", &Collector::clear_plans)
      .def("end_run", &Collector::end_run)
      .def("enable_gather", &Collector::enable_gather, py::arg("capacity") = 65536)
      .def("stage", &Collector::stage, py::arg("name_id"), py::arg("tensor"))
      .def("flush", &Collector::flush)
      .def("add_slot_plan", &Collector::add_slot_plan, py::arg("name_ids"), py::arg("slots"), py::arg("kinds"))
      .def("stage_slot_plan", &Collector::stage_slot_plan, py::arg("handle"), py::arg("n"), py::arg("pad"))
      .def("clear_slot_plans", &Collector::clear_slot_plans)
      .def_readwrite("blocks_per_desc", &Collector::blocks_per_desc)
      .def_readwrite("check_sizes", &Collector::check_sizes)
      .def_readwrite("fault_skip_copy_at", &Collector::fault_skip_copy_at)
      .def_readwrite("use_gather", &Collector::use_gather)
      .def_readwrite("slot_fast", &Collector::slot_fast)
      .def_readwrite("auto_blocks", &Collector::auto_blocks)
      .def_readwrite("auto_bytes_per_block", &Collector::auto_bytes_per_block)
      .def_readwrite("drain_side", &Collector::side_drain)
      .def("side_stream_for_test", [](Collector& c) { return reinterpret_cast<int64_t>(c.side_s); })
      .def("drain_errors", &Collector::drain_errors)
      .def_readonly("active", &Collector::active)
      .def_readonly("learning", &Collector::learning);
  py::class_<HookFwd>(m, "HookFwd")
      .def(py::init<py::object, py::object, int64_t, py::sequence, int64_t, int64_t, bool>(), py::arg("collector"), py::arg("orig"), py::arg("nid_out"),
           py::arg("nids_k"), py::arg("nid_in"), py::arg("nid_st"), py::arg("flush_after"))
      .def("__call__", &HookFwd::call)
      .def_readonly("orig", &HookFwd::orig)
      .def_readonly("flush_after", &HookFwd::flush_after);
  py::class_<Fa2Wrap>(m, "Fa2Wrap")
      .def(py::init<py::object, py::object, py::object, py::object, std::string>(), py::arg("collector"), py::arg("orig_op"), py::arg("tapped_op"),
           py::arg("intern"), py::arg("ns"))
      .def("__call__", &Fa2Wrap::call)
      .def("on_step_begin", &Fa2Wrap::on_step_begin)
      .def("stats", &Fa2Wrap::stats)
      .def_readonly("launches_in_step", &Fa2Wrap::launches_in_step)
      .def_readonly("launch", &Fa2Wrap::launch)
      .def_readwrite("compact", &Fa2Wrap::compact)
      .def_readwrite("zero_after", &Fa2Wrap::zero_after)
      .def_readwrite("bound_words", &Fa2Wrap::bound_words)      // bounded tap: cap in i32 words (0 = off)
      .def_readwrite("seq_base", &Fa2Wrap::seq_base)
      .def_readwrite("hold_launch", &Fa2Wrap::hold_launch)
      .def_readwrite("ms", &Fa2Wrap::ms)
      .def("end_launch", &Fa2Wrap::end_launch);
  py::class_<Fa3Wrap>(m, "Fa3Wrap")
      .def(py::init<py::object, py::object, py::object, py::object, std::string>(), py::arg("collector"), py::arg("orig_op"), py::arg("tapped_op"),
           py::arg("intern"), py::arg("ns"))
      .def("__call__", &Fa3Wrap::call)
      .def("on_step_begin", &Fa3Wrap::on_step_begin)
      .def("stats", &Fa3Wrap::stats)
      .def_readonly("launches_in_step", &Fa3Wrap::launches_in_step)
      .def_readonly("launch", &Fa3Wrap::launch)
      .def_readwrite("zero_after", &Fa3Wrap::zero_after)
      .def_readwrite("ms", &Fa3Wrap::ms);
  m.def("cuda_runtime_version", []() { int v = 0; cudaRuntimeGetVersion(&v); return v; });
}
