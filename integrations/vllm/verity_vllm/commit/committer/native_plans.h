// The native collector's plans (native_collect.cpp), kept for the current commitment run and the previous one.
//
// A plan is keyed by the step's geometry, so every decode step of a growing KV length learns a plan of its own, and a collector that kept
// every plan grew by one plan per committed step for the life of the process. A step reuses a plan learned in its own run or in the run
// before it (the warm-up learns the plans the record run follows), so `end_run` drops the plans the run that just ended did not use: the
// same two generations as the worker's metas (plan_cache.py). A dropped plan that recurs is learned again, at the same offsets: the layout
// is a function of the step's tensors. Its versions start above every version a dropped plan had, so a (key, version) the worker may still
// hold never names a plan learned after the drop. Torch- and CUDA-free, so the CPU tests compile it on its own.
//
// `end_run` runs in the committer's reset, which the served engine waits for, and a served window drops every plan of the window before
// it: freeing them there costs one free per committed tensor of a window. So `end_run` only takes the dropped plans out of the book (their
// map nodes, into `pending_`), and each step frees RELEASE_PER_STEP of them in `use`, twice the rate a run of new plans drops them. What a
// run too short to free them leaves is freed by the next `end_run`, so a dropped plan is freed by the end of the run after the drop, and
// the book and the pending plans together never hold more than the two runs' plans the book held before the reset.
#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace native_plans {

struct Entry {
  int64_t name_id;
  int64_t nbytes;
  int64_t off;                 // byte offset in the arena block
  int64_t gap;                 // zero bytes between the previous entry's end and `off` (0 or 1..3)
  int32_t dtype;               // c10::ScalarType
  std::vector<int64_t> sizes;
};

struct Plan {
  std::vector<Entry> entries;
  int64_t version = 0;
  int64_t used = 0;            // bytes used by the full plan (end of last entry)
  int64_t run = 0;             // the last run a step followed or learned this plan in
};

class Plans {
 public:
  static constexpr int64_t RELEASE_PER_STEP = 2;

  // the plan a step of `key` follows or learns (an empty one when none is held), now a plan of the current run; first frees up to
  // RELEASE_PER_STEP dropped plans
  Plan& use(int64_t key) {
    release(RELEASE_PER_STEP);
    auto ins = by_key_.try_emplace(key);
    Plan& p = ins.first->second;
    if (ins.second) p.version = version_floor_;
    p.run = run_;
    run_used_ = true;
    return p;
  }
  const Plan* find(int64_t key) const {
    auto it = by_key_.find(key);
    return it == by_key_.end() ? nullptr : &it->second;
  }
  // between runs: drop the plans the run that just ended did not use (a run that used none, two resets in a row, drops nothing), after
  // freeing what the steps of that run left of the previous drop
  void end_run() {
    if (!run_used_) return;
    release(static_cast<int64_t>(pending_.size()));
    pending_.reserve(by_key_.size());
    for (auto it = by_key_.begin(); it != by_key_.end();) {
      if (it->second.run == run_) { ++it; continue; }
      retire(it->second);
      pending_entries_ += static_cast<int64_t>(it->second.entries.size());
      pending_.push_back(by_key_.extract(it++));
    }
    ++run_;
    run_used_ = false;
  }
  void clear() {
    for (const auto& kv : by_key_) retire(kv.second);
    by_key_.clear();
    release(static_cast<int64_t>(pending_.size()));
  }
  int64_t size() const { return static_cast<int64_t>(by_key_.size()); }
  int64_t entries() const {
    int64_t n = 0;
    for (const auto& kv : by_key_) n += static_cast<int64_t>(kv.second.entries.size());
    return n;
  }
  int64_t pending() const { return static_cast<int64_t>(pending_.size()); }
  int64_t pending_entries() const { return pending_entries_; }

 private:
  using Map = std::unordered_map<int64_t, Plan>;

  void retire(const Plan& p) { version_floor_ = std::max(version_floor_, p.version + 1); }
  void release(int64_t n) {
    for (; n > 0 && !pending_.empty(); --n) {
      pending_entries_ -= static_cast<int64_t>(pending_.back().mapped().entries.size());
      pending_.pop_back();
    }
  }

  Map by_key_;
  std::vector<Map::node_type> pending_;  // dropped, out of the book, not yet freed
  int64_t pending_entries_ = 0;
  int64_t run_ = 0;
  bool run_used_ = false;
  int64_t version_floor_ = 0;  // above every version a dropped plan had: where a plan created after a drop starts
};

}  // namespace native_plans
