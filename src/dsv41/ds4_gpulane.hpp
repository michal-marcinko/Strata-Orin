// src/dsv41/ds4_gpulane.hpp - the decode path's GPU lane for an integrated GPU (Jetson Orin: one LPDDR5 for the CPU and the
// GPU).  A decoded token's routed experts that VRAM does not hold were all computed by the CPU lane (the ARM cores, ~0.5-0.9 ms
// an IQ1 expert); here they are read into a ring of pinned slots instead (cudaHostAlloc'd: the GPU reads them in place at the
// DRAM rate, 175 GB/s measured - the same as its own allocations) and the GPU computes them with the VRAM experts' kernels
// (~0.05-0.1 ms an expert).  Measured on the Orin (tools/orin-zc/zc.cu): NVMe O_DIRECT reads land in cudaHostAlloc'd memory
// (4.8 GB/s, 4 threads x 1 MB), a page-cache hit is a memcpy from the model's mapping (~30 GB/s, 4 threads).
//
// The host side (this file): a slot a (layer, expert), filled by reader threads in chunks - each chunk from the page cache
// (memcpy from the mapping, when mincore says its pages are there) or unbuffered from the disk - the last chunk sets the slot's
// flag (pinned memory the GPU polls) to the fill's generation.  The GPU's fetch kernel (ds4_pcie.cuh gl_fetch) waits for its
// picks' flags, then points the plan at the slots.  Unbuffered reads need 512-byte aligned offsets: matrix r of every expert
// of a layer sits at the same offset mod 512 in the file (lead[r]; its expert stride is a multiple of 512), so a slot keeps
// matrix r at reg[r] + lead[r] (reg 4096-aligned) and the GPU kernels take the layer's lane layout (those offsets).
//
// Threads: demand(), prefetch(), seen(), cancel_before() from the service thread alone (it owns the slot table); the readers
// touch a slot's counters and flag only.  A slot the GPU may still read (used by request seq, not yet seen past) is not
// given up.  On by default on an integrated GPU, DS4_GL=0 off (ds4_model.cu init_gl).
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::ds4 {

class GpuLane {
public:
    static constexpr uint64_t kAlign = 512;      // the unbuffered reads' alignment (the NVMe's logical block)
    static constexpr unsigned kBad = 0x80000000u;   // a slot's flag: gen | kBad when its fill failed
    struct Geo {                // a layer's experts in the files, and in a slot
        bool ok = false;
        int file[3] = {-1, -1, -1};
        uint64_t off[3] = {}, stride[3] = {}, bytes[3] = {};
        uint64_t lead[3] = {};  // off[r] % kAlign (the same for every expert)
        uint64_t reg[3] = {};   // matrix r's region in the slot (4096-aligned); the matrix at reg + lead
        uint64_t rlen[3] = {};  // its read: lead + bytes rounded up to kAlign
    };
    struct File {
        int dfd = -1, bfd = -1;           // unbuffered, buffered
        const uint8_t* map = nullptr;     // the read-only mapping (page-cache hits are copied from it)
        uint64_t size = 0;
    };
    struct Cfg {
        int slots = 32, threads = 6;
        uint64_t chunk = 1ull << 20;
        bool direct = true;   // misses unbuffered (else buffered reads: the page cache keeps them)
        std::vector<int> cpus;   // reader i on cpus[i % n] (its reads on that CPU's NVMe queue); empty: unpinned
        // the Orin's lost NVMe interrupts (a read's completion waits for the 1 s io_timeout unless another completion on
        // the same queue comes first): a read not done after kick_ms gets a 4 KB read on its CPU's queue from a kicker
        // thread pinned there - that read's interrupt takes the lost completion with it.  0: off (needs pinned readers)
        double kick_ms = 4.0;
        int kick_fd = -1;     // an unbuffered file the kicks read from
    };
    struct Stats {
        std::atomic<int64_t> demand{0}, ready{0}, filling{0}, missed{0}, no_slot{0};
        std::atomic<int64_t> pf{0}, pf_used{0}, pf_dropped{0}, pf_cut{0};
        std::atomic<int64_t> ch_cache{0}, ch_direct{0}, ch_buf{0}, ch_fail{0};
        std::atomic<int64_t> b_cache{0}, b_disk{0};
        std::atomic<int64_t> waited{0}, wait_us{0};   // picks not ready when asked, their wait from the ask to the fill's end
        std::atomic<int64_t> kicks{0}, slow{0}, slow_us{0};   // kicks sent; reads that took > 50 ms, their time
        std::atomic<int64_t> copied_out{0};   // experts the expert cache's mover took from a slot (not the files)
    };
    // DS4_GL_STATS=1: where the lane's picks came from, a layer at a time (printed at exit).  A pick, when its request asks:
    // ready (a guess for it, or a slot an earlier request filled), a guess still queued / being read, not held though a guess
    // was made (no slot, dropped, or given up), or never guessed
    static constexpr int kStatL = 64;
    struct LStats {
        std::atomic<int64_t> demand{0}, ready_guess{0}, ready_reuse{0}, fill_q{0}, fill_r{0}, miss_g{0}, miss_u{0};
        std::atomic<int64_t> waited{0}, wait_us{0};       // picks asked before their fill was done, their wait
        std::atomic<int64_t> pf{0}, pf_used{0}, pf_wasted{0};   // guesses read ahead; asked for; read and given up unasked
        std::atomic<int64_t> pf_dropped{0}, pf_cut{0};          // guesses dropped before their read started, part read
        std::atomic<int64_t> lead_n{0}, lead_us{0};       // guesses asked while still filling: their time from the guess to the ask
        std::atomic<int64_t> qwait_n{0}, qwait_us{0};     // fills: from the fill's queueing to its first read starting
        std::atomic<int64_t> rd_n{0}, rd_us{0};           // fills: from the first read starting to the last one's end
    };

    // the slot geometry: each layer's regions (null geo: none); returns the slot bytes needed (0: no layer fits)
    static uint64_t layout(std::vector<Geo>& geo) {
        uint64_t need = 0;
        for (Geo& g : geo) {
            if (!g.ok) continue;
            uint64_t at = 0;
            for (int r = 0; r < 3; ++r) {
                g.lead[r] = g.off[r] % kAlign;
                g.rlen[r] = (g.lead[r] + g.bytes[r] + kAlign - 1) / kAlign * kAlign;
                g.reg[r] = at;
                at += (g.rlen[r] + 4095) / 4096 * 4096;
            }
            need = std::max(need, at);
        }
        return (need + 65535) / 65536 * 65536;
    }

    bool init(const Cfg& c, const std::vector<Geo>& geo, const std::vector<File>& files, uint8_t* slab, uint64_t stride,
              volatile unsigned* flags) {
        cfg_ = c;
        geo_ = geo;
        files_ = files;
        slab_ = slab;
        stride_ = stride;
        flags_ = flags;
        slots_.reset(new Slot[(size_t) c.slots]);
        where_.resize(geo_.size());
        for (size_t l = 0; l < geo_.size(); ++l) where_[l].assign(1024, -1);
        for (int i = 0; i < 2 * c.slots; ++i) flags_[i] = 0u;
        const int nt = std::max(1, c.threads);
        rd_.reset(new Reader[(size_t) nt]);
        for (int t = 0; t < nt; ++t) {
            rd_[(size_t) t].cpu = c.cpus.empty() ? -1 : c.cpus[(size_t) t % c.cpus.size()];
            th_.emplace_back([this, t] { loop(t); });
        }
        if (c.kick_ms > 0 && c.kick_fd >= 0 && !c.cpus.empty()) {
            for (int t = 0; t < nt; ++t) kth_.emplace_back([this, t] { kicker(t); });
            kth_.emplace_back([this, nt] { watch(nt); });
        }
        return true;
    }
    static std::atomic<GpuLane*>& reporter() {
        static std::atomic<GpuLane*> r{nullptr};
        return r;
    }
    ~GpuLane() {
        GpuLane* me = this;
        if (reporter().compare_exchange_strong(me, nullptr)) report();
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
        kquit_.store(true);
        for (int t = 0; t < (int) th_.size(); ++t) {
            std::lock_guard<std::mutex> lk(rd_[(size_t) t].mu);
            rd_[(size_t) t].cv.notify_all();
        }
        for (auto& t : kth_) t.join();
    }
    bool on() const { return !th_.empty(); }
    int slots() const { return cfg_.slots; }
    uint8_t* base(int s) const { return slab_ + (size_t) s * stride_; }
    const Geo& geo(int layer) const { return geo_[(size_t) layer]; }
    Stats& stats() { return st_; }
    LStats& lst(int layer) { return lst_[(size_t) std::max(0, std::min(kStatL - 1, layer))]; }
    void stats_on(bool v) { stats_on_ = v; }
    /// DS4_GL_STATS: the per-layer table (stderr)
    void report() {
        std::fprintf(stderr, "GPU lane by layer: picks | ready: guess, reuse | still filling: queued, reading | not held: guessed, "
                             "unguessed | waited, ms each | guesses: read, used, wasted (read, never asked), dropped unread, cut part read | guess lead ms (filling) | queue ms, read ms\n");
        LStats t;
        auto add = [](std::atomic<int64_t>& a, const std::atomic<int64_t>& b) { a += b.load(); };
        for (int l = 0; l < kStatL; ++l) {
            LStats& x = lst_[(size_t) l];
            if (x.demand.load() == 0 && x.pf.load() == 0) continue;
            const double d = std::max<double>(1.0, (double) x.demand.load());
            std::fprintf(stderr, "GL L%02d %6lld | %5.1f%% %5.1f%% | %5.1f%% %5.1f%% | %5.1f%% %5.1f%% | %6lld %.2f | %6lld %5.1f%% %5.1f%% %5.1f%% %5.1f%% | %.2f | %.2f %.2f\n",
                         l, (long long) x.demand.load(), 100 * x.ready_guess.load() / d, 100 * x.ready_reuse.load() / d,
                         100 * x.fill_q.load() / d, 100 * x.fill_r.load() / d, 100 * x.miss_g.load() / d, 100 * x.miss_u.load() / d,
                         (long long) x.waited.load(), x.waited.load() ? x.wait_us.load() / 1e3 / x.waited.load() : 0.0,
                         (long long) x.pf.load(), 100.0 * x.pf_used.load() / std::max<int64_t>(1, x.pf.load()),
                         100.0 * x.pf_wasted.load() / std::max<int64_t>(1, x.pf.load()),
                         100.0 * x.pf_dropped.load() / std::max<int64_t>(1, x.pf.load()),
                         100.0 * x.pf_cut.load() / std::max<int64_t>(1, x.pf.load()),
                         x.lead_n.load() ? x.lead_us.load() / 1e3 / x.lead_n.load() : 0.0,
                         x.qwait_n.load() ? x.qwait_us.load() / 1e3 / x.qwait_n.load() : 0.0,
                         x.rd_n.load() ? x.rd_us.load() / 1e3 / x.rd_n.load() : 0.0);
            add(t.demand, x.demand), add(t.ready_guess, x.ready_guess), add(t.ready_reuse, x.ready_reuse), add(t.fill_q, x.fill_q);
            add(t.fill_r, x.fill_r), add(t.miss_g, x.miss_g), add(t.miss_u, x.miss_u), add(t.waited, x.waited), add(t.wait_us, x.wait_us);
            add(t.pf, x.pf), add(t.pf_used, x.pf_used), add(t.pf_wasted, x.pf_wasted), add(t.pf_dropped, x.pf_dropped), add(t.pf_cut, x.pf_cut), add(t.lead_n, x.lead_n), add(t.lead_us, x.lead_us);
            add(t.qwait_n, x.qwait_n), add(t.qwait_us, x.qwait_us), add(t.rd_n, x.rd_n), add(t.rd_us, x.rd_us);
        }
        const double d = std::max<double>(1.0, (double) t.demand.load());
        std::fprintf(stderr, "GL: %lld experts the expert cache moved into VRAM came from a slot\n", (long long) st_.copied_out.load());
        std::fprintf(stderr, "GL all %6lld | %5.1f%% %5.1f%% | %5.1f%% %5.1f%% | %5.1f%% %5.1f%% | %6lld %.2f | %6lld %5.1f%% %5.1f%% %5.1f%% %5.1f%% | %.2f | %.2f %.2f\n",
                     (long long) t.demand.load(), 100 * t.ready_guess.load() / d, 100 * t.ready_reuse.load() / d,
                     100 * t.fill_q.load() / d, 100 * t.fill_r.load() / d, 100 * t.miss_g.load() / d, 100 * t.miss_u.load() / d,
                     (long long) t.waited.load(), t.waited.load() ? t.wait_us.load() / 1e3 / t.waited.load() : 0.0,
                     (long long) t.pf.load(), 100.0 * t.pf_used.load() / std::max<int64_t>(1, t.pf.load()),
                     100.0 * t.pf_wasted.load() / std::max<int64_t>(1, t.pf.load()),
                     100.0 * t.pf_dropped.load() / std::max<int64_t>(1, t.pf.load()),
                     100.0 * t.pf_cut.load() / std::max<int64_t>(1, t.pf.load()),
                     t.lead_n.load() ? t.lead_us.load() / 1e3 / t.lead_n.load() : 0.0,
                     t.qwait_n.load() ? t.qwait_us.load() / 1e3 / t.qwait_n.load() : 0.0,
                     t.rd_n.load() ? t.rd_us.load() / 1e3 / t.rd_n.load() : 0.0);
    }

    /// The GPU has passed every request before seq (the service thread saw request seq published: the device's stream
    /// ran everything enqueued before it)
    void seen(unsigned seq) { seen_ = seq; }

    /// (layer, e) for request seq: its slot (a fill started now if it holds nothing, at the front of the queue), pinned for
    /// the request; *gen the flag value the GPU waits for.  -1: no slot free (the CPU lane computes it)
    int demand(int layer, int e, unsigned seq, unsigned* gen, bool guessed = false) {
        ++st_.demand;
        LStats& ls = lst(layer);
        ++ls.demand;
        int s = where_[(size_t) layer][(size_t) e];
        if (s >= 0) {
            Slot& sp = slots_[(size_t) s];
            const int state = sp.state.load(std::memory_order_acquire);
            if (state == kBadFill) {   // (a failed fill: once more)
                drop(s);
                s = -1;
            } else {
                if (state == kReady) {
                    ++st_.ready;
                    if (sp.pf && !sp.used) ++ls.ready_guess;
                    else ++ls.ready_reuse;
                } else {
                    ++st_.filling;
                    if (sp.started.load(std::memory_order_relaxed) > 0) ++ls.fill_r;
                    else ++ls.fill_q;
                    if (sp.pf && !sp.used) {
                        ++ls.lead_n;
                        ls.lead_us += now_us() - sp.t_fill;
                    }
                    int64_t z = 0;
                    sp.asked.compare_exchange_strong(z, now_us());
                    promote(s);
                }
                if (sp.pf && !sp.used) {
                    ++st_.pf_used;
                    ++lst(sp.layer).pf_used;
                }
                sp.used = true;
                sp.pin = seq;
                sp.touch = ++tick_;
                *gen = sp.gen;
                return s;
            }
        }
        ++st_.missed;
        if (guessed) ++ls.miss_g;
        else ++ls.miss_u;
        s = take();
        if (s < 0) {
            ++st_.no_slot;
            return -1;
        }
        Slot& sp = slots_[(size_t) s];
        sp.used = true;
        sp.pf = false;
        sp.pin = seq;
        sp.asked.store(now_us());
        fill(s, layer, e, 0);
        *gen = sp.gen;
        return s;
    }
    /// (layer, e) read ahead (a guess): into a free slot behind the demand reads, unless held or no slot is free.  prio 1:
    /// the next layer's likely picks, 2: farther or less likely guesses (read only when nothing else waits).  A guess
    /// already being read at a lower priority moves up
    void prefetch(int layer, int e, int prio = 1) {
        prio = std::max(1, std::min(2, prio));
        const int h = where_[(size_t) layer][(size_t) e];
        if (h >= 0) {
            Slot& sp = slots_[(size_t) h];
            if (sp.prio > prio && sp.state.load(std::memory_order_acquire) == kFilling) {
                requeue(h, prio);
                sp.prio = prio;
            }
            return;
        }
        const int s = take();
        if (s < 0) return;
        Slot& sp = slots_[(size_t) s];
        sp.used = false;
        sp.pf = true;
        sp.pin = 0;
        ++st_.pf;
        ++lst(layer).pf;
        fill(s, layer, e, prio);
    }
    /// (another thread: the expert cache's mover) (layer, e)'s three matrices copied out of the slot that holds it read, to
    /// dst + at[r] - false when no slot holds it whole (or the slot was taken for another fill meanwhile: what was copied is
    /// then not trusted).  Lock-free: the slot's generation and state checked before and after the copy
    bool copy_out(int layer, int e, uint8_t* dst, const size_t at[3]) {
        if (layer < 0 || (size_t) layer >= where_.size() || e < 0 || (size_t) e >= where_[(size_t) layer].size()) return false;
        const int s = __atomic_load_n(&where_[(size_t) layer][(size_t) e], __ATOMIC_ACQUIRE);
        if (s < 0 || s >= cfg_.slots) return false;
        Slot& sp = slots_[(size_t) s];
        const unsigned g0 = __atomic_load_n(&sp.gen, __ATOMIC_ACQUIRE);
        if (sp.state.load(std::memory_order_acquire) != kReady || __atomic_load_n(&sp.layer, __ATOMIC_RELAXED) != layer ||
            __atomic_load_n(&sp.e, __ATOMIC_RELAXED) != e)
            return false;
        const Geo& g = geo_[(size_t) layer];
        for (int r = 0; r < 3; ++r) std::memcpy(dst + at[r], base(s) + g.reg[r] + g.lead[r], (size_t) g.bytes[r]);
        std::atomic_thread_fence(std::memory_order_acquire);
        const bool ok = sp.state.load(std::memory_order_acquire) == kReady && __atomic_load_n(&sp.gen, __ATOMIC_ACQUIRE) == g0 &&
                        __atomic_load_n(&sp.layer, __ATOMIC_RELAXED) == layer && __atomic_load_n(&sp.e, __ATOMIC_RELAXED) == e;
        if (ok) ++st_.copied_out;
        return ok;
    }
    /// (the load's check) (layer, e) read into a slot now, whatever it held, and waited for: the slot (-1: failed)
    int fill_now(int layer, int e) {
        const int old = where_[(size_t) layer][(size_t) e];
        if (old >= 0) {
            while (slots_[(size_t) old].state.load(std::memory_order_acquire) == kFilling) std::this_thread::yield();
            drop(old);
        }
        const int s = take();
        if (s < 0) return -1;
        slots_[(size_t) s].used = false;
        slots_[(size_t) s].pf = false;
        fill(s, layer, e, 0);
        while (slots_[(size_t) s].state.load(std::memory_order_acquire) == kFilling) std::this_thread::yield();
        return slots_[(size_t) s].state.load() == kReady && flags_[2 * s] == slots_[(size_t) s].gen &&
                       flags_[2 * s + 1] == slots_[(size_t) s].gen
                   ? s
                   : -1;
    }
    /// Guesses for layers before `layer` (the token has passed them, or its request named its picks) whose reads have not
    /// started: dropped.  started: also those part read - their queued reads dropped, the slot given up once the reads
    /// under way end (it holds nothing the GPU will ask for: its (layer, expert) forgotten now)
    void cancel_before(int layer, bool started = false) {
        std::vector<int> gone, part;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int q = 1; q <= 2; ++q)
                for (auto it = q_[q].begin(); it != q_[q].end();) {
                    Slot& sp = slots_[(size_t) it->slot];
                    if (sp.layer >= 0 && sp.layer < layer && !sp.used && sp.gen == it->gen) {
                        if (sp.started.load(std::memory_order_relaxed) == 0) {
                            if (std::find(gone.begin(), gone.end(), it->slot) == gone.end()) gone.push_back(it->slot);
                            it = q_[q].erase(it);
                            continue;
                        }
                        if (started) {   // (its reads under way finish; the queued ones go, counted down as done-and-bad)
                            if (std::find(part.begin(), part.end(), it->slot) == part.end()) part.push_back(it->slot);
                            sp.bad.store(1, std::memory_order_relaxed);
                            sp.part_bad[it->part].store(1, std::memory_order_relaxed);
                            sp.part_left[it->part].fetch_sub(1, std::memory_order_acq_rel);
                            const bool last = sp.left.fetch_sub(1, std::memory_order_acq_rel) == 1;
                            if (last) sp.state.store(kBadFill, std::memory_order_release);
                            it = q_[q].erase(it);
                            continue;
                        }
                    }
                    ++it;
                }
            for (int s : part) {   // (forgotten under the lock: the readers do not read sp.layer)
                Slot& sp = slots_[(size_t) s];
                ++st_.pf_cut;
                if (sp.layer >= 0) {
                    ++lst(sp.layer).pf_cut;
                    where_[(size_t) sp.layer][(size_t) sp.e] = -1;
                }
                sp.layer = sp.e = -1;
                sp.touch = 0;   // (first to go)
            }
        }
        for (int s : gone) {
            Slot& sp = slots_[(size_t) s];
            ++st_.pf_dropped;
            if (sp.layer >= 0) ++lst(sp.layer).pf_dropped;
            sp.left.store(0);
            sp.part_left[0].store(0);
            sp.part_left[1].store(0);
            sp.state.store(kFree, std::memory_order_release);
            where_[(size_t) sp.layer][(size_t) sp.e] = -1;
            sp.layer = sp.e = -1;
        }
    }

private:
    enum { kFree = 0, kFilling = 1, kReady = 2, kBadFill = 3 };
    struct Slot {
        int layer = -1, e = -1;
        unsigned gen = 0;
        std::atomic<int> state{kFree};
        std::atomic<int> left{0}, started{0}, bad{0};
        std::atomic<int> part_left[2] = {{0}, {0}};   // chunks of the gate / up, of the down still to come
        std::atomic<int> part_bad[2] = {{0}, {0}};
        std::atomic<int64_t> asked{0};   // (us) when a request asked for it before its fill was done (0: not)
        int64_t t_fill = 0;              // (us) when its fill was queued (DS4_GL_STATS)
        std::atomic<int64_t> t_start{0};   // (us) when its first read started
        unsigned pin = 0;          // the last request that uses it (the GPU may read it until a later one is seen)
        uint64_t touch = 0;
        bool pf = false, used = false;
        int prio = 0;              // its reads' queue: 0 demand, 1 near guesses, 2 far guesses
    };
    struct Job {
        int slot = 0, part = 0;   // part 0: the gate / up, 1: the down
        unsigned gen = 0;
        int layer = -1;
        int64_t t_fill = 0;
        int file = 0;
        uint64_t off = 0, len = 0, need = 0;   // file offset (aligned), the read's length, the bytes it must deliver
        uint8_t* dst = nullptr;
    };

    // the kicks (see Cfg::kick_ms): each reader's read start (us, 0: none), its kicker's wake-up
    struct Reader {
        int cpu = -1;
        std::atomic<int64_t> since{0};
        int64_t kicked = 0;
        std::mutex mu;
        std::condition_variable cv;
        bool go = false;
    };
    void drop(int s) {
        Slot& sp = slots_[(size_t) s];
        if (sp.layer >= 0) where_[(size_t) sp.layer][(size_t) sp.e] = -1;
        sp.layer = sp.e = -1;
        sp.state.store(kFree, std::memory_order_release);
    }
    // a slot to fill: a free one, else the least recently touched that no fill writes and the GPU no longer reads
    int take() {
        int best = -1;
        for (int i = 0; i < cfg_.slots; ++i) {
            Slot& sp = slots_[(size_t) i];
            const int state = sp.state.load(std::memory_order_acquire);
            if (state == kFree) return i;
            if (state == kFilling) continue;
            if (sp.used && (int) (sp.pin - seen_) >= 0) continue;   // (the GPU may still read it)
            if (best < 0 || sp.touch < slots_[(size_t) best].touch) best = i;
        }
        if (best >= 0) {
            Slot& b = slots_[(size_t) best];
            if (b.pf && !b.used && b.layer >= 0) ++lst(b.layer).pf_wasted;
            drop(best);
        }
        return best;
    }
    static int64_t now_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void fill(int s, int layer, int e, int prio) {
        Slot& sp = slots_[(size_t) s];
        if (prio > 0) sp.asked.store(0);
        sp.prio = prio;
        const Geo& g = geo_[(size_t) layer];
        sp.layer = layer;
        sp.e = e;
        gen_ = (gen_ + 1) & ~kBad;
        if (gen_ == 0) gen_ = 1;
        sp.gen = gen_;
        sp.touch = ++tick_;
        where_[(size_t) layer][(size_t) e] = s;
        sp.t_fill = now_us();
        std::vector<Job> js;
        for (int r = 0; r < 3; ++r) {
            const uint64_t a0 = g.off[r] + (uint64_t) e * g.stride[r] - g.lead[r];
            for (uint64_t at = 0; at < g.rlen[r]; at += cfg_.chunk) {
                Job j;
                j.slot = s;
                j.part = r < 2 ? 0 : 1;
                j.gen = sp.gen;
                j.layer = layer;
                j.t_fill = sp.t_fill;
                j.file = g.file[r];
                j.off = a0 + at;
                j.len = std::min<uint64_t>(cfg_.chunk, g.rlen[r] - at);
                const uint64_t want_end = g.lead[r] + g.bytes[r];   // (the bytes of the matrix this chunk must bring)
                j.need = want_end > at ? std::min<uint64_t>(j.len, want_end - at) : 0;
                j.dst = base(s) + g.reg[r] + at;
                js.push_back(j);
            }
        }
        sp.bad.store(0, std::memory_order_relaxed);
        sp.started.store(0, std::memory_order_relaxed);
        sp.t_start.store(0, std::memory_order_relaxed);
        int np[2] = {0, 0};
        for (const Job& j : js) ++np[j.part];
        for (int q = 0; q < 2; ++q) {
            sp.part_left[q].store(np[q], std::memory_order_relaxed);
            sp.part_bad[q].store(0, std::memory_order_relaxed);
        }
        sp.left.store((int) js.size(), std::memory_order_relaxed);
        sp.state.store(kFilling, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mu_);
            put(prio, js);
        }
        cv_.notify_all();
    }
    // a guess the token now needs: its queued reads move to the demand queue
    void promote(int s) {
        requeue(s, 0);
        slots_[(size_t) s].prio = 0;
    }
    // a fill's queued reads to queue `to` (ahead of their old one)
    void requeue(int s, int to) {
        std::lock_guard<std::mutex> lk(mu_);
        const unsigned gen = slots_[(size_t) s].gen;
        std::vector<Job> js;
        for (int q = to + 1; q <= 2; ++q)
            for (auto it = q_[q].begin(); it != q_[q].end();) {
                if (it->slot == s && it->gen == gen) {
                    js.push_back(*it);
                    it = q_[q].erase(it);
                } else
                    ++it;
            }
        put(to, js);
    }
    // (under mu_) reads into queue q.  DS4_GL_ORDER=1 (the demand queue): a fill's gate / up reads go ahead of the demand
    // fills' down reads already queued, so every pick's gate / up lands first and the GPU computes them while the downs come
    void put(int q, const std::vector<Job>& js) {
        static const bool order = [] { const char* v = std::getenv("DS4_GL_ORDER"); return v != nullptr && std::atoi(v) != 0; }();
        if (!order || q != 0) {
            q_[q].insert(q_[q].end(), js.begin(), js.end());
            return;
        }
        auto at = q_[0].begin();
        while (at != q_[0].end() && at->part == 0) ++at;
        for (const Job& j : js)
            if (j.part == 0) at = std::next(q_[0].insert(at, j));
        for (const Job& j : js)
            if (j.part != 0) q_[0].push_back(j);
    }
    static bool cached(const uint8_t* p, uint64_t n) {
#if defined(__linux__)
        const uintptr_t pg = 4096, a = (uintptr_t) p & ~(pg - 1);
        const size_t len = (size_t) ((uintptr_t) p + n - a);
        unsigned char v[1024];
        const size_t np = (len + pg - 1) / pg;
        if (np > sizeof v) return false;
        if (mincore((void*) a, len, v) != 0) return false;
        for (size_t i = 0; i < np; ++i)
            if (!(v[i] & 1)) return false;
        return true;
#else
        (void) p, (void) n;
        return false;
#endif
    }
    static bool read_all(int fd, uint8_t* dst, uint64_t len, uint64_t off, uint64_t need) {
#if defined(__linux__)
        uint64_t got = 0;
        while (got < len) {
            const ssize_t r = ::pread(fd, dst + got, (size_t) (len - got), (off_t) (off + got));
            if (r < 0) return false;
            if (r == 0) break;
            got += (uint64_t) r;
            if ((uint64_t) r % kAlign != 0) break;
        }
        return got >= need;
#else
        (void) fd, (void) dst, (void) len, (void) off, (void) need;
        return false;
#endif
    }
    void run(const Job& j, Reader& R) {
        Slot& sp = slots_[(size_t) j.slot];
        const File& f = files_[(size_t) j.file];
        bool ok = false;
        const uint64_t avail = f.size > j.off ? std::min<uint64_t>(j.len, f.size - j.off) : 0;
        if (f.map != nullptr && avail >= j.need && cached(f.map + j.off, avail)) {
            std::memcpy(j.dst, f.map + j.off, (size_t) avail);
            ok = true;
            ++st_.ch_cache;
            st_.b_cache += (int64_t) avail;
        } else if (cfg_.direct && f.dfd >= 0) {
            const int64_t t0 = now_us();
            R.since.store(t0, std::memory_order_relaxed);
            ok = read_all(f.dfd, j.dst, j.len, j.off, j.need);
            R.since.store(0, std::memory_order_relaxed);
            const int64_t dt = now_us() - t0;
            if (dt > 50000) {
                ++st_.slow;
                st_.slow_us += dt;
            }
            if (ok) {
                ++st_.ch_direct;
                st_.b_disk += (int64_t) j.len;
            }
        }
        if (!ok && f.bfd >= 0) {
            const int64_t t0 = now_us();
            R.since.store(t0, std::memory_order_relaxed);
            ok = read_all(f.bfd, j.dst, avail, j.off, j.need);
            R.since.store(0, std::memory_order_relaxed);
            const int64_t dt = now_us() - t0;
            if (dt > 50000) {
                ++st_.slow;
                st_.slow_us += dt;
            }
            if (ok) {
                ++st_.ch_buf;
                st_.b_disk += (int64_t) avail;
            }
        }
        if (!ok) {
            ++st_.ch_fail;
            sp.bad.store(1, std::memory_order_relaxed);
            sp.part_bad[j.part].store(1, std::memory_order_relaxed);
        }
        // the part's last chunk: its flag (the GPU computes the gate / up while the down is still being read)
        if (sp.part_left[j.part].fetch_sub(1, std::memory_order_acq_rel) == 1) {
            const bool pb = sp.part_bad[j.part].load(std::memory_order_relaxed) != 0;
            std::atomic_thread_fence(std::memory_order_release);
            flags_[2 * j.slot + j.part] = pb ? (j.gen | kBad) : j.gen;
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        if (sp.left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            const bool bad = sp.bad.load(std::memory_order_relaxed) != 0;
            const int64_t t1 = now_us();
            if (stats_on_ && j.layer >= 0) {
                LStats& ls = lst(j.layer);
                const int64_t t0 = sp.t_start.load(std::memory_order_relaxed);
                ++ls.qwait_n;
                ls.qwait_us += t0 - j.t_fill;
                ++ls.rd_n;
                ls.rd_us += t1 - t0;
            }
            sp.state.store(bad ? kBadFill : kReady, std::memory_order_release);
            const int64_t a = sp.asked.exchange(0);
            if (a > 0) {
                ++st_.waited;
                st_.wait_us += t1 - a;
                if (j.layer >= 0) {
                    ++lst(j.layer).waited;
                    lst(j.layer).wait_us += t1 - a;
                }
            }
        }
    }
    static void pin(int cpu) {
#if defined(__linux__)
        if (cpu < 0) return;
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof set, &set);
#else
        (void) cpu;
#endif
    }
    void kicker(int t) {
        Reader& R = rd_[(size_t) t];
        pin(R.cpu);
        void* buf = nullptr;
        if (posix_memalign(&buf, 4096, 4096) != 0) return;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(R.mu);
                R.cv.wait(lk, [&] { return R.go || kquit_.load(); });
                if (kquit_.load()) break;
                R.go = false;
            }
#if defined(__linux__)
            if (::pread(cfg_.kick_fd, buf, 4096, 0) < 0) {
            }
#endif
        }
        std::free(buf);
    }
    void watch(int nt) {
        const int64_t lim = (int64_t) (cfg_.kick_ms * 1000.0);
        while (!kquit_.load()) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            const int64_t now = now_us();
            for (int t = 0; t < nt; ++t) {
                Reader& R = rd_[(size_t) t];
                const int64_t s0 = R.since.load(std::memory_order_relaxed);
                if (s0 == 0 || now - s0 < lim || now - R.kicked < lim) continue;
                R.kicked = now;
                ++st_.kicks;
                std::lock_guard<std::mutex> lk(R.mu);
                R.go = true;
                R.cv.notify_one();
            }
        }
    }
    void loop(int t) {
        Reader& R = rd_[(size_t) t];
        pin(R.cpu);
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_ || !q_[0].empty() || !q_[1].empty() || !q_[2].empty(); });
                if (quit_) return;
                const int q = !q_[0].empty() ? 0 : !q_[1].empty() ? 1 : 2;
                j = q_[q].front();
                q_[q].pop_front();
                // (under the lock: cancel_before drops only fills none of whose reads a reader holds)
                Slot& sp = slots_[(size_t) j.slot];
                if (sp.started.fetch_add(1, std::memory_order_relaxed) == 0 && sp.gen == j.gen)
                    sp.t_start.store(now_us(), std::memory_order_relaxed);
            }
            run(j, R);
        }
    }

    Cfg cfg_;
    std::vector<Geo> geo_;
    std::vector<File> files_;
    uint8_t* slab_ = nullptr;
    uint64_t stride_ = 0;
    volatile unsigned* flags_ = nullptr;
    std::unique_ptr<Slot[]> slots_;
    std::vector<std::vector<int>> where_;   // [layer][expert] -> slot (-1)
    unsigned seen_ = 0, gen_ = 0;
    uint64_t tick_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> q_[3];   // the reads: demand, near guesses, far guesses
    bool quit_ = false;
    std::vector<std::thread> th_, kth_;
    std::unique_ptr<Reader[]> rd_;
    std::atomic<bool> kquit_{false};
    Stats st_;
    LStats lst_[kStatL];
    bool stats_on_ = true;
};

}  // namespace strata::ds4
