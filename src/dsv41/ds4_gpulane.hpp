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
// given up.  DS4_GL=1 turns the lane on (ds4_model.cu).
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
        std::atomic<int64_t> pf{0}, pf_used{0}, pf_dropped{0};
        std::atomic<int64_t> ch_cache{0}, ch_direct{0}, ch_buf{0}, ch_fail{0};
        std::atomic<int64_t> b_cache{0}, b_disk{0};
        std::atomic<int64_t> waited{0}, wait_us{0};   // picks not ready when asked, their wait from the ask to the fill's end
        std::atomic<int64_t> kicks{0}, slow{0}, slow_us{0};   // kicks sent; reads that took > 50 ms, their time
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
        for (int i = 0; i < c.slots; ++i) flags_[i] = 0u;
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
    ~GpuLane() {
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

    /// The GPU has passed every request before seq (the service thread saw request seq published: the device's stream
    /// ran everything enqueued before it)
    void seen(unsigned seq) { seen_ = seq; }

    /// (layer, e) for request seq: its slot (a fill started now if it holds nothing, at the front of the queue), pinned for
    /// the request; *gen the flag value the GPU waits for.  -1: no slot free (the CPU lane computes it)
    int demand(int layer, int e, unsigned seq, unsigned* gen) {
        ++st_.demand;
        int s = where_[(size_t) layer][(size_t) e];
        if (s >= 0) {
            Slot& sp = slots_[(size_t) s];
            const int state = sp.state.load(std::memory_order_acquire);
            if (state == kBadFill) {   // (a failed fill: once more)
                drop(s);
                s = -1;
            } else {
                if (state == kReady) ++st_.ready;
                else {
                    ++st_.filling;
                    int64_t z = 0;
                    sp.asked.compare_exchange_strong(z, now_us());
                    promote(s);
                }
                if (sp.pf && !sp.used) ++st_.pf_used;
                sp.used = true;
                sp.pin = seq;
                sp.touch = ++tick_;
                *gen = sp.gen;
                return s;
            }
        }
        ++st_.missed;
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
        fill(s, layer, e, true);
        *gen = sp.gen;
        return s;
    }
    /// (layer, e) read ahead (a guess): into a free slot at the back of the queue, unless held or no slot is free
    void prefetch(int layer, int e) {
        if (where_[(size_t) layer][(size_t) e] >= 0) return;
        const int s = take();
        if (s < 0) return;
        Slot& sp = slots_[(size_t) s];
        sp.used = false;
        sp.pf = true;
        sp.pin = 0;
        ++st_.pf;
        fill(s, layer, e, false);
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
        fill(s, layer, e, true);
        while (slots_[(size_t) s].state.load(std::memory_order_acquire) == kFilling) std::this_thread::yield();
        return slots_[(size_t) s].state.load() == kReady && flags_[s] == slots_[(size_t) s].gen ? s : -1;
    }
    /// Guesses for layers before `layer` (the token has passed them) whose reads have not started: dropped
    void cancel_before(int layer) {
        std::vector<int> gone;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto it = lo_.begin(); it != lo_.end();) {
                Slot& sp = slots_[(size_t) it->slot];
                if (sp.layer < layer && !sp.used && sp.started.load(std::memory_order_relaxed) == 0 && sp.gen == it->gen) {
                    if (std::find(gone.begin(), gone.end(), it->slot) == gone.end()) gone.push_back(it->slot);
                    it = lo_.erase(it);
                } else
                    ++it;
            }
        }
        for (int s : gone) {
            Slot& sp = slots_[(size_t) s];
            ++st_.pf_dropped;
            sp.left.store(0);
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
        std::atomic<int64_t> asked{0};   // (us) when a request asked for it before its fill was done (0: not)
        unsigned pin = 0;          // the last request that uses it (the GPU may read it until a later one is seen)
        uint64_t touch = 0;
        bool pf = false, used = false;
    };
    struct Job {
        int slot = 0;
        unsigned gen = 0;
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
        if (best >= 0) drop(best);
        return best;
    }
    static int64_t now_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void fill(int s, int layer, int e, bool front) {
        Slot& sp = slots_[(size_t) s];
        if (!front) sp.asked.store(0);
        const Geo& g = geo_[(size_t) layer];
        sp.layer = layer;
        sp.e = e;
        gen_ = (gen_ + 1) & ~kBad;
        if (gen_ == 0) gen_ = 1;
        sp.gen = gen_;
        sp.touch = ++tick_;
        where_[(size_t) layer][(size_t) e] = s;
        std::vector<Job> js;
        for (int r = 0; r < 3; ++r) {
            const uint64_t a0 = g.off[r] + (uint64_t) e * g.stride[r] - g.lead[r];
            for (uint64_t at = 0; at < g.rlen[r]; at += cfg_.chunk) {
                Job j;
                j.slot = s;
                j.gen = sp.gen;
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
        sp.left.store((int) js.size(), std::memory_order_relaxed);
        sp.state.store(kFilling, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (front) hi_.insert(hi_.end(), js.begin(), js.end());
            else lo_.insert(lo_.end(), js.begin(), js.end());
        }
        cv_.notify_all();
    }
    // a guess the token now needs: its queued reads move to the demand queue
    void promote(int s) {
        std::lock_guard<std::mutex> lk(mu_);
        const unsigned gen = slots_[(size_t) s].gen;
        for (auto it = lo_.begin(); it != lo_.end();) {
            if (it->slot == s && it->gen == gen) {
                hi_.push_back(*it);
                it = lo_.erase(it);
            } else
                ++it;
        }
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
        }
        if (sp.left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            const bool bad = sp.bad.load(std::memory_order_relaxed) != 0;
            std::atomic_thread_fence(std::memory_order_release);
            flags_[j.slot] = bad ? (j.gen | kBad) : j.gen;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            sp.state.store(bad ? kBadFill : kReady, std::memory_order_release);
            const int64_t a = sp.asked.exchange(0);
            if (a > 0) {
                ++st_.waited;
                st_.wait_us += now_us() - a;
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
                cv_.wait(lk, [&] { return quit_ || !hi_.empty() || !lo_.empty(); });
                if (quit_) return;
                if (!hi_.empty()) {
                    j = hi_.front();
                    hi_.pop_front();
                } else {
                    j = lo_.front();
                    lo_.pop_front();
                }
                // (under the lock: cancel_before drops only fills none of whose reads a reader holds)
                slots_[(size_t) j.slot].started.fetch_add(1, std::memory_order_relaxed);
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
    std::deque<Job> hi_, lo_;
    bool quit_ = false;
    std::vector<std::thread> th_, kth_;
    std::unique_ptr<Reader[]> rd_;
    std::atomic<bool> kquit_{false};
    Stats st_;
};

}  // namespace strata::ds4
