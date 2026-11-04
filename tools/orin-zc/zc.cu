// Orin zero-copy probes for the GPU lane (r4-gpu-lane.md): what the iGPU can read in place, what NVMe can DMA into.
// build: nvcc -O2 -arch=sm_87 -o zc zc.cu -lpthread ; run: ./zc <big model file>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("  %s -> %s\n", #x, cudaGetErrorString(e_)); } } while (0)

__global__ void rd(const int4* __restrict__ p, size_t n, int* out) {
    int acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        int4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) *out = acc;
}
__global__ void rd_sum(const unsigned* __restrict__ p, size_t n, unsigned long long* out) {
    unsigned long long acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) acc += p[i];
    atomicAdd(out, acc);
}
static float gpu_bw(const void* p, size_t bytes, int* o, int reps = 10) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    rd<<<128, 512>>>((const int4*) p, bytes / 16, o);
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r) rd<<<128, 512>>>((const int4*) p, bytes / 16, o);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("  kernel: %s\n", cudaGetErrorString(e)); return -1; }
    return (float) (reps * (double) bytes / (ms / 1e3) / 1e9);
}
static unsigned long long gpu_sum(const void* p, size_t bytes) {
    unsigned long long* d;
    cudaMalloc(&d, 8);
    cudaMemset(d, 0, 8);
    rd_sum<<<128, 512>>>((const unsigned*) p, bytes / 4, d);
    unsigned long long h = 0;
    cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost);
    cudaFree(d);
    return h;
}
static unsigned long long cpu_sum(const void* p, size_t bytes) {
    unsigned long long a = 0;
    const unsigned* q = (const unsigned*) p;
    for (size_t i = 0; i < bytes / 4; ++i) a += q[i];
    return a;
}

// parallel pread of `n` regions of `len` bytes at offs[] into dst + i*len, in chunks of `chunk`, with T threads
static double par_read(int fd, const std::vector<off_t>& offs, size_t len, uint8_t* dst, size_t chunk, int T, bool* ok) {
    std::atomic<size_t> next{0};
    const size_t per = (len + chunk - 1) / chunk, total = offs.size() * per;
    std::atomic<bool> good{true};
    const double t0 = now_ms();
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t)
        th.emplace_back([&] {
            for (;;) {
                const size_t j = next.fetch_add(1);
                if (j >= total) return;
                const size_t r = j / per, c = j % per;
                const size_t o = c * chunk, b = std::min(chunk, len - o);
                ssize_t got = pread(fd, dst + r * len + o, b, offs[r] + (off_t) o);
                if (got != (ssize_t) b) good = false;
            }
        });
    for (auto& x : th) x.join();
    if (ok) *ok = good;
    return now_ms() - t0;
}

int main(int argc, char** argv) {
    const char* file = argc > 1 ? argv[1] : nullptr;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("device %s sm_%d%d integrated %d canMapHostMemory %d\n", prop.name, prop.major, prop.minor, prop.integrated,
           prop.canMapHostMemory);
    int v;
    const struct { cudaDeviceAttr a; const char* n; } attrs[] = {
        {cudaDevAttrPageableMemoryAccess, "pageableMemoryAccess"},
        {cudaDevAttrPageableMemoryAccessUsesHostPageTables, "pageableMemoryAccessUsesHostPageTables"},
        {cudaDevAttrHostRegisterSupported, "hostRegisterSupported"},
        {cudaDevAttrHostRegisterReadOnlySupported, "hostRegisterReadOnlySupported"},
        {cudaDevAttrCanUseHostPointerForRegisteredMem, "canUseHostPointerForRegisteredMem"},
        {cudaDevAttrConcurrentManagedAccess, "concurrentManagedAccess"},
        {cudaDevAttrDirectManagedMemAccessFromHost, "directManagedMemAccessFromHost"},
        {cudaDevAttrUnifiedAddressing, "unifiedAddressing"},
    };
    for (auto& a : attrs) {
        v = -1;
        cudaDeviceGetAttribute(&v, a.a, 0);
        printf("  %s = %d\n", a.n, v);
    }
    int* o;
    cudaMalloc(&o, 4);
    const size_t B = 64ull << 20;

    printf("\n[1] GPU read bandwidth, 64 MB, 128x512 int4 loads\n");
    {
        void* d;
        CK(cudaMalloc(&d, B));
        cudaMemset(d, 1, B);
        printf("  cudaMalloc            %.1f GB/s\n", gpu_bw(d, B, o));
        cudaFree(d);
    }
    {
        void* h;
        CK(cudaHostAlloc(&h, B, cudaHostAllocMapped));
        memset(h, 1, B);
        void* dp = nullptr;
        CK(cudaHostGetDevicePointer(&dp, h, 0));
        printf("  cudaHostAlloc mapped  %.1f GB/s   (dev ptr %s host ptr)\n", gpu_bw(dp, B, o), dp == h ? "==" : "!=");
        memset(h, 2, B);   // CPU writes after GPU read: GPU sees them (I/O coherence)?
        printf("  coherence: gpu sum %llu cpu sum %llu\n", gpu_sum(dp, B), cpu_sum(h, B));
        cudaFreeHost(h);
    }
    {
        void* h = nullptr;
        if (posix_memalign(&h, 4096, B) == 0) {
            memset(h, 1, B);
            const double t0 = now_ms();
            cudaError_t e = cudaHostRegister(h, B, cudaHostRegisterMapped);
            const double t1 = now_ms();
            if (e == cudaSuccess) {
                void* dp = nullptr;
                CK(cudaHostGetDevicePointer(&dp, h, 0));
                printf("  malloc+HostRegister   %.1f GB/s   (register %.2f ms for 64 MB; dev ptr %s host ptr)\n",
                       gpu_bw(dp, B, o), t1 - t0, dp == h ? "==" : "!=");
                const double t2 = now_ms();
                cudaHostUnregister(h);
                printf("  unregister %.2f ms\n", now_ms() - t2);
                // small register (one expert, 7.4 MB) cost
                const size_t E = 7489536 & ~4095ull;
                double tr = 0, tu = 0;
                for (int k = 0; k < 8; ++k) {
                    const double a = now_ms();
                    CK(cudaHostRegister((uint8_t*) h + (k % 4) * (8 << 20), E, cudaHostRegisterMapped));
                    const double b = now_ms();
                    cudaHostUnregister((uint8_t*) h + (k % 4) * (8 << 20));
                    tr += b - a;
                    tu += now_ms() - b;
                }
                printf("  register/unregister of a 7.4 MB expert (anon, resident): %.3f / %.3f ms\n", tr / 8, tu / 8);
            } else
                printf("  malloc+HostRegister: %s\n", cudaGetErrorString(e));
            free(h);
        }
    }
    {
        void* m;
        CK(cudaMallocManaged(&m, B));
        memset(m, 1, B);
        printf("  cudaMallocManaged     %.1f GB/s\n", gpu_bw(m, B, o));
        cudaFree(m);
    }
    if (file == nullptr) return 0;

    // expert-sized regions of the model file
    const size_t E = 7489536;   // an IQ1_M/IQ1_M/IQ1_S expert's blob
    const size_t EA = (E + 4095) & ~4095ull;
    int fdb = open(file, O_RDONLY);
    int fdd = open(file, O_RDONLY | O_DIRECT);
    off_t fsz = lseek(fdb, 0, SEEK_END);
    std::mt19937_64 rng(12345);
    auto regions = [&](int n) {
        std::vector<off_t> r;
        for (int i = 0; i < n; ++i) r.push_back((off_t) ((rng() % (uint64_t) (fsz - (off_t) EA - 4096)) & ~4095ull));
        return r;
    };

    printf("\n[2] O_DIRECT pread into cudaHostAlloc'd memory (cold reads, file %s)\n", file);
    {
        const int N = 16;
        uint8_t* h;
        CK(cudaHostAlloc((void**) &h, N * EA, cudaHostAllocMapped));
        for (size_t chunk : {(size_t) 256 << 10, (size_t) 1 << 20, EA})
            for (int T : {1, 2, 4, 8}) {
                auto offs = regions(N);
                for (auto off : offs) posix_fadvise(fdb, off, EA, POSIX_FADV_DONTNEED);
                bool ok = false;
                const double ms = par_read(fdd, offs, EA, h, chunk, T, &ok);
                printf("  chunk %5zu KB  %d threads: %s  %.2f ms for %d experts = %.2f ms/expert, %.2f GB/s\n", chunk >> 10, T,
                       ok ? "ok" : "FAILED", ms, N, ms / N, N * EA / (ms / 1e3) / 1e9);
            }
        // correctness: O_DIRECT into pinned == buffered read, GPU sees it
        auto offs = regions(1);
        bool ok;
        par_read(fdd, offs, EA, h, 1 << 20, 4, &ok);
        std::vector<uint8_t> ref(EA);
        pread(fdb, ref.data(), EA, offs[0]);
        void* dp;
        cudaHostGetDevicePointer(&dp, h, 0);
        printf("  O_DIRECT bytes == buffered: %s; gpu sum %llu cpu sum %llu\n", memcmp(ref.data(), h, EA) == 0 ? "yes" : "NO",
               gpu_sum(dp, EA & ~15ull), cpu_sum(ref.data(), EA & ~15ull));
        printf("  GPU reads the pinned slots in place: %.1f GB/s\n", gpu_bw(dp, N * EA & ~15ull, o));
        cudaFreeHost(h);
    }

    printf("\n[3] buffered pread (page cache) into pinned memory\n");
    {
        const int N = 16;
        uint8_t* h;
        CK(cudaHostAlloc((void**) &h, N * EA, cudaHostAllocMapped));
        for (int T : {1, 2, 4, 8}) {
            auto offs = regions(N);
            for (auto off : offs) posix_fadvise(fdb, off, EA, POSIX_FADV_DONTNEED);
            bool ok;
            const double cold = par_read(fdb, offs, EA, h, 1 << 20, T, &ok);
            const double warm = par_read(fdb, offs, EA, h, 1 << 20, T, &ok);
            // WILLNEED all first, then read (what lookahead does)
            auto offs2 = regions(N);
            for (auto off : offs2) posix_fadvise(fdb, off, EA, POSIX_FADV_DONTNEED);
            const double a = now_ms();
            for (auto off : offs2) posix_fadvise(fdb, off, EA, POSIX_FADV_WILLNEED);
            const double adv = now_ms() - a;
            const double after = par_read(fdb, offs2, EA, h, 1 << 20, T, &ok);
            printf("  %d threads: cold %.2f ms/expert (%.2f GB/s)  page-cache hit %.3f ms/expert (%.1f GB/s copy)  "
                   "fadvise-WILLNEED %.2f ms then read %.2f ms/expert\n",
                   T, cold / N, N * EA / (cold / 1e3) / 1e9, warm / N, N * EA / (warm / 1e3) / 1e9, adv, after / N);
        }
        cudaFreeHost(h);
    }

    printf("\n[4] cudaHostRegister on an mmap'd file range\n");
    {
        const size_t MAPB = 1ull << 30;
        off_t base = (off_t) ((rng() % (uint64_t) (fsz - (off_t) MAPB)) & ~((1ull << 21) - 1));
        uint8_t* m = (uint8_t*) mmap(nullptr, MAPB, PROT_READ, MAP_SHARED, fdb, base);
        if (m == MAP_FAILED) { printf("  mmap failed\n"); return 0; }
        for (unsigned fl : {(unsigned) (cudaHostRegisterMapped | cudaHostRegisterReadOnly), (unsigned) cudaHostRegisterMapped,
                            (unsigned) (cudaHostRegisterMapped | cudaHostRegisterReadOnly | cudaHostRegisterIoMemory)}) {
            for (int warm = 1; warm >= 0; --warm) {
                uint8_t* p = m + (warm ? 0 : 256 << 20) + (fl & cudaHostRegisterIoMemory ? 512 << 20 : 0) +
                             (fl & cudaHostRegisterReadOnly ? 0 : 128 << 20);
                if (warm) {   // fault it into the page cache + this mapping first
                    volatile unsigned long long s = cpu_sum(p, EA);
                    (void) s;
                }
                const double a = now_ms();
                cudaError_t e = cudaHostRegister(p, EA, fl);
                const double b = now_ms();
                if (e != cudaSuccess) {
                    printf("  flags 0x%x %s: register -> %s\n", fl, warm ? "warm" : "cold", cudaGetErrorString(e));
                    cudaGetLastError();
                    continue;
                }
                void* dp = nullptr;
                CK(cudaHostGetDevicePointer(&dp, p, 0));
                const unsigned long long gs = gpu_sum(dp, EA & ~15ull), cs = cpu_sum(p, EA & ~15ull);
                const float bw = gpu_bw(dp, EA & ~15ull, o, 20);
                const double c = now_ms();
                cudaHostUnregister(p);
                const double d = now_ms();
                printf("  flags 0x%x %s: register %.3f ms, unregister %.3f ms, GPU %.1f GB/s, sums %s, dev ptr %s host ptr\n",
                       fl, warm ? "warm" : "cold", b - a, d - c, bw, gs == cs ? "match" : "DIFFER", dp == p ? "==" : "!=");
            }
        }
        munmap(m, MAPB);
    }

    printf("\n[5] CPU memcpy from mmap'd page cache into pinned memory\n");
    {
        const int N = 16;
        uint8_t* h;
        CK(cudaHostAlloc((void**) &h, N * EA, cudaHostAllocMapped));
        auto offs = regions(N);
        std::vector<uint8_t*> maps;
        for (auto off : offs) {
            uint8_t* m = (uint8_t*) mmap(nullptr, EA, PROT_READ, MAP_SHARED, fdb, off);
            volatile unsigned long long s = cpu_sum(m, EA);
            (void) s;
            maps.push_back(m);
        }
        for (int T : {1, 2, 4, 8, 11}) {
            std::atomic<int> next{0};
            const size_t chunk = 1 << 20, per = (EA + chunk - 1) / chunk;
            const double t0 = now_ms();
            std::vector<std::thread> th;
            for (int t = 0; t < T; ++t)
                th.emplace_back([&] {
                    for (;;) {
                        const int j = next.fetch_add(1);
                        if (j >= (int) (N * per)) return;
                        const size_t r = j / per, c = j % per, off = c * chunk, b = std::min(chunk, EA - off);
                        memcpy(h + r * EA + off, maps[r] + off, b);
                    }
                });
            for (auto& x : th) x.join();
            const double ms = now_ms() - t0;
            printf("  %2d threads: %.3f ms/expert, %.1f GB/s\n", T, ms / N, N * EA / (ms / 1e3) / 1e9);
        }
        for (auto m : maps) munmap(m, EA);
        cudaFreeHost(h);
    }
    return 0;
}
