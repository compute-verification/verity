// verity flock-circuit: one device copy per GPU of a circuit's slot types' CSC (FcTypeUploads' set 0, prove_circuit.cuh), mapped
// into every prover process on that GPU that asks for the same arrays. The first asker allocates it with CUDA's virtual memory
// management (cuMemCreate, exportable as a POSIX file descriptor), copies the arrays in, and hands the descriptor to each later
// asker over a Unix socket in the abstract namespace, named by the GPU's UUID and the arrays' key. A later asker maps the copy
// and never writes it. The memory lives while any process maps it, so the first prover may exit before the others. The key is
// a digest of the arrays' contents (gpu_circuit.rs `types`), so a copy holds exactly the asker's arrays. A process that can't
// take part (no VMM, another network namespace, another user, a failed copy) keeps a copy of its own.
#pragma once
#include <cuda.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <thread>
#include <type_traits>
#include <vector>

extern "C" int flock_cuda_copy(void* dst, const void* src, size_t bytes);

struct FcVmm {
    CUresult (*granularity)(size_t*, const CUmemAllocationProp*, CUmemAllocationGranularity_flags);
    CUresult (*create)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
    CUresult (*release)(CUmemGenericAllocationHandle);
    CUresult (*reserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long);
    CUresult (*addr_free)(CUdeviceptr, size_t);
    CUresult (*map)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
    CUresult (*unmap)(CUdeviceptr, size_t);
    CUresult (*access)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
    CUresult (*export_fd)(void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);
    CUresult (*import_fd)(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType);
    bool ok;
};

// the driver's VMM calls through the runtime, which links no libcuda
static const FcVmm& fc_vmm() {
    static const FcVmm v = [] {
        FcVmm v{};
        auto get = [](const char* name, auto& f) {
            void* p = nullptr;
            cudaDriverEntryPointQueryResult q;
            if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess || q != cudaDriverEntryPointSuccess) {
                cudaGetLastError();
                p = nullptr;
            }
            f = reinterpret_cast<std::remove_reference_t<decltype(f)>>(p);
            return p != nullptr;
        };
        v.ok = get("cuMemGetAllocationGranularity", v.granularity) & get("cuMemCreate", v.create) & get("cuMemRelease", v.release) &
               get("cuMemAddressReserve", v.reserve) & get("cuMemAddressFree", v.addr_free) & get("cuMemMap", v.map) &
               get("cuMemUnmap", v.unmap) & get("cuMemSetAccess", v.access) &
               get("cuMemExportToShareableHandle", v.export_fd) & get("cuMemImportFromShareableHandle", v.import_fd);
        return v;
    }();
    return v;
}

struct FcShareMsg {
    char magic[8];
    uint8_t key[64];
    unsigned long long bytes;
};

static socklen_t fc_share_name(sockaddr_un* a, const unsigned char uuid[16], const uint8_t key[64]) {
    memset(a, 0, sizeof *a);
    a->sun_family = AF_UNIX;
    char* p = a->sun_path + 1;  // the abstract namespace: a leading NUL, and no file to remove
    p += sprintf(p, "verity-flock-csc-");
    for (int i = 0; i < 16; i++) p += sprintf(p, "%02x", uuid[i]);
    *p++ = '-';
    for (int i = 0; i < 24; i++) p += sprintf(p, "%02x", key[i]);
    return (socklen_t)(offsetof(sockaddr_un, sun_path) + (p - a->sun_path));
}

static bool fc_same_user(int c) {
    ucred u{};
    socklen_t n = sizeof u;
    return getsockopt(c, SOL_SOCKET, SO_PEERCRED, &u, &n) == 0 && u.uid == getuid();
}

// `bytes` of h mapped at a fresh address for device dev, read-only unless `write` (read-write where the driver has no
// read-only device mapping); 0 when it can't be
static CUdeviceptr fc_vmm_map(const FcVmm& V, CUmemGenericAllocationHandle h, size_t bytes, size_t align, int dev, bool write,
                              bool* ro) {
    CUdeviceptr va = 0;
    if (V.reserve(&va, bytes, align, 0, 0) != CUDA_SUCCESS) return 0;
    if (V.map(va, bytes, 0, h, 0) != CUDA_SUCCESS) { V.addr_free(va, bytes); return 0; }
    CUmemAccessDesc a{};
    a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    a.location.id = dev;
    a.flags = write ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE : CU_MEM_ACCESS_FLAGS_PROT_READ;
    bool ok = V.access(va, bytes, &a, 1) == CUDA_SUCCESS;
    *ro = ok && !write;
    if (!ok && !write) {
        a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        ok = V.access(va, bytes, &a, 1) == CUDA_SUCCESS;
    }
    if (!ok) { V.unmap(va, bytes); V.addr_free(va, bytes); return 0; }
    return va;
}

// The n arrays h[i] of len[i] words on this GPU, shared under `key` (above): their device addresses in d, and 0; nonzero when
// they aren't shared, and the asker copies them itself
static int fc_share_get(const uint8_t key[64], const uint32_t* const* h, const size_t* len, size_t n, uint32_t** d) {
    const FcVmm& V = fc_vmm();
    int dev = 0;
    cudaDeviceProp dp;
    if (!V.ok) { fprintf(stderr, "FFI circuit: the slot types' CSC not shared (no VMM entry points); a copy of its own\n"); return 1; }
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&dp, dev) != cudaSuccess) { cudaGetLastError(); return 1; }
    std::vector<size_t> off(n);
    size_t used = 0;
    for (size_t i = 0; i < n; i++) { off[i] = used; used += (len[i] * sizeof(uint32_t) + 255) & ~(size_t)255; }
    CUmemAllocationProp p{};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = dev;
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t g = 0;
    if (V.granularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || !g) {
        fprintf(stderr, "FFI circuit: the slot types' CSC not shared (no shareable device memory); a copy of its own\n");
        return 1;
    }
    const size_t bytes = std::max<size_t>((used + g - 1) / g * g, g);
    sockaddr_un a;
    const socklen_t al = fc_share_name(&a, (const unsigned char*)dp.uuid.bytes, key);
    const char* why = "the socket";
    // the first asker binds the name and serves the copy it makes; a later one finds it bound and is served once that copy is
    // made. An asker served nothing (the first one failed and closed) tries again, and may make the copy itself.
    for (int attempt = 0; attempt < 3; attempt++) {
        const int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (s < 0) break;
        if (bind(s, (sockaddr*)&a, al) == 0) {
            CUmemGenericAllocationHandle hd;
            if (listen(s, 256) != 0 || V.create(&hd, bytes, &p, 0) != CUDA_SUCCESS) {
                close(s);
                why = "no room on the device";
                break;
            }
            bool ro = false;
            const CUdeviceptr va = fc_vmm_map(V, hd, bytes, g, dev, true, &ro);
            bool ok = va != 0;
            for (size_t i = 0; ok && i < n; i++) ok = !len[i] || !flock_cuda_copy((char*)va + off[i], h[i], len[i] * sizeof(uint32_t));
            int fd = -1;
            ok = ok && V.export_fd(&fd, hd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) == CUDA_SUCCESS;
            V.release(hd);  // the mapping and the descriptor hold the memory
            if (!ok) {
                if (va) { V.unmap(va, bytes); V.addr_free(va, bytes); }
                if (fd >= 0) close(fd);
                close(s);
                why = "its copy failed";
                break;
            }
            FcShareMsg m{};
            memcpy(m.magic, "fcshare1", 8);
            memcpy(m.key, key, 64);
            m.bytes = bytes;
            std::thread([s, fd, m] {
                for (;;) {
                    const int c = accept4(s, nullptr, nullptr, SOCK_CLOEXEC);
                    if (c < 0) {
                        if (errno == EINTR || errno == ECONNABORTED) continue;
                        // a later asker then finds the name free and keeps a copy of its own
                        close(s);
                        close(fd);
                        return;
                    }
                    if (fc_same_user(c)) {
                        iovec io{(void*)&m, sizeof m};
                        alignas(cmsghdr) char buf[CMSG_SPACE(sizeof(int))] = {};
                        msghdr mh{};
                        mh.msg_iov = &io; mh.msg_iovlen = 1; mh.msg_control = buf; mh.msg_controllen = sizeof buf;
                        cmsghdr* cm = CMSG_FIRSTHDR(&mh);
                        cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS; cm->cmsg_len = CMSG_LEN(sizeof(int));
                        memcpy(CMSG_DATA(cm), &fd, sizeof fd);
                        (void)sendmsg(c, &mh, MSG_NOSIGNAL);
                    }
                    close(c);
                }
            }).detach();
            for (size_t i = 0; i < n; i++) d[i] = (uint32_t*)((char*)va + off[i]);
            fprintf(stderr, "FFI circuit: the slot types' CSC (%zu MiB) made on the device, shared with this GPU's other provers\n",
                    bytes >> 20);
            return 0;
        }
        if (errno != EADDRINUSE) { close(s); break; }
        timeval tv{600, 0};  // the first asker's copy of the arrays takes seconds
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        if (connect(s, (sockaddr*)&a, al) != 0) {
            const int e = errno;
            close(s);
            if (e == ECONNREFUSED) continue;
            break;
        }
        if (!fc_same_user(s)) { close(s); why = "another user's"; break; }
        FcShareMsg m{};
        iovec io{&m, sizeof m};
        alignas(cmsghdr) char buf[CMSG_SPACE(sizeof(int))] = {};
        msghdr mh{};
        mh.msg_iov = &io; mh.msg_iovlen = 1; mh.msg_control = buf; mh.msg_controllen = sizeof buf;
        const ssize_t r = recvmsg(s, &mh, MSG_CMSG_CLOEXEC | MSG_WAITALL);
        close(s);
        int fd = -1;
        for (cmsghdr* cm = CMSG_FIRSTHDR(&mh); r > 0 && cm; cm = CMSG_NXTHDR(&mh, cm))
            if (cm->cmsg_level == SOL_SOCKET && cm->cmsg_type == SCM_RIGHTS) memcpy(&fd, CMSG_DATA(cm), sizeof fd);
        if (r != (ssize_t)sizeof m || fd < 0) {
            if (fd >= 0) close(fd);
            why = r < 0 ? "no answer" : "served nothing";
            if (r < 0) break;
            continue;
        }
        if (memcmp(m.magic, "fcshare1", 8) != 0 || memcmp(m.key, key, 64) != 0 || m.bytes != bytes) {
            close(fd);
            why = "another key's";
            break;
        }
        CUmemGenericAllocationHandle hd;
        const bool imported = V.import_fd(&hd, (void*)(uintptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS;
        close(fd);
        if (!imported) { why = "its import failed"; break; }
        bool ro = false;
        const CUdeviceptr va = fc_vmm_map(V, hd, bytes, g, dev, false, &ro);
        V.release(hd);
        if (!va) { why = "its mapping failed"; break; }
        for (size_t i = 0; i < n; i++) d[i] = (uint32_t*)((char*)va + off[i]);
        fprintf(stderr, "FFI circuit: the slot types' CSC (%zu MiB) mapped %s from another prover's copy on the device\n",
                bytes >> 20, ro ? "read-only" : "read-write");
        return 0;
    }
    cudaGetLastError();
    fprintf(stderr, "FFI circuit: the slot types' CSC not shared (%s); a copy of its own\n", why);
    return 1;
}
