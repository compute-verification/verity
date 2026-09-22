import ctypes, sys, time
t0=time.time()
try:
    cu = ctypes.CDLL("libcuda.so.1")
except OSError as e:
    print("CUINIT FAIL: libcuda.so.1 not loadable:", e); sys.exit(2)
r = cu.cuInit(0); print("cuInit ->", r)
if r != 0: print("CUINIT FAIL rc=%d (999=CUDA_ERROR_UNKNOWN: dead GPU; nvidia-smi may still work)" % r); sys.exit(1)
n = ctypes.c_int(); r = cu.cuDeviceGetCount(ctypes.byref(n)); print("cuDeviceGetCount ->", r, n.value)
dev = ctypes.c_int(); cu.cuDeviceGet(ctypes.byref(dev), 0)
ctx = ctypes.c_void_p(); r = cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev); print("cuCtxCreate ->", r)
if r != 0: print("CUINIT FAIL cuCtxCreate rc=%d" % r); sys.exit(1)
N = 1<<24
dptr = ctypes.c_void_p(); r = cu.cuMemAlloc_v2(ctypes.byref(dptr), N); print("cuMemAlloc 16MiB ->", r)
buf = (ctypes.c_ubyte * N)(); 
for i in range(0, N, 4096): buf[i] = (i//4096) & 0xff
r1 = cu.cuMemcpyHtoD_v2(dptr, buf, N); out = (ctypes.c_ubyte * N)(); r2 = cu.cuMemcpyDtoH_v2(out, dptr, N)
r3 = cu.cuMemsetD8_v2(dptr, 0xAB, N); chk = (ctypes.c_ubyte * 16)(); r4 = cu.cuMemcpyDtoH_v2(chk, dptr, 16)
ok = all(out[i] == ((i//4096) & 0xff) for i in range(0, N, 4096)) and all(c == 0xAB for c in chk)
print("HtoD/DtoH/memset rcs:", r1, r2, r3, r4, "roundtrip_ok:", ok, "t=%.2fs" % (time.time()-t0))
print("CUINIT-OK" if (ok and r1==r2==r3==r4==0) else "CUINIT FAIL: memcpy roundtrip mismatch"); sys.exit(0 if ok else 1)
