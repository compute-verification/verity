// Host harness for sha_witness.cuh: reads (h_in[8], m[16]) u32 records from stdin, writes each compression's z, a, b
// regions (3 x 1,024 u32) to stdout. `cargo test sha_witness_cuh` in flock-live runs it when PSH_HARNESS names the binary.
#include <cstdio>
#include <cstring>
#include "sha_witness.cuh"

int main() {
    uint32_t in[24];
    static uint32_t out[3][1024];
    while (fread(in, 4, 24, stdin) == 24) {
        memset(out, 0, sizeof out);
        psh_witness(in, in + 8, out[0], out[1], out[2]);
        fwrite(out, 4, 3 * 1024, stdout);
    }
    return 0;
}
