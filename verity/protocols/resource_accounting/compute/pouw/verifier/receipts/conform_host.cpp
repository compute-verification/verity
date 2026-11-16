// receipt_folds.cuh on the host, one operation a line on stdin (hex words), its results a line on stdout, for
// receipt_conform.check_header. Build: g++ -O2 -std=c++17 conform_host.cpp (or nvcc -x cu); -DRF_HEADER='"other.cuh"'
// holds another header that defines the same rf_ functions to the same vectors.
#include <cstdio>
#include <cstring>
#ifndef RF_HEADER
#define RF_HEADER "receipt_folds.cuh"
#endif
#include RF_HEADER

int main() {
  char op[16];
  unsigned v[12];
  while (std::scanf("%15s", op) == 1) {
    const int n = !std::strcmp(op, "li") ? 9 : !std::strcmp(op, "duo") ? 10 : !std::strcmp(op, "duow") ? 6
                : !std::strcmp(op, "final") || !std::strcmp(op, "served0") || !std::strcmp(op, "egemmk") ? 2
                : !std::strcmp(op, "egemm0") ? 3
                : !std::strcmp(op, "mix32") || !std::strcmp(op, "keys") || !std::strcmp(op, "served") || !std::strcmp(op, "pfill0") || !std::strcmp(op, "e2ek") ? 1
                : 7;
    for (int i = 0; i < n; i++)
      if (std::scanf("%x", &v[i]) != 1) return 2;
    if (!std::strcmp(op, "ram")) std::printf("%08x\n", rf_ram_step(v[0], v[1], v[2], v[3], v[4], v[5], v[6]));
    else if (!std::strcmp(op, "xin13")) std::printf("%08x\n", rf_xin13_step(v[0], v[1], v[2], v[3], v[4], v[5], v[6]));
    else if (!std::strcmp(op, "xin7")) std::printf("%08x\n", rf_xin7_step(v[0], v[1], v[2], v[3], v[4], v[5], v[6]));
    else if (!std::strcmp(op, "li")) std::printf("%08x\n", rf_li_step(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]));
    else if (!std::strcmp(op, "duo")) {
      uint32_t y = v[0], z = v[1];
      rf_duo_step(y, z, v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]);
      std::printf("%08x %08x\n", y, z);
    } else if (!std::strcmp(op, "duow")) {
      uint32_t y = v[0], z = v[1];
      rf_duo_word(y, z, v[2], v[3], v[4], (int)v[5]);
      std::printf("%08x %08x\n", y, z);
    } else if (!std::strcmp(op, "final")) std::printf("%08x\n", rf_duo_final(v[0], v[1]));
    else if (!std::strcmp(op, "mix32")) std::printf("%08x\n", rf_mix32(v[0]));
    else if (!std::strcmp(op, "keys"))
      std::printf("%d %d %d %d %d\n", rf_key_ok(v[0]), rf_lambda8_ok(v[0]), rf_key_ok_r9(v[0]), rf_kj19(v[0]), rf_key_ok_r11(v[0]));
    else if (!std::strcmp(op, "served")) std::printf("%08x %08x %d\n", rf_served_ka((int)v[0]), rf_served_kk((int)v[0]), rf_served_rot((int)v[0]));
    else if (!std::strcmp(op, "served0")) {
      const uint32_t y = rf_served_y0(v[0], v[1]);
      std::printf("%08x %08x\n", y, rf_served_z0(y));
    } else if (!std::strcmp(op, "pfill0")) std::printf("%08x %08x\n", rf_pfill_y0(v[0]), rf_pfill_z0(v[0]));
    else if (!std::strcmp(op, "egemm0")) std::printf("%08x\n", rf_egemm_xin13_y0(v[0], v[1], (int)v[2]));
    else if (!std::strcmp(op, "egemmk")) std::printf("%08x\n", rf_egemm_key_candidate(v[0], v[1]));
    else if (!std::strcmp(op, "e2ek")) std::printf("%08x\n", rf_e2e_key_candidate(v[0]));
    else return 3;
  }
  return 0;
}
