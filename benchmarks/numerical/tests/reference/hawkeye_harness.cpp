// Drives the pinned Hawkeye simulator's own compiled routines (verity/ml/circuits/silicon/hawkeye/PIN.json, unmodified sources) on
// transition-unit cases, so that hawkeye_close.py and verity.ml.circuits.silicon.silicon can be compared bit for bit against the C++.
//
// Build inside the recorded run (exhaustive_pod.sh), against the pinned checkout $HAWKEYE and a CPU libtorch:
//   g++ -std=c++17 -O2 -I$HAWKEYE -I$TORCH_INC -I$TORCH_INC/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=$ABI \
//       hawkeye_harness.cpp $HAWKEYE/src/Ampere_simulator.cpp $HAWKEYE/src/gfloat.cpp $HAWKEYE/utils/utils.cpp \
//       -L$TORCH_LIB -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$TORCH_LIB -o hawkeye_harness
//
// stdin: one case per line, "<acc: 8 hex> <a: 16 x 4 hex> <b: 16 x 4 hex>".
// stdout: one line per case, in order: "<adapter> <finalised> <matmul>" as 8-hex FP32 words, where
//   adapter    = float(group_sum([c, p0..p7]) then group_sum([., p8..p15])) with c from the D7 adapter (fp32_to_gfloat below),
//   finalised  = the same with the wrapper's finalisation (one more group_sum over the single remaining Gfloat) before the cast,
//   matmul     = Ampere_simulator::matmul on a bf16 row x column with the pinned wrapper (zero start) when |c| == 0, else "-".
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "include/Ampere_simulator.h"
#include "utils/utils.h"

namespace {

// The arbitrary-accumulator adapter of hawkeye_close.fp32_to_gfloat, in C++: zero -> (sign, AMPERE_ZERO_EXPONENT, 0);
// subnormal -> (sign, -126, mantissa); normal -> (sign, e - 127, 1<<23 | mantissa).  Non-finite words are not fed here.
Gfloat fp32_to_gfloat(uint32_t bits) {
    bool sign = (bits >> 31) & 1;
    int exponent_field = (bits >> 23) & 0xFF;
    int32_t mantissa = bits & 0x7FFFFF;
    if (exponent_field == 0) {
        if (mantissa == 0) return Gfloat(sign, AMPERE_ZERO_EXPONENT, 0);
        return Gfloat(sign, FP32_MIN_NONZERO_EXPONENT, mantissa);
    }
    return Gfloat(sign, static_cast<int16_t>(exponent_field - 127), (1 << 23) | mantissa);
}

uint32_t bits_of(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

struct Case {
    uint32_t acc;
    uint16_t a[16];
    uint16_t b[16];
};

}  // namespace

int main() {
    std::vector<Case> cases;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.size() < 8 + 1 + 64 + 1 + 64) continue;
        Case c;
        c.acc = static_cast<uint32_t>(std::strtoul(line.substr(0, 8).c_str(), nullptr, 16));
        for (int i = 0; i < 16; ++i) {
            c.a[i] = static_cast<uint16_t>(std::strtoul(line.substr(9 + 4 * i, 4).c_str(), nullptr, 16));
            c.b[i] = static_cast<uint16_t>(std::strtoul(line.substr(9 + 64 + 1 + 4 * i, 4).c_str(), nullptr, 16));
        }
        cases.push_back(c);
    }

    Ampere_simulator sim;
    std::vector<uint32_t> adapter(cases.size()), finalised(cases.size()), matmul(cases.size(), 0);
    std::vector<char> has_matmul(cases.size(), 0);

    for (size_t n = 0; n < cases.size(); ++n) {
        const Case& c = cases[n];
        std::vector<Gfloat> group;
        group.reserve(AMPERE_ACCUMULATOR_GROUP_SIZE);
        Gfloat accumulator = fp32_to_gfloat(c.acc);
        for (int start = 0; start < 16; start += 8) {
            group.clear();
            group.push_back(accumulator);
            for (int l = start; l < start + 8; ++l) {
                group.push_back(multiply_bfloat16_to_gfloat(c.a[l], c.b[l], AMPERE_ZERO_EXPONENT));
            }
            accumulator = sim.group_sum(group);
        }
        adapter[n] = bits_of(static_cast<float>(accumulator));
        std::vector<Gfloat> last{accumulator};
        finalised[n] = bits_of(static_cast<float>(sim.group_sum(last)));
    }

    // The pinned wrapper on the zero-accumulator cases, in batches: A is m x 16 (row i = a of case i), B is 16 x m column-major
    // (column j = b of case j; strides {1, 16} so the wrapper takes the column-major branch silently); D[i][i] is case i.
    std::vector<size_t> zero_acc;
    for (size_t n = 0; n < cases.size(); ++n) {
        if ((cases[n].acc & 0x7FFFFFFFu) == 0) zero_acc.push_back(n);
    }
    const size_t batch = 64;
    for (size_t start = 0; start < zero_acc.size(); start += batch) {
        const size_t m = std::min(batch, zero_acc.size() - start);
        std::vector<uint16_t> a_buf(m * 16), b_buf(m * 16);
        for (size_t i = 0; i < m; ++i) {
            const Case& c = cases[zero_acc[start + i]];
            std::memcpy(&a_buf[i * 16], c.a, 32);
            std::memcpy(&b_buf[i * 16], c.b, 32);
        }
        auto options = torch::TensorOptions().dtype(torch::kBFloat16);
        torch::Tensor A = torch::from_blob(a_buf.data(), {static_cast<int64_t>(m), 16}, options);
        torch::Tensor B = torch::from_blob(b_buf.data(), {16, static_cast<int64_t>(m)}, {1, 16}, options);
        torch::Tensor D = sim.matmul(A, B);
        auto acc = D.accessor<float, 2>();
        for (size_t i = 0; i < m; ++i) {
            matmul[zero_acc[start + i]] = bits_of(acc[i][i]);
            has_matmul[zero_acc[start + i]] = 1;
        }
    }

    for (size_t n = 0; n < cases.size(); ++n) {
        if (has_matmul[n]) {
            std::printf("%08x %08x %08x\n", adapter[n], finalised[n], matmul[n]);
        } else {
            std::printf("%08x %08x -\n", adapter[n], finalised[n]);
        }
    }
    return 0;
}
