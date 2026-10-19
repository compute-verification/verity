#include "gemms.h"

std::vector<ExtFactory>& ext_registry() {
  static std::vector<ExtFactory> r;
  return r;
}
std::vector<ExtGemm> ext_gemms(int m, int n, int k) {
  std::vector<ExtGemm> v;
  for (const auto f : ext_registry())
    for (const auto& g : f(m, n, k)) v.push_back(g);
  return v;
}
