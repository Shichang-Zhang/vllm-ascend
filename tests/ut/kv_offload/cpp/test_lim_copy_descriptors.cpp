// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#include "csrc/attention/lim_copy_descriptors/op_kernel/lim_copy_descriptors.cpp"

int main() {
  constexpr int block = 4, hot = 8, rowTokens = hot + 2 * block;
  constexpr int rows = 4, hostBlocks = 7, width = 5, stride = 7;
  constexpr uint16_t poison = 0xffff;
  for (int dim : {17, 32, 512}) {
    const int rope = dim == 17 ? 3 : 64;
    std::vector<uint16_t> hostKv(hostBlocks * block * dim), hostRope(hostBlocks * block * rope);
    for (size_t i = 0; i < hostKv.size(); ++i) hostKv[i] = i % 60000;
    for (size_t i = 0; i < hostRope.size(); ++i) hostRope[i] = 100 + i % 60000;
    std::vector<uint16_t> kv(rows * rowTokens * dim, poison), kpe(rows * rowTokens * rope, poison);
    std::vector<uint16_t> expectedKv = kv, expectedRope = kpe;
    std::array<int32_t, rows * stride> source{}, destination{};
    const std::array<int32_t, rows> counts{stride + 5, 5, 7, 7};
    const std::array<int32_t, rows> pool{2, 0, -1, rows};
    std::array<int32_t, rows * width> table{};
    for (int b = 0; b < rows; ++b) {
      for (int j = 0; j < width; ++j) table[b * width + j] = (b + j * 2) % hostBlocks;
      table[b * width + 2] = -1;
      table[b * width + 3] = hostBlocks;
      for (int w = 0; w < stride; ++w) {
        source[b * stride + w] = w * 3;
        destination[b * stride + w] = w;
      }
      source[b * stride + 5] = block * width;   // beyond live table width
      destination[b * stride + 6] = rowTokens;  // beyond the hot row
    }
    source[stride] = -1;
    destination[stride + 1] = -1;
    source[stride + 4] = 4 * block;  // use the last column of the live table
    std::array<int64_t, 2> host{reinterpret_cast<int64_t>(hostKv.data()), reinterpret_cast<int64_t>(hostRope.data())};
    std::array<int64_t, 8> params{block, hot, stride, dim, rope, hostBlocks, rows, rows};
    // Reuse the same buffers with changed geometry/plans, as graph replay
    // does. Poisoned inactive rows and padded pool entries must stay intact.
    for (int active : {4, 1, 2, 0, rows + 1}) {
      std::fill(kv.begin(), kv.end(), poison);
      std::fill(kpe.begin(), kpe.end(), poison);
      std::fill(expectedKv.begin(), expectedKv.end(), poison);
      std::fill(expectedRope.begin(), expectedRope.end(), poison);
      std::array<int64_t, 2> geometry{active, width};
      if (active <= rows) {
        for (int b = 0; b < active; ++b) {
          if (pool[b] < 0 || pool[b] >= rows) continue;
          for (int w = 0; w < std::min(counts[b], stride); ++w) {
            int token = source[b * stride + w], slot = destination[b * stride + w];
            if (token < 0 || token / block >= width || slot < 0 || slot >= rowTokens) continue;
            int phys = table[b * width + token / block];
            if (phys < 0 || phys >= hostBlocks) continue;
            int src = phys * block + token % block, dst = pool[b] * rowTokens + slot;
            std::copy_n(hostKv.data() + src * dim, dim, expectedKv.data() + dst * dim);
            std::copy_n(hostRope.data() + src * rope, rope, expectedRope.data() + dst * rope);
          }
        }
      }
      LimCopyDescriptorsTilingData tiling{6};
      for (int core = 0; core < 6; ++core) {
        AscendC::testBlockIndex = core;
        lim_copy_descriptors(reinterpret_cast<GM_ADDR>(source.data()), reinterpret_cast<GM_ADDR>(destination.data()),
                             reinterpret_cast<GM_ADDR>(const_cast<int32_t*>(counts.data())),
                             reinterpret_cast<GM_ADDR>(table.data()),
                             reinterpret_cast<GM_ADDR>(const_cast<int32_t*>(pool.data())),
                             reinterpret_cast<GM_ADDR>(host.data()), reinterpret_cast<GM_ADDR>(kv.data()),
                             reinterpret_cast<GM_ADDR>(kpe.data()), reinterpret_cast<GM_ADDR>(params.data()),
                             reinterpret_cast<GM_ADDR>(geometry.data()), nullptr, reinterpret_cast<GM_ADDR>(&tiling));
      }
      if (kv != expectedKv || kpe != expectedRope) {
        std::cerr << "copy mismatch dim=" << dim << " active=" << active << '\n';
        return 1;
      }
    }
  }
  return 0;
}
