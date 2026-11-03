// SPDX-License-Identifier: Apache-2.0
// CPU-only AscendC primitives for executing the real LIM copy kernel in UT.
// This models synchronous byte copies, not NPU queues, barriers or timing.
#pragma once

#include <cstdint>
#include <cstring>

#define __aicore__
#define __global__
#define __gm__
using GM_ADDR = uint8_t*;
constexpr int AIC = 1;
constexpr int g_coreType = 0;
struct LimCopyDescriptorsTilingData {
  int64_t usedCoreNum;
};
#define GET_TILING_DATA(name, ptr) auto name = *reinterpret_cast<LimCopyDescriptorsTilingData*>(ptr)
#define TILING_KEY_IS(key) ((key) == 0)

namespace AscendC {
inline uint32_t testBlockIndex = 0;
inline uint32_t GetBlockIdx() { return testBlockIndex; }
enum class QuePosition { VECIN, VECOUT };
template <typename T>
struct LocalTensor {
  T* data;
  LocalTensor operator[](uint64_t offset) { return {data + offset}; }
};
template <typename T>
class GlobalTensor {
 public:
  void SetGlobalBuffer(T* data) { data_ = data; }
  T GetValue(uint64_t index) const { return data_[index]; }
  T* operator[](uint64_t index) { return data_ + index; }

 private:
  T* data_ = nullptr;
};
template <QuePosition Input, QuePosition Output, int Depth>
class TQueBind {
 public:
  uint64_t bytes = 0;
  template <typename T>
  LocalTensor<T> AllocTensor() {
    return {new T[bytes / sizeof(T)]};
  }
  template <typename T>
  void EnQue(LocalTensor<T> tensor) {
    data_ = tensor.data;
  }
  template <typename T>
  LocalTensor<T> DeQue() {
    return {static_cast<T*>(data_)};
  }
  template <typename T>
  void FreeTensor(LocalTensor<T> tensor) {
    delete[] tensor.data;
  }

 private:
  void* data_ = nullptr;
};
class TPipe {
 public:
  template <typename Queue>
  void InitBuffer(Queue& queue, int, uint64_t bytes) {
    queue.bytes = bytes;
  }
};
template <typename T>
struct DataCopyPadExtParams {
  bool pad;
  uint8_t left;
  uint8_t right;
  T value;
};
struct DataCopyExtParams {
  uint16_t count;
  uint32_t bytes;
  uint32_t src;
  uint32_t dst;
  uint32_t reserved;
};
template <typename T>
void DataCopyPad(LocalTensor<T> dst, T* src, DataCopyExtParams params, DataCopyPadExtParams<T>) {
  std::memcpy(dst.data, src, params.bytes);
  // AscendC pads each GM->UB row to 32 bytes, so exercise UB alignment too.
  const auto padding = (32 - params.bytes % 32) % 32;
  std::memset(reinterpret_cast<uint8_t*>(dst.data) + params.bytes, 0, padding);
}
template <typename T>
void DataCopyPad(T* dst, LocalTensor<T> src, DataCopyExtParams params) {
  std::memcpy(dst, src.data, params.bytes);
}
}  // namespace AscendC
