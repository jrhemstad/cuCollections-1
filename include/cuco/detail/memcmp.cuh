
__device__ inline int __cuda_memcmp(void const* __lhs, void const* __rhs, size_t __count)
{
  auto __lhs_c = reinterpret_cast<unsigned char const*>(__lhs);
  auto __rhs_c = reinterpret_cast<unsigned char const*>(__rhs);
  while (__count--) {
    auto const __lhs_v = *__lhs_c++;
    auto const __rhs_v = *__rhs_c++;
    if (__lhs_v < __rhs_v) { return -1; }
    if (__lhs_v > __rhs_v) { return 1; }
  }
  return 0;
}

template <std::size_t alignment, bool large = (alignment > 8)>
struct memcmp_vectorized_impl {
  __device__ static int compare(char const* lhs, char const* rhs, std::size_t size)
  {
    __cuda_memcmp(lhs, rhs, size);
  }
};

template <>
struct memcmp_vectorized_impl<2, false> {
  __device__ static inline int compare(char const* lhs, char const* rhs, std::size_t size)
  {
    for (std::size_t offset = 0; offset < size; offset += 2) {
      auto const lhs2 = *reinterpret_cast<uint16_t const*>(lhs + offset);
      auto const rhs2 = *reinterpret_cast<uint16_t const*>(rhs + offset);
      if (lhs2 < rhs2) return -1;
      if (lhs2 > rhs2) return 1;
    }
    return 0;
  }
};

template <>
struct memcmp_vectorized_impl<4, false> {
  __device__ static inline int compare(char const* lhs, char const* rhs, std::size_t size)
  {
    for (std::size_t offset = 0; offset < size; offset += 4) {
      auto const lhs4 = *reinterpret_cast<uint32_t const*>(lhs + offset);
      auto const rhs4 = *reinterpret_cast<uint32_t const*>(rhs + offset);
      if (lhs4 < rhs4) return -1;
      if (lhs4 > rhs4) return 1;
    }
    return 0;
  }
};

template <>
struct memcmp_vectorized_impl<8, false> {
  __device__ static inline int compare(char const* lhs, char const* rhs, std::size_t size)
  {
    for (std::size_t offset = 0; offset < size; offset += 8) {
      auto const lhs8 = *reinterpret_cast<uint64_t const*>(lhs + offset);
      auto const rhs8 = *reinterpret_cast<uint64_t const*>(rhs + offset);
      if (lhs8 < rhs8) return -1;
      if (lhs8 > rhs8) return 1;
    }
    return 0;
  }
};

template <std::size_t alignment>
struct memcmp_vectorized_impl<alignment, true> : public memcmp_vectorized_impl<8, false> {
};

template <std::size_t native_alignment>
__device__ inline std::size_t determine_runtime_alignment(char const* lhs,
                                                          char const* rhs,
                                                          std::size_t size)
{
  if (native_alignment < 4) {
    auto const lhs_address = reinterpret_cast<std::uintptr_t>(lhs);
    auto const rhs_address = reinterpret_cast<std::uintptr_t>(rhs);
    // Lowest bit set will tell us what the common alignment of the three values is.
    return __ffs(lhs_address | rhs_address | size);
  } else {
    return native_alignment;
  }
}

template <std::size_t native_alignment>
__device__ inline int __memcmp(char const* lhs, char const* rhs, std::size_t size)
{
  auto const runtime_alignment = determine_runtime_alignment<native_alignment>(lhs, rhs, size);

  if (runtime_alignment == native_alignment) {
    return memcmp_vectorized_impl<native_alignment>(lhs, rhs, size);
  } else {
    switch (runtime_alignment) {
      default: return memcmp_vectorized_impl<8>::compare(lhs, rhs, size); break;
      case 3: return memcmp_vectorized_impl<4>::compare(lhs, rhs, size); break;
      case 2: return memcmp_vectorized_impl<2>::compare(lhs, rhs, size); break;
      case 1: return memcmp_vectorized_impl<1>::compare(lhs, rhs, size); break;
    }
  }
}

template <typename T>
__device__ inline int memcmp(T const* lhs, T const* rhs, std::size_t size)
{
  return __memcmp<alignof(T)>(
    reinterpret_cast<char const*>(lhs), reinterpret_cast<char const*>(rhs), size);
}
