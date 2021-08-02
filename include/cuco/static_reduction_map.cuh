/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cooperative_groups.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cub/cub.cuh>
#include <cuda/std/atomic>
#include <memory>

#include <cuco/allocator.hpp>

#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11000) && defined(__CUDA_ARCH__) && \
  (__CUDA_ARCH__ >= 700)
#define CUCO_HAS_CUDA_BARRIER
#endif

#if defined(CUCO_HAS_CUDA_BARRIER)
#include <cuda/barrier>
#endif

#include <cuda.h>
#include <cuco/detail/error.hpp>
#include <cuco/detail/hash_functions.cuh>
#include <cuco/detail/pair.cuh>
#include <cuco/detail/static_reduction_map_kernels.cuh>
#include <cuco/detail/traits.hpp>

namespace cuco {

/**
 * @brief `+` reduction functor that internally uses an atomic fetch-and-add
 * operation.
 *
 * @tparam T The data type used for reduction
 */
template <typename T>
struct reduce_add {
  using value_type            = T;
  static constexpr T identity = 0;

  template <cuda::thread_scope Scope, typename T2>
  __device__ T apply(cuda::atomic<T, Scope>& slot, T2 const& value) const
  {
    return slot.fetch_add(value, cuda::memory_order_relaxed);
  }
};

/**
 * @brief `-` reduction functor that internally uses an atomic fetch-and-add
 * operation.
 *
 * @tparam T The data type used for reduction
 */
template <typename T>
struct reduce_sub {
  using value_type            = T;
  static constexpr T identity = 0;

  template <cuda::thread_scope Scope, typename T2>
  __device__ T apply(cuda::atomic<T, Scope>& slot, T2 const& value) const
  {
    return slot.fetch_sub(value, cuda::memory_order_relaxed);
  }
};

/**
 * @brief `min` reduction functor that internally uses an atomic fetch-and-add
 * operation.
 *
 * @tparam T The data type used for reduction
 */
template <typename T>
struct reduce_min {
  using value_type            = T;
  static constexpr T identity = std::numeric_limits<T>::max();

  template <cuda::thread_scope Scope, typename T2>
  __device__ T apply(cuda::atomic<T, Scope>& slot, T2 const& value) const
  {
    return slot.fetch_min(value, cuda::memory_order_relaxed);
  }
};

/**
 * @brief `max` reduction functor that internally uses an atomic fetch-and-add
 * operation.
 *
 * @tparam T The data type used for reduction
 */
template <typename T>
struct reduce_max {
  using value_type            = T;
  static constexpr T identity = std::numeric_limits<T>::lowest();

  template <cuda::thread_scope Scope, typename T2>
  __device__ T apply(cuda::atomic<T, Scope>& slot, T2 const& value) const
  {
    return slot.fetch_max(value, cuda::memory_order_relaxed);
  }
};

/**
 * @brief Wrapper for a user-defined custom reduction operator.
 * @brief Internally uses an atomic compare-and-swap loop.
 *
 * @tparam T The data type used for reduction
 * @tparam Identity Neutral element under the given reduction group
 * @tparam Op Commutative and associative binary operator
 */
template <typename T,
          T Identity,
          typename Op,
          std::uint32_t BackoffBaseDelay = 8,
          std::uint32_t BackoffMaxDelay  = 256>
struct custom_op {
  using value_type            = T;
  static constexpr T identity = Identity;

  Op op;

  template <cuda::thread_scope Scope, typename T2>
  __device__ T apply(cuda::atomic<T, Scope>& slot, T2 const& value) const
  {
    [[maybe_unused]] unsigned ns = BackoffBaseDelay;

    auto old = slot.load(cuda::memory_order_relaxed);
    while (not slot.compare_exchange_strong(old, op(old, value), cuda::memory_order_relaxed)) {
#if __CUDA_ARCH__ >= 700
      // exponential backoff strategy to reduce atomic contention
      if (true) {
        asm volatile("nanosleep.u32 %0;" ::"r"((unsigned)ns) :);
        if (ns < BackoffMaxDelay) { ns *= 2; }
      }
#endif
    }
    return old;
  }
};

/**
 * @brief A GPU-accelerated, unordered, associative container of key-value
 * pairs that reduces the values associated to the same key according to a
 * functor.
 *
 * Allows constant time concurrent inserts or concurrent find operations (not
 * concurrent insert and find) from threads in device code.
 *
 * Current limitations:
 * - Requires key types where `cuco::is_bitwise_comparable<T>::value` is true
 * - Does not support erasing keys
 * - Capacity is fixed and will not grow automatically
 * - Requires the user to specify sentinel value for the key to indicate empty
 *   slots
 * - Does not support concurrent insert and find operations
 *
 * The `static_reduction_map` supports two types of operations:
 * - Host-side "bulk" operations
 * - Device-side "singular" operations
 *
 * The host-side bulk operations include `insert`, `find`, and `contains`. These
 * APIs should be used when there are a large number of keys to insert or lookup
 * in the map. For example, given a range of keys specified by device-accessible
 * iterators, the bulk `insert` function will insert all keys into the map.
 *
 * The singular device-side operations allow individual threads to perform
 * independent insert or find/contains operations from device code. These
 * operations are accessed through non-owning, trivially copyable "view" types:
 * `device_view` and `mutable_device_view`. The `device_view` class is an
 * immutable view that allows only non-modifying operations such as `find` or
 * `contains`. The `mutable_device_view` class only allows `insert` operations.
 * The two types are separate to prevent erroneous concurrent insert/find
 * operations.
 *
 *  Example:
 *  \code{.cpp}
 *
 * // Empty slots are represented by reserved "sentinel" values. These values should be selected
 * such
 * // that they never occur in your input data.
 * int const empty_key_sentinel = -1;
 *
 * // Number of key/value pairs to be inserted
 * std::size_t const num_elems = 256;
 *
 * // average number of values per distinct key
 * std::size_t const multiplicity = 4;
 *
 * // Compute capacity based on a 50% load factor
 * auto const load_factor     = 0.5;
 * std::size_t const capacity = std::ceil(num_elems / load_factor);
 *
 * // Constructs a map each key with "capacity" slots using -1 as the
 * // empty key sentinel. The initial payload value for empty slots is determined by the identity of
 * // the reduction operation. By using the `reduce_add` operation, all values associated with a
 * // given key will be summed.
 * cuco::static_reduction_map<cuco::reduce_add<int>, int, int> map{capacity, empty_key_sentinel};
 *
 * // Create a sequence of random keys
 * thrust::device_vector<int> insert_keys(num_elems);
 * thrust::transform(thrust::device,
 *                   thrust::make_counting_iterator<std::size_t>(0),
 *                   thrust::make_counting_iterator(insert_keys.size()),
 *                   insert_keys.begin(),
 *                   [=] __device__(auto i) {
 *                     thrust::default_random_engine rng;
 *                     thrust::uniform_int_distribution<int> dist(
 *                       int{1}, static_cast<int>(num_elems / multiplicity));
 *                     rng.discard(i);
 *                     return dist(rng);
 *                   });
 *
 * // Insert each key with a payload of `1` to count the number of times each key was inserted by
 * // using the `reduce_add` op
 * auto zipped = thrust::make_zip_iterator(
 *   thrust::make_tuple(insert_keys.begin(), thrust::make_constant_iterator(1)));
 *
 * // Inserts all pairs into the map, accumulating the payloads with the `reduce_add` operation
 * map.insert(zipped, zipped + insert_keys.size());
 *
 * // Get a `device_view` and passes it to a kernel where threads may perform
 * // `find/contains` lookups
 * kernel<<<...>>>(m.get_device_view());
 * \endcode
 *
 *
 * @tparam Key Arithmetic type used for key
 * @tparam Value Type of the mapped values
 * @tparam Scope The scope in which insert/find operations will be performed by
 * individual threads.
 * @tparam Allocator Type of allocator used for device storage
 */
template <typename ReductionOp,
          typename Key,
          typename Value,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          typename Allocator       = cuco::cuda_allocator<char>>
class static_reduction_map {
  static_assert(
    is_bitwise_comparable<Key>::value,
    "Key type must have unique object representations or have been explicitly declared as safe for "
    "bitwise comparison via specialization of cuco::is_bitwise_comparable<Key>.");

  static_assert(std::is_same<typename ReductionOp::value_type, Value>::value,
                "Type mismatch between ReductionOp::value_type and Value");

 public:
  using value_type         = cuco::pair_type<Key, Value>;
  using key_type           = Key;
  using mapped_type        = Value;
  using atomic_key_type    = cuda::atomic<key_type, Scope>;
  using atomic_mapped_type = cuda::atomic<mapped_type, Scope>;
  using pair_atomic_type   = cuco::pair_type<atomic_key_type, atomic_mapped_type>;
  using atomic_ctr_type    = cuda::atomic<std::size_t, Scope>;
  using allocator_type     = Allocator;
  using slot_allocator_type =
    typename std::allocator_traits<Allocator>::rebind_alloc<pair_atomic_type>;
  using counter_allocator_type =
    typename std::allocator_traits<Allocator>::rebind_alloc<atomic_ctr_type>;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
  static_assert(atomic_key_type::is_always_lock_free,
                "A key type larger than 8B is supported for only sm_70 and up.");
  static_assert(atomic_mapped_type::is_always_lock_free,
                "A value type larger than 8B is supported for only sm_70 and up.");
#endif

  static_reduction_map(static_reduction_map const&) = delete;
  static_reduction_map(static_reduction_map&&)      = delete;
  static_reduction_map& operator=(static_reduction_map const&) = delete;
  static_reduction_map& operator=(static_reduction_map&&) = delete;

  /**
   * @brief Construct a fixed-size map with the specified capacity and sentinel key.
   * @brief Construct a statically sized map with the specified number of slots
   * and sentinel key.
   *
   * The capacity of the map is fixed. Insert operations will not automatically
   * grow the map. Attempting to insert more unique keys than the capacity of
   * the map results in undefined behavior (there should be at least one empty slot).
   *
   * Performance begins to degrade significantly beyond a load factor of ~70%.
   * For best performance, choose a capacity that will keep the load factor
   * below 70%. E.g., if inserting `N` unique keys, choose a capacity of
   * `N * (1/0.7)`.
   *
   * The `empty_key_sentinel` is reserved and undefined behaviour results from
   * attempting to insert said key.
   *
   * @param capacity The total number of slots in the map
   * @param empty_key_sentinel The reserved key value for empty slots
   * @param reduction_op Reduction operator
   * @param alloc Allocator used for allocating device storage
   */
  static_reduction_map(std::size_t capacity,
                       Key empty_key_sentinel,
                       ReductionOp reduction_op = {},
                       Allocator const& alloc   = Allocator{});

  /**
   * @brief Destroys the map and frees its contents.
   *
   */
  ~static_reduction_map();

  /**
   * @brief Inserts all key/value pairs in the range `[first, last)`.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `value_type`
   * @tparam Hash Unary callable type
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of key/value pairs
   * @param last End of the sequence of key/value pairs
   * @param stream CUDA stream used for insert
   * @param hash The unary function to apply to hash each key
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename InputIt,
            typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void insert(InputIt first,
              InputIt last,
              cudaStream_t stream = 0,
              Hash hash           = Hash{},
              KeyEqual key_equal  = KeyEqual{});

  /**
   * @brief Finds the values corresponding to all keys in the range `[first, last)`.
   *
   * If the key `*(first + i)` exists in the map, copies its associated value to `(output_begin +
   * i)`. Else, copies the empty value sentinel.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is
   * convertible to the map's `mapped_type`
   * @tparam Hash Unary callable type
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of values retrieved for each key
   * @param stream CUDA stream used for this operation
   * @param hash The unary function to apply to hash each key
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename InputIt,
            typename OutputIt,
            typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void find(InputIt first,
            InputIt last,
            OutputIt output_begin,
            cudaStream_t stream = 0,
            Hash hash           = Hash{},
            KeyEqual key_equal  = KeyEqual{});

  /**
   * @brief Retrieves all of the keys and their associated values.
   *
   * The order in which keys are returned is implementation defined and not guaranteed to be
   * consistent between subsequent calls to `retrieve_all`.
   *
   * Behavior is undefined if the range beginning at `keys_out` or `values_out` is not large enough
   * to contain the number of keys in the map.
   *
   * @tparam KeyOut Device accessible random access output iterator whose `value_type` is
   * convertible from `key_type`.
   * @tparam ValueOut Device accesible random access output iterator whose `value_type` is
   * convertible from `mapped_type`.
   * @param keys_out Beginning output iterator for keys
   * @param values_out Beginning output iterator for values
   * @param stream CUDA stream used for this operation
   */
  template <typename KeyOut, typename ValueOut>
  void retrieve_all(KeyOut keys_out, ValueOut values_out, cudaStream_t stream = 0);

  /**
   * @brief Indicates whether the keys in the range `[first, last)` are contained in the map.
   *
   * Writes a `bool` to `(output + i)` indicating if the key `*(first + i)` exists in the map.
   *
   * @tparam InputIt Device accessible input iterator whose `value_type` is
   * convertible to the map's `key_type`
   * @tparam OutputIt Device accessible output iterator whose `value_type` is
   * convertible to the map's `mapped_type`
   * @tparam Hash Unary callable type
   * @tparam KeyEqual Binary callable type
   * @param first Beginning of the sequence of keys
   * @param last End of the sequence of keys
   * @param output_begin Beginning of the sequence of booleans for the presence of each key
   * @param stream CUDA stream used
   * @param hash The unary function to apply to hash each key
   * @param key_equal The binary function to compare two keys for equality
   */
  template <typename InputIt,
            typename OutputIt,
            typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
            typename KeyEqual = thrust::equal_to<key_type>>
  void contains(InputIt first,
                InputIt last,
                OutputIt output_begin,
                cudaStream_t stream = 0,
                Hash hash           = Hash{},
                KeyEqual key_equal  = KeyEqual{});

 private:
  class device_view_base {
   protected:
    // Import member type definitions from `static_reduction_map`
    using value_type     = value_type;
    using key_type       = Key;
    using mapped_type    = Value;
    using iterator       = pair_atomic_type*;
    using const_iterator = pair_atomic_type const*;

   private:
    pair_atomic_type* slots_{};     ///< Pointer to flat slots storage
    std::size_t capacity_{};        ///< Total number of slots
    Key empty_key_sentinel_{};      ///< Key value that represents an empty slot
    Value empty_value_sentinel_{};  ///< Initial Value of empty slot
    ReductionOp op_{};              ///< Binary operation reduction function object

   protected:
    __host__ __device__ device_view_base(pair_atomic_type* slots,
                                         std::size_t capacity,
                                         Key empty_key_sentinel,
                                         ReductionOp reduction_op) noexcept
      : slots_{slots},
        capacity_{capacity},
        empty_key_sentinel_{empty_key_sentinel},
        empty_value_sentinel_{ReductionOp::identity},
        op_{reduction_op}
    {
    }

    /**
     * @brief Gets the binary op
     *
     */
    __device__ ReductionOp get_op() const noexcept { return op_; }

    /**
     * @brief Gets slots array.
     *
     * @return Slots array
     */
    __device__ pair_atomic_type* get_slots() noexcept { return slots_; }

    /**
     * @brief Gets slots array.
     *
     * @return Slots array
     */
    __device__ pair_atomic_type const* get_slots() const noexcept { return slots_; }

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * @tparam Hash Unary callable type
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template <typename Hash>
    __device__ iterator initial_slot(Key const& k, Hash hash) noexcept
    {
      return &slots_[hash(k) % capacity_];
    }

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * @tparam Hash Unary callable type
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template <typename Hash>
    __device__ const_iterator initial_slot(Key const& k, Hash hash) const noexcept
    {
      return &slots_[hash(k) % capacity_];
    }

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * To be used for Cooperative Group based probing.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @param g the Cooperative Group for which the initial slot is needed
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template <typename CG, typename Hash>
    __device__ iterator initial_slot(CG const& g, Key const& k, Hash hash) noexcept
    {
      return &slots_[(hash(k) + g.thread_rank()) % capacity_];
    }

    /**
     * @brief Returns the initial slot for a given key `k`
     *
     * To be used for Cooperative Group based probing.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @param g the Cooperative Group for which the initial slot is needed
     * @param k The key to get the slot for
     * @param hash The unary callable used to hash the key
     * @return Pointer to the initial slot for `k`
     */
    template <typename CG, typename Hash>
    __device__ const_iterator initial_slot(CG const& g, Key const& k, Hash hash) const noexcept
    {
      return &slots_[(hash(k) + g.thread_rank()) % capacity_];
    }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot.
     *
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    __device__ iterator next_slot(iterator s) noexcept { return (++s < end()) ? s : begin_slot(); }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot.
     *
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    __device__ const_iterator next_slot(const_iterator s) const noexcept
    {
      return (++s < end()) ? s : begin_slot();
    }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot. To
     * be used for Cooperative Group based probing.
     *
     * @tparam CG The Cooperative Group type
     * @param g The Cooperative Group for which the next slot is needed
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    template <typename CG>
    __device__ iterator next_slot(CG const& g, iterator s) noexcept
    {
      uint32_t index = s - slots_;
      return &slots_[(index + g.size()) % capacity_];
    }

    /**
     * @brief Given a slot `s`, returns the next slot.
     *
     * If `s` is the last slot, wraps back around to the first slot. To
     * be used for Cooperative Group based probing.
     *
     * @tparam CG The Cooperative Group type
     * @param g The Cooperative Group for which the next slot is needed
     * @param s The slot to advance
     * @return The next slot after `s`
     */
    template <typename CG>
    __device__ const_iterator next_slot(CG const& g, const_iterator s) const noexcept
    {
      uint32_t index = s - slots_;
      return &slots_[(index + g.size()) % capacity_];
    }

   public:
    /**
     * @brief Gets the maximum number of elements the hash map can hold.
     *
     * @return The maximum number of elements the hash map can hold
     */
    __host__ __device__ std::size_t get_capacity() const noexcept { return capacity_; }

    /**
     * @brief Gets the sentinel value used to represent an empty key slot.
     *
     * @return The sentinel value used to represent an empty key slot
     */
    __host__ __device__ Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

    /**
     * @brief Gets the sentinel value used to represent an empty value slot.
     *
     * @return The sentinel value used to represent an empty value slot
     */
    __host__ __device__ Value get_empty_value_sentinel() const noexcept
    {
      return empty_value_sentinel_;
    }

    /**
     * @brief Returns iterator to the first slot.
     *
     * @note Unlike `std::map::begin()`, the `begin_slot()` iterator does _not_ point to the first
     * occupied slot. Instead, it refers to the first slot in the array of contiguous slot storage.
     * Iterating from `begin_slot()` to `end_slot()` will iterate over all slots, including those
     * both empty and filled.
     *
     * There is no `begin()` iterator to avoid confusion as it is not possible to provide an
     * iterator over only the filled slots.
     *
     * @return Iterator to the first slot
     */
    __device__ iterator begin_slot() noexcept { return slots_; }

    /**
     * @brief Returns iterator to the first slot.
     *
     * @note Unlike `std::map::begin()`, the `begin_slot()` iterator does _not_ point to the first
     * occupied slot. Instead, it refers to the first slot in the array of contiguous slot storage.
     * Iterating from `begin_slot()` to `end_slot()` will iterate over all slots, including those
     * both empty and filled.
     *
     * There is no `begin()` iterator to avoid confusion as it is not possible to provide an
     * iterator over only the filled slots.
     *
     * @return Iterator to the first slot
     */
    __device__ const_iterator begin_slot() const noexcept { return slots_; }

    /**
     * @brief Returns a const_iterator to one past the last slot.
     *
     * @return A const_iterator to one past the last slot
     */
    __host__ __device__ const_iterator end_slot() const noexcept { return slots_ + capacity_; }

    /**
     * @brief Returns an iterator to one past the last slot.
     *
     * @return An iterator to one past the last slot
     */
    __host__ __device__ iterator end_slot() noexcept { return slots_ + capacity_; }

    /**
     * @brief Returns a const_iterator to one past the last slot.
     *
     * `end()` calls `end_slot()` and is provided for convenience for those familiar with checking
     * an iterator returned from `find()` against the `end()` iterator.
     *
     * @return A const_iterator to one past the last slot
     */
    __host__ __device__ const_iterator end() const noexcept { return end_slot(); }

    /**
     * @brief Returns an iterator to one past the last slot.
     *
     * `end()` calls `end_slot()` and is provided for convenience for those familiar with checking
     * an iterator returned from `find()` against the `end()` iterator.
     *
     * @return An iterator to one past the last slot
     */
    __host__ __device__ iterator end() noexcept { return end_slot(); }
  };

 public:
  /**
   * @brief Mutable, non-owning view-type that may be used in device code to
   * perform singular inserts into the map.
   *
   * `device_mutable_view` is trivially-copyable and is intended to be passed by
   * value.
   *
   * Example:
   * \code{.cpp}
   * cuco::static_reduction_map<cuco::reduce_add<int>int,int> m{100'000, -1};
   *
   * // Inserts a sequence of pairs {{0,0}, {1,1}, ... {i,i}}
   * thrust::for_each(thrust::make_counting_iterator(0),
   *                  thrust::make_counting_iterator(50'000),
   *                  [map = m.get_mutable_device_view()]
   *                  __device__ (auto i) mutable {
   *                     map.insert(thrust::make_pair(i,i));
   *                  });
   * \endcode
   */
  class device_mutable_view : public device_view_base {
   public:
    using value_type     = typename device_view_base::value_type;
    using key_type       = typename device_view_base::key_type;
    using mapped_type    = typename device_view_base::mapped_type;
    using iterator       = typename device_view_base::iterator;
    using const_iterator = typename device_view_base::const_iterator;
    /**
     * @brief Construct a mutable view of the first `capacity` slots of the
     * slots array pointed to by `slots`.
     *
     * @param slots Pointer to beginning of initialized slots array
     * @param capacity The number of slots viewed by this object
     * @param empty_key_sentinel The reserved value for keys to represent empty
     * slots
     * @param empty_value_sentinel The reserved value for mapped values to
     * represent empty slots
     */
    __host__ __device__ device_mutable_view(pair_atomic_type* slots,
                                            std::size_t capacity,
                                            Key empty_key_sentinel,
                                            ReductionOp reduction_op = {}) noexcept
      : device_view_base{slots, capacity, empty_key_sentinel, reduction_op}
    {
    }
    template <typename CG>
    __device__ static device_mutable_view make_from_uninitialized_slots(
      CG const& g,
      pair_atomic_type* slots,
      std::size_t capacity,
      Key empty_key_sentinel,
      ReductionOp reduction_op) noexcept
    {
      device_view_base::initialize_slots(g, slots, capacity, empty_key_sentinel, reduction_op);
      return device_mutable_view{slots, capacity, empty_key_sentinel, reduction_op};
    }
    /**
     * @brief Inserts the specified key/value pair into the map.
     *
     * Returns a pair consisting of an iterator to the inserted element (or to
     * the element that prevented the insertion) and a `bool` denoting whether
     * the insertion took place.
     *
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param insert_pair The pair to insert
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys for
     * equality
     * @return `true` if the insert (of a new key) was successful, `false` otherwise.
     */
    template <typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool insert(value_type const& insert_pair,
                           Hash hash          = Hash{},
                           KeyEqual key_equal = KeyEqual{}) noexcept;
    /**
     * @brief Inserts the specified key/value pair into the map.
     *
     * Returns a pair consisting of an iterator to the inserted element (or to
     * the element that prevented the insertion) and a `bool` denoting whether
     * the insertion took place. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single insert. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `insert` at moderate to high load factors.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     *
     * @param g The Cooperative Group that performs the insert
     * @param insert_pair The pair to insert
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys for
     * equality
     * @return `true` if the insert (of a new key) was successful, `false` otherwise.
     */
    template <typename CG,
              typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool insert(CG const& g,
                           value_type const& insert_pair,
                           Hash hash          = Hash{},
                           KeyEqual key_equal = KeyEqual{}) noexcept;
  };  // class device mutable view

  /**
   * @brief Non-owning view-type that may be used in device code to
   * perform singular find and contains operations for the map.
   *
   * `device_view` is trivially-copyable and is intended to be passed by
   * value.
   *
   */
  class device_view : public device_view_base {
   public:
    using value_type     = typename device_view_base::value_type;
    using key_type       = typename device_view_base::key_type;
    using mapped_type    = typename device_view_base::mapped_type;
    using iterator       = typename device_view_base::iterator;
    using const_iterator = typename device_view_base::const_iterator;
    /**
     * @brief Construct a view of the first `capacity` slots of the
     * slots array pointed to by `slots`.
     *
     * @param slots Pointer to beginning of initialized slots array
     * @param capacity The number of slots viewed by this object
     * @param empty_key_sentinel The reserved value for keys to represent empty
     * slots
     * @param reduction_op The reduction functor
     */
    __host__ __device__ device_view(pair_atomic_type* slots,
                                    std::size_t capacity,
                                    Key empty_key_sentinel,
                                    ReductionOp reduction_op = {}) noexcept
      : device_view_base{slots, capacity, empty_key_sentinel, reduction_op}
    {
    }

    /**
     * @brief Construct a `device_view` from a `device_mutable_view` object
     *
     * @param mutable_map object of type `device_mutable_view`
     */
    __host__ __device__ explicit device_view(device_mutable_view mutable_map)
      : device_view_base{mutable_map.get_slots(),
                         mutable_map.get_capacity(),
                         mutable_map.get_empty_key_sentinel(),
                         mutable_map.get_op()}
    {
    }

    /**
     * @brief Makes a copy of given `device_view` using non-owned memory.
     *
     * This function is intended to be used to create shared memory copies of small static maps,
     * although global memory can be used as well.
     *
     * Example:
     * @code{.cpp}
     * template <typename MapType, int CAPACITY>
     * __global__ void use_device_view(const typename MapType::device_view device_view,
     *                                 map_key_t const* const keys_to_search,
     *                                 map_value_t* const values_found,
     *                                 const size_t number_of_elements)
     * {
     *     const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
     *
     *     __shared__ typename MapType::pair_atomic_type sm_buffer[CAPACITY];
     *
     *     auto g = cg::this_thread_block();
     *
     *     const map_t::device_view sm_static_reduction_map = device_view.make_copy(g,
     *                                                                    sm_buffer);
     *
     *     for (size_t i = g.thread_rank(); i < number_of_elements; i += g.size())
     *     {
     *         values_found[i] = sm_static_reduction_map.find(keys_to_search[i])->second;
     *     }
     * }
     * @endcode
     *
     * @tparam CG The type of the cooperative thread group
     * @param g The cooperative thread group used to copy the slots
     * @param source_device_view `device_view` to copy from
     * @param memory_to_use Array large enough to support `capacity` elements. Object does not take
     * the ownership of the memory
     * @return Copy of passed `device_view`
     */
    template <typename CG>
    __device__ static device_view make_copy(CG const& g,
                                            pair_atomic_type* const memory_to_use,
                                            device_view source_device_view) noexcept
    {
#if defined(CUDA_HAS_CUDA_BARRIER)
      __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
      if (g.thread_rank() == 0) { init(&barrier, g.size()); }
      g.sync();

      cuda::memcpy_async(g,
                         memory_to_use,
                         source_device_view.get_slots(),
                         sizeof(pair_atomic_type) * source_device_view.get_capacity(),
                         barrier);

      barrier.arrive_and_wait();
#else
      pair_atomic_type const* const slots_ptr = source_device_view.get_slots();
      for (std::size_t i = g.thread_rank(); i < source_device_view.get_capacity(); i += g.size()) {
        new (&memory_to_use[i].first)
          atomic_key_type{slots_ptr[i].first.load(cuda::memory_order_relaxed)};
        new (&memory_to_use[i].second)
          atomic_mapped_type{slots_ptr[i].second.load(cuda::memory_order_relaxed)};
      }
      g.sync();
#endif

      return device_view(memory_to_use,
                         source_device_view.get_capacity(),
                         source_device_view.get_empty_key_sentinel(),
                         source_device_view.get_op());
    }

    /**
     * @brief Finds the value corresponding to the key `k`.
     *
     * Returns an iterator to the pair whose key is equivalent to `k`.
     * If no such pair exists, returns `end()`.
     *
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return An iterator to the position at which the key/value pair
     * containing `k` was inserted
     */
    template <typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ iterator find(Key const& k,
                             Hash hash          = Hash{},
                             KeyEqual key_equal = KeyEqual{}) noexcept;

    /** @brief Finds the value corresponding to the key `k`.
     *
     * Returns a const_iterator to the pair whose key is equivalent to `k`.
     * If no such pair exists, returns `end()`.
     *
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return An iterator to the position at which the key/value pair
     * containing `k` was inserted
     */
    template <typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ const_iterator find(Key const& k,
                                   Hash hash          = Hash{},
                                   KeyEqual key_equal = KeyEqual{}) const noexcept;

    /**
     * @brief Finds the value corresponding to the key `k`.
     *
     * Returns an iterator to the pair whose key is equivalent to `k`.
     * If no such pair exists, returns `end()`. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single find. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `find` at moderate to high load factors.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the find
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return An iterator to the position at which the key/value pair
     * containing `k` was inserted
     */
    template <typename CG,
              typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ iterator
    find(CG const& g, Key const& k, Hash hash = Hash{}, KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Finds the value corresponding to the key `k`.
     *
     * Returns a const_iterator to the pair whose key is equivalent to `k`.
     * If no such pair exists, returns `end()`. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single find. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `find` at moderate to high load factors.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the find
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return An iterator to the position at which the key/value pair
     * containing `k` was inserted
     */
    template <typename CG,
              typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ const_iterator find(CG const& g,
                                   Key const& k,
                                   Hash hash          = Hash{},
                                   KeyEqual key_equal = KeyEqual{}) const noexcept;

    /**
     * @brief Indicates whether the key `k` was inserted into the map.
     *
     * If the key `k` was inserted into the map, find returns
     * true. Otherwise, it returns false.
     *
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return A boolean indicating whether the key/value pair
     * containing `k` was inserted
     */
    template <typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool contains(Key const& k,
                             Hash hash          = Hash{},
                             KeyEqual key_equal = KeyEqual{}) noexcept;

    /**
     * @brief Indicates whether the key `k` was inserted into the map.
     *
     * If the key `k` was inserted into the map, find returns
     * true. Otherwise, it returns false. Uses the CUDA Cooperative Groups API to
     * to leverage multiple threads to perform a single contains operation. This provides a
     * significant boost in throughput compared to the non Cooperative Group
     * `contains` at moderate to high load factors.
     *
     * @tparam CG Cooperative Group type
     * @tparam Hash Unary callable type
     * @tparam KeyEqual Binary callable type
     * @param g The Cooperative Group used to perform the contains operation
     * @param k The key to search for
     * @param hash The unary callable used to hash the key
     * @param key_equal The binary callable used to compare two keys
     * for equality
     * @return A boolean indicating whether the key/value pair
     * containing `k` was inserted
     */
    template <typename CG,
              typename Hash     = cuco::detail::MurmurHash3_32<key_type>,
              typename KeyEqual = thrust::equal_to<key_type>>
    __device__ bool contains(CG const& g,
                             Key const& k,
                             Hash hash          = Hash{},
                             KeyEqual key_equal = KeyEqual{}) noexcept;
  };  // class device_view

  /**
   * @brief Gets the maximum number of elements the hash map can hold.
   *
   * @return The maximum number of elements the hash map can hold
   */
  std::size_t get_capacity() const noexcept { return capacity_; }

  /**
   * @brief Gets the number of elements in the hash map.
   *
   * @return The number of elements in the map
   */
  std::size_t get_size() const noexcept { return size_; }

  /**
   * @brief Gets the load factor of the hash map.
   *
   * @return The load factor of the hash map
   */
  float get_load_factor() const noexcept { return static_cast<float>(size_) / capacity_; }

  /**
   * @brief Gets the sentinel value used to represent an empty key slot.
   *
   * @return The sentinel value used to represent an empty key slot
   */
  Key get_empty_key_sentinel() const noexcept { return empty_key_sentinel_; }

  /**
   * @brief Gets the sentinel value used to represent an empty value slot.
   *
   * @return The sentinel value used to represent an empty value slot
   */
  Value get_empty_value_sentinel() const noexcept { return empty_value_sentinel_; }

  /**
   * @brief Constructs a device_view object based on the members of the `static_reduction_map`
   * object.
   *
   * @return A device_view object based on the members of the `static_reduction_map` object
   */
  device_view get_device_view() const noexcept
  {
    return device_view(slots_, capacity_, empty_key_sentinel_, op_);
  }

  /**
   * @brief Constructs a device_mutable_view object based on the members of the
   * `static_reduction_map` object
   *
   * @return A device_mutable_view object based on the members of the `static_reduction_map` object
   */
  device_mutable_view get_device_mutable_view() const noexcept
  {
    return device_mutable_view(slots_, capacity_, empty_key_sentinel_, op_);
  }

 private:
  /// Unsafe access to the slots stripping away their atomic-ness to allow non-atomic access. This
  /// is a temporary solution until we have atomic_ref
  value_type* raw_slots_begin() noexcept { return reinterpret_cast<value_type*>(slots_); }

  value_type const* raw_slots_begin() const noexcept
  {
    return reinterpret_cast<value_type const*>(slots_);
  }

  value_type* raw_slots_end() noexcept { return raw_slots_begin() + get_capacity(); }

  value_type const* raw_slots_end() const noexcept { return raw_slots_begin() + get_capacity(); }

  pair_atomic_type* slots_{nullptr};            ///< Pointer to flat slots storage
  std::size_t capacity_{};                      ///< Total number of slots
  std::size_t size_{};                          ///< Number of keys in map
  Key empty_key_sentinel_{};                    ///< Key value that represents an empty slot
  Value empty_value_sentinel_{};                ///< Initial value of empty slot
  ReductionOp op_{};                            ///< Binary operation reduction function object
  slot_allocator_type slot_allocator_{};        ///< Allocator used to allocate slots
  counter_allocator_type counter_allocator_{};  ///< Allocator used to allocate counters
};
}  // namespace cuco

#include <cuco/detail/static_reduction_map.inl>