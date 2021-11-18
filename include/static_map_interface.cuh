
#include <cuco/allocator.hpp>
#include <cuco/detail/pair.cuh>

#include <cuda/std/atomic>

namespace cuda {
// Strong type wrapper for cudaStream_t
struct stream_view {
};
}  // namespace cuda

namespace cuco {

/// Strong type wrapper to indicate a provided value is a sentinel value
template <typename T>
struct sentinel {
  T value;
};

template <typename T>
struct key_sentinel : sentinel<T>;

template <typename T>
struct payload_sentinel : sentinel<T>;

template <typename Key,
          typename Payload,
          typename Hash,
          typename KeyEqual,
          cuda::thread_scope Scope = cuda::thread_scope_device,
          typename Allocator       = cuco::cuda_allocator<char>>
class static_map {
 public:
  using key_type     = Key;
  using payload_type = Payload;
  using value_type   = cuco::pair<key_type, payload_type>;

  // TODO: When 4 <= sizeof(key_type/value_type) <= 8 we want
  // to use `cuco::pair<key_type, payload_type>` as the storage type and use
  // atomic_ref. Otherwise, we want to use atomic<cuco::pair<key_type,
  // payload_type>>
  // using slot_type = ???

  // Allow specifying these in any order after `capacity`
  static_map(std::size_t capacity,
             key_sentinel<key_type> empty_key_sentinel,
             payload_sentinel<payload_type> empty_payload_sentinel,
             Hash hash,
             KeyEqual key_equal,
             Allocator const& alloc,
             stream_view s);

  /**
   * @brief Destroy the static_map.
   *
   */
  ~static_map();

  /**
   * @brief Returns the number of values in the map.
   *
   * @note: This function may invoke device operations and therefore takes a stream.
   *
   * @param s Stream on which all asynchronous device operations will be ordered
   * @return The number of keys in the map.
   */
  std::size_t size(cuda::stream_view s);

  /**
   * @brief Indicates if the map does not contain any values, i.e., `size() == 0`
   *
   * @note: This function may invoke device operations and therefore takes a stream.
   *
   * @param s Stream on which all asynchronous device operations will be ordered
   * @return Whether the map contains values
   */
  bool is_empty(cuda::stream_view s);

  /**
   * @brief Returns the reserved `key_type` value used to represent an empty slot.
   *
   * @return The `key_type` value representing an empty slot.
   */
  key_type empty_key_sentinel() const noexcept;

  /**
   * @brief Returns the reserved `payload_type` value used to represent an empty slot.
   *
   * @return The `payload_type` value representing an empty slot.
   */
  payload_type empty_payload_sentinel() const noexcept;

  /**
   * @brief Returns the maximum number of values the map can hold.
   *
   * @return The maximum number of values.
   */
  std::size_t capacity() const noexcept;

  /**
   * @brief Clears the contents of the map.
   * 
   * Erases any existing values and resets `size()` to `0`.
   *
   * @param s Stream on which all asynchronous device operations will be ordered
   */
  void clear(cuda::stream_view s);

  /**
   * @brief Reserves enough space for at least `new_capacity` values and reconstructs the hash table.
   * 
   * If `size() > 0`, existing values will be rehashed and inserted into the newly created storage.
   * 
   * Behavior is undefined if `new_capacity < size()`.
   *
   * @param new_capacity The desired capacity 
   * @param s Stream on which all asynchronous device operations will be ordered
   */
  void reserve(std::size_t new_capacity, cuda::stream_view s);

  /**
   * @brief Inserts the contents of `[first, last)` into the map.
   *
   * If any two keys compare equivalent, it is unspecified which is inserted.
   *
   * @tparam PairInputIt Device accessible input iterator whose `value_type` is convertible to
   * `static_map::value_type`.
   * @param first Beginning of the sequence of keys/payloads to insert
   * @param last End of the sequence of keys/payloads to insert
   * @param s Stream on which all asynchronous device operations will be ordered
   */
  template <typename PairInputIt>
  void insert(PairInputIt first, PairInputIt last, cuda::stream_view s);

  /**
   * @brief Inserts the contents of `[first, last)` into the map and indicates the result of each
   * insert.
   *
   * If inserting the key `k` and payload `p` from `*(first+i)` succeeds (i.e., no key equivalent to
   * `k` was already present), stores `*(first_payload_output + i) = p` and
   * `*(first_status_output + i) = true`. Otherwise, if an equivalent key was present, stores the
   * existing payload `p'` `*(first_payload_output + i) = p'` and `*(first_status_output + i) =
   * false`.
   *
   * @tparam PairInputIt Device accessible input iterator whose `value_type` is convertible to
   * `static_map::value_type`.
   * @tparam PayloadOutputIt Device accessible output iterator whose `value_type` is constructible
   * from `payload_type`
   * @tparam StatusOutputIt Device accessible output iterator whose `value_type` is constructible
   * from `bool`
   * @param first Beginning of the sequence of keys/payloads to insert
   * @param last End of the sequence of keys/payloads to insert
   * @param first_payload_output Beginning of the output sequence of inserted or existing payload
   * values
   * @param first_status_output Beginning of the output sequence indicating the success of each
   * insert
   * @param s Stream on which all asynchronous device operations will be ordered
   */
  template <typename PairInputIt, typename PayloadOutputIt, typename StatusOutputIt>
  void insert(PairInputIt first,
              PairInputIt last,
              PayloadOutputIt first_payload_output,
              StatusOutputIt first_status_output,
              cuda::stream_view s);

  /**
   * @brief Inserts the keys from `[first_key, last_key)` with associated payload in
   * `[first_payload, first_payload + std::distance(first_key, last_key))`.
   *
   * If any two keys compare equivalent, it is unspecified which is inserted.
   *
   * Equivalent to:
   * \code{.cpp}
   * auto zipped = thrust::make_zip_iterator(thrust::make_tuple(first_key, first_payload));
   * insert(zipped, zipped + std::distance(first_key, last_key));
   * \endcode
   *
   * @tparam KeyInputIt Device accessible input iterator whose `value_type` is convertible to
   * `key_type`
   * @tparam PayloadInputIt Device accessible input iterator whose `value_type` is convertible to
   * `payload_type`
   * @param first_key Beginning of the sequence of keys to insert
   * @param last_key End of the sequence of keys to insert
   * @param first_payload Beginning of the sequence of associated payloads to insert
   * @param s Stream on which all asynchronous device operations will be ordered
   */
  template <typename KeyInputIt, typename PayloadInputIt>
  void insert(KeyInputIt first_key,
              KeyInputIt last_key,
              PayloadInputIt first_payload,
              cuda::stream_view s);

  /**
   * @brief Inserts the keys from `[first_key, last_key)` with associated payload in
   * `[first_payload, first_payload + std::distance(first_key, last_key))` and indicates the result
   * of each insert.
   *
   * If inserting the key `k` and payload `p` from `*(first_key+i)` and `*(first_payload + i)`
   * succeeds (i.e., no key equivalent to `k` was already present), stores `*(first_payload_output +
   * i) = p` and  `*(first_status_output + i) = true`. Otherwise, if an equivalent key was present,
   * stores the existing payload `p'` `*(first_payload_output + i) = p'` and `*(first_status_output
   * + i) = false`.
   *
   * @tparam KeyInputIt Device accessible input iterator whose `value_type` is convertible to
   * `key_type`
   * @tparam PayloadInputIt Device accessible input iterator whose `value_type` is convertible to
   * `payload_type`
   * @tparam PayloadOutputIt Device accessible output iterator whose `value_type` is constructible
   * from `payload_type`
   * @tparam StatusOutputIt Device accessible output iterator whose `value_type` is constructible
   * from `bool`
   * @param first_key Beginning of the sequence of keys to insert
   * @param last_key End of the sequence of keys to insert
   * @param first_payload Beginning of the sequence of associated payloads to insert
   * @param first_payload_output Beginning of the output sequence of inserted or existing payload
   * values
   * @param first_status_output Beginning of the output sequence indicating the success of each
   * insert
   * @param s Stream on which all asynchronous device operations will be ordered
   */
  template <typename KeyInputIt,
            typename PayloadInputIt,
            typename PayloadOutputIt,
            typename StatusOutputIt>
  void insert(KeyInputIt first_key,
              KeyInputIt last_key,
              PayloadInputIt first_payload,
              PayloadOutputIt first_payload_output,
              StatusOutputIt first_status_output,
              cuda::stream_view s);

  /**
   * @brief Indicates if the keys in `[first, last)` are present in the map.
   *
   * Stores `true/false` to `*(output_begin + i)` if a key equivalent to `*(first + i)` does/does
   * not exist in the map.
   *
   * @tparam KeyInputIt Device accessible input iterator whose `value_type` is convertible to
   * the argument types of `KeyEqual`.
   * @tparam OutputIt Device accessible output iterator whose `value_type` is constructible from
   * `bool`.
   * @param first Beginning of the sequence of keys to search for
   * @param last End of the sequence of keys to search for
   * @param output_begin Beginning of the output sequence indicating the presence of each key.
   * @param s Stream on which all asynchronous device operations will be ordered
   */
  template <typename KeyInputIt, typename OutputIt>
  void is_present(KeyInputIt first, KeyInputIt last, OutputIt output_begin, cuda::stream_view s);

  /**
   * @brief Indicates if the keys in `[first, last)` are absent from the map.
   *
   * Stores `false/true` to `*(output_begin + i)` if a key equivalent to `*(first + i)`  does/does
   * not exist in the map.
   *
   * @tparam KeyInputIt Device accessible input iterator whose `value_type` is convertible to
   * the argument types of `KeyEqual`.
   * @tparam OutputIt Device accessible output iterator whose `value_type` is constructible from
   * `bool`.
   * @param first Beginning of the sequence of keys to search for
   * @param last End of the sequence of keys to search for
   * @param output_begin Beginning of the output sequence indicating the absence of each key.
   * @param s Stream on which all asynchronous device operations will be ordered
   */
  template <typename KeyInputIt, typename OutputIt>
  void is_absent(KeyInputIt first, KeyInputIt last, OutputIt output_begin, cuda::stream_view s);

  /**
   * @brief Applies `F` to every key/payload in the map.
   *
   * Behavior is undefined if the size of the sequence beginning at `output_begin` is less than
   * `size()`.
   *
   * @tparam UnaryFunction Device unary callabe whose argument type is constructible from
   * `value_type`.
   * @tparam OutputIt Device accessible output iterator whose `value_type` is constructible from the
   * result of `F`.
   * @param F The callable to invoke on each key/payload in the map
   * @param s Stream on which all asynchronous device operations will be ordered
   */
  template <typename UnaryFunction, typename OutputIt>
  void for_each(UnaryFunction F, OutputIt output_begin, cuda::stream_view s);
};

}  // namespace cuco