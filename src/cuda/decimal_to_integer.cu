/*
 * Copyright 2026, Sirius Contributors.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_scalar.hpp>

#include <expression_evaluator/decimal_to_integer.hpp>
#include <sirius/exception.hpp>

#include <cstdint>
#include <limits>

namespace sirius {
namespace {

constexpr int block_size = 256;

template <typename Rep, typename Target>
__global__ void decimal_to_integer_kernel(Rep const* input,
                                          cudf::bitmask_type const* input_mask,
                                          cudf::size_type offset,
                                          cudf::size_type size,
                                          __int128_t divisor,
                                          Target* output,
                                          cudf::bitmask_type* output_mask,
                                          int* null_count,
                                          int* overflow)
{
  auto const row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= size) { return; }
  bool valid  = !input_mask || cudf::bit_is_set(input_mask, offset + row);
  output[row] = 0;
  if (valid) {
    auto const value     = static_cast<__int128_t>(input[row]);
    auto rounded         = value / divisor;
    auto const remainder = value % divisor;
    // Divide first: adding half the divisor to the original decimal can overflow its
    // storage representation. The remainder comparison is exact even at scale 38.
    if (divisor > 1) {
      auto const half = divisor / 2;
      if (remainder >= half) { ++rounded; }
      if (remainder <= -half) { --rounded; }
    }
    valid = rounded >= static_cast<__int128_t>(std::numeric_limits<Target>::lowest()) &&
            rounded <= static_cast<__int128_t>(std::numeric_limits<Target>::max());
    if (valid) {
      output[row] = static_cast<Target>(rounded);
    } else {
      atomicExch(overflow, 1);
    }
  }
  if (!valid) {
    atomicAnd(output_mask + row / 32, ~(cudf::bitmask_type{1} << (row % 32)));
    atomicAdd(null_count, 1);
  }
}

template <typename Rep, typename Target>
std::unique_ptr<cudf::column> convert(cudf::column_view input,
                                      cudf::data_type target,
                                      bool try_cast,
                                      __int128_t divisor,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  auto result =
    cudf::make_numeric_column(target, input.size(), cudf::mask_state::ALL_VALID, stream, mr);
  if (input.is_empty()) { return result; }
  rmm::device_scalar<int> null_count(0, stream, mr);
  rmm::device_scalar<int> overflow(0, stream, mr);
  decimal_to_integer_kernel<Rep, Target>
    <<<(static_cast<int64_t>(input.size()) + block_size - 1) / block_size,
       block_size,
       0,
       stream.value()>>>(input.data<Rep>(),
                         input.null_mask(),
                         input.offset(),
                         input.size(),
                         divisor,
                         result->mutable_view().data<Target>(),
                         result->mutable_view().null_mask(),
                         null_count.data(),
                         overflow.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  result->set_null_count(null_count.value(stream));
  if (!try_cast && overflow.value(stream)) {
    throw invalid_input_exception("Decimal-to-integer cast out of range");
  }
  return result;
}

template <typename Rep>
std::unique_ptr<cudf::column> dispatch_target(cudf::column_view input,
                                              cudf::data_type target,
                                              bool try_cast,
                                              __int128_t divisor,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  switch (target.id()) {
    case cudf::type_id::INT8:
      return convert<Rep, int8_t>(input, target, try_cast, divisor, stream, mr);
    case cudf::type_id::INT16:
      return convert<Rep, int16_t>(input, target, try_cast, divisor, stream, mr);
    case cudf::type_id::INT32:
      return convert<Rep, int32_t>(input, target, try_cast, divisor, stream, mr);
    case cudf::type_id::INT64:
      return convert<Rep, int64_t>(input, target, try_cast, divisor, stream, mr);
    case cudf::type_id::UINT8:
      return convert<Rep, uint8_t>(input, target, try_cast, divisor, stream, mr);
    case cudf::type_id::UINT16:
      return convert<Rep, uint16_t>(input, target, try_cast, divisor, stream, mr);
    case cudf::type_id::UINT32:
      return convert<Rep, uint32_t>(input, target, try_cast, divisor, stream, mr);
    case cudf::type_id::UINT64:
      return convert<Rep, uint64_t>(input, target, try_cast, divisor, stream, mr);
    default: throw invalid_input_exception("Expected an 8–64-bit integer target");
  }
}

}  // namespace

std::unique_ptr<cudf::column> cast_decimal_to_integer(cudf::column_view input,
                                                      cudf::data_type target,
                                                      bool try_cast,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto const scale = input.type().scale();
  if (scale > 0 || scale < -38) {
    throw invalid_input_exception("Decimal-to-integer cast requires a SQL decimal scale");
  }
  __int128_t divisor = 1;
  for (int i = 0; i < -scale; ++i) {
    divisor *= 10;
  }
  switch (input.type().id()) {
    case cudf::type_id::DECIMAL32:
      return dispatch_target<int32_t>(input, target, try_cast, divisor, stream, mr);
    case cudf::type_id::DECIMAL64:
      return dispatch_target<int64_t>(input, target, try_cast, divisor, stream, mr);
    case cudf::type_id::DECIMAL128:
      return dispatch_target<__int128_t>(input, target, try_cast, divisor, stream, mr);
    default: throw invalid_input_exception("Expected a decimal source column");
  }
}

}  // namespace sirius
