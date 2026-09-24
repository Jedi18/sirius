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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace sirius {

/// SQL decimal-to-integer conversion: nearest integer, ties away from zero, checked range.
/// Supports DECIMAL32/64/128 with scales [-38, 0] and signed/unsigned 8–64-bit targets.
/// NULL inputs remain NULL. Overflow throws for CAST, or produces NULL for TRY_CAST.
std::unique_ptr<cudf::column> cast_decimal_to_integer(cudf::column_view input,
                                                      cudf::data_type target,
                                                      bool try_cast,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr);

}  // namespace sirius
