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

#include "ast_test_support.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <expression_evaluator/decimal_to_integer.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/gpu_expression_translator_internal.hpp>
#include <sirius/exception.hpp>

#include <limits>

using namespace sirius::expr_test;

TEST_CASE("decimal integer kernel handles slices nulls empty input and storage edges",
          "[expression_evaluator][decimal_cast]")
{
  rmm::cuda_stream stream;
  auto mr     = cudf::get_current_device_resource_ref();
  auto column = cudf::make_fixed_point_column(
    cudf::data_type{cudf::type_id::DECIMAL64, -2}, 70, cudf::mask_state::ALL_VALID, stream, mr);
  std::vector<int64_t> raw(70, -150);
  raw[0]  = 999999;
  raw[33] = std::numeric_limits<int64_t>::max();  // NULL payload must not overflow.
  raw[34] = 12750;
  raw[35] = -12849;
  raw[36] = -12850;
  REQUIRE(cudaMemcpyAsync(column->mutable_view().data<int64_t>(),
                          raw.data(),
                          raw.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  cudf::set_null_mask(column->mutable_view().null_mask(), 33, 34, false, stream);
  column->set_null_count(1);
  auto view = cudf::slice(column->view(), {1, 69}, stream).front();
  auto result =
    sirius::cast_decimal_to_integer(view, cudf::data_type{cudf::type_id::INT8}, true, stream, mr);
  auto values = copy_column_to_host<int8_t>(result->view());
  auto valid  = copy_valids_to_host(result->view());
  REQUIRE(result->size() == 68);
  REQUIRE(result->null_count() == 3);
  for (size_t i = 0; i < values.size(); ++i) {
    CAPTURE(i);
    REQUIRE(valid[i] == (i != 32 && i != 33 && i != 35));
    if (valid[i]) { REQUIRE(values[i] == (i == 34 ? -128 : -2)); }
  }
  REQUIRE_THROWS_AS(
    sirius::cast_decimal_to_integer(view, cudf::data_type{cudf::type_id::INT8}, false, stream, mr),
    sirius::invalid_input_exception);
  auto empty = cudf::slice(column->view(), {1, 1}, stream).front();
  REQUIRE(
    sirius::cast_decimal_to_integer(empty, cudf::data_type{cudf::type_id::INT8}, false, stream, mr)
      ->size() == 0);
  cudf::set_null_mask(column->mutable_view().null_mask(), 0, column->size(), false, stream);
  column->set_null_count(column->size());
  auto all_null = sirius::cast_decimal_to_integer(
    column->view(), cudf::data_type{cudf::type_id::INT8}, false, stream, mr);
  REQUIRE(all_null->null_count() == column->size());

  // Widening raw carriers before rounding avoids overflowing int64 at either edge.
  raw        = {std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::lowest()};
  auto edges = cudf::make_fixed_point_column(
    cudf::data_type{cudf::type_id::DECIMAL64, -1}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
  REQUIRE(cudaMemcpyAsync(edges->mutable_view().data<int64_t>(),
                          raw.data(),
                          raw.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  auto rounded = sirius::cast_decimal_to_integer(
    edges->view(), cudf::data_type{cudf::type_id::INT64}, false, stream, mr);
  REQUIRE(copy_column_to_host<int64_t>(rounded->view()) ==
          std::vector<int64_t>{922337203685477581LL, -922337203685477581LL});
}

TEST_CASE("decimal carrier restoration stays separate from semantic integer casts",
          "[expression_evaluator][decimal_cast]")
{
  using sirius::logical_type;
  using sirius::type_id;
  auto const decimal32 = logical_type::make_decimal(9, 2);
  auto const decimal64 = logical_type::make_decimal(18, 2);
  auto const bigint    = logical_type::make(type_id::BIGINT);
  auto stream          = cudf::get_default_stream();
  auto mr              = cudf::get_current_device_resource_ref();
  auto column          = cudf::make_fixed_point_column(
    cudf::data_type{cudf::type_id::DECIMAL32, -2}, 3, cudf::mask_state::UNALLOCATED, stream, mr);
  std::vector<int32_t> raw{150, -150, 249};
  REQUIRE(cudaMemcpy(column->mutable_view().data<int32_t>(),
                     raw.data(),
                     raw.size() * sizeof(int32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  for (auto strategy : {sirius::expression_evaluator_strategy::MATERIALIZE,
                        sirius::expression_evaluator_strategy::AST_INTERPRET,
                        sirius::expression_evaluator_strategy::AST_JIT}) {
    auto restore = std::make_unique<ast_node>(sirius::ast::cast{
      make_ref_typed(0, decimal32), decimal64, false, sirius::ast::cast_kind::carrier_restore});
    sirius::expression_evaluator restore_evaluator(*restore, mr, stream, strategy, 0);
    auto restored = restore_evaluator.evaluate(cudf::table_view{{column->view()}});
    REQUIRE(restored->view().column(0).type() == cudf::data_type{cudf::type_id::DECIMAL64, -2});
    REQUIRE(copy_column_to_host<int64_t>(restored->view().column(0)) ==
            std::vector<int64_t>{150, -150, 249});

    auto semantic = make_cast(std::move(restore), bigint, false);
    REQUIRE(semantic->cudf_ast_op_count() == 0);
    sirius::expression_evaluator evaluator(*semantic, mr, stream, strategy, 0);
    auto output = evaluator.evaluate(cudf::table_view{{column->view()}});
    REQUIRE(copy_column_to_host<int64_t>(output->view().column(0)) ==
            std::vector<int64_t>{2, -2, 2});
    sirius::gpu_expression_translator translator(stream, mr);
    REQUIRE_FALSE(translator.translate_expression(*semantic).has_value());
  }
}
