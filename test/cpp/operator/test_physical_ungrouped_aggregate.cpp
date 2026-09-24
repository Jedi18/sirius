/*
 * Copyright 2025, Sirius Contributors.
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

#include "helper/type_conversions.hpp"
#include "operator_test_utils.hpp"
#include "operator_type_traits.hpp"

#include <catch.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <expression/ast/node.hpp>
#include <op/sirius_physical_grouped_aggregate_merge.hpp>
#include <op/sirius_physical_ungrouped_aggregate.hpp>
#include <op/sirius_physical_ungrouped_aggregate_merge.hpp>

#include <cstdint>
#include <iterator>
#include <memory>

using namespace duckdb;
using namespace sirius::op;
using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::test::operator_utils;

namespace {
inline uint64_t int128_low64(__int128_t value)
{
  return static_cast<uint64_t>(static_cast<unsigned __int128>(value));
}

inline int64_t int128_high64(__int128_t value)
{
  return static_cast<int64_t>(static_cast<unsigned __int128>(value) >> 64);
}

// Translate a vector of DuckDB expressions into Sirius AST nodes (size/order
// preserved, null slot for an unsupported shape).
inline duckdb::vector<std::unique_ptr<sirius::ast::node>> translate_expressions(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> out;
  out.reserve(exprs.size());
  for (auto& e : exprs) {
    out.push_back(e ? sirius::ast::from_duckdb(*e) : nullptr);
  }
  return out;
}
}  // namespace

// Helper to create a dummy AggregateFunction since we only need the name and types for the GPU
// operator
AggregateFunction MakeDummyAggregate(const std::string& name,
                                     const duckdb::vector<LogicalType>& args,
                                     const LogicalType& ret_type)
{
  return AggregateFunction(
    name, args, ret_type, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
}

TEMPLATE_TEST_CASE("sirius_physical_ungrouped_aggregate computes SUM/MIN/MAX/COUNT",
                   "[physical_ungrouped_aggregate]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   decimal64_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  // Create values for batches
  auto vals = Traits::sample_values();
  // Ensure we have at least 4 values to split across 2 batches
  while (vals.size() < 4) {
    vals.insert(vals.end(), vals.begin(), vals.end());
  }
  if (vals.size() > 4) { vals.resize(4); }

  std::vector<typename Traits::type> batch1_vals(vals.begin(), vals.begin() + 2);
  std::vector<typename Traits::type> batch2_vals(vals.begin() + 2, vals.begin() + 4);

  std::shared_ptr<data_batch> b1, b2;

  if constexpr (Traits::is_decimal) {
    b1 = make_decimal64_batch(*space, batch1_vals, Traits::scale);
    b2 = make_decimal64_batch(*space, batch2_vals, Traits::scale);
  } else {
    b1 = make_numeric_batch<typename Traits::type>(*space, batch1_vals, Traits::cudf_type);
    b2 = make_numeric_batch<typename Traits::type>(*space, batch2_vals, Traits::cudf_type);
  }

  // 1. SUM(col0)
  // 2. MIN(col0)
  // 3. MAX(col0)
  // 4. COUNT(col0)
  // 5. COUNT(*)

  auto make_aggregates = [&](duckdb::vector<duckdb::LogicalType>& ret_types) {
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;

    // SUM
    {
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
      children.push_back(make_uniq<BoundReferenceExpression>(Traits::logical_type(), 0));
      aggregates.push_back(make_uniq<BoundAggregateExpression>(
        MakeDummyAggregate("sum", {Traits::logical_type()}, Traits::logical_type()),
        std::move(children),
        nullptr,
        nullptr,
        AggregateType::NON_DISTINCT));
      ret_types.push_back(Traits::logical_type());
    }

    // MIN
    {
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
      children.push_back(make_uniq<BoundReferenceExpression>(Traits::logical_type(), 0));
      aggregates.push_back(make_uniq<BoundAggregateExpression>(
        MakeDummyAggregate("min", {Traits::logical_type()}, Traits::logical_type()),
        std::move(children),
        nullptr,
        nullptr,
        AggregateType::NON_DISTINCT));
      ret_types.push_back(Traits::logical_type());
    }

    // MAX
    {
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
      children.push_back(make_uniq<BoundReferenceExpression>(Traits::logical_type(), 0));
      aggregates.push_back(make_uniq<BoundAggregateExpression>(
        MakeDummyAggregate("max", {Traits::logical_type()}, Traits::logical_type()),
        std::move(children),
        nullptr,
        nullptr,
        AggregateType::NON_DISTINCT));
      ret_types.push_back(Traits::logical_type());
    }

    // COUNT
    {
      duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
      children.push_back(make_uniq<BoundReferenceExpression>(Traits::logical_type(), 0));
      aggregates.push_back(make_uniq<BoundAggregateExpression>(
        MakeDummyAggregate(
          "count", {Traits::logical_type()}, LogicalType(duckdb::LogicalTypeId::BIGINT)),
        std::move(children),
        nullptr,
        nullptr,
        AggregateType::NON_DISTINCT));
      ret_types.push_back(LogicalType(duckdb::LogicalTypeId::BIGINT));
    }

    // COUNT_STAR
    {
      aggregates.push_back(make_uniq<BoundAggregateExpression>(
        MakeDummyAggregate("count_star", {}, LogicalType(duckdb::LogicalTypeId::BIGINT)),
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
        nullptr,
        nullptr,
        AggregateType::NON_DISTINCT));
      ret_types.push_back(LogicalType(duckdb::LogicalTypeId::BIGINT));
    }

    return aggregates;
  };

  duckdb::vector<duckdb::LogicalType> local_types;
  auto local_aggregates = make_aggregates(local_types);
  duckdb::vector<duckdb::LogicalType> merge_types;
  auto merge_aggregates = make_aggregates(merge_types);

  sirius_physical_ungrouped_aggregate local_op(
    sirius::from_duckdb_vec(local_types),
    translate_expressions(std::move(local_aggregates)),
    0,
    duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
  sirius_physical_ungrouped_aggregate_merge merge_op(
    sirius::from_duckdb_vec(merge_types),
    translate_expressions(std::move(merge_aggregates)),
    0,
    duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);

  auto local_out1 = local_op.execute(pipelineable_operator_data({b1}), cudf::get_default_stream());
  auto local_out2 = local_op.execute(pipelineable_operator_data({b2}), cudf::get_default_stream());
  auto local_out1_batches =
    dynamic_cast<const pipelineable_operator_data&>(*local_out1).get_data_batches();
  auto local_out2_batches =
    dynamic_cast<const pipelineable_operator_data&>(*local_out2).get_data_batches();
  std::vector<std::shared_ptr<data_batch>> merge_inputs;
  merge_inputs.insert(merge_inputs.end(),
                      std::make_move_iterator(local_out1_batches.begin()),
                      std::make_move_iterator(local_out1_batches.end()));
  merge_inputs.insert(merge_inputs.end(),
                      std::make_move_iterator(local_out2_batches.begin()),
                      std::make_move_iterator(local_out2_batches.end()));
  auto out = merge_op.execute(pipelineable_operator_data(merge_inputs), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);

  REQUIRE(view.num_columns() == 5);
  REQUIRE(view.num_rows() == 1);

  // Verify
  // SUM may widen the output type (e.g. DECIMAL64 -> DECIMAL128), so read with agg_output_type.
  // MIN/MAX are not widened for decimals, so read with min_max_output_type (= type for decimals).
  auto sum_out        = copy_column_to_host<typename Traits::agg_output_type>(view.column(0));
  auto min_out        = copy_column_to_host<typename Traits::min_max_output_type>(view.column(1));
  auto max_out        = copy_column_to_host<typename Traits::min_max_output_type>(view.column(2));
  auto count_out      = copy_column_to_host<int64_t>(view.column(3));
  auto count_star_out = copy_column_to_host<int64_t>(view.column(4));

  // Compute expected SUM in agg_output_type (DECIMAL64 SUM upcasts to DECIMAL128).
  // Compute expected MIN/MAX in min_max_output_type (DECIMAL64 stays DECIMAL64).
  typename Traits::agg_output_type expected_sum = 0;
  typename Traits::min_max_output_type expected_min =
    static_cast<typename Traits::min_max_output_type>(vals[0]);
  typename Traits::min_max_output_type expected_max =
    static_cast<typename Traits::min_max_output_type>(vals[0]);

  for (auto v : vals) {
    expected_sum += static_cast<typename Traits::agg_output_type>(v);
    auto v_mm = static_cast<typename Traits::min_max_output_type>(v);
    if (v_mm < expected_min) expected_min = v_mm;
    if (v_mm > expected_max) expected_max = v_mm;
  }

  // Approximate check for floats
  if constexpr (std::is_floating_point_v<typename Traits::type>) {
    REQUIRE(sum_out[0] == Approx(static_cast<typename Traits::type>(expected_sum)));
    REQUIRE(min_out[0] == Approx(static_cast<typename Traits::type>(expected_min)));
    REQUIRE(max_out[0] == Approx(static_cast<typename Traits::type>(expected_max)));
  } else if constexpr (std::is_same_v<typename Traits::agg_output_type, __int128_t>) {
    // SUM output is DECIMAL128 (__int128_t).
    REQUIRE(int128_high64(sum_out[0]) == int128_high64(expected_sum));
    REQUIRE(int128_low64(sum_out[0]) == int128_low64(expected_sum));
    // MIN/MAX output is DECIMAL64 (int64_t) — not widened.
    REQUIRE(min_out[0] == expected_min);
    REQUIRE(max_out[0] == expected_max);
  } else {
    REQUIRE(sum_out[0] == expected_sum);
    REQUIRE(min_out[0] == expected_min);
    REQUIRE(max_out[0] == expected_max);
  }

  REQUIRE(count_out[0] == 4);
  REQUIRE(count_star_out[0] == 4);
}

TEMPLATE_TEST_CASE("sirius_physical_ungrouped_aggregate resolves AVG in merge",
                   "[physical_ungrouped_aggregate]",
                   int32_t,
                   int64_t,
                   float,
                   double,
                   decimal64_tag)
{
  using Traits = gpu_type_traits<TestType>;

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);

  auto vals = Traits::sample_values();
  while (vals.size() < 4) {
    vals.insert(vals.end(), vals.begin(), vals.end());
  }
  if (vals.size() > 4) { vals.resize(4); }

  std::vector<typename Traits::type> batch1_vals(vals.begin(), vals.begin() + 2);
  std::vector<typename Traits::type> batch2_vals(vals.begin() + 2, vals.begin() + 4);

  std::shared_ptr<data_batch> b1, b2;
  if constexpr (Traits::is_decimal) {
    b1 = make_decimal64_batch(*space, batch1_vals, Traits::scale);
    b2 = make_decimal64_batch(*space, batch2_vals, Traits::scale);
  } else {
    b1 = make_numeric_batch<typename Traits::type>(*space, batch1_vals, Traits::cudf_type);
    b2 = make_numeric_batch<typename Traits::type>(*space, batch2_vals, Traits::cudf_type);
  }

  auto make_avg_aggregates = [&](duckdb::vector<duckdb::LogicalType>& ret_types) {
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
    children.push_back(make_uniq<BoundReferenceExpression>(Traits::logical_type(), 0));
    auto return_type =
      Traits::is_decimal ? Traits::logical_type() : LogicalType(duckdb::LogicalTypeId::DOUBLE);
    aggregates.push_back(make_uniq<BoundAggregateExpression>(
      MakeDummyAggregate("avg", {Traits::logical_type()}, return_type),
      std::move(children),
      nullptr,
      nullptr,
      AggregateType::NON_DISTINCT));
    ret_types.push_back(return_type);
    return aggregates;
  };

  duckdb::vector<duckdb::LogicalType> local_types;
  auto local_aggregates = make_avg_aggregates(local_types);
  local_types.push_back(LogicalType(duckdb::LogicalTypeId::BIGINT));
  duckdb::vector<duckdb::LogicalType> merge_types;
  auto merge_aggregates = make_avg_aggregates(merge_types);

  sirius_physical_ungrouped_aggregate local_op(
    sirius::from_duckdb_vec(local_types),
    translate_expressions(std::move(local_aggregates)),
    0,
    duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
  sirius_physical_ungrouped_aggregate_merge merge_op(
    sirius::from_duckdb_vec(merge_types),
    translate_expressions(std::move(merge_aggregates)),
    0,
    duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);

  auto local_out1 = local_op.execute(pipelineable_operator_data({b1}), cudf::get_default_stream());
  auto local_out2 = local_op.execute(pipelineable_operator_data({b2}), cudf::get_default_stream());
  auto local_out1_batches =
    dynamic_cast<const pipelineable_operator_data&>(*local_out1).get_data_batches();
  auto local_out2_batches =
    dynamic_cast<const pipelineable_operator_data&>(*local_out2).get_data_batches();
  std::vector<std::shared_ptr<data_batch>> merge_inputs;
  merge_inputs.insert(merge_inputs.end(),
                      std::make_move_iterator(local_out1_batches.begin()),
                      std::make_move_iterator(local_out1_batches.end()));
  merge_inputs.insert(merge_inputs.end(),
                      std::make_move_iterator(local_out2_batches.begin()),
                      std::make_move_iterator(local_out2_batches.end()));

  auto out = merge_op.execute(pipelineable_operator_data(merge_inputs), cudf::get_default_stream());
  REQUIRE(dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches().size() == 1);

  auto view = sirius::get_cudf_table_view(
    *dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()[0]);
  REQUIRE(view.num_columns() == 1);
  REQUIRE(view.num_rows() == 1);

  if constexpr (Traits::is_decimal) {
    REQUIRE(view.column(0).type() == cudf::data_type{cudf::type_id::DECIMAL64, Traits::scale});
    auto avg_out                       = copy_column_to_host<typename Traits::type>(view.column(0));
    typename Traits::type expected_sum = 0;
    for (auto v : vals) {
      expected_sum += v;
    }
    // The sample values divide exactly in their fixed-point representation.
    REQUIRE(avg_out[0] == expected_sum / static_cast<typename Traits::type>(vals.size()));
  } else {
    auto avg_out        = copy_column_to_host<double>(view.column(0));
    double expected_sum = 0.0;
    for (auto v : vals) {
      expected_sum += static_cast<double>(v);
    }
    double expected_avg = expected_sum / static_cast<double>(vals.size());
    REQUIRE(avg_out[0] == Approx(expected_avg));
  }
}

// Use synthetic local partials to exercise the >INT64 merge domain without
// materializing billions of INT32 rows. Each partial is an exact sum of n
// copies of +/- INT32_MAX, the narrow-integer AVG planner's supported domain.
// The total count also exceeds INT64, exercising the unsigned denominator.
TEST_CASE("integer AVG decimal partials remain exact across multiple batches",
          "[physical_ungrouped_aggregate][integer_aggregate]")
{
  auto memory_manager = initialize_memory_manager();
  auto* space         = memory_manager->get_memory_space(Tier::GPU, 0);
  REQUIRE(space);
  auto stream              = default_stream();
  auto mr                  = get_resource_ref(*space);
  auto decimal_type        = LogicalType::DECIMAL(38, 0);
  constexpr int64_t n      = int64_t{1} << 62;
  constexpr int64_t value  = 2147483647;
  const __int128_t partial = static_cast<__int128_t>(n) * value;

  auto decimal_column = [&](const __int128_t v) {
    auto col = cudf::make_fixed_point_column(
      cudf::data_type{cudf::type_id::DECIMAL128, 0}, 1, cudf::mask_state::UNALLOCATED, stream, mr);
    REQUIRE(
      cudaMemcpy(col->mutable_view().data<__int128_t>(), &v, sizeof(v), cudaMemcpyHostToDevice) ==
      cudaSuccess);
    return col;
  };
  auto int_column = [&](int64_t v, cudf::type_id type = cudf::type_id::INT64) {
    auto col = cudf::make_numeric_column(
      cudf::data_type{type}, 1, cudf::mask_state::UNALLOCATED, stream, mr);
    REQUIRE(
      cudaMemcpy(col->mutable_view().data<int64_t>(), &v, sizeof(v), cudaMemcpyHostToDevice) ==
      cudaSuccess);
    return col;
  };
  // Also verify the local reduction writes a DECIMAL128 partial and declares
  // the same schema, so no information is lost before the merge tests below.
  {
    const std::vector<__int128_t> values{partial, partial};
    auto column = cudf::make_fixed_point_column(
      cudf::data_type{cudf::type_id::DECIMAL128, 0}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
    REQUIRE(cudaMemcpy(column->mutable_view().data<__int128_t>(),
                       values.data(),
                       values.size() * sizeof(__int128_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(column));
    auto input = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                         *space,
                                         stream,
                                         sirius::telemetry::batch_telemetry_info{});
    duckdb::vector<duckdb::unique_ptr<Expression>> arguments;
    arguments.push_back(make_uniq<BoundReferenceExpression>(decimal_type, 0));
    duckdb::vector<duckdb::unique_ptr<Expression>> expressions;
    expressions.push_back(make_uniq<BoundAggregateExpression>(
      MakeDummyAggregate("avg", {decimal_type}, LogicalType::DOUBLE),
      std::move(arguments),
      nullptr,
      nullptr,
      AggregateType::NON_DISTINCT));
    sirius_physical_ungrouped_aggregate local(
      sirius::from_duckdb_vec(duckdb::vector<LogicalType>{LogicalType::DOUBLE}),
      translate_expressions(std::move(expressions)),
      2,
      TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
    REQUIRE(sirius::to_duckdb(local.get_local_output_types()[0]) == decimal_type);
    auto result = local.execute(pipelineable_operator_data({input}), stream);
    auto const& batches =
      dynamic_cast<const pipelineable_operator_data&>(*result).get_data_batches();
    REQUIRE(batches.size() == 1);
    auto view = sirius::get_cudf_table_view(*batches[0]);
    REQUIRE(view.column(0).type().id() == cudf::type_id::DECIMAL128);
    REQUIRE(copy_column_to_host<__int128_t>(view.column(0))[0] == 2 * partial);
    REQUIRE(view.column(1).type().id() == cudf::type_id::UINT64);
    REQUIRE(copy_column_to_host<uint64_t>(view.column(1))[0] == 2);
  }
  for (bool grouped : {false, true}) {
    INFO("grouped=" << grouped);
    duckdb::vector<duckdb::unique_ptr<Expression>> expressions;
    duckdb::vector<duckdb::unique_ptr<Expression>> arguments;
    arguments.push_back(make_uniq<BoundReferenceExpression>(decimal_type, grouped ? 1 : 0));
    expressions.push_back(make_uniq<BoundAggregateExpression>(
      MakeDummyAggregate("avg", {decimal_type}, LogicalType::DOUBLE),
      std::move(arguments),
      nullptr,
      nullptr,
      AggregateType::NON_DISTINCT));

    std::vector<std::shared_ptr<data_batch>> partials;
    for (int sign : {1, 1, -1}) {
      std::vector<std::unique_ptr<cudf::column>> columns;
      if (grouped) { columns.push_back(int_column(7)); }
      columns.push_back(decimal_column(sign * partial));
      columns.push_back(int_column(n, cudf::type_id::UINT64));
      partials.push_back(sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                                 *space,
                                                 stream,
                                                 sirius::telemetry::batch_telemetry_info{}));
    }
    REQUIRE(partials.size() == 3);
    std::unique_ptr<operator_data> result;
    if (grouped) {
      duckdb::vector<std::unique_ptr<sirius::ast::node>> groups;
      BoundReferenceExpression key(LogicalType::BIGINT, 0);
      groups.push_back(sirius::ast::from_duckdb(key));
      sirius_physical_grouped_aggregate_merge merger(
        sirius::from_duckdb_vec(
          duckdb::vector<LogicalType>{LogicalType::BIGINT, LogicalType::DOUBLE}),
        translate_expressions(std::move(expressions)),
        std::move(groups),
        1);
      result = merger.execute(pipelineable_operator_data(partials), stream);
    } else {
      sirius_physical_ungrouped_aggregate_merge merger(
        sirius::from_duckdb_vec(duckdb::vector<LogicalType>{LogicalType::DOUBLE}),
        translate_expressions(std::move(expressions)),
        1,
        TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
      result = merger.execute(pipelineable_operator_data(partials), stream);
    }
    auto const& batches =
      dynamic_cast<const pipelineable_operator_data&>(*result).get_data_batches();
    REQUIRE(batches.size() == 1);
    auto view = sirius::get_cudf_table_view(*batches[0]);
    REQUIRE(view.num_rows() == 1);
    auto averages = copy_column_to_host<double>(view.column(grouped ? 1 : 0));
    REQUIRE(averages[0] == Approx(static_cast<double>(value) / 3));
  }
}
