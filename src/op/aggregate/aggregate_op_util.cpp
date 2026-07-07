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

#include "op/aggregate/aggregate_op_util.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/common/assert.hpp"
#include "expression/aggregate_id.hpp"
#include "expression/ast/node.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sirius {
namespace op {

namespace {

// Single place that builds the "Unsupported aggregate function: <name>" diagnostic so the
// message (and the aggregate_id -> name lookup) is not repeated at every rejection site.
[[noreturn]] void throw_unsupported_aggregate(sirius::aggregate_id fid,
                                              std::string_view detail = {})
{
  auto const name = sirius::to_duckdb_aggregate_name(fid);
  throw std::runtime_error(detail.empty()
                             ? std::format("Unsupported aggregate function: {}", name)
                             : std::format("Unsupported aggregate function: {} {}", name, detail));
}

}  // namespace

std::optional<cudf::aggregation::Kind> to_cudf_aggregation_kind(sirius::aggregate_id id)
{
  switch (id) {
    case sirius::aggregate_id::sum:
    case sirius::aggregate_id::sum_no_overflow: return cudf::aggregation::Kind::SUM;
    case sirius::aggregate_id::count: return cudf::aggregation::Kind::COUNT_VALID;
    case sirius::aggregate_id::count_star: return cudf::aggregation::Kind::COUNT_ALL;
    case sirius::aggregate_id::min: return cudf::aggregation::Kind::MIN;
    case sirius::aggregate_id::max: return cudf::aggregation::Kind::MAX;
    case sirius::aggregate_id::avg:
    case sirius::aggregate_id::first: return std::nullopt;
  }
  return std::nullopt;
}

CudfAggregateDefinitions convert_duckdb_aggregates_to_cudf(
  const duckdb::vector<std::unique_ptr<sirius::ast::node>>& groups_p,
  const duckdb::vector<std::unique_ptr<sirius::ast::node>>& expressions)
{
  CudfAggregateDefinitions result;

  // 1. Extract group_idx from groups_p
  for (const auto& group : groups_p) {
    auto const& ref =
      sirius::ast::require_reference(group.get(), "convert_duckdb_aggregates_to_cudf group");
    result.group_idx.push_back(static_cast<int>(ref.column_index));
  }

  // 2. Extract aggregates (cudf::aggregation::Kind) from expressions
  for (const auto& aggregate : expressions) {
    auto const& aggr = sirius::ast::require_aggregate(
      aggregate.get(), "convert_duckdb_aggregates_to_cudf aggregate");
    auto const fid       = aggr.function();
    auto const& children = aggr.arguments();

    // Handle AVG specially: it expands into SUM + COUNT_VALID
    if (fid == sirius::aggregate_id::avg) {
      D_ASSERT(children.size() == 1);
      D_ASSERT(children[0]->is_reference());
      auto col_idx = static_cast<int>(children[0]->as_reference().column_index);

      size_t sum_position = result.cudf_aggregates.size();
      result.cudf_aggregates.push_back(cudf::aggregation::Kind::SUM);
      result.cudf_aggregate_idx.push_back(col_idx);
      result.cudf_aggregate_struct_col_indices.push_back({});
      result.cudf_aggregates.push_back(cudf::aggregation::Kind::COUNT_VALID);
      result.cudf_aggregate_idx.push_back(col_idx);
      result.cudf_aggregate_struct_col_indices.push_back({});
      result.aggregate_slots.push_back(
        AggregateSlot{true, false, sum_position, sirius::get_cudf_type(aggr.return_type())});
      result.has_avg = true;
      continue;
    }

    // Handle COUNT(DISTINCT col) and COUNT(DISTINCT (col1, col2, ...)):
    // Use COLLECT_SET locally; merge via MERGE_SETS; then count list elements.
    // For multi-column, a struct column is synthesized from the component columns.
    if (aggr.distinct() && fid == sirius::aggregate_id::count) {
      D_ASSERT(children.size() == 1);
      auto const& child = *children[0];
      size_t position   = result.cudf_aggregates.size();
      result.cudf_aggregates.push_back(cudf::aggregation::Kind::COLLECT_SET);

      if (child.is_reference()) {
        // Single-column case: COUNT(DISTINCT col)
        result.cudf_aggregate_idx.push_back(static_cast<int>(child.as_reference().column_index));
        result.cudf_aggregate_struct_col_indices.push_back({});
      } else {
        // Multi-column case: COUNT(DISTINCT (col1, col2, ...)) — child is a struct_pack expression
        D_ASSERT(child.is_function_call());
        auto const& func_expr = child.as_function_call();
        std::vector<int> struct_indices;
        for (auto const& arg : func_expr.arguments()) {
          D_ASSERT(arg->is_reference());
          struct_indices.push_back(static_cast<int>(arg->as_reference().column_index));
        }
        D_ASSERT(!struct_indices.empty());
        result.cudf_aggregate_idx.push_back(-1);  // sentinel: struct column, see gpu_aggregate_impl
        result.cudf_aggregate_struct_col_indices.push_back(std::move(struct_indices));
      }

      result.aggregate_slots.push_back(AggregateSlot{false, true, position});
      result.has_count_distinct = true;
      continue;
    }

    auto const agg_kind = to_cudf_aggregation_kind(fid);
    if (!agg_kind) { throw_unsupported_aggregate(fid); }
    size_t current_position = result.cudf_aggregates.size();
    result.cudf_aggregates.push_back(*agg_kind);

    // 3. Extract aggregate_idx from the children of the aggregate expression
    if (children.empty()) {
      // COUNT(*) has no children - use 0 as a placeholder (will be handled by COUNT_ALL)
      if (fid == sirius::aggregate_id::count_star) {
        result.cudf_aggregate_idx.push_back(0);
      } else {
        throw_unsupported_aggregate(fid, "with no children");
      }
    } else {
      if (children.size() == 1) {
        // Extract the column index from the first child (most aggregates have one child)
        D_ASSERT(children[0]->is_reference());
        result.cudf_aggregate_idx.push_back(
          static_cast<int>(children[0]->as_reference().column_index));
      } else {
        throw_unsupported_aggregate(fid, "with " + std::to_string(children.size()) + " children");
      }
    }
    result.cudf_aggregate_struct_col_indices.push_back({});
    result.aggregate_slots.push_back(AggregateSlot{false, false, current_position});
  }

  return result;
}

std::shared_ptr<cucascade::data_batch> finalize_merged_grouped_aggregate(
  std::shared_ptr<cucascade::data_batch> merged,
  int num_group_cols,
  const std::vector<AggregateSlot>& aggregate_slots,
  bool has_avg,
  bool has_count_distinct,
  rmm::cuda_stream_view stream)
{
  // No AVG / COUNT DISTINCT post-processing needed: the merged batch is already final.
  if (!has_avg && !has_count_distinct) { return merged; }

  // Post-merge projection: handle AVG (SUM/COUNT) and COUNT DISTINCT (list element count).
  // Release ownership of the merged table's columns so we can move (not copy) them. Acquire
  // EXCLUSIVE lock since release_table() is a mutating operation.
  auto merged_mut  = merged->to_mutable();
  auto* space      = merged_mut.get_memory_space();
  auto mr          = space->get_default_allocator();
  auto& gpu_rep    = merged_mut.get_data()->cast<cucascade::gpu_table_representation>();
  auto merged_cols = gpu_rep.release_table(stream)->release();

  std::vector<std::unique_ptr<cudf::column>> output_cols;

  // Move group key columns (zero-copy)
  for (int i = 0; i < num_group_cols; ++i) {
    output_cols.push_back(std::move(merged_cols[i]));
  }

  // Process each original aggregate
  for (auto const& slot : aggregate_slots) {
    if (slot.is_avg) {
      int sum_col_idx   = num_group_cols + static_cast<int>(slot.cudf_idx);
      int count_col_idx = num_group_cols + static_cast<int>(slot.cudf_idx) + 1;

      auto sum_view   = merged_cols[sum_col_idx]->view();
      auto count_view = merged_cols[count_col_idx]->view();

      std::unique_ptr<cudf::column> avg_col;
      bool is_decimal = sirius::IsCudfTypeDecimal(slot.output_type);
      if (is_decimal) {
        // DECIMAL: divide directly in fixed-point to preserve precision
        avg_col = cudf::binary_operation(
          sum_view, count_view, cudf::binary_operator::DIV, slot.output_type, stream, mr);
      } else {
        // Non-DECIMAL: cast to FLOAT64 and divide
        auto sum_f64 = cudf::cast(sum_view, cudf::data_type{cudf::type_id::FLOAT64}, stream, mr);
        auto count_f64 =
          cudf::cast(count_view, cudf::data_type{cudf::type_id::FLOAT64}, stream, mr);
        avg_col = cudf::binary_operation(sum_f64->view(),
                                         count_f64->view(),
                                         cudf::binary_operator::DIV,
                                         cudf::data_type{cudf::type_id::FLOAT64},
                                         stream,
                                         mr);
      }

      output_cols.push_back(std::move(avg_col));
    } else if (slot.is_count_distinct) {
      // The merged column is a LIST column (output of MERGE_SETS). Count elements per row to
      // produce the final distinct count, then cast to INT64.
      int col_idx      = num_group_cols + static_cast<int>(slot.cudf_idx);
      auto list_view   = cudf::lists_column_view(merged_cols[col_idx]->view());
      auto count_int32 = cudf::lists::count_elements(list_view, stream, mr);
      auto count_int64 =
        cudf::cast(count_int32->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
      output_cols.push_back(std::move(count_int64));
    } else {
      // Move non-AVG, non-count-distinct aggregate columns directly (zero-copy)
      int col_idx = num_group_cols + static_cast<int>(slot.cudf_idx);
      output_cols.push_back(std::move(merged_cols[col_idx]));
    }
  }

  auto output_table = std::make_unique<cudf::table>(std::move(output_cols), stream, mr);
  return sirius::make_data_batch(std::move(output_table), *space, stream);
}

}  // namespace op
}  // namespace sirius
