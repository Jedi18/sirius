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

#include "op/sirius_physical_grouped_aggregate.hpp"

#include "data/data_batch_utils.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/aggregate/gpu_aggregate_impl.hpp"
#include "op/merge/gpu_merge_impl.hpp"

#include <nvtx3/nvtx3.hpp>

namespace sirius {
namespace op {

sirius_physical_grouped_aggregate::sirius_physical_grouped_aggregate(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> groups_p,
  std::size_t estimated_cardinality)
  : sirius_physical_grouped_aggregate(std::move(types),
                                      std::move(expressions),
                                      std::move(groups_p),
                                      {},
                                      {},
                                      estimated_cardinality,
                                      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES,
                                      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES)
{
}

// expressions is the list of aggregates to be computed. Each aggregates has a bound_ref expression
// to a column groups_p is the list of group by columns. Each group by column is a bound_ref
// expression to a column grouping_sets_p is the list of grouping set. Each grouping set is a set of
// indexes to the group by columns. Seems like DuckDB group the groupby columns into several sets
// and for every grouping set there is one radix_table grouping_functions_p is a list of indexes to
// the groupby expressions (groups_p) for each grouping_sets. The first level of the vector is the
// grouping set and the second level is the indexes to the groupby expression for that set.
sirius_physical_grouped_aggregate::sirius_physical_grouped_aggregate(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> groups_p,
  duckdb::vector<duckdb::GroupingSet> grouping_sets_p,
  duckdb::vector<duckdb::unsafe_vector<std::size_t>> grouping_functions_p,
  std::size_t estimated_cardinality,
  duckdb::TupleDataValidityType /*group_validity*/,
  duckdb::TupleDataValidityType /*distinct_validity*/)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::HASH_GROUP_BY, std::move(types), estimated_cardinality),
    grouping_sets(std::move(grouping_sets_p))
{
  auto cudf_defs                    = convert_duckdb_aggregates_to_cudf(groups_p, expressions);
  group_idx                         = std::move(cudf_defs.group_idx);
  cudf_aggregates                   = std::move(cudf_defs.cudf_aggregates);
  cudf_aggregate_idx                = std::move(cudf_defs.cudf_aggregate_idx);
  cudf_aggregate_struct_col_indices = std::move(cudf_defs.cudf_aggregate_struct_col_indices);
  aggregate_slots                   = std::move(cudf_defs.aggregate_slots);
  has_avg                           = cudf_defs.has_avg;
  has_count_distinct                = cudf_defs.has_count_distinct;
}

std::unique_ptr<operator_data> sirius_physical_grouped_aggregate::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_grouped_aggregate::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  std::vector<std::shared_ptr<::cucascade::data_batch>> results;
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& input_batch : input_batches) {
    auto* batch_space = input_batch.get_memory_space();
    if (!batch_space) { continue; }
    space       = batch_space;
    auto result = gpu_aggregate_impl::local_grouped_aggregate(input_batch,
                                                              group_idx,
                                                              cudf_aggregates,
                                                              cudf_aggregate_idx,
                                                              cudf_aggregate_struct_col_indices,
                                                              stream,
                                                              *batch_space);
    results.push_back(std::move(result));
  }

  // Two-phase (default): emit the per-batch partials; partition + merge finalize them.
  if (!_collapsed) { return std::make_unique<pipelineable_operator_data>(results); }

  // Collapsed single task (#990 B-bypass): all input arrived in this one task, so combine the
  // partials and finalize HERE — producing the final result — then sink() routes it straight to
  // the downstream consumer, skipping partition + merge entirely. This reuses the same merge +
  // finalization the two-phase MERGE_GROUP_BY operator uses, so results are identical.
  if (results.empty() || space == nullptr) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<::cucascade::data_batch>>{});
  }
  std::shared_ptr<::cucascade::data_batch> merged;
  if (results.size() == 1) {
    merged = results[0];
  } else {
    std::vector<::cucascade::read_only_data_batch> partials_ro;
    partials_ro.reserve(results.size());
    for (auto const& partial : results) {
      partials_ro.push_back(partial->to_read_only());
    }
    merged = gpu_merge_impl::merge_grouped_aggregate(
      partials_ro, static_cast<int>(group_idx.size()), cudf_aggregates, stream, *space);
  }
  auto finalized = finalize_merged_grouped_aggregate(merged,
                                                     static_cast<int>(group_idx.size()),
                                                     aggregate_slots,
                                                     has_avg,
                                                     has_count_distinct,
                                                     stream);
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<::cucascade::data_batch>>{finalized});
}

std::unique_ptr<operator_data> sirius_physical_grouped_aggregate::get_next_task_input_data()
{
  // Not collapse-capable → today's behavior: one task per input batch.
  if (!_collapse_capable) { return sirius_physical_operator::get_next_task_input_data(); }

  std::lock_guard<std::mutex> lg(_collapse_mutex);

  // Already decided NOT to collapse → keep serving per-batch tasks like the base operator.
  if (_collapse_decided && !_collapsed) {
    return sirius_physical_operator::get_next_task_input_data();
  }
  // Collapsed → the single task already drained everything.
  if (_collapse_decided && _collapsed) { return nullptr; }

  // First scheduling of a collapse-capable aggregate. Its input edge is a FULL barrier, so all
  // input is present now; measure the true input size (same measurement the partition operator
  // uses in determine_num_partitions) and decide.
  std::uint64_t total_bytes = 0;
  for (auto& [port_name, port_ptr] : ports) {
    if (!port_ptr->repo) { continue; }
    for (auto batch_id : port_ptr->repo->get_batch_ids(0)) {
      auto batch = port_ptr->repo->get_data_batch_by_id(batch_id, 0);
      if (!batch) { continue; }
      auto ro = batch->to_read_only();
      if (ro.get_data()) { total_bytes += ro.get_data()->get_size_in_bytes(); }
    }
  }
  _collapse_decided = true;

  if (total_bytes > _single_task_budget_bytes) {
    // Estimate said "small" but the data is actually large → fall back to the normal per-batch
    // partial path feeding partition+merge. No collapse, no bypass (the runtime guard).
    _collapsed = false;
    return sirius_physical_operator::get_next_task_input_data();
  }

  // Collapse: drain ALL input batches into a single task (see ungrouped merge for the pattern).
  _collapsed = true;
  std::vector<std::shared_ptr<::cucascade::data_batch>> input_batch;
  for (auto& [port_name, port_ptr] : ports) {
    if (!port_ptr->repo) { continue; }
    while (auto batch = port_ptr->repo->pop_next_data_batch()) {
      input_batch.push_back(std::move(batch));
    }
  }
  if (input_batch.empty()) { return nullptr; }
  return std::make_unique<pipelineable_operator_data>(input_batch);
}

void sirius_physical_grouped_aggregate::sink(const operator_data& output_data,
                                             rmm::cuda_stream_view stream)
{
  // Non-collapsed → default fan-out (pushes to the partition next-port, as today).
  if (!_collapsed) {
    sirius_physical_operator::sink(output_data, stream);
    return;
  }
  // Collapsed (B-bypass): execute() already produced the FINALIZED result, so route it
  // straight to the downstream consumer's port and NOT to partition. Partition+merge then
  // receive no input and are never scheduled; the pipeline-completion machinery still marks
  // them finished and releases downstream's FULL barrier. Deposit-then-finish: this push
  // completes within the aggregate task's sink, before pipeline completion propagates.
  auto& pipelineable_output = dynamic_cast<const pipelineable_operator_data&>(output_data);
  for (auto& batch : pipelineable_output.get_data_batches()) {
    for (auto& next_port_info : get_next_ports_after_sink()) {
      if (next_port_info.next_operator == _bypass_downstream) {
        next_port_info.next_operator->push_data_batch(next_port_info.next_operator_port_name,
                                                      batch);
      }
    }
  }
}
}  // namespace op
}  // namespace sirius
