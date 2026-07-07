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

#pragma once

#include "cudf/aggregation.hpp"
#include "cudf/types.hpp"
#include "duckdb/execution/operator/aggregate/distinct_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/radix_partitioned_hashtable.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/storage/data_table.hpp"
#include "expression/ast/node.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/sirius_physical_operator.hpp"

#include <memory>
#include <mutex>
#include <numeric>

namespace sirius {
namespace op {

class sirius_physical_grouped_aggregate : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::HASH_GROUP_BY;

 public:
  sirius_physical_grouped_aggregate(duckdb::vector<sirius::logical_type> types,
                                    duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
                                    duckdb::vector<std::unique_ptr<sirius::ast::node>> groups,
                                    std::size_t estimated_cardinality);

  sirius_physical_grouped_aggregate(
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
    duckdb::vector<std::unique_ptr<sirius::ast::node>> groups,
    duckdb::vector<duckdb::GroupingSet> grouping_sets,
    duckdb::vector<duckdb::unsafe_vector<std::size_t>> grouping_functions,
    std::size_t estimated_cardinality,
    duckdb::TupleDataValidityType group_validity,
    duckdb::TupleDataValidityType distinct_validity);

  duckdb::vector<duckdb::GroupingSet> grouping_sets;

  // TODO: we may need some of these variables later when we implement grouping sets

  // //! The grouping sets
  // duckdb::GroupedAggregateData grouped_aggregate_data;

  // //! The radix partitioned hash tables (one per grouping set)
  // duckdb::vector<duckdb::HashAggregateGroupingData> groupings;
  // duckdb::unique_ptr<duckdb::DistinctAggregateCollectionInfo> distinct_collection_info;
  // //! A recreation of the input chunk, with nulls for everything that isn't a group
  // duckdb::vector<sirius::logical_type> input_group_types;

  // // Filters given to sink and friends
  // duckdb::unsafe_vector<std::size_t> non_distinct_filter;
  // duckdb::unsafe_vector<std::size_t> distinct_filter;

  // duckdb::unordered_map<duckdb::Expression*, size_t> filter_indexes;

  // Grouped aggregatge definitions for cudf compute
  std::vector<int> group_idx;
  std::vector<cudf::aggregation::Kind> cudf_aggregates;
  std::vector<int> cudf_aggregate_idx;
  std::vector<std::vector<int>> cudf_aggregate_struct_col_indices;

  // AVG decomposition metadata
  std::vector<AggregateSlot> aggregate_slots;
  bool has_avg            = false;
  bool has_count_distinct = false;

 public:
  std::vector<int> get_output_grouping_indices() const
  {
    std::vector<int> indices(group_idx.size());
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
  }

  // Source interface
  bool is_source() const override { return true; }

  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }

  // Sink interface
  bool is_sink() const override { return true; }

  bool sink_order_dependent() const override { return false; }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  // --- #990 single-task collapse (Option B / B-bypass) ---
  //
  // When the converter decides (from the estimate) that this aggregate MIGHT be small enough
  // to run in one task, it marks it collapse-capable via enable_collapse(): the aggregate's
  // input edge is set to a FULL barrier (so all input is present when scheduled), and a
  // "bypass" next-port to `bypass_downstream` (the operator that would otherwise read from
  // MERGE) is added alongside the normal aggregate->partition next-port.
  //
  // At runtime get_next_task_input_data() measures the actual input bytes: if the whole input
  // fits in one task (<= budget) it drains everything into a single task and sets _collapsed,
  // so execute() aggregates + finalizes in one pass and sink() routes the finalized result
  // straight to the downstream repo — leaving partition+merge with no input (never scheduled).
  // If the input is larger, it degrades to the normal per-batch partial path feeding partition.
  void enable_collapse(sirius_physical_operator* bypass_downstream,
                       std::size_t single_task_budget_bytes)
  {
    _collapse_capable         = true;
    _bypass_downstream        = bypass_downstream;
    _single_task_budget_bytes = single_task_budget_bytes;
  }

  std::unique_ptr<operator_data> get_next_task_input_data() override;
  void sink(const operator_data& output_data, rmm::cuda_stream_view stream) override;

 private:
  //! Set by the converter (plan time): this aggregate may collapse to a single task.
  bool _collapse_capable = false;
  //! Downstream consumer to route to when collapsed (identifies the bypass next-port).
  sirius_physical_operator* _bypass_downstream = nullptr;
  //! One-task byte budget; input measured <= this collapses (0 => not collapse-capable).
  std::size_t _single_task_budget_bytes = 0;
  //! Whether the collapse decision has been made yet (measured input once).
  bool _collapse_decided = false;
  //! Runtime decision, set in get_next_task_input_data before execute()/sink() of that task.
  bool _collapsed = false;
  //! Guards the collapse decision / drain against concurrent task-creator threads.
  std::mutex _collapse_mutex;
};

}  // namespace op
}  // namespace sirius
