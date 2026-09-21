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

//! Consumer-side tests for the group-by memory-aware bypass prototype (issue #1746 point 2).
//!
//! These drive the real sirius_physical_grouped_aggregate_merge::get_partition_strategy, so they
//! cover the operator's own classification — aggregate partial-state kinds, the key/aggregate
//! column split, and the downstream walk — rather than re-checking the pure policy's arithmetic
//! (test_group_by_bypass_policy.cpp does that). They need no GPU: the metadata the PARTITION would
//! collect is supplied directly, which is also how the "unknown metadata" cases are expressed.

#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "op/aggregate/group_by_bypass_policy.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <cudf/types.hpp>

#include <catch.hpp>

#include <memory>
#include <utility>
#include <vector>

using namespace sirius::op;
using sirius::op::group_by_bypass::decision_reason;

namespace {

using namespace sirius::test::operator_utils;

/// AUTO is ceil(total_bytes / hash_partition_bytes); this is comfortably above 1 partition at the
/// merge's default target, so every test starts from an AUTO > 1 the prototype could overturn.
constexpr uint64_t kBigInputBytes = 8ULL * 1024 * 1024 * 1024;

group_by_bypass_metadata ample_metadata(std::vector<bypass_column_meta> columns)
{
  group_by_bypass_metadata meta;
  meta.upstream_complete            = true;
  meta.single_gpu_resident          = true;
  meta.distinct_memory_spaces       = 1;
  meta.columns                      = std::move(columns);
  meta.total_rows                   = 1'000'000;
  meta.admissible_additional_budget = 32ULL * 1024 * 1024 * 1024;
  meta.space_capacity               = 48ULL * 1024 * 1024 * 1024;
  meta.target_device_id             = 3;  // deliberately not 0
  meta.num_batches                  = 8;
  meta.total_bytes                  = kBigInputBytes;
  meta.headroom_fraction            = 0.25;
  return meta;
}

partition_sizing_input sizing_input(const group_by_bypass_metadata* meta)
{
  return partition_sizing_input{kBigInputBytes,
                                /*is_build_side=*/false,
                                /*build_foldable=*/false,
                                /*combined_total_bytes=*/kBigInputBytes,
                                meta};
}

/// A merge with `aggregations` over one INT64 group key, wired under a RESULT_COLLECTOR so the
/// downstream walk finds a bounded collection path.
struct merge_fixture {
  explicit merge_fixture(const std::vector<std::string>& aggregations, bool attach_collector = true)
  {
    std::vector<std::size_t> agg_indexes(aggregations.size(), 1);
    auto agg = sirius::test::create_aggregate_expressions<gpu_type_traits<int64_t>>(
      {0}, aggregations, agg_indexes);
    merge = duckdb::make_uniq<sirius_physical_grouped_aggregate_merge>(
      std::move(agg.output_types), std::move(agg.aggregates), std::move(agg.groups), 1'000'000);
    merge->operator_id = 7;
    if (attach_collector) {
      collector = duckdb::make_uniq<sirius_physical_operator>(
        SiriusPhysicalOperatorType::RESULT_COLLECTOR, duckdb::vector<sirius::logical_type>{}, 0);
      collector->operator_id = 8;
      // Stamps merge->_parent_op without descending into the collector (which would try to cast
      // a plain operator to sirius_physical_result_collector).
      sirius::planner::sirius_physical_plan_generator::set_parent_ops(*merge, collector.get());
    }
  }

  duckdb::unique_ptr<sirius_physical_operator> collector;
  duckdb::unique_ptr<sirius_physical_grouped_aggregate_merge> merge;
};

}  // namespace

TEST_CASE("merge group-by keeps the automatic count when the bypass prototype is off",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  merge_fixture f{{"sum"}};
  // No metadata is exactly what the PARTITION passes with the setting off.
  auto const strategy = f.merge->get_partition_strategy(sizing_input(nullptr));
  CHECK(strategy.num_partitions ==
        natural_num_partitions(kBigInputBytes, sirius::config::DEFAULT_HASH_PARTITION_BYTES, 1));
  CHECK(strategy.num_partitions > 1);
  CHECK_FALSE(strategy.broadcast);
  CHECK_FALSE(strategy.build_probe);
  // Nothing was decided, so nothing can claim a reservation floor.
  CHECK_FALSE(f.merge->last_bypass_decision().has_value());
  CHECK(f.merge->mandatory_peak_memory_floor({8, kBigInputBytes}) == 0);
}

TEST_CASE("merge group-by selects P=1 for a supported complete candidate with ample budget",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  merge_fixture f{{"sum"}};
  auto const meta     = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                        {static_cast<int>(cudf::type_id::INT64), 8, false}});
  auto const strategy = f.merge->get_partition_strategy(sizing_input(&meta));

  CHECK(strategy.num_partitions == 1);
  auto const decision = f.merge->last_bypass_decision();
  REQUIRE(decision.has_value());
  CHECK(decision->reason == decision_reason::bypass_selected);
  CHECK(decision->prototype_activated);

  // A selected bypass — and only a selected bypass — asserts a reservation floor.
  auto const floor = f.merge->mandatory_peak_memory_floor({8, kBigInputBytes});
  CHECK(floor == decision->model.additional_needed);
  CHECK(floor > 0);
  // The floor is the modelled requirement, not the headroom-inflated one: headroom is a selection
  // margin, not memory the merge is claimed to touch.
  CHECK(floor < decision->model.required_bytes);
}

TEST_CASE("merge group-by rejects P=1 when the modelled requirement does not fit",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  merge_fixture f{{"sum"}};
  auto meta = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                              {static_cast<int>(cudf::type_id::INT64), 8, false}});
  meta.admissible_additional_budget = 4ULL * 1024 * 1024;  // far below the modelled need

  auto const strategy = f.merge->get_partition_strategy(sizing_input(&meta));
  CHECK(strategy.num_partitions ==
        natural_num_partitions(kBigInputBytes, sirius::config::DEFAULT_HASH_PARTITION_BYTES, 1));
  CHECK(strategy.num_partitions > 1);

  auto const decision = f.merge->last_bypass_decision();
  REQUIRE(decision.has_value());
  CHECK(decision->reason == decision_reason::insufficient_budget);
  CHECK_FALSE(decision->prototype_activated);
  // A rejected candidate must not leave a floor behind for the partitioned tasks that now run.
  CHECK(f.merge->mandatory_peak_memory_floor({8, kBigInputBytes}) == 0);
}

TEST_CASE("merge group-by rejects aggregate states outside the whitelist",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  // The whitelist is checked against the physical partial states, not the SQL output types. AVG
  // and COUNT(DISTINCT) both need post-merge projection whose allocations are not modelled.
  auto expect_reason = [](const std::vector<std::string>& aggregations,
                          std::size_t num_state_columns,
                          decision_reason expected) {
    merge_fixture f{aggregations};
    std::vector<bypass_column_meta> columns{{static_cast<int>(cudf::type_id::INT64), 8, false}};
    for (std::size_t i = 0; i < num_state_columns; ++i) {
      columns.push_back({static_cast<int>(cudf::type_id::INT64), 8, false});
    }
    auto const meta     = ample_metadata(std::move(columns));
    auto const strategy = f.merge->get_partition_strategy(sizing_input(&meta));
    auto const decision = f.merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    INFO("expected " << group_by_bypass::reason_name(expected) << ", got "
                     << group_by_bypass::reason_name(decision->reason));
    CHECK(decision->reason == expected);
    CHECK(strategy.num_partitions > 1);
  };

  SECTION("AVG expands to SUM + COUNT and needs a post-merge divide")
  {
    // AVG contributes two cudf aggregate slots, so the physical input carries two state columns.
    expect_reason({"avg"}, 2, decision_reason::unsupported_state);
  }

  SECTION("a supported count is accepted — it merges as a SUM of partial counts")
  {
    merge_fixture f{{"count"}};
    auto const meta = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                      {static_cast<int>(cudf::type_id::INT64), 8, false}});
    CHECK(f.merge->get_partition_strategy(sizing_input(&meta)).num_partitions == 1);
  }

  SECTION("a variable-width physical column is rejected even with a supported aggregate")
  {
    merge_fixture f{{"sum"}};
    // A STRING key reports no fixed width; its concatenated size is not modelled.
    auto const meta     = ample_metadata({{static_cast<int>(cudf::type_id::STRING), 0, false},
                                          {static_cast<int>(cudf::type_id::INT64), 8, false}});
    auto const strategy = f.merge->get_partition_strategy(sizing_input(&meta));
    auto const decision = f.merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    CHECK(decision->reason == decision_reason::unsupported_state);
    CHECK(strategy.num_partitions > 1);
  }

  SECTION("a floating-point partial state is outside the v1 envelope")
  {
    merge_fixture f{{"sum"}};
    auto const meta = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                      {static_cast<int>(cudf::type_id::FLOAT64), 8, false}});
    auto const decision_strategy = f.merge->get_partition_strategy(sizing_input(&meta));
    auto const decision          = f.merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    CHECK(decision->reason == decision_reason::unsupported_state);
    CHECK(decision_strategy.num_partitions > 1);
  }
}

TEST_CASE("merge group-by rejects an unsupported or unknown downstream",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  SECTION("no parent at all is an unknown consumer shape")
  {
    merge_fixture f{{"sum"}, /*attach_collector=*/false};
    auto const meta     = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                          {static_cast<int>(cudf::type_id::INT64), 8, false}});
    auto const strategy = f.merge->get_partition_strategy(sizing_input(&meta));
    auto const decision = f.merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    CHECK(decision->reason == decision_reason::unsupported_downstream);
    CHECK(strategy.num_partitions > 1);
  }

  SECTION("a fused TOP_N downstream is rejected")
  {
    merge_fixture f{{"sum"}, /*attach_collector=*/false};
    auto top_n = duckdb::make_uniq<sirius_physical_operator>(
      SiriusPhysicalOperatorType::TOP_N, duckdb::vector<sirius::logical_type>{}, 0);
    top_n->operator_id = 9;
    sirius::planner::sirius_physical_plan_generator::set_parent_ops(*f.merge, top_n.get());

    auto const meta     = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                          {static_cast<int>(cudf::type_id::INT64), 8, false}});
    auto const strategy = f.merge->get_partition_strategy(sizing_input(&meta));
    auto const decision = f.merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    CHECK(decision->reason == decision_reason::unsupported_downstream);
    CHECK(strategy.num_partitions > 1);
  }

  SECTION("a PROJECTION on the way to the collector is supported and charged a copy")
  {
    merge_fixture f{{"sum"}, /*attach_collector=*/false};
    auto collector = duckdb::make_uniq<sirius_physical_operator>(
      SiriusPhysicalOperatorType::RESULT_COLLECTOR, duckdb::vector<sirius::logical_type>{}, 0);
    collector->operator_id = 8;
    auto projection        = duckdb::make_uniq<sirius_physical_operator>(
      SiriusPhysicalOperatorType::PROJECTION, duckdb::vector<sirius::logical_type>{}, 0);
    projection->operator_id = 9;
    projection->children.push_back(std::move(f.merge));
    sirius::planner::sirius_physical_plan_generator::set_parent_ops(*projection, collector.get());
    auto* merge = &projection->children[0]->Cast<sirius_physical_grouped_aggregate_merge>();

    auto const meta = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                      {static_cast<int>(cudf::type_id::INT64), 8, false}});
    CHECK(merge->get_partition_strategy(sizing_input(&meta)).num_partitions == 1);
    auto const decision = merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    CHECK(decision->reason == decision_reason::bypass_selected);
    // The retained transformed copy is charged, so this needs strictly more than a bare collector.
    CHECK(decision->model.downstream_bytes == decision->model.output_bytes);
  }
}

TEST_CASE("merge group-by rejects projected, multi-GPU and unreadable candidates",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  auto base_columns = []() {
    return std::vector<bypass_column_meta>{{static_cast<int>(cudf::type_id::INT64), 8, false},
                                           {static_cast<int>(cudf::type_id::INT64), 8, false}};
  };

  SECTION("upstream still running keeps point 3's projected choice")
  {
    merge_fixture f{{"sum"}};
    auto meta              = ample_metadata(base_columns());
    meta.upstream_complete = false;
    auto const strategy    = f.merge->get_partition_strategy(sizing_input(&meta));
    auto const decision    = f.merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    CHECK(decision->reason == decision_reason::projected_input);
    CHECK(strategy.num_partitions ==
          natural_num_partitions(kBigInputBytes, sirius::config::DEFAULT_HASH_PARTITION_BYTES, 1));
  }

  SECTION("more than one admitted GPU keeps existing behaviour")
  {
    merge_fixture f{{"sum"}};
    f.merge->set_num_gpus(4);
    auto const meta     = ample_metadata(base_columns());
    auto const strategy = f.merge->get_partition_strategy(sizing_input(&meta));
    auto const decision = f.merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    CHECK(decision->reason == decision_reason::multi_gpu);
    // Including the multi-GPU floor the existing policy applies.
    CHECK(strategy.num_partitions ==
          natural_num_partitions(kBigInputBytes, sirius::config::DEFAULT_HASH_PARTITION_BYTES, 4));
  }

  SECTION("input spread across memory spaces is not a single-GPU candidate")
  {
    merge_fixture f{{"sum"}};
    auto meta                   = ample_metadata(base_columns());
    meta.distinct_memory_spaces = 2;
    meta.single_gpu_resident    = false;
    f.merge->get_partition_strategy(sizing_input(&meta));
    auto const decision = f.merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    CHECK(decision->reason == decision_reason::unsupported_residency);
  }

  SECTION("unreadable column metadata is rejected, not treated as an empty schema")
  {
    merge_fixture f{{"sum"}};
    auto meta    = ample_metadata(base_columns());
    meta.columns = std::nullopt;
    f.merge->get_partition_strategy(sizing_input(&meta));
    auto const decision = f.merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    CHECK(decision->reason == decision_reason::unsupported_state);
  }

  SECTION("a schema that disagrees with this merge's own layout is rejected")
  {
    merge_fixture f{{"sum"}};
    auto meta = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false}});
    f.merge->get_partition_strategy(sizing_input(&meta));
    auto const decision = f.merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    CHECK(decision->reason == decision_reason::unsupported_state);
  }

  SECTION("an unknown budget is rejected rather than assumed infinite")
  {
    merge_fixture f{{"sum"}};
    auto meta                         = ample_metadata(base_columns());
    meta.admissible_additional_budget = std::nullopt;
    f.merge->get_partition_strategy(sizing_input(&meta));
    auto const decision = f.merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    CHECK(decision->reason == decision_reason::unknown_metadata);
  }

  SECTION("a row count above cuDF's concatenation limit is rejected without wrapping")
  {
    merge_fixture f{{"sum"}};
    auto meta           = ample_metadata(base_columns());
    meta.total_rows     = group_by_bypass::CUDF_MAX_ROWS + 1;
    auto const strategy = f.merge->get_partition_strategy(sizing_input(&meta));
    auto const decision = f.merge->last_bypass_decision();
    REQUIRE(decision.has_value());
    CHECK(decision->reason == decision_reason::size_overflow);
    CHECK(strategy.num_partitions > 1);
  }
}

TEST_CASE("merge group-by does not claim an activation when AUTO already chose one partition",
          "[physical_grouped_aggregate_merge][group_by_bypass]")
{
  merge_fixture f{{"sum"}};
  auto meta        = ample_metadata({{static_cast<int>(cudf::type_id::INT64), 8, false},
                                     {static_cast<int>(cudf::type_id::INT64), 8, false}});
  meta.total_bytes = 1024;
  // A small input the existing policy already resolves to a single partition.
  auto const strategy =
    f.merge->get_partition_strategy(partition_sizing_input{1024, false, false, 1024, &meta});

  CHECK(strategy.num_partitions == 1);
  auto const decision = f.merge->last_bypass_decision();
  REQUIRE(decision.has_value());
  CHECK(decision->reason == decision_reason::already_one);
  CHECK_FALSE(decision->prototype_activated);
  // The pre-existing choice must not pick up the prototype's reservation behaviour.
  CHECK(f.merge->mandatory_peak_memory_floor({1, 1024}) == 0);
}
