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

// sirius_physical_grouped_aggregate_merge::split_input is what the executor's OOM recovery
// leans on: the pieces must be a key-disjoint, row-complete decomposition of the merge's input,
// and merging them separately must give exactly the one-shot merge.

#include "../operator_test_utils.hpp"
#include "../operator_type_traits.hpp"
#include "aggregate_test_utils.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>

#include <catch.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::operator_utils;

namespace {

using Traits = gpu_type_traits<int64_t>;

/// A merge over GROUP BY col0 with SUM(col1), the shape every HASH_GROUP_BY feeds it.
std::unique_ptr<sirius_physical_grouped_aggregate_merge> make_sum_merge()
{
  auto agg   = sirius::test::create_aggregate_expressions<Traits>({0}, {"sum"}, {1});
  auto merge = std::make_unique<sirius_physical_grouped_aggregate_merge>(
    std::move(agg.output_types), std::move(agg.aggregates), std::move(agg.groups), 0);
  merge->operator_id = 0;
  return merge;
}

/// One partial-aggregate batch: distinct keys `first_key .. first_key + n`, value = key * weight.
std::shared_ptr<cucascade::data_batch> make_partial(cucascade::memory::memory_space& space,
                                                    int64_t first_key,
                                                    int64_t n,
                                                    int64_t weight)
{
  std::vector<int64_t> keys(static_cast<std::size_t>(n));
  std::vector<int64_t> values(static_cast<std::size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    keys[static_cast<std::size_t>(i)]   = first_key + i;
    values[static_cast<std::size_t>(i)] = (first_key + i) * weight;
  }
  return make_two_column_batch<int64_t, int64_t>(space, keys, values, cudf::type_id::INT64);
}

std::vector<int64_t> host_column(const cudf::table_view& view, int col)
{
  return copy_column_to_host<int64_t>(view.column(col));
}

std::size_t rows_of(const operator_data& data)
{
  auto const& p    = dynamic_cast<const pipelineable_operator_data&>(data);
  std::size_t rows = 0;
  for (auto const& ro : p.get_read_only_batches()) {
    rows += static_cast<std::size_t>(sirius::get_cudf_table_view(ro).num_rows());
  }
  return rows;
}

}  // namespace

TEST_CASE("merge split_input: pieces are key-disjoint and together hold every input row",
          "[physical_grouped_aggregate_merge][oom_split]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = default_stream();
  auto merge  = make_sum_merge();

  // Three partials over overlapping key ranges, as three HASH_GROUP_BY tasks would produce.
  constexpr int64_t kKeys = 1000;
  std::vector<std::shared_ptr<cucascade::data_batch>> input{make_partial(*space, 0, kKeys, 1),
                                                            make_partial(*space, 0, kKeys, 2),
                                                            make_partial(*space, 500, kKeys, 3)};
  partitioned_operator_data partitioned(input, /*partition_idx=*/3);

  auto pieces = merge->split_input(partitioned, 2, /*split_round=*/0, stream);
  stream.sync();
  REQUIRE(pieces.size() == 2);

  // Each piece keeps the input's shape and slot so the executor pins it to the same GPU.
  std::size_t total_rows = 0;
  std::map<int64_t, int> piece_of_key;
  for (std::size_t j = 0; j < pieces.size(); ++j) {
    auto const* piece = dynamic_cast<const partitioned_operator_data*>(pieces[j].get());
    REQUIRE(piece != nullptr);
    REQUIRE(piece->get_partition_idx() == std::optional<std::size_t>{3});
    REQUIRE_FALSE(piece->get_data_batches().empty());
    for (auto const& ro : piece->get_read_only_batches()) {
      auto view = sirius::get_cudf_table_view(ro);
      REQUIRE(view.num_columns() == 2);
      REQUIRE(view.num_rows() > 0);
      total_rows += static_cast<std::size_t>(view.num_rows());
      for (auto key : host_column(view, 0)) {
        auto [it, inserted] = piece_of_key.emplace(key, static_cast<int>(j));
        // A key seen before must have gone to the same piece: that is what lets each piece be
        // merged on its own.
        if (!inserted) { REQUIRE(it->second == static_cast<int>(j)); }
      }
    }
  }
  CHECK(total_rows == static_cast<std::size_t>(3 * kKeys));
  CHECK(piece_of_key.size() == static_cast<std::size_t>(kKeys + 500));
  // Both pieces got a non-trivial share: murmur3 over 1500 keys does not pile up on one side.
  auto const input_rows = static_cast<std::size_t>(3 * kKeys);
  for (std::size_t j = 0; j < pieces.size(); ++j) {
    CHECK(rows_of(*pieces[j]) > input_rows / 4);
    CHECK(rows_of(*pieces[j]) < input_rows - input_rows / 4);
  }
}

TEST_CASE("merge split_input: merging the pieces separately equals the one-shot merge",
          "[physical_grouped_aggregate_merge][oom_split]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = default_stream();
  auto merge  = make_sum_merge();

  constexpr int64_t kKeys = 2000;
  std::vector<std::shared_ptr<cucascade::data_batch>> input{make_partial(*space, 0, kKeys, 1),
                                                            make_partial(*space, 0, kKeys, 5),
                                                            make_partial(*space, 1000, kKeys, 7),
                                                            make_partial(*space, 1500, 10, 11)};
  pipelineable_operator_data whole(input);

  auto whole_out = merge->execute(whole, stream);
  auto whole_batches =
    dynamic_cast<const pipelineable_operator_data&>(*whole_out).get_data_batches();
  REQUIRE(whole_batches.size() == 1);

  // Two rounds deep, as the executor would do after two splits.
  auto first = merge->split_input(whole, 2, /*split_round=*/0, stream);
  REQUIRE(first.size() == 2);
  std::vector<std::unique_ptr<operator_data>> leaves;
  for (auto& piece : first) {
    auto second = merge->split_input(*piece, 2, /*split_round=*/1, stream);
    REQUIRE(second.size() == 2);
    for (auto& leaf : second) {
      leaves.push_back(std::move(leaf));
    }
  }
  stream.sync();

  std::vector<cucascade::read_only_data_batch> leaf_outputs;
  std::vector<std::shared_ptr<cucascade::data_batch>> keep_alive;
  std::size_t merged_rows = 0;
  for (auto const& leaf : leaves) {
    auto out = merge->execute(*leaf, stream);
    for (auto const& batch :
         dynamic_cast<const pipelineable_operator_data&>(*out).get_data_batches()) {
      keep_alive.push_back(batch);
      leaf_outputs.push_back(batch->to_read_only());
      merged_rows += static_cast<std::size_t>(sirius::get_cudf_table_view(*batch).num_rows());
    }
  }
  // Every group appears in exactly one leaf output, so the leaf outputs' row counts add up to
  // the one-shot merge's row count (no lost groups, no duplicated groups).
  CHECK(merged_rows ==
        static_cast<std::size_t>(sirius::get_cudf_table_view(*whole_batches[0]).num_rows()));
  auto combined = gpu_merge_impl::concat(leaf_outputs, stream, *space);
  stream.sync();
  CHECK(sirius::test::expect_data_batches_equivalent(whole_batches[0], combined, /*sort=*/true));
}

TEST_CASE("merge split_input: a slot the PARTITION already hashed still splits evenly",
          "[physical_grouped_aggregate_merge][oom_split][hash_partition]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = default_stream();
  auto merge  = make_sum_merge();

  // What the plan does: hash-partition on the key with cuDF's default seed into P = 2 slots.
  auto all = make_partial(*space, 0, 4000, 1);
  std::vector<std::shared_ptr<cucascade::data_batch>> slots;
  {
    auto ro = all->to_read_only();
    slots   = gpu_partition_impl::hash_partition(ro, {0}, 2, stream, *space);
  }
  stream.sync();
  REQUIRE(slots.size() == 2);
  auto const slot0_rows = sirius::get_cudf_table_view(*slots[0]).num_rows();
  REQUIRE(slot0_rows > 1000);

  // Re-hashing slot 0 with the default seed is the degenerate case the resplit seed avoids:
  // every row has hash % 2 == 0, so a 2-way split leaves one side empty.
  {
    auto ro        = slots[0]->to_read_only();
    auto same_seed = gpu_partition_impl::hash_partition(ro, {0}, 2, stream, *space);
    stream.sync();
    REQUIRE(same_seed.size() == 2);
    CHECK(sirius::get_cudf_table_view(*same_seed[0]).num_rows() == slot0_rows);
    CHECK(sirius::get_cudf_table_view(*same_seed[1]).num_rows() == 0);
  }

  // split_input hashes with its own seed, so the slot divides again.
  partitioned_operator_data slot_input({slots[0]}, /*partition_idx=*/0);
  auto pieces = merge->split_input(slot_input, 2, /*split_round=*/0, stream);
  stream.sync();
  REQUIRE(pieces.size() == 2);
  auto const r0 = rows_of(*pieces[0]);
  auto const r1 = rows_of(*pieces[1]);
  CHECK(r0 + r1 == static_cast<std::size_t>(slot0_rows));
  CHECK(r0 > static_cast<std::size_t>(slot0_rows) / 4);
  CHECK(r1 > static_cast<std::size_t>(slot0_rows) / 4);
}

TEST_CASE("merge split_input: a single key cannot be split and says so",
          "[physical_grouped_aggregate_merge][oom_split]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = default_stream();
  auto merge  = make_sum_merge();

  SECTION("one key across several batches")
  {
    std::vector<std::shared_ptr<cucascade::data_batch>> input{make_partial(*space, 42, 1, 1),
                                                              make_partial(*space, 42, 1, 2),
                                                              make_partial(*space, 42, 1, 3)};
    pipelineable_operator_data data(input);
    auto pieces = merge->split_input(data, 2, 0, stream);
    stream.sync();
    REQUIRE(pieces.size() == 1);
    CHECK(rows_of(*pieces[0]) == 3);
    CHECK(dynamic_cast<const partitioned_operator_data*>(pieces[0].get()) == nullptr);
  }

  SECTION("one row")
  {
    std::vector<std::shared_ptr<cucascade::data_batch>> input{make_partial(*space, 7, 1, 1)};
    partitioned_operator_data data(input, 5);
    auto pieces = merge->split_input(data, 2, 0, stream);
    REQUIRE(pieces.size() == 1);
    auto const* piece = dynamic_cast<const partitioned_operator_data*>(pieces[0].get());
    REQUIRE(piece != nullptr);
    CHECK(piece->get_partition_idx() == std::optional<std::size_t>{5});
  }

  SECTION("fewer than two pieces is rejected as a request")
  {
    std::vector<std::shared_ptr<cucascade::data_batch>> input{make_partial(*space, 0, 10, 1)};
    pipelineable_operator_data data(input);
    CHECK_THROWS_AS(merge->split_input(data, 1, 0, stream), std::invalid_argument);
  }
}

TEST_CASE("merge split_input: empty input batches contribute nothing and are dropped",
          "[physical_grouped_aggregate_merge][oom_split]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  auto stream = default_stream();
  auto merge  = make_sum_merge();

  std::vector<std::shared_ptr<cucascade::data_batch>> input{
    make_partial(*space, 0, 0, 1), make_partial(*space, 0, 600, 1), make_partial(*space, 0, 0, 1)};
  pipelineable_operator_data data(input);
  auto pieces = merge->split_input(data, 2, 0, stream);
  stream.sync();
  REQUIRE(pieces.size() == 2);
  std::size_t batches = 0;
  for (auto const& piece : pieces) {
    auto const& p = dynamic_cast<const pipelineable_operator_data&>(*piece);
    batches += p.get_data_batches().size();
    for (auto const& ro : p.get_read_only_batches()) {
      CHECK(sirius::get_cudf_table_view(ro).num_rows() > 0);
    }
  }
  // The one non-empty input batch yields one batch per piece; the empty ones none.
  CHECK(batches == 2);
  CHECK(rows_of(*pieces[0]) + rows_of(*pieces[1]) == 600);
}
