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

// Unit tests for grouped_merge_partition_strategy(): the partition count and GPU subset of a
// GROUP BY merge exchange. GPU-free; the end-to-end placement is covered by the [mgpu] tests in
// test_physical_grouped_aggregate_merge_mgpu.cpp.

#include "op/partition_placement.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"

#include <catch.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

using sirius::op::effective_min_bytes_per_gpu;
using sirius::op::effective_min_bytes_to_trigger_partitioning;
using sirius::op::GROUPED_MERGE_MAX_ROWS_PER_PARTITION;
using sirius::op::grouped_merge_partition_strategy;
using sirius::op::partition_strategy;

namespace {

constexpr uint64_t kMiB = 1024ull * 1024;
constexpr uint64_t kGiB = 1024 * kMiB;
constexpr uint64_t kH   = 5 * kGiB;  // Live default hash_partition_bytes on a large GPU.

std::vector<int> const kFourGpus{0, 1, 2, 3};

/// The strategy with the derived (0) cliff and bytes-per-GPU, and few rows.
partition_strategy defaults(uint64_t bytes,
                            std::vector<int> const& gpus = kFourGpus,
                            std::size_t operator_id      = 0,
                            uint64_t rows                = 1000)
{
  return grouped_merge_partition_strategy(bytes, rows, kH, 0, 0, gpus, operator_id);
}

}  // namespace

TEST_CASE("grouped_merge_partition_strategy - 0 defaults resolve to 2 x H and H",
          "[grouped_merge_partition_strategy][unit]")
{
  CHECK(effective_min_bytes_to_trigger_partitioning(0, kH) == 2 * kH);
  CHECK(effective_min_bytes_to_trigger_partitioning(3 * kGiB, kH) == 3 * kGiB);
  CHECK(effective_min_bytes_to_trigger_partitioning(0, UINT64_MAX) == UINT64_MAX);
  CHECK(effective_min_bytes_per_gpu(0, kH) == kH);
  CHECK(effective_min_bytes_per_gpu(7 * kGiB, kH) == 7 * kGiB);

  // Just below 2 x H stays on one partition; at 2 x H the merge partitions.
  CHECK(defaults(2 * kH - 1).num_partitions == 1);
  CHECK(defaults(2 * kH).num_partitions == 2);
}

TEST_CASE("grouped_merge_partition_strategy - below the cliff one partition on one GPU",
          "[grouped_merge_partition_strategy][unit]")
{
  for (uint64_t const bytes : {uint64_t{0}, 70 * kMiB, 2600 * kMiB, kH + 1, 2 * kH - 1}) {
    auto const s = defaults(bytes);
    INFO("bytes=" << bytes);
    CHECK(s.num_partitions == 1);
    CHECK(s.placement.num_partitions() == 1);
    CHECK(s.placement.devices().size() == 1);
    CHECK_FALSE(s.broadcast);
    CHECK_FALSE(s.build_probe);
  }
}

TEST_CASE("grouped_merge_partition_strategy - worked examples at H = 5 GiB on 4 GPUs",
          "[grouped_merge_partition_strategy][unit]")
{
  // 12 GiB: k = ceil(12 / 5) = 3, P = max(ceil(12 / 5), 3) = 3.
  auto const twelve = defaults(12 * kGiB);
  CHECK(twelve.num_partitions == 3);
  CHECK(twelve.placement.devices().size() == 3);

  // 40 GiB: k = clamp(8, 1, 4) = 4, P = 8 round-robin over all four GPUs.
  auto const forty = defaults(40 * kGiB);
  CHECK(forty.num_partitions == 8);
  CHECK(forty.placement.devices() == kFourGpus);
}

TEST_CASE("grouped_merge_partition_strategy - k clamps to the GPU count and P >= k",
          "[grouped_merge_partition_strategy][unit]")
{
  // A small bytes-per-GPU asks for many GPUs; only four exist.
  auto const s =
    grouped_merge_partition_strategy(2 * kH, 1000, kH, 0, /*min_bytes_per_gpu=*/1, kFourGpus, 0);
  CHECK(s.placement.devices().size() == 4);
  // ceil(bytes / H) is 2, but every chosen GPU needs a partition.
  CHECK(s.num_partitions == 4);
  CHECK(s.placement.num_partitions() == 4);

  for (uint64_t const bytes : {2 * kH, 3 * kH + 1, 17 * kH, 100 * kH}) {
    auto const t = defaults(bytes);
    INFO("bytes=" << bytes);
    CHECK(t.placement.num_partitions() == static_cast<std::size_t>(t.num_partitions));
    CHECK(static_cast<std::size_t>(t.num_partitions) >= t.placement.devices().size());
    CHECK(t.placement.devices().size() <= kFourGpus.size());
  }
}

TEST_CASE("grouped_merge_partition_strategy - explicit cliff and bytes-per-GPU",
          "[grouped_merge_partition_strategy][unit]")
{
  // A cliff of 1 byte partitions everything; 1 GiB per GPU spreads 3 GiB over 3 GPUs while the
  // count still follows H.
  auto const s = grouped_merge_partition_strategy(3 * kGiB, 1000, kGiB, 1, kGiB, kFourGpus, 0);
  CHECK(s.num_partitions == 3);
  CHECK(s.placement.devices().size() == 3);

  // A cliff above the input keeps a multi-H input on one partition.
  auto const t = grouped_merge_partition_strategy(4 * kH, 1000, kH, 5 * kH, 0, kFourGpus, 0);
  CHECK(t.num_partitions == 1);
  CHECK(t.placement.devices().size() == 1);
}

TEST_CASE("grouped_merge_partition_strategy - the row guard splits below the cliff",
          "[grouped_merge_partition_strategy][unit]")
{
  constexpr uint64_t kMaxRows = GROUPED_MERGE_MAX_ROWS_PER_PARTITION;
  CHECK(defaults(kGiB, kFourGpus, 0, kMaxRows).num_partitions == 1);

  auto const over = defaults(kGiB, kFourGpus, 0, kMaxRows + 1);
  CHECK(over.num_partitions == 2);
  // Below the cliff the extra partitions stay on one GPU.
  CHECK(over.placement.devices().size() == 1);

  // Above the cliff the row guard can exceed the byte count.
  auto const big = defaults(2 * kH, kFourGpus, 0, 5 * kMaxRows);
  CHECK(big.num_partitions == 5);
  CHECK(big.placement.devices().size() == 2);
}

TEST_CASE("grouped_merge_partition_strategy - single-GPU merges rotate by operator id",
          "[grouped_merge_partition_strategy][unit]")
{
  auto const a = defaults(70 * kMiB, kFourGpus, /*operator_id=*/5);
  auto const b = defaults(70 * kMiB, kFourGpus, /*operator_id=*/6);
  REQUIRE(a.placement.devices().size() == 1);
  REQUIRE(b.placement.devices().size() == 1);
  CHECK(a.placement.devices() != b.placement.devices());
  CHECK(a.placement.devices() == std::vector<int>{1});
  CHECK(b.placement.devices() == std::vector<int>{2});

  // Rotation picks from the admitted ids, not from 0..n-1.
  auto const c = defaults(70 * kMiB, std::vector<int>{2, 5}, /*operator_id=*/1);
  CHECK(c.placement.devices() == std::vector<int>{5});
}

TEST_CASE("grouped_merge_partition_strategy - one GPU partitions only above the cliff",
          "[grouped_merge_partition_strategy][unit]")
{
  std::vector<int> const one{0};
  CHECK(defaults(2 * kH - 1, one).num_partitions == 1);
  CHECK(defaults(2 * kH, one).num_partitions == 2);
  auto const s = defaults(40 * kGiB, one);
  CHECK(s.num_partitions == 8);
  CHECK(s.placement.devices() == one);
}

TEST_CASE("grouped_merge_partition_strategy - no GPU list plans one GPU, unpinned",
          "[grouped_merge_partition_strategy][unit]")
{
  std::vector<int> const none;
  auto const small = defaults(70 * kMiB, none);
  CHECK(small.num_partitions == 1);
  CHECK_FALSE(small.placement.any_pinned());

  // Single-GPU arithmetic: the count follows H, with k clamped to 1.
  auto const large = defaults(40 * kGiB, none);
  CHECK(large.num_partitions == 8);
  CHECK(large.placement.num_partitions() == 8);
  CHECK_FALSE(large.placement.any_pinned());
}

TEST_CASE("grouped_merge_partition_strategy rejects a zero partition target",
          "[grouped_merge_partition_strategy][unit]")
{
  CHECK_THROWS_AS(grouped_merge_partition_strategy(kGiB, 1, 0, 0, 0, kFourGpus, 0),
                  std::invalid_argument);
}
