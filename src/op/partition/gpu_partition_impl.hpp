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

#include "telemetry/data_batch_probe.hpp"

#include <cudf/cudf_utils.hpp>
#include <cudf/hashing.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <vector>

namespace sirius {

namespace telemetry {
class telemetry_context;
}  // namespace telemetry

namespace op {

/**
 * @brief Functionalities for partitioning the input data batch into multiple output batches.
 *
 * Provide functionalities including:
 * - Hash partitioning with specified partitioning columns;
 * - Evenly partitioning to evenly split the input table.
 *
 * Require caller to have already upgraded input data batches into `gpu_table_representation`.
 */
class gpu_partition_impl {
 public:
  /// Hash seed for re-splitting data that the plan-level PARTITION already hash-partitioned.
  ///
  /// The PARTITION operator places a row in slot `murmur3(keys, DEFAULT_HASH_SEED) % P`. Splitting
  /// one of those slots again with the same seed is degenerate whenever the new count shares a
  /// factor with `P` (every row of slot `p` has `hash % 2 == p % 2`, so a 2-way split of a 2- or
  /// 4-way partition leaves one piece empty). Each split round therefore hashes with its own seed,
  /// none of which is the default, so the rounds' boundaries are independent of each other and of
  /// the plan's.
  [[nodiscard]] static constexpr uint32_t resplit_hash_seed(uint32_t split_round) noexcept
  {
    constexpr uint32_t kResplitSeedBase = 0x9E3779B9u;  // never cudf::DEFAULT_HASH_SEED (0)
    return kResplitSeedBase + split_round;
  }

  /**
   * @brief Perform hash partitioning on the input data batch.
   *
   * @param input The input batch to be hash partitioned.
   * @param partition_key_idx Column ids of the partitioning columns.
   * @param partition_key_cast_types Per-key target types for the partition hash.
   * @param num_partitions Number of partitions.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   * @param telemetry_info Telemetry lineage for the output batches.
   * @param seed Murmur3 seed. The plan-level PARTITION uses cuDF's default; a re-split of
   *             already-partitioned data must pass @ref resplit_hash_seed instead.
   *
   * @return The output data batches, one per partition, in partition order. Partitions that
   *         received no rows are returned as zero-row batches.
   */
  static std::vector<std::shared_ptr<cucascade::data_batch>> hash_partition(
    const cucascade::read_only_data_batch& input,
    const std::vector<int>& partition_key_idx,
    const std::vector<cudf::data_type>& partition_key_cast_types,
    int num_partitions,
    ::cuda::stream_ref stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info,
    uint32_t seed);

  /// Existing call path, using cuDF's default seed.
  static std::vector<std::shared_ptr<cucascade::data_batch>> hash_partition(
    const cucascade::read_only_data_batch& input,
    const std::vector<int>& partition_key_idx,
    const std::vector<cudf::data_type>& partition_key_cast_types,
    int num_partitions,
    ::cuda::stream_ref stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {});

  /// Overload without cast types (all keys hashed as-is). Kept for backward compatibility.
  static std::vector<std::shared_ptr<cucascade::data_batch>> hash_partition(
    const cucascade::read_only_data_batch& input,
    const std::vector<int>& partition_key_idx,
    int num_partitions,
    ::cuda::stream_ref stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {})
  {
    return hash_partition(
      input, partition_key_idx, {}, num_partitions, stream, memory_space, telemetry_info);
  }

  /**
   * @brief Perform evenly partitioning on the input data batch.
   *
   * @param input The input batch to be evenly partitioned.
   * @param num_partitions Number of partitions.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param memory_space The memory space used to allocate memory for the output data batch.
   *
   * @return The output data batches.
   */
  static std::vector<std::shared_ptr<cucascade::data_batch>> evenly_partition(
    const cucascade::read_only_data_batch& input,
    int num_partitions,
    ::cuda::stream_ref stream,
    cucascade::memory::memory_space& memory_space,
    const telemetry::batch_telemetry_info& telemetry_info = {});
};

}  // namespace op
}  // namespace sirius
