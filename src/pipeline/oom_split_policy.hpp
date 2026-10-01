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

#include <cstdint>

namespace sirius::pipeline {

/// What the GPU executor does with a task that ran out of memory at an operator.
enum class oom_recovery_action : uint8_t {
  /// Re-run the same input with a larger reservation (the pre-existing behaviour).
  retry,
  /// Hash-split the input at the failing operator into `pieces` smaller inputs, each becoming
  /// its own task, and run those instead.
  split,
};

/// The knobs of the split decision. Mirrors the `oom_split_*` operator_params.
struct oom_split_config {
  /// Plain retries an input gets before the executor splits it instead. 0 splits on the first
  /// OOM; the default lets one retry absorb a transient contention OOM first.
  uint32_t split_after_retries = 1;
  /// How many times one task's input may be split along its lineage. 0 disables splitting. The
  /// pieces at the deepest level fall back to plain retries, bounded by the executor's retry cap.
  uint32_t max_split_depth = 4;
  /// Pieces per split round. Fixed: a binary split halves the working set per round while
  /// keeping the task count per lineage at most 2^max_split_depth.
  static constexpr uint32_t pieces_per_split = 2;
};

/// The per-lineage facts the decision reads, all carried on the task's local state.
struct oom_recovery_state {
  /// The operator that failed can split its input (sirius_physical_operator::supports_input_split).
  bool operator_supports_split = false;
  /// A previous split attempt on this input could not divide it (one key or one row).
  bool split_exhausted = false;
  /// Splits already applied along this input's lineage.
  uint32_t split_depth = 0;
  /// OOM reschedules of this exact input since it was created or last split, INCLUDING the one
  /// being decided.
  uint32_t ooms_since_split = 0;
};

/**
 * @brief Decide between retrying an OOM'd task and splitting its input.
 *
 * Pure so it can be tested without a GPU. The rules, in order:
 *  1. An operator that cannot split, or an input that proved unsplittable, is always retried.
 *  2. The first `split_after_retries` OOMs of an input are retried with a larger reservation,
 *     exactly as before this policy existed: that path resolves OOMs caused by other tasks
 *     holding memory, where splitting would only add a partition pass.
 *  3. After that the input is split, as long as the lineage is below `max_split_depth`.
 *  4. At the depth cap the pieces are retried; the executor's retry cap bounds that.
 */
[[nodiscard]] constexpr oom_recovery_action decide_oom_recovery(
  const oom_split_config& config, const oom_recovery_state& state) noexcept
{
  if (!state.operator_supports_split || state.split_exhausted) {
    return oom_recovery_action::retry;
  }
  if (state.ooms_since_split <= config.split_after_retries) { return oom_recovery_action::retry; }
  if (state.split_depth >= config.max_split_depth) { return oom_recovery_action::retry; }
  return oom_recovery_action::split;
}

}  // namespace sirius::pipeline
