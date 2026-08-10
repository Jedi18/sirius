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

#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

namespace sirius {
namespace op {
class sirius_physical_operator;
}  // namespace op

namespace pipeline {

class sirius_pipeline;

/**
 * @brief Projected total bytes that will flow through a point in the plan.
 *
 * A total for the whole query, not a so-far figure: "how many bytes will this port have
 * received once its producer is done?". See docs/super-sirius/data-size-estimation.md.
 */
struct data_size_estimate {
  std::size_t bytes = 0;
  /// Measured rather than projected: the walk anchored on a finished pipeline or an exactly
  /// known source total, with no learned ratio applied anywhere along the chain.
  bool exact = false;
  /// Pipelines traversed. Diagnostic — ratio error compounds per hop.
  std::size_t hops = 0;
  /// Completed tasks behind the *weakest* ratio in the chain; 0 when the estimate is exact.
  /// Diagnostic: separates "we sampled too early" from "the model is wrong" when one misses.
  std::size_t ratio_samples = 0;
};

/// Tuning for a single estimation call.
struct size_estimate_options {
  /// Use a 1:1 ratio for a pipeline with no completed task, instead of returning nullopt.
  /// Trades accuracy for always getting an answer; never marks the result exact.
  bool assume_unit_ratio = false;
  /// Recursion guard against pathological plan depth, and defensively against graph cycles.
  std::size_t max_hops = 16;
  /// Sample floors below which a ratio is treated as absent (so @ref assume_unit_ratio still
  /// applies). The fan-in floor is far higher because that ratio is systematically biased low
  /// while tasks are in flight, where a single-input ratio is merely noisy. See
  /// docs/super-sirius/data-size-estimation.md#fan-in.
  std::size_t min_ratio_samples        = 4;
  std::size_t min_fan_in_ratio_samples = 16;
};

/**
 * @brief Project the total bytes arriving at @p op's @p port_id input port.
 *
 * @return nullopt for a missing port, a dependency-only port (null repo), a port with no
 *         producer, or when the upstream walk cannot produce an estimate.
 */
[[nodiscard]] std::optional<data_size_estimate> estimate_port_total_input_bytes(
  op::sirius_physical_operator& op, std::string_view port_id, size_estimate_options options = {});

/**
 * @brief Project the total bytes @p pipeline will emit over the whole query.
 *
 * Walks upstream to the first known total, then chains each intervening pipeline's measured
 * output/input ratio back down. Four terminating cases, in the order the implementation tries
 * them (see data_size_estimator.cpp for why each is shaped as it is):
 *
 *  1. finished pipeline — its recorded output total, exactly;
 *  2. fan-in — follow only the source's nominated primary port;
 *  3. leaf — anchor on the source's own total;
 *  4. single producer — recurse, then apply this pipeline's ratio.
 *
 * @return nullopt whenever any link in the chain is unknown.
 */
[[nodiscard]] std::optional<data_size_estimate> estimate_pipeline_total_output_bytes(
  sirius_pipeline& pipeline, size_estimate_options options = {});

}  // namespace pipeline
}  // namespace sirius
