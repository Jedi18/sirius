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

#include "pipeline/data_size_estimator.hpp"

#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sirius {
namespace pipeline {

namespace {

/// How many distinct data-carrying producer pipelines feed a pipeline, and (when there is
/// exactly one) which. See @ref scan_producer_pipelines.
struct producer_scan {
  std::size_t count      = 0;
  sirius_pipeline* first = nullptr;
};

/**
 * @brief Count the data-carrying ports feeding @p pipeline and identify the first producer.
 *
 * Mirrors sirius_pipeline::get_ingress_ports_info, which returns the producer *operator* where
 * we need the producer *pipeline*. `operators` spans source through sink after is_ready(), so
 * walking it reaches build-side ports on the sink too. Dependency-only ports (null repo) and
 * ports with no producer carry no bytes and are skipped.
 *
 * Deduplication by port pointer is load-bearing: `operators` normally already contains the
 * source and sink, so without it every port is seen twice and a single-input pipeline reads as
 * a fan-in. A reserved vector rather than a hash set because this runs on every task-creation
 * poll until a projection latches, and port counts are single digits.
 */
producer_scan scan_producer_pipelines(sirius_pipeline& pipeline)
{
  producer_scan result;
  std::vector<const op::sirius_physical_operator::port*> seen;
  seen.reserve(4);

  auto collect_from = [&](op::sirius_physical_operator* candidate) {
    if (candidate == nullptr) { return; }
    for (auto port_id : candidate->get_port_ids()) {
      auto* p = candidate->get_port(port_id);
      if (p == nullptr || p->repo == nullptr || !p->src_pipeline) { continue; }
      if (std::find(seen.begin(), seen.end(), p) != seen.end()) { continue; }
      seen.push_back(p);
      if (result.count == 0) { result.first = p->src_pipeline.get(); }
      result.count++;
    }
  };

  for (auto& op_ref : pipeline.get_operators()) {
    collect_from(&op_ref.get());
  }
  // Sink-only pipelines leave `operators` empty; source/sink are still set.
  collect_from(pipeline.get_source().get());
  collect_from(pipeline.get_sink().get());

  return result;
}

/// The weaker of two sample counts, treating 0 as "no ratio applied yet" rather than as a
/// minimum. Keeps @ref data_size_estimate::ratio_samples reporting the least-supported ratio in
/// the chain.
std::size_t weaker_sample_count(std::size_t so_far, std::size_t candidate)
{
  return so_far == 0 ? candidate : std::min(so_far, candidate);
}

/// Scale @p bytes by @p ratio, refusing anything that would not survive the narrowing to
/// std::size_t. Ratios are quotients of measured byte counts and are normally small and finite,
/// but a corrupt or extreme pair must not produce UB in llround or a wrapped total that then
/// sizes a partition count.
std::optional<std::size_t> scale_checked(std::size_t bytes, double ratio)
{
  if (!std::isfinite(ratio) || ratio < 0.0) { return std::nullopt; }
  double const scaled = static_cast<double>(bytes) * ratio;
  if (!std::isfinite(scaled) || scaled < 0.0) { return std::nullopt; }
  // llround is UB outside the integral type's range; cap at what a byte count can represent.
  constexpr double kMaxBytes = 9.0e18;  // comfortably inside both int64 and size_t
  if (scaled > kMaxBytes) { return std::nullopt; }
  return static_cast<std::size_t>(std::llround(scaled));
}

std::optional<data_size_estimate> apply_pipeline_ratio(sirius_pipeline& pipeline,
                                                       data_size_estimate input,
                                                       const size_estimate_options& options)
{
  auto const totals = pipeline.get_memory_history().totals();
  auto ratio        = pipeline.get_memory_history().output_to_input_ratio();
  // Too few completed tasks is treated exactly as "no ratio yet": the number exists but one or
  // two batches are not evidence that it describes the pipeline, and the consumer latches the
  // first estimate it is given rather than refining it later. See min_ratio_samples.
  if (!ratio.has_value() || totals.records < options.min_ratio_samples) {
    if (!options.assume_unit_ratio) { return std::nullopt; }
    ratio = 1.0;
  }
  auto const scaled = scale_checked(input.bytes, *ratio);
  if (!scaled.has_value()) { return std::nullopt; }
  return data_size_estimate{
    .bytes = *scaled,
    // A learned ratio is a projection, never a measurement.
    .exact         = false,
    .hops          = input.hops,
    .ratio_samples = weaker_sample_count(input.ratio_samples, totals.records),
  };
}

std::optional<data_size_estimate> estimate_output_bytes_impl(sirius_pipeline& pipeline,
                                                             const size_estimate_options& options,
                                                             std::size_t hops)
{
  if (hops > options.max_hops) { return std::nullopt; }

  // 1. Finished pipeline: its recorded output total is the exact answer. Safe to read because
  //    pipeline_finished is only set once tasks_created == tasks_completed, and each task
  //    records its output before mark_task_completed() runs in its destructor.
  //
  //    Zero records is NOT reported as zero bytes: record() drops tasks whose input_basis is 0,
  //    so it means "no evidence", not "produced nothing". Reporting 0 would size a large input
  //    onto one partition; callers fall back to waiting, which is also right when the pipeline
  //    genuinely emitted nothing.
  if (pipeline.is_pipeline_finished()) {
    auto const totals = pipeline.get_memory_history().totals();
    if (totals.records == 0) { return std::nullopt; }
    return data_size_estimate{
      .bytes = totals.output_bytes,
      .exact = true,
      .hops  = hops,
    };
  }

  auto const producers = scan_producer_pipelines(pipeline);

  // 2. Fan-in: follow only the source's nominated primary input.
  //
  //    The recorded input_basis is unusable here: a STANDARD join pairs each probe batch with
  //    every build batch and borrows rather than pops, so the same bytes enter input_basis once
  //    per pairing and its sum is a cross product, not an input volume. Ask the operator for
  //    probe bytes counted once per batch instead, and divide the pipeline's output total by
  //    that. An operator nominating no primary port (CTE, delim-join wiring) yields nullopt.
  if (producers.count > 1) {
    auto* source = pipeline.get_source().get();
    if (source == nullptr) { return std::nullopt; }

    auto const port_name = source->primary_input_port();
    auto const consumed  = source->consumed_primary_input_bytes();
    if (!port_name.has_value() || !consumed.has_value() || *consumed == 0) { return std::nullopt; }

    // This ratio reads low while tasks are in flight, and worst at the first opportunity to
    // sample it. See size_estimate_options::min_fan_in_ratio_samples.
    auto const totals = pipeline.get_memory_history().totals();
    if (totals.records < options.min_fan_in_ratio_samples) { return std::nullopt; }

    // get_port throws on an unknown name, and the name here comes from an operator override —
    // a throw would escape into get_next_task_hint(). Treat a bad nomination as "no estimate".
    auto const ids = source->get_port_ids();
    if (std::find(ids.begin(), ids.end(), *port_name) == ids.end()) { return std::nullopt; }

    auto* p = source->get_port(*port_name);
    if (p == nullptr || p->repo == nullptr || !p->src_pipeline) { return std::nullopt; }

    auto upstream = estimate_output_bytes_impl(*p->src_pipeline, options, hops + 1);
    if (!upstream.has_value()) { return std::nullopt; }

    // The residual in-flight bias is bounded by min_fan_in_ratio_samples above, not corrected
    // here: task counts cannot be used to discount it, because `consumed` does not advance once
    // per task. See data-size-estimation.md.
    auto const ratio  = static_cast<double>(totals.output_bytes) / static_cast<double>(*consumed);
    auto const scaled = scale_checked(upstream->bytes, ratio);
    if (!scaled.has_value()) { return std::nullopt; }
    return data_size_estimate{
      .bytes         = *scaled,
      .exact         = false,
      .hops          = upstream->hops,
      .ratio_samples = weaker_sample_count(upstream->ratio_samples, totals.records),
    };
  }

  // 3. Leaf: anchor on what the source operator knows about its own total.
  if (producers.count == 0) {
    auto* source = pipeline.get_source().get();
    if (source == nullptr) { return std::nullopt; }

    if (auto input_bytes = source->total_source_input_bytes()) {
      return apply_pipeline_ratio(
        pipeline, data_size_estimate{.bytes = *input_bytes, .exact = true, .hops = hops}, options);
    }
    // Already an output quantity — returned as-is. Scaling it by the pipeline ratio (which is
    // derived from pre-filter input bytes) would double-count filter selectivity.
    if (auto output_bytes = source->total_source_output_bytes()) {
      return data_size_estimate{.bytes = *output_bytes, .exact = false, .hops = hops};
    }
    return std::nullopt;
  }

  // 4. Single producer: recurse, then scale by this pipeline's ratio.
  if (producers.first == nullptr) { return std::nullopt; }
  auto upstream = estimate_output_bytes_impl(*producers.first, options, hops + 1);
  if (!upstream.has_value()) { return std::nullopt; }
  return apply_pipeline_ratio(pipeline, *upstream, options);
}

}  // namespace

std::optional<data_size_estimate> estimate_pipeline_total_output_bytes(
  sirius_pipeline& pipeline, size_estimate_options options)
{
  return estimate_output_bytes_impl(pipeline, options, /*hops=*/0);
}

std::optional<data_size_estimate> estimate_port_total_input_bytes(op::sirius_physical_operator& op,
                                                                  std::string_view port_id,
                                                                  size_estimate_options options)
{
  // get_port throws when the name is unknown; a missing port is "no estimate", not an error.
  auto ids = op.get_port_ids();
  if (std::find(ids.begin(), ids.end(), port_id) == ids.end()) { return std::nullopt; }

  auto* p = op.get_port(port_id);
  if (p == nullptr || p->repo == nullptr || !p->src_pipeline) { return std::nullopt; }
  return estimate_pipeline_total_output_bytes(*p->src_pipeline, options);
}

}  // namespace pipeline
}  // namespace sirius
