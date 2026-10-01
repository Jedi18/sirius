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

// End-to-end OOM recovery by input splitting, through the real gpu_pipeline_executor and
// gpu_pipeline_task: a MERGE_GROUP_BY whose execute() reports out-of-memory whenever its input
// exceeds a row budget. The executor must retry, then hash-split the preserved input into pieces
// that fit, and the pieces' outputs must equal the one-shot merge with no group lost or doubled.
// When nothing can ever fit, recovery must stay bounded and publish nothing.

#include "catch.hpp"
#include "data/data_batch_utils.hpp"
#include "exec/channel.hpp"
#include "exec/config.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_operator.hpp"
#include "operator/aggregate/aggregate_test_utils.hpp"
#include "operator/operator_test_utils.hpp"
#include "operator/operator_type_traits.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"
#include "pipeline/task_request.hpp"
#include "utils/telemetry_utils.hpp"

#include <rmm/error.hpp>

#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace sirius::test::operator_utils;
using Traits = gpu_type_traits<int64_t>;

constexpr std::size_t kGpuCapacity = 1100ULL * 1024 * 1024;

//------------------------------------------------------------------------------
// A grouped-aggregate merge that "does not fit" above a row budget.
//------------------------------------------------------------------------------
class oom_injecting_merge : public sirius::op::sirius_physical_grouped_aggregate_merge {
 public:
  static std::unique_ptr<oom_injecting_merge> make()
  {
    auto agg = sirius::test::create_aggregate_expressions<Traits>({0}, {"sum"}, {1});
    return std::make_unique<oom_injecting_merge>(
      std::move(agg.output_types), std::move(agg.aggregates), std::move(agg.groups));
  }

  oom_injecting_merge(duckdb::vector<sirius::logical_type> types,
                      duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
                      duckdb::vector<std::unique_ptr<sirius::ast::node>> groups)
    : sirius_physical_grouped_aggregate_merge(
        std::move(types), std::move(expressions), std::move(groups), 0)
  {
  }

  std::unique_ptr<sirius::op::operator_data> execute(const sirius::op::operator_data& input,
                                                     ::cuda::stream_ref stream) override
  {
    executes.fetch_add(1, std::memory_order_relaxed);
    std::size_t rows = 0;
    auto const& p    = dynamic_cast<const sirius::op::pipelineable_operator_data&>(input);
    for (auto const& ro : p.get_read_only_batches()) {
      rows += static_cast<std::size_t>(sirius::get_cudf_table_view(ro).num_rows());
    }
    {
      std::lock_guard<std::mutex> lock(rows_mutex);
      rows_seen.push_back(rows);
    }
    if (rows > fail_above_rows.load(std::memory_order_relaxed)) {
      injected_ooms.fetch_add(1, std::memory_order_relaxed);
      throw rmm::out_of_memory("injected: a merge over " + std::to_string(rows) +
                               " rows does not fit");
    }
    return sirius_physical_grouped_aggregate_merge::execute(input, stream);
  }

  std::atomic<std::size_t> fail_above_rows{0};
  std::atomic<int> executes{0};
  std::atomic<int> injected_ooms{0};
  std::mutex rows_mutex;
  std::vector<std::size_t> rows_seen;
};

//------------------------------------------------------------------------------
// A sink that keeps every batch it is handed.
//------------------------------------------------------------------------------
class collecting_sink : public sirius::op::sirius_physical_operator {
 public:
  collecting_sink()
    : sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::FILTER,
                               sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),
                               0)
  {
  }
  std::string get_name() const override { return "collecting_sink"; }
  bool is_sink() const override { return true; }

  void sink(const sirius::op::operator_data& input, ::cuda::stream_ref) override
  {
    auto const& p = dynamic_cast<const sirius::op::pipelineable_operator_data&>(input);
    std::lock_guard<std::mutex> lock(mutex);
    for (auto const& batch : p.get_data_batches()) {
      batches.push_back(batch);
    }
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> snapshot()
  {
    std::lock_guard<std::mutex> lock(mutex);
    return batches;
  }

  std::mutex mutex;
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
};

//------------------------------------------------------------------------------
// Fixture: real memory manager + executor, a one-operator pipeline (merge -> sink).
//------------------------------------------------------------------------------
struct split_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* mem_space = nullptr;
  sirius::exec::channel<std::unique_ptr<sirius::pipeline::task_request>> request_channel;
  std::unique_ptr<sirius::pipeline::gpu_pipeline_executor> executor;
  std::shared_ptr<sirius::pipeline::completion_handler> completion =
    std::make_shared<sirius::pipeline::completion_handler>();

  std::shared_ptr<sirius::operator_params> params;
  std::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline;
  std::unique_ptr<oom_injecting_merge> merge;
  std::unique_ptr<collecting_sink> sink;
  std::shared_ptr<sirius::pipeline::sirius_pipeline_task_global_state> global_state;

  ~split_fixture()
  {
    // Stop first so no worker still references the operators below; then destroy whatever is
    // still queued while the pipeline and operators it reports completion to are alive.
    if (executor) {
      executor->stop();
      executor->drain_leftover_tasks();
    }
    request_channel.close();
  }

  bool setup(uint32_t split_after_retries, uint32_t max_split_depth, int num_threads = 2)
  {
    try {
      cucascade::memory::reservation_manager_configurator builder;
      builder.set_number_of_gpus(1)
        .set_gpu_usage_limit(kGpuCapacity)
        .set_reservation_fraction_per_gpu(0.95)
        .set_per_numa_region_capacity(1ULL * 1024 * 1024 * 1024)
        .use_gpu_id_as_host_id()
        .track_reservation_per_stream(false)
        .set_reservation_fraction_per_numa_region(0.75);
      manager =
        std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    } catch (const std::exception&) {
      return false;
    }
    mem_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
    if (!mem_space) { return false; }

    auto telemetry = sirius::test::make_test_telemetry_context();
    sirius::exec::thread_pool_config config;
    config.num_threads        = num_threads;
    config.thread_name_prefix = "oom-split";
    executor                  = std::make_unique<sirius::pipeline::gpu_pipeline_executor>(
      config, mem_space, request_channel.make_publisher(), nullptr, telemetry);

    params                          = std::make_shared<sirius::operator_params>();
    params->oom_split_after_retries = split_after_retries;
    params->oom_split_max_depth     = max_split_depth;
    sirius::pipeline::pipeline_build_context const ctx{telemetry, true, 1, params};
    pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(ctx);
    pipeline->set_pipeline_id(7);
    merge = oom_injecting_merge::make();
    sink  = std::make_unique<collecting_sink>();

    sirius::pipeline::sirius_pipeline_build_state build_state;
    build_state.set_pipeline_source(*pipeline, *merge);
    build_state.add_pipeline_operator(*pipeline, *merge);
    build_state.set_pipeline_sink(*pipeline, *sink, 1);
    std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines{pipeline};
    sirius::pipeline::assign_operator_ids(pipelines);
    merge->set_pipeline(pipeline);
    sink->set_pipeline(pipeline);

    global_state =
      std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(pipeline, telemetry);
    global_state->set_completion_handler(completion);
    executor->start();
    return true;
  }

  /// One partial-aggregate batch: distinct keys first_key..first_key+n, value = key * weight.
  std::shared_ptr<cucascade::data_batch> make_partial(int64_t first_key, int64_t n, int64_t weight)
  {
    std::vector<int64_t> keys(static_cast<std::size_t>(n));
    std::vector<int64_t> values(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      keys[static_cast<std::size_t>(i)]   = first_key + i;
      values[static_cast<std::size_t>(i)] = (first_key + i) * weight;
    }
    return make_two_column_batch<int64_t, int64_t>(*mem_space, keys, values, cudf::type_id::INT64);
  }

  /// The task the task creator would build for one merge partition.
  void schedule_merge_task(std::vector<std::shared_ptr<cucascade::data_batch>> batches)
  {
    auto input = std::make_unique<sirius::op::partitioned_operator_data>(std::move(batches), 0);
    auto local =
      std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(input));
    auto task = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
      1, std::vector<cucascade::shared_data_repository*>{}, std::move(local), global_state);
    executor->schedule(std::move(task));
  }

  /// Wait for the pipeline to finish or the query to fail; false on timeout.
  bool wait_for_outcome(std::chrono::seconds timeout)
  {
    auto const start = std::chrono::steady_clock::now();
    while (!pipeline->is_pipeline_finished() && !completion->has_error()) {
      if (std::chrono::steady_clock::now() - start > timeout) { return false; }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    // Let the failing attempt's lambda unwind so the executor is quiescent for assertions.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return true;
  }

  /// key -> SUM over every batch the sink received; REQUIREs each key appears exactly once.
  std::map<int64_t, int64_t> collected_sums()
  {
    std::map<int64_t, int64_t> sums;
    for (auto const& batch : sink->snapshot()) {
      auto view   = sirius::get_cudf_table_view(*batch);
      auto keys   = copy_column_to_host<int64_t>(view.column(0));
      auto values = copy_column_to_host<int64_t>(view.column(1));
      REQUIRE(keys.size() == values.size());
      for (std::size_t i = 0; i < keys.size(); ++i) {
        auto [it, inserted] = sums.emplace(keys[i], values[i]);
        INFO("group " << keys[i] << " was published more than once");
        REQUIRE(inserted);
      }
    }
    return sums;
  }
};

/// Four overlapping partials over kKeys-wide key ranges; returns the expected SUM per key.
constexpr int64_t kKeys = 1000;

std::map<int64_t, int64_t> expected_sums()
{
  std::map<int64_t, int64_t> sums;
  auto add = [&](int64_t first, int64_t n, int64_t weight) {
    for (int64_t k = first; k < first + n; ++k) {
      sums[k] += k * weight;
    }
  };
  add(0, kKeys, 1);
  add(0, kKeys, 2);
  add(kKeys / 2, kKeys, 3);
  add(kKeys, kKeys, 5);
  return sums;
}

std::vector<std::shared_ptr<cucascade::data_batch>> make_input(split_fixture& f)
{
  return {f.make_partial(0, kKeys, 1),
          f.make_partial(0, kKeys, 2),
          f.make_partial(kKeys / 2, kKeys, 3),
          f.make_partial(kKeys, kKeys, 5)};
}
constexpr std::size_t kTotalRows = static_cast<std::size_t>(4 * kKeys);

}  // namespace

TEST_CASE(
  "OOM split: after the plain retry the merge input is split and the pieces reproduce "
  "the full merge",
  "[gpu_pipeline_executor][oom][oom_split]")
{
  split_fixture f;
  if (!f.setup(/*split_after_retries=*/1, /*max_split_depth=*/4)) {
    WARN("Skipping OOM split test — no GPU available.");
    return;
  }
  // The whole input (4000 rows) does not fit; a half (~2000) does.
  f.merge->fail_above_rows = kTotalRows * 3 / 4;
  f.schedule_merge_task(make_input(f));

  REQUIRE(f.wait_for_outcome(std::chrono::seconds(60)));
  REQUIRE_FALSE(f.completion->has_error());

  // Attempt 1 OOMs, the plain retry OOMs, the third attempt splits, two pieces run.
  CHECK(f.merge->injected_ooms.load() == 2);
  CHECK(f.merge->executes.load() == 4);
  // original + plain retry + split task + 2 pieces, every one accounted for.
  CHECK(f.pipeline->get_tasks_created() == 5);
  CHECK(f.pipeline->get_tasks_completed() == 5);
  CHECK(f.sink->snapshot().size() == 2);

  auto const got  = f.collected_sums();
  auto const want = expected_sums();
  REQUIRE(got.size() == want.size());
  for (auto const& [key, sum] : want) {
    INFO("group " << key);
    REQUIRE(got.count(key) == 1);
    CHECK(got.at(key) == sum);
  }
}

TEST_CASE("OOM split: pieces that still do not fit are split again, down to ones that do",
          "[gpu_pipeline_executor][oom][oom_split]")
{
  split_fixture f;
  if (!f.setup(/*split_after_retries=*/0, /*max_split_depth=*/4)) {
    WARN("Skipping OOM split test — no GPU available.");
    return;
  }
  // Neither the whole input nor a half fits; a quarter (~1000 rows) does.
  f.merge->fail_above_rows = kTotalRows * 3 / 8;
  f.schedule_merge_task(make_input(f));

  REQUIRE(f.wait_for_outcome(std::chrono::seconds(60)));
  REQUIRE_FALSE(f.completion->has_error());

  // 1 (whole) + 2 (halves) OOM; 4 quarters succeed. Zero plain retries configured.
  CHECK(f.merge->injected_ooms.load() == 3);
  CHECK(f.merge->executes.load() == 7);
  // original + its split task + 2 halves + their 2 split tasks + 4 quarters.
  CHECK(f.pipeline->get_tasks_created() == 10);
  CHECK(f.pipeline->get_tasks_completed() == 10);
  CHECK(f.sink->snapshot().size() == 4);

  auto const got  = f.collected_sums();
  auto const want = expected_sums();
  REQUIRE(got.size() == want.size());
  for (auto const& [key, sum] : want) {
    INFO("group " << key);
    REQUIRE(got.count(key) == 1);
    CHECK(got.at(key) == sum);
  }
}

TEST_CASE(
  "OOM split: an input that never fits fails the query within the depth cap and "
  "publishes nothing",
  "[gpu_pipeline_executor][oom][oom_split][max_retries]")
{
  split_fixture f;
  if (!f.setup(/*split_after_retries=*/0, /*max_split_depth=*/2)) {
    WARN("Skipping OOM split test — no GPU available.");
    return;
  }
  f.merge->fail_above_rows = 0;  // nothing ever fits
  f.schedule_merge_task(make_input(f));

  REQUIRE(f.wait_for_outcome(std::chrono::seconds(120)));
  REQUIRE(f.completion->has_error());
  CHECK(f.sink->snapshot().empty());

  // Depth 0 (1 run), depth 1 (2 runs), depth 2 (4 runs); below the cap no further split, so the
  // smallest input ever merged is a quarter: four distinct non-zero sizes at most.
  std::vector<std::size_t> rows;
  {
    std::lock_guard<std::mutex> lock(f.merge->rows_mutex);
    rows = f.merge->rows_seen;
  }
  std::size_t smallest = kTotalRows;
  for (auto r : rows) {
    smallest = std::min(smallest, r);
  }
  CHECK(smallest > kTotalRows / 8);  // never split past depth 2 (quarters)
  CHECK(f.merge->injected_ooms.load() >= 1 + 2 + 4);
  // Every attempt of the lineage counts towards the executor's retry cap, so the number of runs
  // is bounded by (pieces) x (cap).
  CHECK(f.merge->executes.load() <= 7 * 100);

  f.executor->drain_and_wait();
  CHECK(f.executor->is_task_queue_empty());
}

TEST_CASE("OOM split: a depth cap of zero keeps the pre-existing plain retry path",
          "[gpu_pipeline_executor][oom][oom_split][max_retries]")
{
  split_fixture f;
  if (!f.setup(/*split_after_retries=*/0, /*max_split_depth=*/0)) {
    WARN("Skipping OOM split test — no GPU available.");
    return;
  }
  f.merge->fail_above_rows = 0;
  f.schedule_merge_task(make_input(f));

  REQUIRE(f.wait_for_outcome(std::chrono::seconds(120)));
  REQUIRE(f.completion->has_error());
  CHECK(f.sink->snapshot().empty());
  // Every run saw the whole input: no split ever happened.
  std::lock_guard<std::mutex> lock(f.merge->rows_mutex);
  for (auto r : f.merge->rows_seen) {
    CHECK(r == kTotalRows);
  }
  // The original attempt plus MAX_RETRIES rescheduled attempts.
  CHECK(f.merge->executes.load() == 101);
}

TEST_CASE("OOM split: a single-key input cannot be split and falls back to plain retries",
          "[gpu_pipeline_executor][oom][oom_split][max_retries]")
{
  split_fixture f;
  if (!f.setup(/*split_after_retries=*/0, /*max_split_depth=*/4)) {
    WARN("Skipping OOM split test — no GPU available.");
    return;
  }
  f.merge->fail_above_rows = 0;
  f.schedule_merge_task(
    {f.make_partial(42, 1, 1), f.make_partial(42, 1, 2), f.make_partial(42, 1, 3)});

  REQUIRE(f.wait_for_outcome(std::chrono::seconds(120)));
  REQUIRE(f.completion->has_error());
  CHECK(f.sink->snapshot().empty());
  // Every attempt ran the three unsplittable rows: the split task found one key, marked the
  // input exhausted and ran it; from then on only plain retries happened, up to the cap.
  std::lock_guard<std::mutex> lock(f.merge->rows_mutex);
  REQUIRE_FALSE(f.merge->rows_seen.empty());
  for (auto r : f.merge->rows_seen) {
    CHECK(r == 3);
  }
  CHECK(f.merge->executes.load() == 101);
}
