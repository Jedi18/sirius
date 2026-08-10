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

#include "catch.hpp"
#include "pipeline/data_size_estimator.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cucascade/data/data_repository.hpp>

#include <cstddef>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using sirius::op::MemoryBarrierType;
using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::pipeline::estimate_pipeline_total_output_bytes;
using sirius::pipeline::estimate_port_total_input_bytes;
using sirius::pipeline::pipeline_build_context;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;
using sirius::pipeline::size_estimate_options;
using sirius::pipeline::task_memory_record;

namespace {

constexpr std::size_t kMiB = 1024ull * 1024ull;

/// Leaf-source stand-in whose two "what is my total?" answers are set per test.
struct test_source_operator : sirius_physical_operator {
  test_source_operator() : sirius_physical_operator(SiriusPhysicalOperatorType::PROJECTION, {}, 0)
  {
  }

  [[nodiscard]] std::optional<std::size_t> total_source_input_bytes() const override
  {
    return input_total;
  }
  [[nodiscard]] std::optional<std::size_t> total_source_output_bytes() const override
  {
    return output_total;
  }

  // Fan-in nomination: which port carries the primary (probe-equivalent) input, and how many
  // bytes have been taken from it. Both unset by default, matching every non-join operator.
  [[nodiscard]] std::optional<std::string_view> primary_input_port() const override
  {
    if (primary_port.empty()) { return std::nullopt; }
    return std::string_view{primary_port};
  }
  [[nodiscard]] std::optional<std::size_t> consumed_primary_input_bytes() const override
  {
    return consumed_primary;
  }

  std::optional<std::size_t> input_total;
  std::optional<std::size_t> output_total;
  std::string primary_port;  ///< empty = nominates none
  std::optional<std::size_t> consumed_primary;
};

/// Pipeline whose finished state is set directly, instead of being driven through the task
/// counters that update_pipeline_status() normally maintains.
struct test_pipeline : sirius_pipeline {
  explicit test_pipeline(const pipeline_build_context& ctx) : sirius_pipeline(ctx) {}

  [[nodiscard]] bool is_pipeline_finished() const override { return finished; }

  bool finished = false;
};

/// Builds a chain/DAG of single-operator pipelines wired through data-carrying ports, mirroring
/// how the planner wires real pipelines. Each pipeline's operator doubles as source and sink.
class estimator_dag {
 public:
  /// Add a pipeline and return it. `source` is the operator standing in for its leaf source.
  test_pipeline& add()
  {
    auto pipeline = duckdb::make_shared_ptr<test_pipeline>(_ctx);
    auto op       = std::make_unique<test_source_operator>();
    op->set_pipeline(pipeline);
    _bs.set_pipeline_source(*pipeline, *op);
    _bs.set_pipeline_sink(*pipeline, sirius::optional_ptr<sirius_physical_operator>(op.get()), 0);
    auto& ref = *pipeline;
    _ops.push_back(std::move(op));
    _pipelines.push_back(std::move(pipeline));
    return ref;
  }

  /// Wire a data-flow edge from -> to. `with_repo=false` models a dependency-only port.
  void connect(test_pipeline& from,
               test_pipeline& to,
               MemoryBarrierType barrier    = MemoryBarrierType::FULL,
               bool with_repo               = true,
               const std::string& port_name = "")
  {
    auto* consumer_op = to.get_source().get();
    auto* producer_op = from.get_sink().get();

    _names.push_back(port_name.empty() ? "e" + std::to_string(_names.size()) : port_name);
    std::string_view name = _names.back();

    auto port  = std::make_unique<sirius_physical_operator::port>();
    port->type = barrier;
    if (with_repo) {
      _repos.push_back(std::make_unique<cucascade::shared_data_repository>());
      port->repo = _repos.back().get();
    } else {
      port->repo = nullptr;
    }
    port->src_pipeline  = shared_for(from);
    port->dest_pipeline = shared_for(to);
    consumer_op->add_port(name, std::move(port));

    producer_op->add_next_port_after_sink(
      sirius_physical_operator::next_port_info{consumer_op, name, uuid::now_v7()});
  }

  /// The operator standing in as `pipeline`'s leaf source.
  static test_source_operator& source_of(test_pipeline& pipeline)
  {
    return static_cast<test_source_operator&>(*pipeline.get_source());
  }

 private:
  duckdb::shared_ptr<sirius_pipeline> shared_for(test_pipeline& pipeline) const
  {
    for (const auto& p : _pipelines) {
      if (p.get() == &pipeline) { return p; }
    }
    return nullptr;
  }

  pipeline_build_context _ctx{nullptr, true};
  sirius_pipeline_build_state _bs;
  std::vector<std::unique_ptr<test_source_operator>> _ops;
  std::vector<std::unique_ptr<cucascade::shared_data_repository>> _repos;
  std::deque<std::string> _names;  // stable storage backing the port-name string_views
  std::vector<duckdb::shared_ptr<sirius_pipeline>> _pipelines;
};

/// Record one completed task on `pipeline`.
void record_task(test_pipeline& pipeline, std::size_t in_bytes, std::size_t out_bytes)
{
  pipeline.get_memory_history().record(task_memory_record{in_bytes, in_bytes * 2, out_bytes});
}

/// Give `pipeline` a measured output/input ratio backed by `samples` completed tasks.
///
/// Defaults to exactly what size_estimate_options::min_ratio_samples requires, so a pipeline
/// set up this way has a ratio the estimator will trust; cases that are *about* the sample
/// count pass it explicitly. The bytes are split evenly across the samples, so the aggregate
/// ratio is `out_bytes / in_bytes` regardless of the count (every value here is a multiple of
/// kMiB, which divides exactly).
void record_ratio(test_pipeline& pipeline,
                  std::size_t in_bytes,
                  std::size_t out_bytes,
                  std::size_t samples = size_estimate_options{}.min_ratio_samples)
{
  for (std::size_t i = 0; i < samples; ++i) {
    record_task(pipeline, in_bytes / samples, out_bytes / samples);
  }
}

}  // namespace

TEST_CASE("data_size_estimator: a finished pipeline reports its exact recorded output",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer = dag.add();
  record_ratio(producer, 100 * kMiB, 40 * kMiB);
  record_ratio(producer, 100 * kMiB, 60 * kMiB);
  producer.finished = true;

  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 100 * kMiB);
  CHECK(est->exact);
  CHECK(est->hops == 0);
}

TEST_CASE("data_size_estimator: a finished pipeline with no records is unknown, not zero",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer    = dag.add();
  producer.finished = true;
  // record() drops tasks with a zero input basis (a scan_info with no a-priori size estimate),
  // so "no records" carries no information about how much the pipeline emitted. Reporting 0
  // here would size a large input onto a single partition.

  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer).has_value());
}

TEST_CASE("data_size_estimator: an unfinished leaf scales its known total by the measured ratio",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer = dag.add();
  // The source will feed 1 GiB; the pipeline has so far turned 100 MiB of input into 25 MiB.
  estimator_dag::source_of(producer).input_total = 1024 * kMiB;
  record_ratio(producer, 100 * kMiB, 25 * kMiB);

  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 256 * kMiB);
  // A learned ratio is a projection even though the leaf total is known exactly.
  CHECK_FALSE(est->exact);
}

TEST_CASE("data_size_estimator: no ratio yet means no estimate, unless unit ratio is allowed",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer                                 = dag.add();
  estimator_dag::source_of(producer).input_total = 512 * kMiB;
  // No task has completed, so the pipeline has no measured input->output ratio.

  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer).has_value());

  auto est = estimate_pipeline_total_output_bytes(producer,
                                                  size_estimate_options{
                                                    .assume_unit_ratio = true,
                                                  });
  REQUIRE(est.has_value());
  CHECK(est->bytes == 512 * kMiB);
  CHECK_FALSE(est->exact);
}

TEST_CASE("data_size_estimator: a single-input ratio from too few tasks is not trusted",
          "[data_size_estimator][estimation]")
{
  auto const min_samples = size_estimate_options{}.min_ratio_samples;
  REQUIRE(min_samples > 1);  // otherwise this case asserts nothing

  estimator_dag dag;
  auto& producer                                 = dag.add();
  estimator_dag::source_of(producer).input_total = 1024 * kMiB;

  // One task short of the floor. The ratio exists and is arithmetically usable, but a filter
  // over clustered data can make a single batch's selectivity nothing like the table's — and
  // the consumer latches the first estimate it gets, so an unlucky sample is never corrected.
  for (std::size_t i = 0; i < min_samples - 1; ++i) {
    record_task(producer, 100 * kMiB, 25 * kMiB);
  }
  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer).has_value());

  // An untrusted ratio is treated exactly as no ratio, so the unit-ratio escape still applies
  // and yields the unscaled source total.
  auto assumed = estimate_pipeline_total_output_bytes(producer,
                                                      size_estimate_options{
                                                        .assume_unit_ratio = true,
                                                      });
  REQUIRE(assumed.has_value());
  CHECK(assumed->bytes == 1024 * kMiB);
  CHECK_FALSE(assumed->exact);

  // The floor is a knob, not a constant: on the very same history, a caller willing to act on
  // thinner evidence lowers it and gets the measured ratio applied (1 GiB x 0.25).
  auto lowered = estimate_pipeline_total_output_bytes(producer,
                                                      size_estimate_options{
                                                        .min_ratio_samples = 1,
                                                      });
  REQUIRE(lowered.has_value());
  CHECK(lowered->bytes == 256 * kMiB);

  // One more task clears the default floor, and the same answer arrives without the override.
  record_task(producer, 100 * kMiB, 25 * kMiB);
  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 256 * kMiB);
  CHECK(est->ratio_samples == min_samples);
}

TEST_CASE("data_size_estimator: a source that knows nothing yields no estimate",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer = dag.add();
  record_ratio(producer, 100 * kMiB, 50 * kMiB);
  // Both source totals stay nullopt — this is the streaming-source case.

  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer).has_value());
  // A unit ratio does not invent a total that the source cannot supply.
  CHECK_FALSE(estimate_pipeline_total_output_bytes(producer,
                                                   size_estimate_options{
                                                     .assume_unit_ratio = true,
                                                   })
                .has_value());
}

TEST_CASE("data_size_estimator: an output-level source total bypasses the pipeline ratio",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer = dag.add();
  // Only the post-filter output total is known (the scan's cardinality fallback).
  estimator_dag::source_of(producer).output_total = 300 * kMiB;
  // A ratio exists but must NOT be applied: it is derived from pre-filter input bytes, so
  // using it here would count filter selectivity twice.
  record_ratio(producer, 100 * kMiB, 10 * kMiB);

  auto est = estimate_pipeline_total_output_bytes(producer);
  REQUIRE(est.has_value());
  CHECK(est->bytes == 300 * kMiB);
  CHECK_FALSE(est->exact);
}

TEST_CASE("data_size_estimator: ratios compose along a multi-hop chain",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& scan     = dag.add();
  auto& mid      = dag.add();
  auto& consumer = dag.add();
  dag.connect(scan, mid);
  dag.connect(mid, consumer);

  estimator_dag::source_of(scan).input_total = 1024 * kMiB;
  record_ratio(scan, 100 * kMiB, 50 * kMiB);  // ratio 0.5  -> 512 MiB leaves the scan
  record_ratio(mid, 100 * kMiB, 25 * kMiB);   // ratio 0.25 -> 128 MiB leaves mid

  auto& consumer_op = *consumer.get_source();
  auto est = estimate_port_total_input_bytes(consumer_op, consumer_op.get_port_ids().front());
  REQUIRE(est.has_value());
  CHECK(est->bytes == 128 * kMiB);
  CHECK_FALSE(est->exact);
  CHECK(est->hops == 1);  // consumer's port -> mid (hop 0) -> scan (hop 1)
}

TEST_CASE("data_size_estimator: the walk stops at the first finished pipeline",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& scan     = dag.add();
  auto& mid      = dag.add();
  auto& consumer = dag.add();
  dag.connect(scan, mid);
  dag.connect(mid, consumer);

  // mid is done, so its recorded output total is authoritative and the scan is never consulted
  // (deliberately left with no source total, which would otherwise abort the walk).
  record_ratio(mid, 100 * kMiB, 200 * kMiB);
  mid.finished = true;

  auto& consumer_op = *consumer.get_source();
  auto est = estimate_port_total_input_bytes(consumer_op, consumer_op.get_port_ids().front());
  REQUIRE(est.has_value());
  CHECK(est->bytes == 200 * kMiB);
  CHECK(est->exact);
  CHECK(est->hops == 0);
}

namespace {

/// A `probe -> join <- build`, `join -> consumer` shape. Returns the consumer's only port name
/// so callers can estimate through it.
struct fan_in_dag {
  estimator_dag dag;
  test_pipeline* probe    = nullptr;
  test_pipeline* build    = nullptr;
  test_pipeline* join     = nullptr;
  test_pipeline* consumer = nullptr;

  fan_in_dag()
  {
    probe    = &dag.add();
    build    = &dag.add();
    join     = &dag.add();
    consumer = &dag.add();
    dag.connect(*probe, *join, MemoryBarrierType::PARTIAL, true, "default");
    dag.connect(*build, *join, MemoryBarrierType::FULL, true, "build");
    dag.connect(*join, *consumer);
  }

  /// Record exactly `n` completed tasks on the join pipeline so it clears
  /// size_estimate_options::min_fan_in_ratio_samples. The ratio is unaffected by the count.
  /// Uses record_task rather than record_ratio so `n` means n tasks, not n groups of them.
  void record_join_ratio(std::size_t in_bytes, std::size_t out_bytes, std::size_t n = 16)
  {
    for (std::size_t i = 0; i < n; ++i) {
      record_task(*join, in_bytes / n, out_bytes / n);
    }
  }

  /// Drive the join pipeline's task counters. The estimator must ignore these entirely — see
  /// the "does not correct its ratio from task counts" case for why.
  void set_join_task_counters(std::size_t created, std::size_t completed)
  {
    for (std::size_t i = 0; i < created; ++i) {
      join->mark_task_created();
    }
    for (std::size_t i = 0; i < completed; ++i) {
      join->mark_task_completed();
    }
  }

  /// Probe side projects 1 GiB; build side is deliberately given a wildly different total so a
  /// test can prove it is not consulted.
  void give_both_sides_totals()
  {
    estimator_dag::source_of(*probe).input_total = 1024 * kMiB;
    record_ratio(*probe, 100 * kMiB, 100 * kMiB);  // ratio 1.0 -> 1 GiB leaves the probe side
    estimator_dag::source_of(*build).input_total = 1 * kMiB;
    record_ratio(*build, 100 * kMiB, 100 * kMiB);
  }

  std::optional<sirius::pipeline::data_size_estimate> estimate()
  {
    auto& op = *consumer->get_source();
    return estimate_port_total_input_bytes(op, op.get_port_ids().front());
  }
};

}  // namespace

TEST_CASE("data_size_estimator: a fan-in that nominates no primary port yields no estimate",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  f.record_join_ratio(100 * kMiB, 50 * kMiB);
  estimator_dag::source_of(*f.join).consumed_primary = 100 * kMiB;
  // primary_port left empty — this is the CTE / delim-join case, which must still bail out.

  CHECK_FALSE(f.estimate().has_value());
}

TEST_CASE("data_size_estimator: a fan-in with no consumed primary bytes yields no estimate",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  f.record_join_ratio(100 * kMiB, 50 * kMiB);
  estimator_dag::source_of(*f.join).primary_port = "default";
  // No probe batch taken yet, so there is nothing to form a ratio against.
  estimator_dag::source_of(*f.join).consumed_primary = 0;

  CHECK_FALSE(f.estimate().has_value());
}

TEST_CASE("data_size_estimator: a fan-in scales the primary upstream by output-per-primary-byte",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  // The join pipeline has emitted 50 MiB while taking 200 MiB of probe input: ratio 0.25.
  // Note the ratio comes from consumed_primary, NOT from the recorded input_basis — that is
  // the whole point, since a STANDARD join's input_basis double-counts re-paired batches.
  f.record_join_ratio(999 * kMiB, 50 * kMiB);  // input_basis deliberately bogus
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  auto est = f.estimate();
  REQUIRE(est.has_value());
  CHECK(est->bytes == 256 * kMiB);  // 1 GiB (probe projection) x 0.25
  CHECK_FALSE(est->exact);
  CHECK(est->hops == 1);  // consumer -> join (0) -> probe side (1)
}

TEST_CASE("data_size_estimator: a fan-in with too few completed tasks yields no estimate",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  // A fan-in ratio divides a completion-accrued numerator by a live denominator, so it reads
  // low while tasks are in flight — and worst at the first opportunity to sample it. One
  // completed task is not enough evidence.
  f.record_join_ratio(100 * kMiB, 50 * kMiB, /*n=*/1);
  CHECK_FALSE(f.estimate().has_value());

  // Clearing the threshold unblocks it. (The reported sample count is the weakest link in the
  // chain, not the join's own — see the next case.)
  f.record_join_ratio(100 * kMiB, 50 * kMiB, /*n=*/16);
  CHECK(f.estimate().has_value());
}

TEST_CASE("data_size_estimator: a fan-in does not correct its ratio from task counts",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();  // probe side projects 1 GiB
  f.record_join_ratio(100 * kMiB, 50 * kMiB);
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  auto const before = f.estimate();
  REQUIRE(before.has_value());
  CHECK(before->bytes == 256 * kMiB);  // 1 GiB x (50/200)

  // An earlier version scaled the denominator by completed/created to discount tasks still in
  // flight. That is only valid if every task consumes an equal share of `consumed`, and a join's
  // do not: `consumed` advances on a probe batch's FIRST pairing, so with B build batches only
  // one task in B moves it at all. Leaving tasks in flight must not move the projection.
  f.set_join_task_counters(/*created=*/25, /*completed=*/20);
  auto const after = f.estimate();
  REQUIRE(after.has_value());
  CHECK(after->bytes == before->bytes);
}

TEST_CASE("data_size_estimator: a projection that would overflow is reported as unknown",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& scan = dag.add();

  // A near-maximal input paired with an expanding ratio: the product does not fit in a byte
  // count. Report nothing rather than a wrapped total that would then size a partition count.
  estimator_dag::source_of(scan).input_total = std::numeric_limits<std::size_t>::max();
  record_ratio(scan, 1 * kMiB, 1024 * kMiB);

  CHECK_FALSE(estimate_pipeline_total_output_bytes(scan).has_value());

  // The same input with a ratio that keeps the product in range is still answered.
  estimator_dag::source_of(scan).input_total = 1024 * kMiB;
  CHECK(estimate_pipeline_total_output_bytes(scan).has_value());
}

TEST_CASE("data_size_estimator: ratio_samples reports the weakest ratio in the chain",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();  // probe side gets exactly min_ratio_samples records
  f.record_join_ratio(100 * kMiB, 50 * kMiB, /*n=*/40);
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  auto est = f.estimate();
  REQUIRE(est.has_value());
  // The join has 40 samples but the probe pipeline has only the bare minimum — the chain is
  // only as trustworthy as its weakest link.
  CHECK(est->ratio_samples == size_estimate_options{}.min_ratio_samples);
}

TEST_CASE("data_size_estimator: a fan-in ignores the non-primary side entirely",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  f.give_both_sides_totals();
  f.record_join_ratio(100 * kMiB, 50 * kMiB);
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  auto const before = f.estimate();
  REQUIRE(before.has_value());

  // Move the build side dramatically; the projection must not budge.
  estimator_dag::source_of(*f.build).input_total = 4096 * kMiB;
  record_ratio(*f.build, 100 * kMiB, 800 * kMiB);

  auto const after = f.estimate();
  REQUIRE(after.has_value());
  CHECK(after->bytes == before->bytes);
}

TEST_CASE("data_size_estimator: a fan-in whose primary upstream is unknown yields no estimate",
          "[data_size_estimator][estimation][fan_in]")
{
  fan_in_dag f;
  // Build side is fully knowable; probe side is not (its source knows no total). The strict
  // fallback means we report nothing rather than substituting the side we happen to know.
  estimator_dag::source_of(*f.build).input_total = 1 * kMiB;
  record_ratio(*f.build, 100 * kMiB, 100 * kMiB);
  f.record_join_ratio(100 * kMiB, 50 * kMiB);
  estimator_dag::source_of(*f.join).primary_port     = "default";
  estimator_dag::source_of(*f.join).consumed_primary = 200 * kMiB;

  CHECK_FALSE(f.estimate().has_value());
}

TEST_CASE("data_size_estimator: max_hops bounds the upstream walk",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& scan = dag.add();
  auto& a    = dag.add();
  auto& b    = dag.add();
  dag.connect(scan, a);
  dag.connect(a, b);

  estimator_dag::source_of(scan).input_total = 100 * kMiB;
  record_ratio(scan, 100 * kMiB, 100 * kMiB);
  record_ratio(a, 100 * kMiB, 100 * kMiB);
  record_ratio(b, 100 * kMiB, 100 * kMiB);

  CHECK(estimate_pipeline_total_output_bytes(b, size_estimate_options{.max_hops = 2}).has_value());
  CHECK_FALSE(
    estimate_pipeline_total_output_bytes(b, size_estimate_options{.max_hops = 1}).has_value());
}

TEST_CASE("data_size_estimator: unknown and dependency-only ports yield no estimate",
          "[data_size_estimator][estimation]")
{
  estimator_dag dag;
  auto& producer = dag.add();
  auto& consumer = dag.add();
  // A dependency-only edge: no repository, so no bytes ever flow through it.
  dag.connect(producer, consumer, MemoryBarrierType::FULL, /*with_repo=*/false);

  producer.finished = true;
  record_ratio(producer, 100 * kMiB, 100 * kMiB);

  auto& consumer_op = *consumer.get_source();
  CHECK_FALSE(estimate_port_total_input_bytes(consumer_op, "no_such_port").has_value());
  CHECK_FALSE(
    estimate_port_total_input_bytes(consumer_op, consumer_op.get_port_ids().front()).has_value());
}
