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

// Per-operator MGPU integration test for grouped_aggregate_merge.
//
// grouped_aggregate_merge is partition-based (like hash_join) — it builds a
// cuco GPU hash table per partition. cuco tables cannot span GPUs, so every
// task for a given partition must pin to the same GPU, which the merge's
// round-robin partition placement stamps on each task's input. These TEST_CASEs exercise
// the cross-GPU routing path in isolation:
//
//   1. High-cardinality GROUP BY with hash_partition_bytes small enough to
//      force multi-partition execution. The merge input is far above
//      min_bytes_per_gpu (= hash_partition_bytes), so grouped_merge_partition_strategy
//      spreads it over both GPUs and both see pipeline_task dispatches.
//   2. Single-key GROUP BY (all k=0). Tiny aggregate below the partitioning
//      cliff; just a correctness check that the one-partition
//      grouped_aggregate_merge still works under num_gpus=2.
//   3. COUNT(*)-only GROUP BY across both GPUs — sanity check that the
//      COUNT aggregation-type enum dispatches correctly through the merge
//      path when SUM is not in play.
//   4. The same high-cardinality GROUP BY with a large hash_partition_bytes:
//      the merge input is below the cliff (2 x hash_partition_bytes), so the
//      merge runs as one partition on one GPU instead of one per GPU.
//
// Cases 1 and 4 read the merge's sizing decision from its debug log line.

#include "mgpu_test_utils.hpp"

#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb.hpp>
#include <log/logging.hpp>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sirius::test::mgpu;

namespace {

constexpr int kNumFiles    = 8;
constexpr int kRowsPerFile = 500000;

mgpu_env_params make_params()
{
  mgpu_env_params p;
  p.cache                    = "none";
  p.hash_partition_bytes     = 1'000'000;  // 1 MiB → force many partitions
  p.pipeline_num_threads     = 4;
  p.task_creator_num_threads = 4;
  return p;
}

/// One merge_group_by sizing decision, parsed from its debug log line.
struct merge_sizing {
  int partitions;
  int gpus_used;
  int gpus_admitted;
  std::string placement;
};

/// Every merge_group_by sizing line in the log files under @p log_dir.
std::vector<merge_sizing> parse_merge_sizing(fs::path const& log_dir)
{
  std::vector<merge_sizing> result;
  std::regex const re(
    R"(merge_group_by id \d+ sized (\d+) partitions on (\d+) of (\d+) GPUs .* placement (\[.*\]))");
  if (!fs::exists(log_dir)) { return result; }
  for (auto const& entry : fs::directory_iterator(log_dir)) {
    if (!entry.is_regular_file()) { continue; }
    std::ifstream f(entry.path());
    std::string line;
    while (std::getline(f, line)) {
      std::smatch m;
      if (std::regex_search(line, m, re)) {
        result.push_back({std::stoi(m[1]), std::stoi(m[2]), std::stoi(m[3]), m[4]});
      }
    }
  }
  return result;
}

fs::path make_tmp_dir(std::string const& tag)
{
  auto tmp =
    fs::temp_directory_path() / ("sirius-mgpu-gagg-" + std::to_string(::getpid()) + "-" + tag);
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);
  return tmp;
}

}  // namespace

TEST_CASE("grouped_aggregate_merge - group by with high cardinality distributes across both GPUs",
          "[mgpu][operator-mgpu][grouped_aggregate_merge][gpu_execution][multi_gpu]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("highcard");
  auto yaml = tmp / "mgpu.yaml";

  // Independent keys across files: range(0..500000) each — total ~4M rows
  // with ~500k distinct keys. At ~16 B/row that is ~64 MiB input; with
  // hash_partition_bytes=1 MiB sirius_physical_partition produces many
  // partitions (>= num_gpus floor of 2).
  generate_parquet_surface(
    tmp,
    "SELECT range AS k, range * 2 AS v FROM range(" + std::to_string(kRowsPerFile) + ")",
    kNumFiles);

  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

  auto glob = parquet_glob(tmp);
  auto inner_query =
    "SELECT k, SUM(v) AS sum_v, COUNT(*) AS cnt "
    "FROM read_parquet('" +
    glob +
    "') "
    "GROUP BY k "
    "ORDER BY k "
    "LIMIT 50";

  std::map<int, size_t> tasks_per_gpu;
  scoped_log_dir log_dir(tmp / "log", "debug");
  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    require_gpu_matches_cpu(*con, inner_query);
    auto& scheduler = env.get_task_scheduler(*con);
    con.reset();
    scheduler.visit_executors(
      [&](int device_id, const sirius::pipeline::gpu_pipeline_executor& exec) {
        tasks_per_gpu[device_id] = exec.get_metrics().tasks_executed;
      });
    sirius::log::get_sink()->flush();
  }

  INFO("gpu0 tasks=" << tasks_per_gpu[0] << " gpu1 tasks=" << tasks_per_gpu[1]);
  REQUIRE(tasks_per_gpu.count(0));
  REQUIRE(tasks_per_gpu.count(1));
  REQUIRE(tasks_per_gpu.at(0) >= 1);
  REQUIRE(tasks_per_gpu.at(1) >= 1);

  // The merge itself is spread over both GPUs.
  auto const sizing = parse_merge_sizing(log_dir.path());
  REQUIRE_FALSE(sizing.empty());
  for (auto const& s : sizing) {
    INFO("partitions=" << s.partitions << " gpus=" << s.gpus_used << " placement=" << s.placement);
    CHECK(s.gpus_admitted == 2);
    CHECK(s.gpus_used == 2);
    CHECK(s.partitions >= 2);
  }

  std::error_code ec;
  fs::remove_all(tmp, ec);
}

TEST_CASE("grouped_aggregate_merge - a merge input below the cliff runs on one GPU",
          "[mgpu][operator-mgpu][grouped_aggregate_merge][gpu_execution][multi_gpu]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("belowcliff");
  auto yaml = tmp / "mgpu.yaml";

  // The high-cardinality surface of test #1 (~100 MB of merge input). With a
  // 256 MB partition target the cliff is 512 MB, so the merge keeps one
  // partition. Before #1746 the num_gpus floor split it in two across both GPUs.
  generate_parquet_surface(
    tmp,
    "SELECT range AS k, range * 2 AS v FROM range(" + std::to_string(kRowsPerFile) + ")",
    kNumFiles);

  auto params                 = make_params();
  params.hash_partition_bytes = 256'000'000;
  write_mgpu_yaml(yaml, params);
  REQUIRE(fs::exists(yaml));

  auto glob = parquet_glob(tmp);
  auto inner_query =
    "SELECT k, SUM(v) AS sum_v, COUNT(*) AS cnt "
    "FROM read_parquet('" +
    glob +
    "') "
    "GROUP BY k "
    "ORDER BY k "
    "LIMIT 50";

  scoped_log_dir log_dir(tmp / "log", "debug");
  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    require_gpu_matches_cpu(*con, inner_query);
    con.reset();
    sirius::log::get_sink()->flush();
  }

  auto const sizing = parse_merge_sizing(log_dir.path());
  REQUIRE_FALSE(sizing.empty());
  for (auto const& s : sizing) {
    INFO("partitions=" << s.partitions << " gpus=" << s.gpus_used << " placement=" << s.placement);
    CHECK(s.gpus_admitted == 2);
    CHECK(s.gpus_used == 1);
    CHECK(s.partitions == 1);
    // Every task of the one partition is pinned to that partition's GPU.
    CHECK(std::regex_match(s.placement, std::regex(R"(\[0->\d+\])")));
  }

  std::error_code ec;
  fs::remove_all(tmp, ec);
}

TEST_CASE("grouped_aggregate_merge - group by with single key forces single-GPU path",
          "[mgpu][operator-mgpu][grouped_aggregate_merge][gpu_execution][multi_gpu]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("singlekey");
  auto yaml = tmp / "mgpu.yaml";

  // Every row shares k=0 → aggregate result is a single group. Below the
  // small-table-bytes threshold where the num_gpus floor is skipped, so
  // only one partition exists and the merge runs on a single GPU. We just
  // assert correctness here, not distribution.
  generate_parquet_surface(
    tmp,
    "SELECT 0 AS k, range * 2 AS v FROM range(" + std::to_string(kRowsPerFile) + ")",
    kNumFiles);

  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

  auto glob = parquet_glob(tmp);
  auto inner_query =
    "SELECT k, SUM(v) AS sum_v, COUNT(*) AS cnt "
    "FROM read_parquet('" +
    glob +
    "') "
    "GROUP BY k "
    "ORDER BY k";

  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    require_gpu_matches_cpu(*con, inner_query);
    con.reset();
  }

  std::error_code ec;
  fs::remove_all(tmp, ec);
}

TEST_CASE("grouped_aggregate_merge - count(*)-only aggregate across two GPUs",
          "[mgpu][operator-mgpu][grouped_aggregate_merge][gpu_execution][multi_gpu]")
{
  if (!require_two_gpus()) return;

  auto tmp  = make_tmp_dir("countstar");
  auto yaml = tmp / "mgpu.yaml";

  // Same high-cardinality surface as test #1 so we hit the multi-partition
  // MGPU path, but the aggregate is COUNT(*) only — catches cudf
  // aggregation-type enum dispatch issues that SUM would otherwise mask.
  generate_parquet_surface(
    tmp,
    "SELECT range AS k, range * 2 AS v FROM range(" + std::to_string(kRowsPerFile) + ")",
    kNumFiles);

  write_mgpu_yaml(yaml, make_params());
  REQUIRE(fs::exists(yaml));

  auto glob = parquet_glob(tmp);
  auto inner_query =
    "SELECT k, COUNT(*) AS cnt "
    "FROM read_parquet('" +
    glob +
    "') "
    "GROUP BY k "
    "ORDER BY k "
    "LIMIT 50";

  std::map<int, size_t> tasks_per_gpu;
  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    require_gpu_matches_cpu(*con, inner_query);
    auto& scheduler = env.get_task_scheduler(*con);
    con.reset();
    scheduler.visit_executors(
      [&](int device_id, const sirius::pipeline::gpu_pipeline_executor& exec) {
        tasks_per_gpu[device_id] = exec.get_metrics().tasks_executed;
      });
  }

  INFO("gpu0 tasks=" << tasks_per_gpu[0] << " gpu1 tasks=" << tasks_per_gpu[1]);
  REQUIRE(tasks_per_gpu.count(0));
  REQUIRE(tasks_per_gpu.count(1));
  REQUIRE(tasks_per_gpu.at(0) >= 1);
  REQUIRE(tasks_per_gpu.at(1) >= 1);

  std::error_code ec;
  fs::remove_all(tmp, ec);
}
