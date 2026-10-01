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

// The OOM split decision is a pure function of the lineage state, so it is pinned down here
// without a GPU. The executor integration (real OOMs, real pieces) lives in
// test_oom_split_reschedule.cpp.

#include "catch.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "pipeline/oom_split_policy.hpp"

#include <cudf/hashing.hpp>

using sirius::pipeline::decide_oom_recovery;
using sirius::pipeline::oom_recovery_action;
using sirius::pipeline::oom_recovery_state;
using sirius::pipeline::oom_split_config;

TEST_CASE("oom split policy: an operator that cannot split is always retried",
          "[pipeline][oom_split]")
{
  oom_split_config config;
  config.split_after_retries = 0;
  config.max_split_depth     = 4;
  for (uint32_t ooms = 1; ooms < 10; ++ooms) {
    oom_recovery_state state;
    state.operator_supports_split = false;
    state.ooms_since_split        = ooms;
    CHECK(decide_oom_recovery(config, state) == oom_recovery_action::retry);
  }
}

TEST_CASE("oom split policy: the configured plain retries come before the first split",
          "[pipeline][oom_split]")
{
  oom_split_config config;
  config.split_after_retries = 2;
  config.max_split_depth     = 4;
  oom_recovery_state state;
  state.operator_supports_split = true;

  // The nth OOM is decided with ooms_since_split == n.
  state.ooms_since_split = 1;
  CHECK(decide_oom_recovery(config, state) == oom_recovery_action::retry);
  state.ooms_since_split = 2;
  CHECK(decide_oom_recovery(config, state) == oom_recovery_action::retry);
  state.ooms_since_split = 3;
  CHECK(decide_oom_recovery(config, state) == oom_recovery_action::split);

  SECTION("zero plain retries splits on the very first OOM")
  {
    config.split_after_retries = 0;
    state.ooms_since_split     = 1;
    CHECK(decide_oom_recovery(config, state) == oom_recovery_action::split);
  }
}

TEST_CASE("oom split policy: the depth cap turns further OOMs back into plain retries",
          "[pipeline][oom_split]")
{
  oom_split_config config;
  config.split_after_retries = 0;
  config.max_split_depth     = 2;
  oom_recovery_state state;
  state.operator_supports_split = true;
  state.ooms_since_split        = 1;

  state.split_depth = 0;
  CHECK(decide_oom_recovery(config, state) == oom_recovery_action::split);
  state.split_depth = 1;
  CHECK(decide_oom_recovery(config, state) == oom_recovery_action::split);
  state.split_depth = 2;
  CHECK(decide_oom_recovery(config, state) == oom_recovery_action::retry);
  state.split_depth = 7;
  CHECK(decide_oom_recovery(config, state) == oom_recovery_action::retry);

  SECTION("a depth cap of zero disables splitting entirely")
  {
    config.max_split_depth = 0;
    state.split_depth      = 0;
    state.ooms_since_split = 50;
    CHECK(decide_oom_recovery(config, state) == oom_recovery_action::retry);
  }
}

TEST_CASE("oom split policy: an input that proved unsplittable is never split again",
          "[pipeline][oom_split]")
{
  oom_split_config config;
  config.split_after_retries = 0;
  config.max_split_depth     = 4;
  oom_recovery_state state;
  state.operator_supports_split = true;
  state.split_exhausted         = true;
  state.ooms_since_split        = 3;
  CHECK(decide_oom_recovery(config, state) == oom_recovery_action::retry);
}

TEST_CASE("oom split policy: defaults retry once, then split, up to four levels",
          "[pipeline][oom_split]")
{
  oom_split_config const config;
  CHECK(config.split_after_retries == 1);
  CHECK(config.max_split_depth == 4);
  CHECK(oom_split_config::pieces_per_split == 2);

  oom_recovery_state state;
  state.operator_supports_split = true;
  state.ooms_since_split        = 1;
  CHECK(decide_oom_recovery(config, state) == oom_recovery_action::retry);
  state.ooms_since_split = 2;
  CHECK(decide_oom_recovery(config, state) == oom_recovery_action::split);
}

TEST_CASE("resplit hash seeds differ from cuDF's default and from each other",
          "[pipeline][oom_split][hash_partition]")
{
  using sirius::op::gpu_partition_impl;
  // The plan-level PARTITION hashes with cudf::DEFAULT_HASH_SEED; a re-split with that seed is
  // degenerate (see resplit_hash_seed), so no round may ever produce it.
  for (uint32_t round = 0; round < 16; ++round) {
    CHECK(gpu_partition_impl::resplit_hash_seed(round) != cudf::DEFAULT_HASH_SEED);
    for (uint32_t other = 0; other < round; ++other) {
      CHECK(gpu_partition_impl::resplit_hash_seed(round) !=
            gpu_partition_impl::resplit_hash_seed(other));
    }
  }
}
