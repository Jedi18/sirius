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

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

using IntegerAggregateFixture = sirius::test::GpuExecutionFixture;

namespace {
void expect_integer_aggregate_rejection(IntegerAggregateFixture& fx, std::string const& sql)
{
  fx.expect_plan_fallback_matches_cpu(sql);
  fx.run_ok("SET enable_duckdb_fallback = false");
  auto before = sirius::test::get_transparent_execution_stats(*fx.con);
  auto result = fx.con->Query(sql);
  auto after  = sirius::test::get_transparent_execution_stats(*fx.con);
  fx.run_ok("SET enable_duckdb_fallback = true");
  REQUIRE(result);
  REQUIRE(result->HasError());
  INFO(result->GetError());
  REQUIRE(result->GetError().find("Integer ") != std::string::npos);
  REQUIRE(result->GetError().find("accumulator") != std::string::npos);
  REQUIRE(after.executions == before.executions);
}
}  // namespace

TEST_CASE_METHOD(IntegerAggregateFixture,
                 "wide integer SUM and AVG reject before local or merge overflow",
                 "[integration][gpu_execution][integer_aggregate]")
{
  run_ok("SET disabled_optimizers = 'in_clause,compressed_materialization,late_materialization'");
  run_ok("CREATE TABLE wide_agg(g INTEGER, s BIGINT, u UBIGINT)");
  run_ok(
    "INSERT INTO wide_agg VALUES "
    "(1, 9223372036854775807, 18446744073709551615),"
    "(1, 9223372036854775807, 18446744073709551615),"
    "(2, -9223372036854775808, 9223372036854775808),"
    "(2, -9223372036854775808, 9223372036854775808),"
    "(3, -9223372036854775807, 1),"
    "(3, 9223372036854775807, 9223372036854775807),"
    "(3, NULL, NULL), (4, NULL, NULL)");
  run_ok("CHECKPOINT");
  for (auto const* column : {"s", "u"}) {
    for (auto const* function : {"sum", "avg"}) {
      auto expression = std::string(function) + "(" + column + ")";
      expect_integer_aggregate_rejection(*this, "SELECT " + expression + " FROM wide_agg");
      expect_integer_aggregate_rejection(*this,
                                         "SELECT g, " + expression + " FROM wide_agg GROUP BY g");
    }
  }
}

TEST_CASE_METHOD(IntegerAggregateFixture,
                 "integer aggregate planning rejects unproven domains even without rows",
                 "[integration][gpu_execution][integer_aggregate]")
{
  // Keep an aggregate in the plan and prevent statistics from proving an empty SUM safe.
  run_ok(
    "SET disabled_optimizers = 'statistics_propagation,compressed_materialization,"
    "late_materialization'");
  run_ok("CREATE TABLE empty_agg(s BIGINT, u UBIGINT)");
  run_ok("CHECKPOINT");
  expect_integer_aggregate_rejection(*this, "SELECT SUM(s), AVG(s) FROM empty_agg");
  expect_integer_aggregate_rejection(*this, "SELECT SUM(u), AVG(u) FROM empty_agg");
  run_ok("INSERT INTO empty_agg VALUES (NULL, NULL)");
  run_ok("CHECKPOINT");
  expect_integer_aggregate_rejection(*this, "SELECT SUM(s), AVG(s) FROM empty_agg");
  expect_integer_aggregate_rejection(*this, "SELECT SUM(u), AVG(u) FROM empty_agg");
}

TEST_CASE_METHOD(IntegerAggregateFixture,
                 "narrow integer AVG keeps exact decimal partials and DOUBLE SQL results",
                 "[integration][gpu_execution][integer_aggregate]")
{
  run_ok("SET disabled_optimizers = 'in_clause,compressed_materialization,late_materialization'");
  run_ok("CREATE TABLE narrow_agg(g INTEGER, s INTEGER, u UINTEGER)");
  run_ok(
    "INSERT INTO narrow_agg VALUES "
    "(1, 2147483647, 4294967295), (1, 2147483647, 4294967295),"
    "(2, -2147483648, 0), (2, 2147483647, 4294967295),"
    "(2, NULL, NULL), (3, NULL, NULL)");
  run_ok("CHECKPOINT");
  compare_gpu_vs_cpu("SELECT AVG(s), AVG(u) FROM narrow_agg");
  compare_gpu_vs_cpu("SELECT g, AVG(s), AVG(u) FROM narrow_agg GROUP BY g");
  compare_gpu_vs_cpu("SELECT AVG(s), AVG(u) FROM narrow_agg WHERE g = 3");
  auto result = con->Query("SELECT AVG(s), AVG(u) FROM narrow_agg");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->types[0] == duckdb::LogicalType::DOUBLE);
  REQUIRE(result->types[1] == duckdb::LogicalType::DOUBLE);
}

TEST_CASE_METHOD(IntegerAggregateFixture,
                 "proven integer SUM and other unsigned aggregates stay on GPU",
                 "[integration][gpu_execution][integer_aggregate]")
{
  run_ok("CREATE TABLE safe_agg(g INTEGER, s INTEGER, b BIGINT, u UBIGINT)");
  run_ok(
    "INSERT INTO safe_agg VALUES "
    "(1, 12, 12, 18446744073709551615), (1, -3, -3, 9223372036854775808),"
    "(2, 7, 7, 0), (2, NULL, NULL, NULL)");
  run_ok("CHECKPOINT");
  compare_gpu_vs_cpu("SELECT SUM(s), SUM(b), COUNT(u), MIN(u), MAX(u) FROM safe_agg");
  compare_gpu_vs_cpu(
    "SELECT g, SUM(s), SUM(b), COUNT(u), MIN(u), MAX(u) "
    "FROM safe_agg GROUP BY g");
}

TEST_CASE_METHOD(IntegerAggregateFixture,
                 "integer SUM fallback preserves DuckDB HUGEINT overflow errors",
                 "[integration][gpu_execution][integer_aggregate]")
{
  run_ok("CREATE TABLE overflow_agg(v HUGEINT)");
  run_ok(
    "INSERT INTO overflow_agg VALUES "
    "(170141183460469231731687303715884105727), (1)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution = true");
  auto before   = sirius::test::get_transparent_execution_stats(*con);
  auto fallback = con->Query("SELECT SUM(v) FROM overflow_agg");
  auto after    = sirius::test::get_transparent_execution_stats(*con);
  REQUIRE(fallback);
  REQUIRE(fallback->HasError());
  REQUIRE(after.fallbacks == before.fallbacks + 1);
  REQUIRE(after.executions == before.executions);
  run_ok("SET gpu_execution = false");
  auto cpu = con->Query("SELECT SUM(v) FROM overflow_agg");
  run_ok("SET gpu_execution = true");
  REQUIRE(cpu);
  REQUIRE(cpu->HasError());
  REQUIRE(fallback->GetError() == cpu->GetError());
}
