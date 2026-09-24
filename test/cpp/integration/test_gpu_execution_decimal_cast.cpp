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
#include <utils/parquet_fixture_utils.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

class DecimalCastFixture : public sirius::test::GpuExecutionFixture {
 public:
  // The DuckDB-native GPU decoder does not support DECIMAL128 yet. Feed wide decimals
  // through Parquet so the cast, rather than an unrelated scan fallback, is exercised.
  void expose(std::string const& table, bool parquet)
  {
    std::string source = table;
    if (parquet) {
      auto const file = scratch.file_literal(std::to_string(next_file++) + ".parquet");
      run_ok("SET gpu_execution = false;");
      run_ok("COPY " + table + " TO " + file + " (FORMAT PARQUET);");
      run_ok("SET gpu_execution = true;");
      source = "read_parquet(" + file + ")";
    }
    run_ok("CREATE OR REPLACE VIEW " + table + "_input AS SELECT * FROM " + source + ";");
  }

 private:
  sirius::test::scratch_dir scratch{"decimal_cast"};
  unsigned next_file{0};
};

TEST_CASE_METHOD(DecimalCastFixture,
                 "decimal casts round column values in every evaluation strategy",
                 "[integration][gpu_execution][decimal_cast]")
{
  auto const strategy = GENERATE("materialize", "ast_interpret", "ast_jit");
  run_ok(std::string("SET expression_evaluator_strategy = '") + strategy + "';");
  for (auto const precision : {9, 18, 38}) {
    CAPTURE(strategy, precision);
    run_ok("CREATE OR REPLACE TABLE dec_cast(d DECIMAL(" + std::to_string(precision) + ",2));");
    run_ok(
      "INSERT INTO dec_cast VALUES (-2.51),(-2.50),(-2.49),(-1.50),(-0.51),(-0.50),"
      "(-0.49),(-0.01),(0),(0.01),(0.49),(0.50),(0.51),(1.50),(2.49),(2.50),(2.51),(NULL);");
    run_ok("CHECKPOINT;");
    expose("dec_cast", precision > 18);
    // Column references prevent constant folding. Nested arithmetic forces an AST parent.
    compare_gpu_vs_cpu(
      "SELECT CAST(d AS TINYINT), CAST(d AS SMALLINT), CAST(d AS INTEGER),"
      " CAST(d AS BIGINT), (CAST(d AS BIGINT) % 7) + 1 FROM dec_cast_input;");
    compare_gpu_vs_cpu(
      "SELECT TRY_CAST(d AS UTINYINT), TRY_CAST(d AS USMALLINT),"
      " TRY_CAST(d AS UINTEGER), TRY_CAST(d AS UBIGINT) FROM dec_cast_input;");
    compare_gpu_vs_cpu("SELECT d FROM dec_cast_input WHERE CAST(d AS BIGINT) % 7 = 3;");
    run_ok(
      "INSERT INTO dec_cast VALUES (127.49),(127.50),(-128.49),(-128.50),"
      "(255.49),(255.50);");
    run_ok("CHECKPOINT;");
    expose("dec_cast", precision > 18);
    compare_gpu_vs_cpu(
      "SELECT TRY_CAST(d AS TINYINT), TRY_CAST(d AS UTINYINT) FROM dec_cast_input;");
  }
  run_ok("CREATE TABLE dec_scale0(d DECIMAL(38,0));");
  run_ok(
    "INSERT INTO dec_scale0 VALUES (0),(127),(-128),(9223372036854775807),"
    "(-9223372036854775808),(9223372036854775808),(NULL);");
  run_ok("CREATE TABLE dec_scale38(d DECIMAL(38,38));");
  // Quote these inputs: DuckDB parses unquoted 38-place fractional literals as DOUBLE.
  run_ok(
    "INSERT INTO dec_scale38 VALUES ('0.49999999999999999999999999999999999999'),"
    "('0.50000000000000000000000000000000000000'),"
    "('0.99999999999999999999999999999999999999'),"
    "('-0.49999999999999999999999999999999999999'),"
    "('-0.50000000000000000000000000000000000000'),"
    "('-0.99999999999999999999999999999999999999'),(NULL);");
  run_ok("CHECKPOINT;");
  expose("dec_scale0", true);
  expose("dec_scale38", true);
  compare_gpu_vs_cpu("SELECT TRY_CAST(d AS BIGINT), TRY_CAST(d AS UBIGINT) FROM dec_scale0_input;");
  compare_gpu_vs_cpu("SELECT CAST(d AS BIGINT), TRY_CAST(d AS UBIGINT) FROM dec_scale38_input;");
}

TEST_CASE_METHOD(DecimalCastFixture,
                 "decimal casts check the rounded value against integer limits",
                 "[integration][gpu_execution][decimal_cast]")
{
  auto const strategy = GENERATE("materialize", "ast_interpret", "ast_jit");
  run_ok(std::string("SET expression_evaluator_strategy = '") + strategy + "';");
  struct bounds {
    std::string target;
    std::string minimum;
    std::string maximum;
    bool is_unsigned;
  };
  for (auto const& b : {bounds{"TINYINT", "-128", "127", false},
                        bounds{"SMALLINT", "-32768", "32767", false},
                        bounds{"INTEGER", "-2147483648", "2147483647", false},
                        bounds{"BIGINT", "-9223372036854775808", "9223372036854775807", false},
                        bounds{"UTINYINT", "0", "255", true},
                        bounds{"USMALLINT", "0", "65535", true},
                        bounds{"UINTEGER", "0", "4294967295", true},
                        bounds{"UBIGINT", "0", "18446744073709551615", true}}) {
    CAPTURE(strategy, b.target);
    auto const lower = b.is_unsigned ? "-0" : b.minimum;
    run_ok("CREATE OR REPLACE TABLE dec_limits(d DECIMAL(38,2));");
    run_ok("INSERT INTO dec_limits VALUES (" + b.maximum + ".49),(" + b.maximum + ".00),(" + lower +
           ".49),(" + b.minimum + ".00),(0),(NULL);");
    run_ok("CHECKPOINT;");
    expose("dec_limits", true);
    compare_gpu_vs_cpu("SELECT CAST(d AS " + b.target + ") FROM dec_limits_input;");
    run_ok("INSERT INTO dec_limits VALUES (" + b.maximum + ".50),(" + b.maximum + ".51),(" + lower +
           ".50),(" + lower + ".51);");
    run_ok("CHECKPOINT;");
    expose("dec_limits", true);
    compare_gpu_vs_cpu("SELECT TRY_CAST(d AS " + b.target + ") FROM dec_limits_input;");

    // CAST must fail rather than return wrapped/truncated values. With fallback enabled the
    // runtime policy retries on DuckDB, preserving DuckDB's conversion error.
    auto const query = "SELECT CAST(d AS " + b.target + ") FROM dec_limits_input;";
    run_ok("SET enable_duckdb_fallback = false;");
    auto gpu_error = con->Query(query);
    REQUIRE(gpu_error);
    REQUIRE(gpu_error->HasError());
    REQUIRE(gpu_error->GetError().find("Decimal-to-integer cast out of range") !=
            std::string::npos);
    run_ok("SET enable_duckdb_fallback = true;");
    auto const before   = sirius::test::get_transparent_execution_stats(*con);
    auto fallback_error = con->Query(query);
    auto const after    = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(fallback_error);
    REQUIRE(fallback_error->HasError());
    REQUIRE(fallback_error->GetError().find("Failed to cast decimal value") != std::string::npos);
    REQUIRE(after.runtime_fallbacks == before.runtime_fallbacks + 1);
    run_ok("SET gpu_execution = false;");
    auto cpu_error = con->Query(query);
    run_ok("SET gpu_execution = true;");
    REQUIRE(cpu_error);
    REQUIRE(cpu_error->HasError());
    REQUIRE(cpu_error->GetError().find("Failed to cast decimal value") != std::string::npos);
  }
}

TEST_CASE_METHOD(DecimalCastFixture,
                 "decimal casts preserve unsupported integer widths through CPU fallback",
                 "[integration][gpu_execution][decimal_cast]")
{
  run_ok("CREATE TABLE dec_huge(d DECIMAL(38,2));");
  run_ok(
    "INSERT INTO dec_huge VALUES (9223372036854775808.50),"
    "(-9223372036854775809.50),(18446744073709551616.50),(NULL);");
  run_ok("CHECKPOINT;");
  expose("dec_huge", true);
  expect_plan_fallback_matches_cpu("SELECT CAST(d AS HUGEINT) FROM dec_huge_input;");
  expect_plan_fallback_matches_cpu("SELECT TRY_CAST(d AS UHUGEINT) FROM dec_huge_input;");
  // DuckDB's INT16 decimal storage remains CPU-only; no cuDF DECIMAL16 exists.
  run_ok("CREATE TABLE dec_small(d DECIMAL(4,2));");
  run_ok("INSERT INTO dec_small VALUES (1.49),(1.50),(-1.50),(-0.50),(NULL);");
  run_ok("CHECKPOINT;");
  expect_plan_fallback_matches_cpu("SELECT CAST(d AS BIGINT) FROM dec_small;");
}
