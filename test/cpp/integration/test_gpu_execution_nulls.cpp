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
#include <utils/sirius_test_env.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct NullsConfigEnvGuard {
  explicit NullsConfigEnvGuard(const std::string& path)
  {
    setenv("SIRIUS_CONFIG_FILE", path.c_str(), 1);
  }
  ~NullsConfigEnvGuard() { unsetenv("SIRIUS_CONFIG_FILE"); }
};

static std::vector<std::vector<std::string>> collect_rows(duckdb::MaterializedQueryResult& result)
{
  std::vector<std::vector<std::string>> rows;
  for (duckdb::idx_t r = 0; r < result.RowCount(); r++) {
    std::vector<std::string> row;
    row.reserve(result.ColumnCount());
    for (duckdb::idx_t c = 0; c < result.ColumnCount(); c++) {
      row.push_back(result.GetValue(c, r).ToString());
    }
    rows.push_back(std::move(row));
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

/**
 * Fixture that creates small in-memory tables with NULLs and provides
 * compare_gpu_vs_cpu() to verify the GPU transparent path matches DuckDB CPU.
 *
 * Tables:
 *   null_int(id, a, b)  — integers; a and b each have 2 NULL rows out of 5
 *   null_str(id, s, n)  — mixed VARCHAR/integer; s and n each have 2 NULLs
 *   null_left(id, k)    — left side for join tests; k has 2 NULLs
 *   null_right(k, v)    — right side for join tests; one NULL key row
 */
class NullHandlingFixture {
 public:
  NullHandlingFixture()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      auto cfg_path = fs::path(__FILE__).parent_path() / "integration.yaml";
      REQUIRE(fs::exists(cfg_path));
      config_guard = std::make_unique<NullsConfigEnvGuard>(cfg_path.string());
      db           = std::make_unique<duckdb::DuckDB>(nullptr);
      con          = std::make_unique<duckdb::Connection>(*db);
    }

    con->Query("SET gpu_execution = true;");
    setup_tables();
  }

  void compare_gpu_vs_cpu(const std::string& query)
  {
    con->Query("SET gpu_execution = true;");
    auto gpu_result = con->Query(query);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) { UNSCOPED_INFO("GPU error: " << gpu_result->GetError()); }
    REQUIRE_FALSE(gpu_result->HasError());

    con->Query("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    con->Query("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    if (cpu_result->HasError()) { UNSCOPED_INFO("CPU error: " << cpu_result->GetError()); }
    REQUIRE_FALSE(cpu_result->HasError());

    REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
    REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

    auto& gpu_mat = gpu_result->Cast<duckdb::MaterializedQueryResult>();
    auto& cpu_mat = cpu_result->Cast<duckdb::MaterializedQueryResult>();
    auto gpu_rows = collect_rows(gpu_mat);
    auto cpu_rows = collect_rows(cpu_mat);

    for (duckdb::idx_t r = 0; r < gpu_rows.size(); r++) {
      for (duckdb::idx_t c = 0; c < gpu_rows[r].size(); c++) {
        if (gpu_rows[r][c] != cpu_rows[r][c]) {
          UNSCOPED_INFO("Row " << r << " Col " << c << ": GPU=[" << gpu_rows[r][c] << "] CPU=["
                               << cpu_rows[r][c] << "]");
        }
        REQUIRE(gpu_rows[r][c] == cpu_rows[r][c]);
      }
    }
  }

 private:
  void require_ok(const std::string& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
    REQUIRE_FALSE(result->HasError());
  }

  void setup_tables()
  {
    // id  a     b
    //  1  1    10
    //  2  NULL 20
    //  3  3    NULL
    //  4  NULL NULL
    //  5  5    50
    require_ok("DROP TABLE IF EXISTS null_int");
    require_ok("CREATE TEMP TABLE null_int (id INTEGER, a INTEGER, b INTEGER)");
    require_ok(
      "INSERT INTO null_int VALUES (1,1,10),(2,NULL,20),(3,3,NULL),(4,NULL,NULL),(5,5,50)");

    // id  s         n
    //  1  'alpha'   1
    //  2  NULL      2
    //  3  'gamma'   NULL
    //  4  NULL      NULL
    //  5  'epsilon' 5
    require_ok("DROP TABLE IF EXISTS null_str");
    require_ok("CREATE TEMP TABLE null_str (id INTEGER, s VARCHAR, n INTEGER)");
    require_ok(
      "INSERT INTO null_str VALUES "
      "(1,'alpha',1),(2,NULL,2),(3,'gamma',NULL),(4,NULL,NULL),(5,'epsilon',5)");

    // id  k
    //  1  1
    //  2  NULL
    //  3  3
    //  4  NULL
    //  5  5
    require_ok("DROP TABLE IF EXISTS null_left");
    require_ok("CREATE TEMP TABLE null_left (id INTEGER, k INTEGER)");
    require_ok("INSERT INTO null_left VALUES (1,1),(2,NULL),(3,3),(4,NULL),(5,5)");

    // k     v
    //  1   100
    //  3   300
    //  NULL 999
    require_ok("DROP TABLE IF EXISTS null_right");
    require_ok("CREATE TEMP TABLE null_right (k INTEGER, v INTEGER)");
    require_ok("INSERT INTO null_right VALUES (1,100),(3,300),(NULL,999)");
  }

 public:
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  std::unique_ptr<NullsConfigEnvGuard> config_guard;
};

}  // namespace

//===----------------------------------------------------------------------===//
// Filter
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - filter equality excludes NULLs",
                 "[integration][gpu_execution][filter][nulls]")
{
  compare_gpu_vs_cpu("SELECT id FROM null_int WHERE a = 3 ORDER BY id");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - filter IS NULL",
                 "[integration][gpu_execution][filter][nulls]")
{
  compare_gpu_vs_cpu("SELECT id FROM null_int WHERE a IS NULL ORDER BY id");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - filter IS NOT NULL",
                 "[integration][gpu_execution][filter][nulls]")
{
  compare_gpu_vs_cpu("SELECT id FROM null_int WHERE a IS NOT NULL ORDER BY id");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - filter on nullable varchar",
                 "[integration][gpu_execution][filter][nulls]")
{
  compare_gpu_vs_cpu("SELECT id FROM null_str WHERE s IS NULL ORDER BY id");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - compound filter with nullable columns",
                 "[integration][gpu_execution][filter][nulls]")
{
  compare_gpu_vs_cpu("SELECT id FROM null_int WHERE a IS NOT NULL AND b IS NOT NULL ORDER BY id");
}

//===----------------------------------------------------------------------===//
// Ungrouped aggregate
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - COUNT(*) counts all rows",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  compare_gpu_vs_cpu("SELECT count(*) FROM null_int");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - COUNT(col) skips NULLs",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  compare_gpu_vs_cpu("SELECT count(a) FROM null_int");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - SUM skips NULLs",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  compare_gpu_vs_cpu("SELECT sum(a), sum(b) FROM null_int");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - MIN and MAX skip NULLs",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  compare_gpu_vs_cpu("SELECT min(a), max(a), min(b), max(b) FROM null_int");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - aggregate over all-NULL column is NULL",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  compare_gpu_vs_cpu("SELECT sum(a), min(a), max(a) FROM null_int WHERE a IS NULL");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - COUNT(*) vs COUNT(col) differ when NULLs present",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  compare_gpu_vs_cpu("SELECT count(*), count(a), count(b) FROM null_int");
}

//===----------------------------------------------------------------------===//
// Grouped aggregate
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - NULL keys form their own group",
                 "[integration][gpu_execution][group_by][nulls]")
{
  compare_gpu_vs_cpu("SELECT a, count(*) AS cnt FROM null_int GROUP BY a ORDER BY a NULLS FIRST");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - SUM with NULL inputs in group",
                 "[integration][gpu_execution][group_by][nulls]")
{
  compare_gpu_vs_cpu("SELECT id, sum(b) AS s FROM null_int GROUP BY id ORDER BY id");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - multi-column GROUP BY with NULLs",
                 "[integration][gpu_execution][group_by][nulls]")
{
  compare_gpu_vs_cpu(
    "SELECT a, b, count(*) AS cnt "
    "FROM null_int GROUP BY a, b "
    "ORDER BY a NULLS FIRST, b NULLS FIRST");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - HAVING on nullable aggregate",
                 "[integration][gpu_execution][having][nulls]")
{
  compare_gpu_vs_cpu(
    "SELECT a, count(*) AS cnt FROM null_int GROUP BY a "
    "HAVING count(*) > 1 ORDER BY a NULLS FIRST");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - HAVING filters groups where sum IS NULL",
                 "[integration][gpu_execution][having][nulls]")
{
  compare_gpu_vs_cpu(
    "SELECT id, sum(b) AS s FROM null_int GROUP BY id "
    "HAVING sum(b) IS NOT NULL ORDER BY id");
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - arithmetic NULL propagation",
                 "[integration][gpu_execution][expression][nulls]")
{
  compare_gpu_vs_cpu("SELECT id, a + 1, b - 1, a * b FROM null_int ORDER BY id");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - COALESCE replaces NULLs",
                 "[integration][gpu_execution][expression][nulls]")
{
  compare_gpu_vs_cpu("SELECT id, coalesce(a, 0), coalesce(b, -1) FROM null_int ORDER BY id");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - CASE WHEN IS NULL",
                 "[integration][gpu_execution][expression][nulls]")
{
  compare_gpu_vs_cpu(
    "SELECT id, CASE WHEN a IS NULL THEN -1 ELSE a END AS a_safe "
    "FROM null_int ORDER BY id");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - NULL propagation through varchar expressions",
                 "[integration][gpu_execution][expression][nulls]")
{
  compare_gpu_vs_cpu("SELECT id, s || ' suffix', coalesce(s, 'none') FROM null_str ORDER BY id");
}

//===----------------------------------------------------------------------===//
// Join
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - inner join NULL keys do not match",
                 "[integration][gpu_execution][join][nulls]")
{
  compare_gpu_vs_cpu(
    "SELECT l.id, l.k, r.v "
    "FROM null_left l "
    "INNER JOIN null_right r ON l.k = r.k "
    "ORDER BY l.id");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - left join preserves NULL-key rows from left",
                 "[integration][gpu_execution][join][nulls]")
{
  compare_gpu_vs_cpu(
    "SELECT l.id, l.k, r.v "
    "FROM null_left l "
    "LEFT JOIN null_right r ON l.k = r.k "
    "ORDER BY l.id");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - left join count(*) vs count(col) differ",
                 "[integration][gpu_execution][join][nulls]")
{
  compare_gpu_vs_cpu(
    "SELECT count(*), count(r.v) "
    "FROM null_left l "
    "LEFT JOIN null_right r ON l.k = r.k");
}

TEST_CASE_METHOD(NullHandlingFixture,
                 "gpu_execution nulls - inner join with aggregate on nullable column",
                 "[integration][gpu_execution][join][nulls]")
{
  compare_gpu_vs_cpu(
    "SELECT l.k, count(*), sum(r.v) "
    "FROM null_left l "
    "INNER JOIN null_right r ON l.k = r.k "
    "GROUP BY l.k ORDER BY l.k");
}
