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
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

class RegexpReplaceFixture : public sirius::test::GpuExecutionFixture {
 public:
  RegexpReplaceFixture()
  {
    // Persist actual columns so constant folding cannot evaluate the replacement on CPU.
    run_ok("CREATE TABLE regex_input(id INTEGER, s VARCHAR);");
    run_ok(R"sql(INSERT INTO regex_input VALUES
      (1, 'aba aba aba'), (2, 'no match'), (3, NULL), (4, ''),
      (5, 'a'), (6, 'aa'), (7, 'baaa'), (8, '123-456-789'),
      (9, 'éabaéaba'), (10, 'first' || chr(10) || 'aba aba'),
      (11, 'https://www.example.com/a/b'), (12, 'http://other.example/x'),
      (13, 'APPLE Orange APPLE'), (14, 'a' || chr(10) || 'a');)sql");
    run_ok("CHECKPOINT;");
  }
};

}  // namespace

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace replaces only the first match on GPU",
                 "[integration][gpu_execution][regexp_replace]")
{
  for (auto const* expression : {
         "regexp_replace(s, 'a', 'X')",
         "regexp_replace(s, 'aba', 'X')",
         "regexp_replace(s, 'a', '')",
         "regexp_replace(s, '^a', 'X')",
         "regexp_replace(s, 'a$', 'X')",
         "regexp_replace(s, '[0-9]+', '#')",
         "regexp_replace(s, '', 'X')",
         "regexp_replace(s, '^', 'X')",
         "regexp_replace(s, '$', 'X')",
         "regexp_replace(s, 'a*', 'X')",
         "regexp_replace(s, 'a?', 'X')",
       }) {
    INFO(expression);
    compare_gpu_vs_cpu(std::string{"SELECT id, "} + expression + " FROM regex_input");
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace backreferences replace only the first match on GPU",
                 "[integration][gpu_execution][regexp_replace]")
{
  for (auto const* expression : {
         R"(regexp_replace(s, '(a)', '<\1>'))",
         R"(regexp_replace(s, '(a)(b)', '\2-\1-\2'))",
         R"(regexp_replace(s, '(aba)', '\12'))",
         R"(regexp_replace(s, 'aba', '<\0>'))",
         R"(regexp_replace(s, '[A-Z]', '\0\0'))",
         R"(regexp_replace(s, '\s', '\0\0'))",
         R"(regexp_replace(s, '^(a)', '<\1>'))",
         R"(regexp_replace(s, '(a)$', '<\1>'))",
         R"(regexp_replace(s, '([0-9]+)', '<\1>'))",
         R"(regexp_replace(s, '(a*)', '<\1>'))",
         R"(regexp_replace(s, '(a?)', '<\1>'))",
         R"(regexp_replace(s, '()', '<\1>'))",
         R"(regexp_replace(s, '(a)|(b)', '<\1><\2>'))",
         R"(regexp_replace(s, '^https?://(?:www\.)?([^/]+)/.*$', '\1'))",
       }) {
    INFO(expression);
    compare_gpu_vs_cpu(std::string{"SELECT id, "} + expression + " FROM regex_input");
  }
}

TEST_CASE_METHOD(RegexpReplaceFixture,
                 "regexp_replace options and unsupported rewrites fall back during planning",
                 "[integration][gpu_execution][regexp_replace]")
{
  // Four-argument calls were never supported by the evaluator. All explicit options must
  // retain DuckDB semantics, including global replacement and combinations of flags.
  for (auto const* options : {"", "g", "c", "i", "l", "m", "n", "p", "s", "gi", "gs"}) {
    INFO(options);
    expect_plan_fallback_matches_cpu(std::string{"SELECT id, regexp_replace(s, 'a.', 'X', '"} +
                                     options + "') FROM regex_input");
    expect_plan_fallback_matches_cpu(
      std::string{R"(SELECT id, regexp_replace(s, '(a)', '<\1>', ')"} + options +
      "') FROM regex_input");
  }
  for (auto const* expression : {
         R"(regexp_replace(s, '(a)', '\\1'))",
         R"(regexp_replace(s, '(a)', '\1${1}'))",
         "regexp_replace(s, s, 'X')",
         "regexp_replace(s, 'a', s)",
       }) {
    INFO(expression);
    expect_plan_fallback_matches_cpu(std::string{"SELECT id, "} + expression + " FROM regex_input");
  }
}
