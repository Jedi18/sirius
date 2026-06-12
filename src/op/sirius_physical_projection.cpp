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

#include "op/sirius_physical_projection.hpp"

#include "config.hpp"
#include "data/data_batch_utils.hpp"
#include "expression/ast/node.hpp"
#include "expression_executor/gpu_expression_executor.hpp"

#include <nvtx3/nvtx3.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <duckdb/common/exception.hpp>

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

namespace sirius {
namespace op {

namespace {

// A projection can reuse (move out) its input columns instead of copying them only when every
// output is a *distinct* column reference. Requiring references (no computation) and distinct
// indices (each reused column is moved exactly once, never copied) keeps the reuse path
// allocation-free. That is what makes consuming the input batch safe under OOM-driven task
// rescheduling: an allocation-free projection can never OOM after it has stolen its input, so the
// pipeline never has to replay a now-emptied input batch.
bool can_reuse_input_columns(const duckdb::vector<std::unique_ptr<sirius::ast::node>>& select_list)
{
  std::unordered_set<std::uint32_t> seen_indices;
  seen_indices.reserve(select_list.size());
  for (auto const& node : select_list) {
    if (node == nullptr || !node->is_reference()) { return false; }
    if (!seen_indices.insert(node->as_reference().column_index).second) {
      return false;  // a duplicate reference would require copying the column
    }
  }
  return true;
}

}  // namespace

sirius_physical_projection::sirius_physical_projection(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list,
  std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::PROJECTION, std::move(types), estimated_cardinality),
    select_list(std::move(select_list))
{
}

std::unique_ptr<operator_data> sirius_physical_projection::execute(const operator_data& input_data,
                                                                   rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_projection::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_data_batches();

  /// TODO: the operator should choose the execution strategy based on statistics and a deeper
  /// understand of the trade-offs between the different strategies. See:
  /// https://github.com/sirius-db/sirius/issues/636
  sirius::gpu_expression_executor gpu_expression_executor(
    select_list, cudf::get_current_device_resource_ref(), stream);

  // When every output is a distinct pass-through column reference, the projection can reuse the
  // input columns zero-copy instead of deep-copying them (see can_reuse_input_columns and
  // gpu_expression_executor::execute(std::unique_ptr<cudf::table>)).
  bool const reuse_input_columns = can_reuse_input_columns(select_list);

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  for (auto const& batch : input_batches) {
    cucascade::memory::memory_space* space = nullptr;
    std::unique_ptr<cudf::table> owned_input;

    // Reuse columns only when this task is the sole owner of the batch (use_count() == 1). A shared
    // batch -- e.g. the pipeline's subscribed source input, or a batch fanned out to multiple
    // downstream consumers -- must not be mutated, so it falls back to the copying read-only path
    // below. try_to_mutable() is non-blocking; a sole owner never contends, so a (rare) failure
    // also falls back.
    if (reuse_input_columns && batch.use_count() == 1) {
      if (auto mutable_batch = batch->try_to_mutable()) {
        space = mutable_batch->get_memory_space();
        owned_input =
          mutable_batch->get_data()->cast<cucascade::gpu_table_representation>().release_table(
            stream);
      }
    }

    std::unique_ptr<cudf::table> projected_table;
    if (owned_input != nullptr) {
      // Sole owner: hand ownership to the executor so pass-through columns are moved, not copied.
      projected_table = gpu_expression_executor.execute(std::move(owned_input));
    } else {
      // Shared (or non-reusable) input: evaluate against a read-only view, copying as before.
      auto read_only_batch = batch->to_read_only();
      space                = read_only_batch.get_memory_space();
      projected_table =
        gpu_expression_executor.execute(sirius::get_cudf_table_view(read_only_batch));
    }

    output_batches.push_back(sirius::make_data_batch(std::move(projected_table), *space, stream));
  }
  return std::make_unique<pipelineable_operator_data>(output_batches);
}

}  // namespace op
}  // namespace sirius
