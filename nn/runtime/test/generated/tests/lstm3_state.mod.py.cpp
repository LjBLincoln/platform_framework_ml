// DO NOT EDIT;
// Generated by ml/nn/runtime/test/specs/generate_test.sh
#include "../../TestGenerated.h"

namespace lstm3_state {
std::vector<MixedTypedExample> examples = {
// Generated lstm3_state test
#include "generated/examples/lstm3_state.example.cpp"
};
// Generated model constructor
#include "generated/models/lstm3_state.model.cpp"
} // namespace lstm3_state
TEST_F(GeneratedTests, lstm3_state) {
    execute(lstm3_state::CreateModel,
            lstm3_state::is_ignored,
            lstm3_state::examples);
}
