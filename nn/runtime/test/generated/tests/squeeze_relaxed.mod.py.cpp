// DO NOT EDIT;
// Generated by ml/nn/runtime/test/specs/generate_test.sh
#include "../../TestGenerated.h"

namespace squeeze_relaxed {
std::vector<MixedTypedExample> examples = {
// Generated squeeze_relaxed test
#include "generated/examples/squeeze_relaxed.example.cpp"
};
// Generated model constructor
#include "generated/models/squeeze_relaxed.model.cpp"
} // namespace squeeze_relaxed
TEST_F(GeneratedTests, squeeze_relaxed) {
    execute(squeeze_relaxed::CreateModel,
            squeeze_relaxed::is_ignored,
            squeeze_relaxed::examples);
}
