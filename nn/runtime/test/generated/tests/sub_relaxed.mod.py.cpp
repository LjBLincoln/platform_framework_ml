// DO NOT EDIT;
// Generated by ml/nn/runtime/test/specs/generate_test.sh
#include "../../TestGenerated.h"

namespace sub_relaxed {
std::vector<MixedTypedExample> examples = {
// Generated sub_relaxed test
#include "generated/examples/sub_relaxed.example.cpp"
};
// Generated model constructor
#include "generated/models/sub_relaxed.model.cpp"
} // namespace sub_relaxed
TEST_F(GeneratedTests, sub_relaxed) {
    execute(sub_relaxed::CreateModel,
            sub_relaxed::is_ignored,
            sub_relaxed::examples);
}
