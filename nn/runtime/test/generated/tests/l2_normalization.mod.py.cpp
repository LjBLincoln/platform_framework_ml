// DO NOT EDIT;
// Generated by ml/nn/runtime/test/specs/generate_test.sh
#include "../../TestGenerated.h"

namespace l2_normalization {
std::vector<MixedTypedExample> examples = {
// Generated l2_normalization test
#include "generated/examples/l2_normalization.example.cpp"
};
// Generated model constructor
#include "generated/models/l2_normalization.model.cpp"
} // namespace l2_normalization
TEST_F(GeneratedTests, l2_normalization) {
    execute(l2_normalization::CreateModel,
            l2_normalization::is_ignored,
            l2_normalization::examples);
}
