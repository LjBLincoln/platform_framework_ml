// DO NOT EDIT;
// Generated by ml/nn/runtime/test/specs/generate_test.sh
#include "../../TestGenerated.h"

namespace softmax_float_1 {
std::vector<MixedTypedExample> examples = {
// Generated softmax_float_1 test
#include "generated/examples/softmax_float_1.example.cpp"
};
// Generated model constructor
#include "generated/models/softmax_float_1.model.cpp"
} // namespace softmax_float_1
TEST_F(GeneratedTests, softmax_float_1) {
    execute(softmax_float_1::CreateModel,
            softmax_float_1::is_ignored,
            softmax_float_1::examples);
}
