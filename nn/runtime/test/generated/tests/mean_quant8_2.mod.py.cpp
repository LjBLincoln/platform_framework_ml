// DO NOT EDIT;
// Generated by ml/nn/runtime/test/specs/generate_test.sh
#include "../../TestGenerated.h"

namespace mean_quant8_2 {
std::vector<MixedTypedExample> examples = {
// Generated mean_quant8_2 test
#include "generated/examples/mean_quant8_2.example.cpp"
};
// Generated model constructor
#include "generated/models/mean_quant8_2.model.cpp"
} // namespace mean_quant8_2
TEST_F(GeneratedTests, mean_quant8_2) {
    execute(mean_quant8_2::CreateModel,
            mean_quant8_2::is_ignored,
            mean_quant8_2::examples);
}
