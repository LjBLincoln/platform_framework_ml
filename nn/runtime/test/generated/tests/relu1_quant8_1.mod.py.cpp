// DO NOT EDIT;
// Generated by ml/nn/runtime/test/specs/generate_test.sh
#include "../../TestGenerated.h"

namespace relu1_quant8_1 {
std::vector<MixedTypedExample> examples = {
// Generated relu1_quant8_1 test
#include "generated/examples/relu1_quant8_1.example.cpp"
};
// Generated model constructor
#include "generated/models/relu1_quant8_1.model.cpp"
} // namespace relu1_quant8_1
TEST_F(GeneratedTests, relu1_quant8_1) {
    execute(relu1_quant8_1::CreateModel,
            relu1_quant8_1::is_ignored,
            relu1_quant8_1::examples);
}
