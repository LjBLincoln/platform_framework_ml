// DO NOT EDIT;
// Generated by ml/nn/runtime/test/specs/generate_test.sh
#include "../../TestGenerated.h"

namespace local_response_norm_float_4 {
std::vector<MixedTypedExample> examples = {
// Generated local_response_norm_float_4 test
#include "generated/examples/local_response_norm_float_4.example.cpp"
};
// Generated model constructor
#include "generated/models/local_response_norm_float_4.model.cpp"
} // namespace local_response_norm_float_4
TEST_F(GeneratedTests, local_response_norm_float_4) {
    execute(local_response_norm_float_4::CreateModel,
            local_response_norm_float_4::is_ignored,
            local_response_norm_float_4::examples);
}
