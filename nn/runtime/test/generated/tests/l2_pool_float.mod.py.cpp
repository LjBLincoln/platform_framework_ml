// DO NOT EDIT;
// Generated by ml/nn/runtime/test/specs/generate_test.sh
#include "../../TestGenerated.h"

namespace l2_pool_float {
std::vector<MixedTypedExample> examples = {
// Generated l2_pool_float test
#include "generated/examples/l2_pool_float.example.cpp"
};
// Generated model constructor
#include "generated/models/l2_pool_float.model.cpp"
} // namespace l2_pool_float
TEST_F(GeneratedTests, l2_pool_float) {
    execute(l2_pool_float::CreateModel,
            l2_pool_float::is_ignored,
            l2_pool_float::examples);
}
