// DO NOT EDIT;
// Generated by ml/nn/runtime/test/specs/generate_test.sh
#include "../../TestGenerated.h"

namespace svdf2 {
std::vector<MixedTypedExample> examples = {
// Generated svdf2 test
#include "generated/examples/svdf2.example.cpp"
};
// Generated model constructor
#include "generated/models/svdf2.model.cpp"
} // namespace svdf2
TEST_F(GeneratedTests, svdf2) {
    execute(svdf2::CreateModel,
            svdf2::is_ignored,
            svdf2::examples);
}
