// DO NOT EDIT;
// Generated by ml/nn/runtime/test/specs/generate_test.sh
#include "../../TestGenerated.h"

namespace svdf {
std::vector<MixedTypedExample> examples = {
// Generated svdf test
#include "generated/examples/svdf.example.cpp"
};
// Generated model constructor
#include "generated/models/svdf.model.cpp"
} // namespace svdf
TEST_F(GeneratedTests, svdf) {
    execute(svdf::CreateModel,
            svdf::is_ignored,
            svdf::examples);
}
