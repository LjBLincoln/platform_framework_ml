# How to add an NDK-based test from a TFLite model and example

`tflite_example_converter` is a small tool that converts a TFLite model and examples to a form that only depends on NNAPI. This README shows how we could use the tool to turn a TFLite test into an NNAPI test.

## Build `tflite_example_converter` in TFLite

First, copy `google3/tflite_example_converter.cc` to a CitC client, say at `$G3`:

    cp google3/tflite_example_converter.cc \
    $G3/third_party/tensorflow/contrib/lite/testing

Then, edit BUILD of the same directory so blaze can build tflite_example_converter, add a target like:

    cc_binary(
      name = "tflite_example_converter",
      srcs = [
          "tflite_example_converter.cc",
      ],
      deps = [
          ":parse_testdata_lib",
          "//third_party/tensorflow/contrib/lite:builtin_ops",
          "//third_party/tensorflow/contrib/lite:framework",
      ],
    )

# Convert a TFLite model and examples

  Let's take a single-op model like "add" as an example.

    cd $G3/third_party/tensorflow/contrib/lite/testing/generated_data/simple

  run:

      $G3/blaze-bin/third_party/tensorflow/contrib/lite/testing/\
      tflite_example_converter add.bin add_tests.txt

  This step creates two files, one in C++, representing the test inputs,
  and another in python, for the test generator.

  The output would look like:

    Model file generated at: add.bin.model.py
    Example file generated at: add_tests.txt.example.cc


  run `test_generator.py` on the generated model file, i.e.:

`./test_generator.py $G3/google3/third_party/tensorflow/contrib/lite/testing/generated_data/simple/add.bin.model.py`

   and it would print out a C++ function like:

    void CreateModel(Model *model) {
      OperandType type0(Type::TENSOR_FLOAT32, {1, 8, 8, 3});
      // Phase 1, operands
      auto op1 = model->addOperand(&type0);
      auto op2 = model->addOperand(&type0);
      auto op0 = model->addOperand(&type0);
      // Phase 2, operations
      model->addOperation(ANEURALNETWORKS_ADD, {op1, op1}, {op0});
      model->addOperation(ANEURALNETWORKS_ADD, {op0, op1}, {op2});
      // Phase 3, inputs and outputs
      model->setInputsAndOutputs(
        {op1},
        {op2});
      assert(model->isValid());
    }

   Save the output as a C++ source file to `frameworks/ml/nn/runtime/test/generated/model`, and copy the example to
   `frameworks/ml/nn/runtime/test/generated/examples`.

   Then edit frameworks/ml/nn/runtime/test/Conv.cpp to include the generated files similar to how other tests are included. Add a new Google test `TEST_F` as needed.

