# model
model = Model()

i1 = Input("input", "TENSOR_QUANT8_ASYMM", "0.0f, 127.5f, {1, 4}") # batch = 1, depth = 1
beta = Float32Scalar("beta", 0.00001) # close to 0
output = Output("output", "TENSOR_QUANT8_ASYMM", "0.0f, 1.0f, {1, 4}")

# model 1
model = model.Operation("SOFTMAX", i1, beta).To(output)

# Example 1. Input in operand 0,
input0 = {i1: [1, 2, 10, 20]}

output0 = {output: [64, 64, 64, 64]}

# Instantiate an example
Example((input0, output0))
