# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 1, 1, 3}")
i2 = Output("op2", "TENSOR_FLOAT32", "{1, 1, 1, 3}")

# Color wize (channel-wise) normalization
model = model.Operation("L2_NORMALIZATION", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0 - 1 color, 3 channels
          [0, 3, 4]}

output0 = {i2: # output 0
           [0, .6, .8]}

# Instantiate an example
Example((input0, output0))
