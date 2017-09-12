# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # input 0
i2 = Output("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # output 0
model = model.Operation("RELU6", i1).To(i2)
# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-10.0, -0.5, 0.5, 10.0]}
output0 = {i2: # output 0
          [0.0, 0.0, 0.5, 6.0]}
# Instantiate an example
Example((input0, output0))

