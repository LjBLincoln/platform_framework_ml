# model
model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "0.0f, 127.5f, {1, 2, 2, 1}") # input 0
i2 = Output("op2", "TENSOR_QUANT8_ASYMM", "0.0f, 127.5f, {1, 2, 2, 1}") # output 0
model = model.Operation("RELU6", i1).To(i2)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 1, 11, 12]}
output0 = {i2: # output 0
          [0, 1, 11, 12]}
# Instantiate an example
Example((input0, output0))

# Example 2. Input in operand 0,
input1 = {i1: # input 0
          [13, 14, 254, 255]}
output1 = {i2: # output 0
          [12, 12, 12, 12]}
# Instantiate an example
Example((input1, output1))
