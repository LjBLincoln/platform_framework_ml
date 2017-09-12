# model
model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "0.0f, 127.5f, {1, 2, 2, 1}") # input 0
o = Output("op2", "TENSOR_QUANT8_ASYMM", "0.0f, 127.5f, {1, 2, 2, 1}") # output 0
model = model.Operation("RELU1", i1).To(o)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 1, 2, 3]}
output0 = {o: # output 0
          [0, 1, 2, 2]}

# Instantiate one example
Example((input0, output0))

# Example 2. Input in operand 0,
input1 = {i1: # input 0
          [4, 10, 100, 255]}
output1 = {o: # output 0
          [2, 2, 2, 2]}

# Instantiate another example
Example((input1, output1))
