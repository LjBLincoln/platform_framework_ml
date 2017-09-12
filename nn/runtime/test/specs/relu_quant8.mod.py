# model
model = Model()
# input 0
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "-127.5f, 127.5f, {1, 2, 2, 1}")
# output 0
o = Output("op2", "TENSOR_QUANT8_ASYMM", "-127.5f, 127.5f, {1, 2, 2, 1}")
model = model.Operation("RELU", i1).To(o)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 1, 126, 127]}
output0 = {o: # output 0
           [128, 128, 128, 128]}

# Instantiate an example
Example((input0, output0))

# Example 2. Input in operand 0,
input1 = {i1: # input 0
          [128, 129, 254, 255]}
output1 = {o: # output 0
           [128, 129, 254, 255]}

# Instantiate another example
Example((input1, output1))
