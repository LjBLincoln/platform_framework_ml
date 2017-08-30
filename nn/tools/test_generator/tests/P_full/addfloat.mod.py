# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2}") # input 0
i2 = Input("op2", "TENSOR_FLOAT32", "{2}") # input 0
b0 = Int32Bias("b0", 0) # an int32_t bias
i3 = Output("op3", "TENSOR_FLOAT32", "{2}") # output 0
model = model.Operation("ADD", i1, i2, b0).To(i3)

# Example 1. Input in operand 0,
input0 = {0: # input 0
          [1.0, 2.0],
          1: # input 1
          [3.0, 4.0]}

output0 = {0: # output 0
           [4.0, 6.0]}

# Instantiate an example
Example((input0, output0))



