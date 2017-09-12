# model
model = Model()

i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "0.0f, 127.0f, {1, 2, 2, 1}")
i3 = Output("op3", "TENSOR_QUANT8_ASYMM", "0.0f, 127.0f, {1, 2, 2, 1}")
model = model.Operation("LOGISTIC", i1).To(i3)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [0, 1, 2, 127]}

output0 = {i3: # output 0
           [128, 159, 187, 255]}

# Instantiate an example
Example((input0, output0))
