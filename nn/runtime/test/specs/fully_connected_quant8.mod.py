# model

model = Model()
in0 = Input("op1", "TENSOR_QUANT8_ASYMM", "0.0f, 127.5f, {3}")
weights = Input("op2", "TENSOR_QUANT8_ASYMM", "0.0f, 127.5f, {1, 1}")
bias = Input("b0", "TENSOR_QUANT8_ASYMM", "0.0f, 63.75f, {1}")
out0 = Output("op3", "TENSOR_QUANT8_ASYMM", "0.0f, 255.0f,{3}")
act = Int32Scalar("act", 0)
model = model.Operation("FULLY_CONNECTED", in0, weights, bias, act).To(out0)

# Example 1. Input in operand 0,
input0 = {in0: # input 0
             [2, 32, 16],
         weights: [2],
         bias: [4]}
output0 = {out0: # output 0
               [2, 17, 9]}

# Instantiate an example
Example((input0, output0))
