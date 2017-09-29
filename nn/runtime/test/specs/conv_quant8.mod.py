model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{1, 3, 3, 1}, 0.5f, 0")
f1 = Input("op2", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 1}, 0.5f, 0")
b1 = Input("op3", "TENSOR_QUANT8_ASYMM", "{1}, 0.25f, 0")
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
# output dimension:
#     (i1.height - f1.height + 1) x (i1.width - f1.width + 1)
output = Output("op4", "TENSOR_QUANT8_ASYMM", "{1, 2, 2, 1}, 1.f, 0")

model = model.Operation("CONV_2D", i1, f1, b1, pad0, pad0, pad0, pad0, stride, stride, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [8, 8, 8, 8, 4, 8, 8, 8, 8],
          f1:
          [2, 2, 2, 2],
          b1:
          [4]}
# (i1 (conv) f1) + b1
output0 = {output: # output 0
           [15, 15, 15, 15]}

# Instantiate an example
Example((input0, output0))
