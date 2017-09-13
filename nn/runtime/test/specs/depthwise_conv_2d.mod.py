model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 2}")
f1 = Input("op2", "TENSOR_FLOAT32", "{2, 2, 2, 2}")
b1 = Input("op3", "TENSOR_FLOAT32", "{2}")
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
cm = Int32Scalar("channelMultiplier", 1)
output = Output("op4", "TENSOR_FLOAT32", "{2}")

model = model.Operation("DEPTHWISE_CONV_2D",
                        i1, f1, b1,
                        pad0, pad0, pad0, pad0,
                        stride, stride,
                        cm, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [10, 21,  10, 22,  10, 23,  10, 24],
          f1:
          [.25, 0,  .25, 1,  .25, 0,  .25, 1],
          b1:
          [0, 4]}
# (i1 (conv) f1) + b1
output0 = {output: # output 0
           [10, 50]}

# Instantiate an example
Example((input0, output0))
