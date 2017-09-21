model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "0.0f, 127.5f, {1, 2, 2, 2}")
f1 = Input("op2", "TENSOR_QUANT8_ASYMM", "0.0f, 127.5f, {1, 2, 2, 2}")
b1 = Input("op3", "TENSOR_INT32", "0.0f, 63.75f, {2}")
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
cm = Int32Scalar("channelMultiplier", 1)
output = Output("op4", "TENSOR_QUANT8_ASYMM", "0.0f, 255.0f, {2}")

model = model.Operation("DEPTHWISE_CONV_2D",
                        i1, f1, b1,
                        pad0, pad0, pad0, pad0,
                        stride, stride,
                        cm, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [4, 16, 4, 32, 4, 64, 4, 128],
          f1:
          [2, 4,  2, 0,  2, 2,  2, 0],
          b1:
          [0, 0]}
# (i1 (depthconv) f1)
output0 = {output: # output 0
           [8, 48]}

# Instantiate an example
Example((input0, output0))
