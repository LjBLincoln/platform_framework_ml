# model
model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "0.0f, 127.5f, {1, 2, 2, 1}") # input 0
cons1 = Int32Scalar("cons1", 1)
act = Int32Scalar("act", 0)
i3 = Output("op3", "TENSOR_QUANT8_ASYMM", "0.0f, 127.5f, {1, 2, 2, 1}") # output 0
model = model.Operation("MAX_POOL_2D", i1, cons1, cons1, cons1, cons1, cons1, act).To(i3)
# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 2, 3, 4]}
output0 = {i3: # output 0
          [1, 2, 3, 4]}
# Instantiate an example
Example((input0, output0))
