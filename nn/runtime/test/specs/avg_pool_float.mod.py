# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # input 0
cons1 = Int32Scalar("cons1", 1)
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
i3 = Output("op3", "TENSOR_FLOAT32", "{1, 2, 2, 1}") # output 0
model = model.Operation("AVERAGE_POOL_2D", i1, pad0, pad0, pad0, pad0, cons1, cons1, cons1, cons1, act).To(i3)
# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1.0, 2.0, 3.0, 4.0]}
output0 = {i3: # output 0
          [1.0, 2.0, 3.0, 4.0]}
# Instantiate an example
Example((input0, output0))

