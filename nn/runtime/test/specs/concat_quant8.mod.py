# model
model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM", "{2, 3}, 0.5f, 0") # input 0
i2 = Input("op2", "TENSOR_QUANT8_ASYMM", "{2, 3}, 0.5f, 0") # input 1
axis1 = Int32Scalar("axis1", 1)
act0 = Int32Scalar("act0", 0)
r = Output("result", "TENSOR_QUANT8_ASYMM", "{2, 6}, 0.5f, 0") # output
model = model.Operation("CONCATENATION", i1, i2, axis1, act0).To(r)

# Example 1.
input0 = {i1: [1, 2, 3, 4, 5, 6],
          i2: [7, 8, 9, 10, 11, 12]}
output0 = {r: [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12]}

# Instantiate an example
Example((input0, output0))
