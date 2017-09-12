# model
model = Model()
i1 = Input("op1", "TENSOR_FLOAT32", "{2, 3}") # input tensor 0
i2 = Input("op2", "TENSOR_FLOAT32", "{2, 3}") # input tensor 1
axis0 = Int32Scalar("axis0", 0)
act0 = Int32Scalar("act0", 0)
r = Output("result", "TENSOR_FLOAT32", "{4, 3}") # output
model = model.Operation("CONCATENATION", i1, i2, axis0, act0).To(r)

# Example 1.
input0 = {i1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          i2: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}
output0 = {r: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}

# Instantiate an example
Example((input0, output0))
