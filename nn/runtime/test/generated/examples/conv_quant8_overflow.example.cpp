// Generated file (from: conv_quant8_overflow.mod.py). Do not edit
// Begin of an example
{
//Input(s)
{ // See tools/test_generator/include/TestHarness.h:MixedTyped
  // int -> FLOAT32 map
  {},
  // int -> INT32 map
  {},
  // int -> QUANT8_ASYMM map
  {{0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}}, {1, {10, 40, 70, 20, 50, 80, 30, 60, 90}}, {2, {0, 0, 0}}}
},
//Output(s)
{ // See tools/test_generator/include/TestHarness.h:MixedTyped
  // int -> FLOAT32 map
  {},
  // int -> INT32 map
  {},
  // int -> QUANT8_ASYMM map
  {{0, {75, 90, 107, 165, 203, 242, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}}}
}
}, // End of an example
