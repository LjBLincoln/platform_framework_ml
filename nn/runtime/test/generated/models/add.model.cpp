// Generated file (from: add.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type1(Type::INT32, {});
  OperandType type0(Type::TENSOR_FLOAT32, {2});
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto op2 = model->addOperand(&type0);
  auto b0 = model->addOperand(&type1);
  auto op3 = model->addOperand(&type0);
  // Phase 2, operations
  static int32_t b0_init[] = {0};
  model->setOperandValue(b0, b0_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_ADD, {op1, op2, b0}, {op3});
  // Phase 3, inputs and outputs
  model->setInputsAndOutputs(
    {op1, op2},
    {op3});
  assert(model->isValid());
}
