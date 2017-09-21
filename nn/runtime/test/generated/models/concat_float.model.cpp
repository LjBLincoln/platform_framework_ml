// Generated file (from: concat_float.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type1(Type::INT32, {});
  OperandType type0(Type::TENSOR_FLOAT32, {2, 3});
  OperandType type2(Type::TENSOR_FLOAT32, {4, 3});
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto op2 = model->addOperand(&type0);
  auto axis0 = model->addOperand(&type1);
  auto act0 = model->addOperand(&type1);
  auto result = model->addOperand(&type2);
  // Phase 2, operations
  static int32_t axis0_init[] = {0};
  model->setOperandValue(axis0, axis0_init, sizeof(int32_t) * 1);
  static int32_t act0_init[] = {0};
  model->setOperandValue(act0, act0_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_CONCATENATION, {op1, op2, axis0, act0}, {result});
  // Phase 3, inputs and outputs
  model->setInputsAndOutputs(
    {op1, op2},
    {result});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
