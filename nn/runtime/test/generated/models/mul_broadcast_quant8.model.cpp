// Generated file (from: mul_broadcast_quant8.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type2(Type::INT32, {});
  OperandType type0(Type::TENSOR_QUANT8_ASYMM, {1, 2}, 1.0, 0);
  OperandType type1(Type::TENSOR_QUANT8_ASYMM, {2, 2}, 1.0, 0);
  OperandType type3(Type::TENSOR_QUANT8_ASYMM, {2, 2}, 2.0, 0);
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto op2 = model->addOperand(&type1);
  auto act = model->addOperand(&type2);
  auto op3 = model->addOperand(&type3);
  // Phase 2, operations
  static int32_t act_init[] = {0};
  model->setOperandValue(act, act_init, sizeof(int32_t) * 1);
  model->addOperation(ANEURALNETWORKS_MUL, {op1, op2, act}, {op3});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {op1, op2},
    {op3});
  assert(model->isValid());
  model->finish();
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
