// Generated file (from: sigmoid_quant8.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type1(Type::TENSOR_QUANT8_ASYMM, 0.0f, 1.0f, {1, 2, 2, 1});
  OperandType type0(Type::TENSOR_QUANT8_ASYMM, 0.0f, 127.0f, {1, 2, 2, 1});
  // Phase 1, operands
  auto op1 = model->addOperand(&type0);
  auto op3 = model->addOperand(&type1);
  // Phase 2, operations
  model->addOperation(ANEURALNETWORKS_LOGISTIC, {op1}, {op3});
  // Phase 3, inputs and outputs
  model->setInputsAndOutputs(
    {op1},
    {op3});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
