// Generated file (from: embedding_lookup.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type0(Type::TENSOR_FLOAT32, {3, 2, 4});
  OperandType type1(Type::TENSOR_FLOAT32, {3});
  // Phase 1, operands
  auto value = model->addOperand(&type0);
  auto index = model->addOperand(&type1);
  auto output = model->addOperand(&type0);
  // Phase 2, operations
  model->addOperation(ANEURALNETWORKS_EMBEDDING_LOOKUP, {value, index}, {output});
  // Phase 3, inputs and outputs
  model->identifyInputsAndOutputs(
    {value, index},
    {output});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
