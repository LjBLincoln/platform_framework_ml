// Generated file (from: rnn.mod.py). Do not edit
void CreateModel(Model *model) {
  OperandType type2(Type::TENSOR_FLOAT32, {16, 16});
  OperandType type1(Type::TENSOR_FLOAT32, {16, 8});
  OperandType type3(Type::TENSOR_FLOAT32, {16});
  OperandType type4(Type::TENSOR_FLOAT32, {2, 16});
  OperandType type0(Type::TENSOR_FLOAT32, {2, 8});
  OperandType type5(Type::TENSOR_INT32, {1});
  // Phase 1, operands
  auto input = model->addOperand(&type0);
  auto weights = model->addOperand(&type1);
  auto recurrent_weights = model->addOperand(&type2);
  auto bias = model->addOperand(&type3);
  auto hidden_state = model->addOperand(&type4);
  auto activation_param = model->addOperand(&type5);
  auto output = model->addOperand(&type4);
  // Phase 2, operations
  model->addOperation(ANEURALNETWORKS_RNN, {input, weights, recurrent_weights, bias, hidden_state, activation_param}, {output});
  // Phase 3, inputs and outputs
  model->setInputsAndOutputs(
    {input, weights, recurrent_weights, bias, hidden_state, activation_param},
    {output});
  assert(model->isValid());
}

bool is_ignored(int i) {
  static std::set<int> ignore = {};
  return ignore.find(i) != ignore.end();
}
