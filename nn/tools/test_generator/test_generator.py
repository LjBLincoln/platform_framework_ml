#!/usr/bin/python3

# Copyright 2017, The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NN model compiler

Compile models and examples into NDK-based unit tests
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import contextlib

@contextlib.contextmanager
def smart_open(filename=None):
  if filename and filename != '-':
    fh = open(filename, 'w')
  else:
    fh = sys.stdout

  try:
    yield fh
  finally:
    if fh is not sys.stdout:
      fh.close()

class Phase(object):
  def __init__(self):
    self.__contents = []

  def append(self, x):
    self.__contents.append(x)

  def dump(self, filename):
    for x in self.__contents:
      print ("  " + x + ";", file=filename)

# Tracking objects inside a model with a not necessarily unique name and
# an unique number
class NamedObject(object):
  __serial = 0

  def __init__(self, name = "NamedObject"):
    self.__name = name
    self.__id = NamedObject.serial()
    NamedObject.__serial += 1

  def ID(self):
    return self.__id

  def serial():
    return NamedObject.__serial

  def get_name(self):
    return self.__name

  def __str__(self):
    return self.get_name()

  def __hash__(self):
    return self.__id

# Object that can be traversed during topological sorting phase
class Traversable(object):
  def traversable(self):
    return True

class Nontraversable(object):
  def traversable(self):
    return False

# Object that can take input from other objects
class Uses(object):
  all_uses = set()
  def __init__(self, ins = []):
    self.ins = ins.copy()
    Uses.all_uses.add(self)
    for i in ins:
      i.outs.add(self)

# Object that other objects takes its definition from
class Definitions(object):
  def __init__(self, outs = []):
    self.outs = set(outs)
    for o in outs:
      o.ins.append(self)

class Type(object):
  __types =  {}
  __type_serial = 0 # types have their own numbering
  def __init__(self, vt = None, shape = None):
    self.__vt = vt
    self.__shape = shape
    if vt is None or shape is None:
      self.__name = None
      return

    key = str(self)
    if key not in Type.__types:
      self.__id = Type.__type_serial
      Type.__types[str(self)] = self
      Type.__type_serial += 1
    else:
      self.__id = Type.__types[key].__id
    self.__name = "type" + str(self.__id)

  def get_name(self):
    return self.__name

  def __str__(self):
    return (", ".join([self.__vt, self.__shape]))

  def __hash__(self):
    return self.__id

  def dump(filename):
    for key, value in sorted(Type.__types.items()):
      print ("  OperandType " + str(value.__name) + "(Type::" + str(key) + ");", file=filename)



# A value is a typed, named object
class Value(NamedObject):
  def __init__(self, name, vt):
    NamedObject.__init__(self, name)
    self.type = vt

# An operand that can be fed into operations. Also, an operand is always
# declared before operations.
class Operand(Value):
  # All operand declarations in string
  operands = Phase()

  def __init__(self, name, vt):
    Value.__init__(self, name, vt)
    def_string = (
        "auto " + self.get_name() + " = "\
            "model->addOperand(&" + vt.get_name() + ")")
    Operand.operands.append(def_string)

  # By default, produce nothing (when asked by the Topological Sort phase)
  def Definition(self):
    pass

  def Reference(self):
    return NamedObject.__str__(self)

  # Print a set of operands in curly braces
  def print_operands(operands):
    return [ x.Reference() for x in operands ]

# A user-declared input operand
class Input(Operand, Definitions, Traversable):
  # for enumerating inputs
  __next_number = 0
  # Holds reference to all Inputs; used by Topoligcal sort as starting nodes.
  __inputs = set()

  def __init__(self, name, vt, shape):
    Operand.__init__(self, name, Type(vt, shape))
    Definitions.__init__(self)
    Input.__inputs.add(self)
    self.number = Input.__next_number
    Input.__next_number += 1

  def is_internal(self):
    return False

  def get_inputs(exclude_internal = None):
    if exclude_internal is not None:
      external = { x for x in Input.__inputs if not x.is_internal() }
      return external
    else:
      return Input.__inputs

# A user-declared output operand
class Output(Operand, Uses, Nontraversable):
  # for enumerating outputs
  __next_number = 0
  __outputs = set()

  def __init__(self, name, vt, shape):
    Operand.__init__(self, name, Type(vt, shape))
    Uses.__init__(self)
    Output.__outputs.add(self)
    self.number = Output.__next_number
    Output.__next_number += 1

  def get_outputs():
    return Output.__outputs

class ModelArgument:
  __arguments = []

  def __init__(self, arg_type, arg_name):
    self.__arg_type = arg_type
    self.__arg_name = arg_name
    ModelArgument.__arguments.append(" ".join([arg_type, arg_name]))

  def get_arg_type(self):
    return self.__arg_type

  def get_arg_name(self):
    return self.__arg_name

  def get_arguments():
    return ModelArgument.__arguments

class Parameter(Input):
  __type_lookup = {
      "INT32": "int32_t",
      "FLOAT32": "float",
      "TENSOR_INT32": "int32_t",
      "TENSOR_FLOAT32": "float",
      "TENSOR_QUANT8_ASYMM": "uint8_t",
    }

  def __init__(self, name, vt, shape, initializer):
    Input.__init__(self, name, vt, shape)
    self.initializer = initializer
    self.cpptype = Parameter.__type_lookup[vt]
  def is_internal(self):
    return True
  def Definition(self):
    init_name = self.get_name() + "_init"
    initializer = [str(x) for x in self.initializer]
    if self.cpptype == "float":
      initializer = [ x+"f" for x in initializer]
    init = self.cpptype + " " + init_name + "[]"
    init = "static " + init + " = {" + ", ".join(initializer) + "};"
    args = [ self.get_name(), init_name,
            "sizeof(" + self.cpptype + ") * " + str(len(self.initializer)) ]
    stmt = "\n  ".join([init,
                      "model->setOperandValue(" + ", ".join(args)+");"])
    return stmt

class Int32Scalar(Parameter):
  def __init__(self, name, value):
    Parameter.__init__(self, name, "INT32", "{1}", [value])

class Float32Scalar(Parameter):
  def __init__(self, name, value):
    Parameter.__init__(self, name, "FLOAT32", "{1}", [value])

# A compiler-generated intermediate result from an operation
class IntermediateResult(Operand, Definitions, Uses, Traversable):
  def __init__(self, src: Value):
    tmp_name = "tmp" + str(NamedObject.serial())
    Operand.__init__(self, tmp_name, src.type)
    Definitions.__init__(self)
    Uses.__init__(self, [src])

# An explicitly declared intermediate result
class Internal(Operand, Definitions, Uses, Traversable):
  def __init__(self, name, vt, shape):
    Operand.__init__(self, name, Type(vt, shape))
    Definitions.__init__(self)
    Uses.__init__(self)

# An operation in a model
class Operation(Value, Definitions, Uses, Traversable):
  def __init__(self, optype, ins, outs):
    Value.__init__(self, optype, ins[0].type)
    Definitions.__init__(self, outs)
    Uses.__init__(self, ins)
    self.optype = optype

  def __str__(self):
    inputs = [ str(x) for x in self.ins ]
    return "Operation:" + self.optype + " " + ", ".join(inputs)

  def Reference(self):
    return "operation" + str(self.ID());

  def Definition(self):
    inputs = Operand.print_operands(self.ins);
    outputs = Operand.print_operands(self.outs);
    return "model->addOperation(ANEURALNETWORKS_"+self.optype+", " + \
        "{"+", ".join(inputs)+"}, {" + ", ".join(outputs) + "});"

# Main interface
class Model(object):
  def __init__(self):
    self.__currentOp = None

  # TODO turn this into generic binary operations
  def Add(self, i1: Value, i2 = None) -> Operation:
    ins = [i1]
    if i2 is not None:
      ins.append(i2)
    if self.__currentOp is not None:
      ir = IntermediateResult(self.__currentOp)
      self.__currentOp = ir
      ins.append(self.__currentOp)

    op = Operation("ADD", ins, [])

    self.__currentOp = op
    return self

  def Operation(self, op_name, *args):
    ins = [i for i in args]
    outs = []
    op = Operation(op_name, ins, outs)
    self.__currentOp = op
    return self

  def RawAdd(self, i1: Value, i2: Value, o = None) -> Operation:
    ins = [i1, i2]
    outs = []
    if o is not None:
      outs = [o]
    op = Operation("ADD", ins, outs)

    self.__currentOp = op
    return self

  # See CpuExecutor::executeOperation() for the arguments of each op
  def AveragePool(self, input, padding, stride_width, stride_height, filter_width, filter_height, activation):
    ins = [input, padding, stride_width,
           stride_height, filter_width, filter_height, activation]
    outs = []
    op = Operation("AVERAGE_POOL", ins, outs)
    self.__currentOp = op
    return self

  def Concatenation(self, *args):
    ins = [i for i in args]
    outs = []
    op = Operation("CONCATENATION", ins, outs)
    self.__currentOp = op
    return self

  def Conv(self, filter, bias, input, padding, stride_width, stride_height, activation):
    ins = [filter, bias, input, padding, stride_width,
           stride_height, activation]
    outs = []
    op = Operation("CONV", ins, outs)
    self.__currentOp = op
    return self

  def DepthWiseConv(self, filter, bias, input, padding, stride_width, stride_height, depth_multiplier, activation):
    ins = [filter, bias, input, padding, stride_width,
           stride_height, depth_multiplier, activation]
    outs = []
    op = Operation("DEPTHWISE_CONV", ins, outs)
    self.__currentOp = op
    return self

  def FullyConnected(self, input, weights, bias, activation):
    ins = [input, weights, bias, activation]
    outs = []
    op = Operation("FULLY_CONNECTED", ins, outs)
    self.__currentOp = op
    return self

  def Logistic(self, input):
    ins = [input]
    outs = []
    op = Operation("LOGISTIC", ins, outs)
    self.__currentOp = op
    return self

  def L2Pool(self, input, padding, stride_width, stride_height, filter_width, filter_height, activation):
    ins = [input, padding, stride_width,
           stride_height, filter_width, filter_height, activation]
    outs = []
    op = Operation("L2_POOL", ins, outs)
    self.__currentOp = op
    return self

  def MaxPool(self, input, padding, stride_width, stride_height, filter_width, filter_height, activation):
    ins = [input, padding, stride_width,
           stride_height, filter_width, filter_height, activation]
    outs = []
    op = Operation("MAX_POOL", ins, outs)
    self.__currentOp = op
    return self

  def SoftMax(self, input, beta):
    ins = [input, beta]
    outs = []
    op = Operation("SOFTMAX", ins, outs)
    self.__currentOp = op
    return self

  def Out(self, o: Value) -> Operation:
    self.__currentOp.outs.add(o)
    o.ins.append(self.__currentOp)
    return self

  def To(self, o:Value):
    ret = Model.Out(self, o)
    self.__currentOp = None
    return self

class Example():
  __examples = []
  def __init__(self, list_of_examples):
    Example.__examples.append(list_of_examples)

  def dump_dict(d):
    ret = []
    for k, v in d.items():
      init = ", ".join([str(i)+'f' for i in v])
      key = str(k)
      if type(k) is not int:
        key = str(k.number)
      ret.append('{%s, {%s}}' %(key, init))
    return ", ".join(ret)

  def dump(example_file):
    if len(Example.__examples) > 0:
      print ('// Generated file. Do not edit', file = example_file)
    for i, o in Example.__examples:
      print ('// Begin of an example', file = example_file)
      print ('{', file = example_file)
      inputs = Example.dump_dict(i)
      outputs = Example.dump_dict(o)
      print ('//Input(s)\n{' + inputs + '},', file = example_file)
      print ('//Output(s)\n{' + outputs + '}', file = example_file)
      print ('}, // End of an example', file = example_file)

def TopologicalSort(model_file):
  start = Input.get_inputs().copy()
  deps = { x: set(x.ins) for x in Uses.all_uses }

  while len(start) > 0:
    cur = start.pop()
    op = cur.Definition()
    if op is not None:
      print ("  " + op, file = model_file)
    for o in cur.outs:
      deps[o].remove(cur)
      if len(deps[o]) == 0 and o.traversable():
        start.add(o)

# Take a model from command line
def import_source():
  parser = argparse.ArgumentParser()
  parser.add_argument("spec", help='the spec file')
  parser.add_argument("-m", "--model", help='the output model file', default='-')
  parser.add_argument("-e", "--example", help='the output example file', default='-')
  args = parser.parse_args()

  if os.path.exists(args.spec):
    exec(open(args.spec).read())

  return (args.model, args.example)

if __name__ == '__main__':
  (model, example) = import_source()
  # Boilerplate
  args = ""
  if len(ModelArgument.get_arguments()) > 0:
    args = ", " + ", ".join(ModelArgument.get_arguments())

  print ("Output model:" + model, file = sys.stderr)
  print ("Output example:" + example, file = sys.stderr)

  with smart_open(model) as model_file:
    print ('// Generated file. Do not edit', file = model_file)
    print ("void CreateModel(Model *model" + args + ") {", file=model_file)

    # Phase 0: types
    Type.dump(model_file)
    # Phase 1: add operands
    print ("  // Phase 1, operands", file=model_file)
    Operand.operands.dump(model_file)

    # Phase 2: operations
    print ("  // Phase 2, operations", file=model_file)
    TopologicalSort(model_file)

    # Phase 3: add inputs and outputs
    print ("  // Phase 3, inputs and outputs", file=model_file)
    inputs = Operand.print_operands(Input.get_inputs(True));
    outputs = Operand.print_operands(Output.get_outputs());
    print ("  model->setInputsAndOutputs(\n" +
           "    {"+", ".join(inputs)+"},\n    {" + ", ".join(outputs) + "});",
           file=model_file)
    # Boilerplate
    print ("  assert(model->isValid());", file=model_file);
    print ("}", file=model_file)

  with smart_open(example) as example_file:
    Example.dump(example_file)


