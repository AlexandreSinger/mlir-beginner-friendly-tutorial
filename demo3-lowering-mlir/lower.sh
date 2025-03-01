#!/bin/bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LLVM_BIN_DIR=$SCRIPT_DIR/../build/bin

# Clean up demo 2. This is just to put a clean starting point in this folder.
# linalg-on-tensor is the entry point for this flow and represents the logical
# computation the user wishes to perform, without any information on the device
# we are targetting.
$LLVM_BIN_DIR/mlir-opt \
    $SCRIPT_DIR/../demo2-entering-mlir/demo2.mlir \
    -o $SCRIPT_DIR/demo3-0-linalg-on-tensor.mlir

# Lower tensors to memrefs. Since we plan on targeting CPUs, we cannot stay at
# the tensor abstraction since every chunk of data needs to have an allocated
# buffer somewhere in memory. We can make use of the one-shot-bufferize pass
# in MLIR to convert the Tensors into MemRefs. MemRefs are practically wrappers
# around allocated pointers to memory with shape information.
$LLVM_BIN_DIR/mlir-opt \
    $SCRIPT_DIR/demo3-0-linalg-on-tensor.mlir \
    -one-shot-bufferize="bufferize-function-boundaries=true" \
    -canonicalize \
    -o $SCRIPT_DIR/demo3-1-linalg-on-memref.mlir

# Convert the linalg operations into loops in the SCF dialect. We are assuming
# that our target device cannot compute MatMul and ReLU directly; so we convert
# these kernels into loops. This is a lower level of abstraction than the linalg
# algorithms themselves and closer to what a CPU can execute.
$LLVM_BIN_DIR/mlir-opt \
    $SCRIPT_DIR/demo3-1-linalg-on-memref.mlir \
    -convert-linalg-to-loops \
    -o $SCRIPT_DIR/demo3-2-loops.mlir

# Convert the loops into branches. Many CPU architectures use branch instructions
# to represent loops. We lower the loops in our kernel into branches; further
# lowering the control flow abstraction to as close to CPUs as we can.
$LLVM_BIN_DIR/mlir-opt \
    $SCRIPT_DIR/demo3-2-loops.mlir \
    -convert-scf-to-cf \
    -o $SCRIPT_DIR/demo3-3-branches.mlir

# Finally, we convert everything to the LLVM dialect. We have lowered the
# abstraction of the data and control flow low enough that we can convert
# basically one-to-one to the LLVM dialect. We target LLVM here since LLVM
# contains standard backend code which can generate real assembly for the kernel
# we wrote.
$LLVM_BIN_DIR/mlir-opt \
    $SCRIPT_DIR/demo3-3-branches.mlir \
    -convert-func-to-llvm \
    -convert-cf-to-llvm \
    -finalize-memref-to-llvm \
    -convert-arith-to-llvm \
    -reconcile-unrealized-casts \
    -canonicalize \
    -o $SCRIPT_DIR/demo3-4-llvm.mlir

# Once we are fully in the LLVM dialect, we can use the mlir-translate tool to
# convert the LLVM dialect MLIR code into LLVMIR.
$LLVM_BIN_DIR/mlir-translate \
    $SCRIPT_DIR/demo3-4-llvm.mlir \
    --mlir-to-llvmir \
    -o $SCRIPT_DIR/demo3-5.ll

# This LLVMIR code can then be further optimized and lowered into assembly code
# to run on a CPU. This is not shown for this tutorial.

