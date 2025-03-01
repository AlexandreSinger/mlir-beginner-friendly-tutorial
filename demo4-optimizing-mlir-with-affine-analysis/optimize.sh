#!/bin/bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LLVM_BIN_DIR=$SCRIPT_DIR/../build/bin

# Lets take the same example from Demo 2 again.
$LLVM_BIN_DIR/mlir-opt \
    $SCRIPT_DIR/../demo2-entering-mlir/demo2.mlir \
    -o $SCRIPT_DIR/demo4-0-linalg-on-tensor.mlir

# Bufferize it.
$LLVM_BIN_DIR/mlir-opt \
    $SCRIPT_DIR/demo4-0-linalg-on-tensor.mlir \
    -one-shot-bufferize="bufferize-function-boundaries=true" \
    -o $SCRIPT_DIR/demo4-1-linalg-on-memref.mlir

# Then lower to the Affine level of abstraction (instead of loops).
# In the Affine dialect, it provides analytical models to represent how memory
# is accessed in a kernel. This is incredibly useful when performing
# optimizations on the buffer level.
$LLVM_BIN_DIR/mlir-opt \
    $SCRIPT_DIR/demo4-1-linalg-on-memref.mlir \
    -convert-linalg-to-affine-loops \
    -o $SCRIPT_DIR/demo4-2-affine.mlir

# One optimization we can perform is loop fusion. The kernel we are optimizing
# has two loops with similar iteration loops. The Affine dialect is able to
# analyze the memory access patterns of these two loops and recognize that they
# can be fused safely. This cannot be done in the loops dialect, since it cannot
# be certain that the loops are independent. The benefit of loop fusion in this
# case is that it removes the allocation of a 1MB memref! Beyond just saving
# memory, the reduces the memory footprint of the entire kernel which can
# improve cache locality.
$LLVM_BIN_DIR/mlir-opt \
    $SCRIPT_DIR/demo4-2-affine.mlir \
    -affine-loop-fusion \
    -o $SCRIPT_DIR/demo4-3-affine-fused.mlir

# Speaking of cache locality, tiling is another technique used to improve the
# cache locality. My computer has an L2 cache size of 1MiB, which is around
# 1024 kB. We can tell the affine-loop-tile pass that we have this cache size
# and it will tile the loops to optimize the core parts of the code to have a
# footprint of 1024 kB.
# Note: This can make the MLIR code look very strange since it does some
#       weirdness with the induction variables. To make it more readible for
#       this tutorial, I normalize and canonicalize everything.
$LLVM_BIN_DIR/mlir-opt \
    $SCRIPT_DIR/demo4-3-affine-fused.mlir \
    -affine-loop-tile="cache-size=1024" \
    -affine-loop-normalize \
    -canonicalize \
    -o $SCRIPT_DIR/demo4-4-affine-fused-tiled.mlir

# Once we have completed all of the Affine analysis we wish to perform, we lower
# out of the affine dialect into the scf and memref dialects. This can then be
# further lowered into LLVM, as shown in Demo 3.
$LLVM_BIN_DIR/mlir-opt \
    $SCRIPT_DIR/demo4-4-affine-fused-tiled.mlir \
    -lower-affine \
    -canonicalize \
    -o $SCRIPT_DIR/demo4-5-loops.mlir

