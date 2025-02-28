# mlir-beginner-friendly-tutorial

This is a beginner-friendly tutorial on MLIR from the perspective of a user of
MLIR, not a compiler engineer. This tutorial will introduce why MLIR exists and
how it is used to compile code at different levels of abstraction. This tutorial
will focus on working with the "core" dialects of MLIR.

# 0: Building MLIR

In this repository, you will find a Git submodule of LLVM. This was the most
recent version of LLVM that was available when I wrote this tutorial. There is
nothing special about it, but I provided it here so the results of the tutorial
will always match in the future. Make sure you have initialized the submodule
using:
```sh
git submodule init
git submodule update
```

These build instructions are based on the Getting Started page provided by MLIR:
https://mlir.llvm.org/getting_started/

This tutorial uses the Ninja generator to build MLIR, this can be installed using:
```sh
apt-get install ninja-build
```

Create a build folder and run the following CMake command. After run the final
command to build MLIR and check that it built successfully.

```sh
mkdir build
cd build
cmake -G Ninja ../llvm-project/llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON
ninja check-mlir
```

Note: That last command will take a very long time to run. This will build all
of the LLVM and MLIR code necessary to check that MLIR was built correctly for
your system. I recommend giving it many cores using `-j<num_cores>`.

After this command completes, you should have the executables you need to perform
this tutorial.

