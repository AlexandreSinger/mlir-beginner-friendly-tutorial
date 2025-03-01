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

# Demo 1: Motivating MLIR

Demo 1 is a demonstration on what writing high-performance code is like without
MLIR. It demonstrates a common technique where the user writes their code at a
high level (just basic linear-algebra operations) and use high-performance
libraries to make the code run fast. This is the same technique that is used
for libraries like TensorFlow and PyTorch in Python; this is written in C++ so
the actual instructions can be shown in more detail, but the same concepts apply
to Python.

`demo1.cpp` shows the high-level code that a user may write. This code is a
basic Fully-Connected layer one might find in a Deep Neural Network. To
implement this layer, one just needs to take the input vector (which is often
represented as a matrix due to batching) and multiplying it by a weight matrix.
The output of this is then passed into an activation function (in this case
ReLu) to get the output of the layer. The user writes this code at a high-level,
without concern for performance, which makes the code easier to write and work
with. It also makes the code more portable to other targets.

In order to write such high-level code, one must make use of high-performance
libraries. In this case, I wrote a basic library consisting of a tensor class
(`tensor.h`) and a linear algebra kernel library (`linalg_lib.h`). The code I
wrote in these libraries is the bare minimum required to make a library like
this and is not optimized at all. The key idea of this library is that the user
has no control over what is written here (sometimes this library is not even
accessible and is hidden as a binary); experts on the architecture build these
libraries using in-depth knowledge about the device (BLAS is a good example of
one of these libraries). The matmul kernel in `linalg.h` discusses different
optimizations one may perform for improved performance, however every
optimizations requires in-depth knowledge of the target architecture.

The benefit of this approach is that users do not need to be experts of the
device they are programming on and can make use of high-performance architectures
for their applications (like AI, HPC, etc.). This also allows for portability
between different accelerators and generations of chips. The downside is that
these high-performance libraries create a large barrier to entry for new chips
and require a lot of time and knowledge to design.

The key thing to notice from this demo is that the optimizations performed on
these kernels are often the same. To get good performance on MatMul, one has to
tile; but how much to tile is what changes.
There are scheduling libraries that
try to resolve this issue by separating the kernels from the optimizations;
however, ideally the compiler should be leveraged to perform these optimizations
since it knows this information about the architecture.
The problem is that, in order to compile this code to an executable, the code
must be written as bare for loops and memory accesses which lowers the abstraction
level too early for the compiler to do these optimizations.
This is where MLIR comes in. MLIR is a compiler framework which allows for
different levels of abstraction ("dialects") to be represented within its Intermediate
Representation. This allows for compiler passes to be written which perform
these high-level optimizations on kernels. These passes and dialects, if written
well, can be reused by different compilers to achieve good performance on all
devices (even hardware accelerators, which usually remain at very high levels of
abstraction).

There are other motivations for using MLIR, this is just a motivation that I felt
best encapsulates the "Multi-Level" aspect of MLIR.

