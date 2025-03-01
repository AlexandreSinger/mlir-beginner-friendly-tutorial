# mlir-beginner-friendly-tutorial

This is a beginner-friendly tutorial on MLIR from the perspective of a user of
MLIR, not a compiler engineer. This tutorial will introduce why MLIR exists and
how it is used to compile code at different levels of abstraction. This tutorial
will focus on working with the "core" dialects of MLIR.

# Demo 0: Building MLIR

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

# Demo 2: Entering MLIR

Now that we have motivated why we may want to use MLIR, lets convert the high-level
code from demo 1 into MLIR:
```cpp
int main(void) {
    Tensor<float, 256, 512> FC_INPUT;
    Tensor<float, 512, 1024> FC_WEIGHT;
    Tensor<float, 256, 1024> FC_OUTPUT = matmul(FC_INPUT, FC_WEIGHT);
    Tensor<float, 256, 1024> OUT = relu(FC_OUTPUT);
}
```

Often, custom tools are created which can convert code such as the code above
into MLIR automatically. For this example, such a tool is trivial to write since
I chose the API for the library to match the Tensor + Linalg level of abstraction
in MLIR. Often times people do things the other way around, the build a level of
abstraction in MLIR which matches their pre-existing APIs; however, for this
tutorial, I wanted to use the core MLIR dialects.

I chose to enter the Tensor + Linalg level of abstraction for this demo since
this is a common abstraction used by the key users of MLIR (TensorFlow and
PyTorch). It is also a very interesting level of abstraction.

I converted the code above into MLIR code by hand. This code can be found in
`demo2.mlir`. For this tutorial it does not matter if the MLIR code was generated
by a tool or not. In this MLIR file, you will find comments where I describe how
to read the Intermediate Representation at this level of abstraction.

MLIR provides a tool called `mlir-opt` which is used to test MLIR code. This
tool runs passes on MLIR code (which will be described in the next demo) and
verfies that the MLIR code is valid between passes. Since it runs validation so
often, this tool is often just used for testing / debugging; while custom tools
based on this one are used when building a real compiler flow. For this part of
the demo, I want to use this tool to ensure that the MLIR code I wrote by hand
is valid. After following the steps in Demo 0, you should have `mlir-opt` already
built in the `build/bin` folder. To use it, we perform the following command:
```sh
./build/bin/mlir-opt demo2-entering-mlir/demo2.mlir
```

`mlir-opt` will print an error if there is a syntax error with what I wrote. In
this case we get no error. You will notice that this tool will print the IR after
parsing. This renames many of the values and remove any comments which are not
necessary for compilation. You can print this result to a file using the `-o`
option.

