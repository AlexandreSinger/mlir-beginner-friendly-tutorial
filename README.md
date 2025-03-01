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
```mlir
#map = affine_map<(i, j) -> (i, j)>
module {
func.func @main() -> tensor<256x1024xf32> {
    %FC_INPUT = tensor.empty() : tensor<256x512xf32>
    %FC_WEIGHT = tensor.empty() : tensor<512x1024xf32>
    %c_init = arith.constant 0.0 : f32
    %matmul_init = tensor.splat %c_init : tensor<256x1024xf32>
    %FC_OUTPUT = linalg.matmul
                    ins(%FC_INPUT, %FC_WEIGHT : tensor<256x512xf32>, tensor<512x1024xf32>)
                    outs(%matmul_init : tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %relu_init = tensor.empty() : tensor<256x1024xf32>
    %OUT = linalg.generic { indexing_maps = [#map, #map],
                            iterator_types = ["parallel", "parallel"]}
               ins(%FC_OUTPUT : tensor<256x1024xf32>)
               outs(%relu_init : tensor<256x1024xf32>) {
               ^bb0(%in: f32, %out: f32):
                    %c0 = arith.constant 0.0 : f32
                    %cmp = arith.cmpf ugt, %in, %c0 : f32
                    %sel = arith.select %cmp, %in, %c0 : f32
                    linalg.yield %sel : f32
               } -> tensor<256x1024xf32>
    func.return %OUT : tensor<256x1024xf32>
}
```

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

# Demo 3: Lowering MLIR

Since MLIR code exists at different levels of abstraction, we need to be able to
lower from a higher level of abstraction to a lower one in order to compile to
a given target. This demo will ignore optimizing at the different levels and only
show how to lower the MLIR code from Demo 2 to LLVMIR for compiling onto a CPU.
Our goal is to take the high-level code we wrote in Demo 2, and lower it to
the level of abstraction closest to assembly language.

The script `lower.sh` lowers the code from Demo 2 step-by-step from the high-level
linalg representation all the way to LLVMIR. You can run this script by doing:
```sh
bash demo3-lowering-mlir/lower.sh
```
We will now walk through what each step in this script is doing.

The MLIR code from Demo 2 is at a level of abstraction called "Linalg on Tensor",
which is shown in the following code:
```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main() -> tensor<256x1024xf32> {
    %0 = tensor.empty() : tensor<256x512xf32>
    %1 = tensor.empty() : tensor<512x1024xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %splat = tensor.splat %cst : tensor<256x1024xf32>
    %2 = linalg.matmul ins(%0, %1 : tensor<256x512xf32>, tensor<512x1024xf32>) outs(%splat : tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %3 = tensor.empty() : tensor<256x1024xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<256x1024xf32>) outs(%3 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_0 = arith.constant 0.000000e+00 : f32
      %5 = arith.cmpf ugt, %in, %cst_0 : f32
      %6 = arith.select %5, %in, %cst_0 : f32
      linalg.yield %6 : f32
    } -> tensor<256x1024xf32>
    return %4 : tensor<256x1024xf32>
  }
}
```
You will notice that at this abstraction level there is no concept of what
device this code is running on. Tensors, by design, do not contain any information
on how the data is stored in memory, and the linalg operations have practically
no information on how the linear algebra operations will be performed. It is
just a high-level description of an algorithm.

The first thing we need to do is lower the Tensors into MemRefs. Tensors are
abstract data types which only represent the data being created / used. We need
this data to exist somewhere in memory in buffers. MLIR provides specialized
passes to convert tensors into buffers for each of the dialects. A list of all
these passes can be found [here](https://mlir.llvm.org/docs/Passes/#bufferization-passes).
In this case, we want to use the `one-shot-bufferize` pass. This performs all
the bufferization over all the dialects at once in "one-shot". If you do not need
fine-grained control over how the buffers are created, this is a good pass to use.
We will be using `mlir-opt` to run this pass. See the `lower.sh` script for how
to use it. After performing bufferization, the code is lowered to a new level
of abstraction called "Linalg on MemRef":
```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main() -> memref<256x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    linalg.map outs(%alloc_1 : memref<256x1024xf32>)
      () {
        linalg.yield %cst : f32
      }
    linalg.matmul ins(%alloc, %alloc_0 : memref<256x512xf32>, memref<512x1024xf32>) outs(%alloc_1 : memref<256x1024xf32>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_1 : memref<256x1024xf32>) outs(%alloc_2 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.cmpf ugt, %in, %cst : f32
      %1 = arith.select %0, %in, %cst : f32
      linalg.yield %1 : f32
    }
    return %alloc_2 : memref<256x1024xf32>
  }
}
```
At this level of abstraction, you will notice that we can no longer just create
empty tensors anymore; now, we have to actually allocate the memory onto the
device. What changed here is that now all data buffers have real pointers
underneath that have been allocated within the kernel. This is the first time
we have seen MemRefs in this tutorial. These are incredibly imporant Types found
in core MLIR. MemRefs represent memory buffers. They are wrappers around pointers.
MemRefs truly are just pointers with shape / type information attached to them.
They "break" SSA by allowing users to write to locations within the MemRef without
having to create a new Value. I should be clear that the MemRef values themselves
are still SSA (for example `%alloc` is still an SSA value), but because we are
working with pointers, it is challenging to perform data-flow analysis on the
data contained within MemRefs. This demonstrates how moving from one level of
abstraction to another leads to necessary losses in information.

Now that the Tensors have been lowered into buffers, and the linalg operations
are working on these buffers, we can now lower these linalg ops to real algorithms.
CPUs do not come with ops to compute the matmul over two buffers (normally), so
we need to convert these ops into actual for-loops that can be executed. This
will lower our abstraction level further from what is often called "graph-level"
linalg operations to actual instructions. This will look similar to what one
would write in C++. We can convert the linalg dialect to loops using the
`convert-linalg-to-loops` pass found
[here](https://mlir.llvm.org/docs/Passes/#-convert-linalg-to-loops).
This will produce the following code:
```mlir
module {
  func.func @main() -> memref<256x1024xf32> {
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
      }
    }
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        scf.for %arg2 = %c0 to %c512 step %c1 {
          %0 = memref.load %alloc[%arg0, %arg2] : memref<256x512xf32>
          %1 = memref.load %alloc_0[%arg2, %arg1] : memref<512x1024xf32>
          %2 = memref.load %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %0 = memref.load %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
        %1 = arith.cmpf ugt, %0, %cst : f32
        %2 = arith.select %1, %0, %cst : f32
        memref.store %2, %alloc_2[%arg0, %arg1] : memref<256x1024xf32>
      }
    }
    return %alloc_2 : memref<256x1024xf32>
  }
}
```
Notice that the linalg operations have been converted to practically the same
code we wrote in Demo 1 in C++! Another thing to notice is that we are now
operating directly on the MemRef buffers, instead of naming high-level operations
we want to perform. Thus, now we are directly loading from and storing to these
buffers. Due to all of this, the length of the kernel also increases. This is the
consequence of lowering the abstraction level. Code gets less abstract and more
detailed.
To represent loops in MLIR, we use the Strutured Control Flow dialect which
creates ops for things like For loops, if statment, while loops, etc.
This code is at the abstraction level that is closest to C++, but in
order to compile this code we need to lower more towards assembly.

Assembly language does not have high-level control flow, like for loops. Due to
this, we have to lower the for loops to something that is compatible with assembly.
For most CPUs, this is done using branch instructions. General branching ops
are provided by the control-flow dialect (`cf` dialect). We can convert the scf
dialect into the cf dialect using the conversion pass `convert-scf-to-cf`. I should
note that all passes available in core MLIR can be found [here](https://mlir.llvm.org/docs/Passes/).
I will not show the resulting MLIR code here since it becomes very verbose and
much harder to read. This is an important point, we are losing more and more
high-level information as we lower further. This makes it harder to perform
optimizations. This is a key idea in MLIR: perform optimizations at the level
of abstraction that is most convenient, never later. After this point, we are
at the level of abstraction that is as close to CPUs as we can get in core MLIR.

Now that we are at the CPU level of abstraction, we want to make use of LLVM to
lower the rest of the way. This is very convenient since LLVM has been built over
several years to convert CPU level code all the way down to assembly. We want to
make use of this lowering! In order to emit LLVMIR, we need to convert all of
our MLIR code to the LLVM dialect. This is done by converting each of the dialects we have
in our kernel into the LLVM dialect. See the `lower.sh` script for which passes I chose to
use for this particular kernel. After this point the code is as close to LLVMIR
as we can get in MLIR.

The final tool that we will use is a translation tool called `mlir-translate`
which will turn the MLIR code in the LLVM dialect into LLVMIR. This is different
from `mlir-opt` since `mlir-opt` always takes in valid MLIR code and spits out
valid MLIR code. The goal of `mlir-translate` is to take MLIR code and translate
it into another target language; in our case, it is LLVMIR. After this step we
have valid LLVMIR code which can be compiled further using LLVM all the way to
executable assembly. I will not show this in this tutorial since it is out of
scope.

# Demo 4: Optimizing MLIR with Affine Analysis

MLIR can do much more than just lower high-level code to CPUs and other targets.
What makes MLIR so powerful is its ability to do high-level analysis and
optimization as it is lowering. One analysis that is built-into MLIR's core
dialects is Affine Analysis.

In a nutshell, Affine Analysis uses analytical models to represent how code
accesses buffers. By using these analytical models, we can deduce the memory
access patterns of buffers which allow us to do exciting optimization like
fusing complicated loops together and tiling loops to improve cache locality.

Affine Analysis is provided as part of the affine dialect in MLIR. This is a
level of abstraction that exists below the linalg dialect, but above the loops
dialect. What we can do is as we are lowering from the Linalg dialect to the scf
dialect, we can make a pit-stop into the affine dialect, perform optimizations,
and the continue into the scf dialect.

The script `optimize.sh` walks through how to enter into the affine dialect from
our Demo 2 code, perform loop fusion and tiling, and then lower into the scf
dialect.

We can convert the linalg dialect into the affine dialect using the
`convert-linalg-to-affine-loops` pass after performing bufferization. This
produces the following code:
```mlir
module {
  func.func @main() -> memref<256x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 1024 {
        affine.store %cst, %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
      }
    }
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 1024 {
        affine.for %arg2 = 0 to 512 {
          %0 = affine.load %alloc[%arg0, %arg2] : memref<256x512xf32>
          %1 = affine.load %alloc_0[%arg2, %arg1] : memref<512x1024xf32>
          %2 = affine.load %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 1024 {
        %0 = affine.load %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
        %1 = arith.cmpf ugt, %0, %cst : f32
        %2 = arith.select %1, %0, %cst : f32
        affine.store %2, %alloc_2[%arg0, %arg1] : memref<256x1024xf32>
      }
    }
    return %alloc_2 : memref<256x1024xf32>
  }
}
```
You will notice that the affine dialect looks very similar to the scf on memref
level of abstraction; however, this dialect comes with gaurentees on how memrefs
are accessed within the kernel, which makes performing Affine Analysis much
easier. Now that we are in the affine dialect, we can make use of the optimizations
provided by the dialect.

The first optimizations I want to perform is loop fusion. You will notice that
the original kernel separates out the zeroing of output buffer, computing the
matmul, and performing the ReLU into different loop nests. We know that the
memory accesses between these loops are independent, but without Affine Analysis
the compiler struggles to realize this. Here, we use the `affine-loop-fusion`
pass which produces the following code:
```mlir
module {
  func.func @main() -> memref<256x1024xf32> {
    %alloc = memref.alloc() : memref<1x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 1024 {
        affine.store %cst, %alloc[0, 0] : memref<1x1xf32>
        affine.for %arg2 = 0 to 512 {
          %3 = affine.load %alloc_0[%arg0, %arg2] : memref<256x512xf32>
          %4 = affine.load %alloc_1[%arg2, %arg1] : memref<512x1024xf32>
          %5 = affine.load %alloc[0, 0] : memref<1x1xf32>
          %6 = arith.mulf %3, %4 : f32
          %7 = arith.addf %5, %6 : f32
          affine.store %7, %alloc[0, 0] : memref<1x1xf32>
        }
        %0 = affine.load %alloc[0, 0] : memref<1x1xf32>
        %1 = arith.cmpf ugt, %0, %cst : f32
        %2 = arith.select %1, %0, %cst : f32
        affine.store %2, %alloc_2[%arg0, %arg1] : memref<256x1024xf32>
      }
    }
    return %alloc_2 : memref<256x1024xf32>
  }
}
```
Notice that the compiler was able to recognize that the three loop nests could
be fused and combined them all into 1! This has a massive affect on the performance
of the kernel. Not only do we not have to iterate over 256x1024 elements three
times, but notice that one of the buffers dissapeared! The ReLU function needed
to allocate a 1MB buffer to store the result of the activation; however, using
affine analysis, the compiler realized that this buffer was not necessary and
removed it. This greatly reduces the memory footprint of the kernel which will
help with cache locality.

Speaking of cache locality, we can further improve our cache hit rate by performing
tiling. As mentioned in Demo 1, tiling is a target-specific optimization which
requires knowledge on the size of the caches and the size of the memory accessed
within the loop nests. Fortunitely, we can use Affine Analysis to deduce the size
of the memory footprint within the loop nests and we can tell the affine dialect
how large our cache is! For my computer, the cache size is around 1024 kB, so I
can perform loop tiling using `affine-loop-tile="cache-size=1024"`. I also
perform some other cleanup passes to make the code more legible below:
```mlir
module {
  func.func @main() -> memref<256x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<1x1xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    affine.for %arg0 = 0 to 128 {
      affine.for %arg1 = 0 to 512 {
        affine.for %arg2 = 0 to 2 {
          affine.for %arg3 = 0 to 2 {
            affine.store %cst, %alloc[0, 0] : memref<1x1xf32>
            affine.for %arg4 = 0 to 512 {
              %3 = affine.load %alloc_0[%arg2 + %arg0 * 2, %arg4] : memref<256x512xf32>
              %4 = affine.load %alloc_1[%arg4, %arg3 + %arg1 * 2] : memref<512x1024xf32>
              %5 = affine.load %alloc[0, 0] : memref<1x1xf32>
              %6 = arith.mulf %3, %4 : f32
              %7 = arith.addf %5, %6 : f32
              affine.store %7, %alloc[0, 0] : memref<1x1xf32>
            }
            %0 = affine.load %alloc[0, 0] : memref<1x1xf32>
            %1 = arith.cmpf ugt, %0, %cst : f32
            %2 = arith.select %1, %0, %cst : f32
            affine.store %2, %alloc_2[%arg2 + %arg0 * 2, %arg3 + %arg1 * 2] : memref<256x1024xf32>
          }
        }
      }
    }
    return %alloc_2 : memref<256x1024xf32>
  }
}
```
Using Affine Analysis, the compiler decided to tile the two outermost loops by
2x2. This partitions the inner calculations into tiles of size 2x2 which the
compiler believes will maximize cache locality.

Now that we have performed the optimizations we care about in the Affine dialect,
we can continue the lowering process to the scf dialect using the `lower-affine`
pass. This produces the output below:
```mlir
module {
  func.func @main() -> memref<256x1024xf32> {
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<1x1xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    scf.for %arg0 = %c0 to %c128 step %c1 {
      scf.for %arg1 = %c0 to %c512 step %c1 {
        scf.for %arg2 = %c0 to %c2 step %c1 {
          scf.for %arg3 = %c0 to %c2 step %c1 {
            memref.store %cst, %alloc[%c0, %c0] : memref<1x1xf32>
            scf.for %arg4 = %c0 to %c512 step %c1 {
              %7 = arith.muli %arg0, %c2 overflow<nsw> : index
              %8 = arith.addi %arg2, %7 : index
              %9 = memref.load %alloc_0[%8, %arg4] : memref<256x512xf32>
              %10 = arith.muli %arg1, %c2 overflow<nsw> : index
              %11 = arith.addi %arg3, %10 : index
              %12 = memref.load %alloc_1[%arg4, %11] : memref<512x1024xf32>
              %13 = memref.load %alloc[%c0, %c0] : memref<1x1xf32>
              %14 = arith.mulf %9, %12 : f32
              %15 = arith.addf %13, %14 : f32
              memref.store %15, %alloc[%c0, %c0] : memref<1x1xf32>
            }
            %0 = memref.load %alloc[%c0, %c0] : memref<1x1xf32>
            %1 = arith.cmpf ugt, %0, %cst : f32
            %2 = arith.select %1, %0, %cst : f32
            %3 = arith.muli %arg0, %c2 overflow<nsw> : index
            %4 = arith.addi %arg2, %3 : index
            %5 = arith.muli %arg1, %c2 overflow<nsw> : index
            %6 = arith.addi %arg3, %5 : index
            memref.store %2, %alloc_2[%4, %6] : memref<256x1024xf32>
          }
        }
      }
    }
    return %alloc_2 : memref<256x1024xf32>
  }
}
```

Let's compare the output above to the kernel in the scf dialect without performing
Affine Analysis:
```mlir
module {
  func.func @main() -> memref<256x1024xf32> {
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
      }
    }
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        scf.for %arg2 = %c0 to %c512 step %c1 {
          %0 = memref.load %alloc[%arg0, %arg2] : memref<256x512xf32>
          %1 = memref.load %alloc_0[%arg2, %arg1] : memref<512x1024xf32>
          %2 = memref.load %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %0 = memref.load %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
        %1 = arith.cmpf ugt, %0, %cst : f32
        %2 = arith.select %1, %0, %cst : f32
        memref.store %2, %alloc_2[%arg0, %arg1] : memref<256x1024xf32>
      }
    }
    return %alloc_2 : memref<256x1024xf32>
  }
}
```
Notice that the kernel is much more concise, makes fewer allocations, and has
better cache locality. All of this we got for free by using analysis provided
by us being at a higher level of abstraction.

This kernel in the scf dialect can then be lowered in the same way as Demo 3.

