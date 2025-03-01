/// @file
/// @author Alex Singer
/// @date   March 2025
/// @brief  The high-level code from demo 1 converted into MLIR.
///
/// Usually tools are used to automatically convert the user's code into MLIR
/// code. For this tutorial I wrote this kernel by hand, but it should be
/// trivial to build a tool to do this conversion for this particular example.

// This is used by the linalg.generic op later. See that section for more info.
#map = affine_map<(i, j) -> (i, j)>

// This is the top-level container operation. It is part of the "builtin" dialect.
//
// This is the first occurence in this tutorial of something called an
// "Operation" in MLIR. Operations have application-specific semantics, meaning
// that their functionality is defined by the context in which they are used in.
// Applications have names, may return results, may take operands, may have
// properties, may have attributes, and may have regions which can contain other
// Operations. In this case, the ModuleOp contains one region. 
//
// A dialect in MLIR is a collection of Operations at some level of abstraction.
// For the builtin dialect, these are used for MLIR-specific organization. The
// built-in dialect often has no meaning to the code itself, and is used within
// MLIR to help with the language specification of MLIR.
//
// This operation contains all of the code which will be compiled through MLIR.
// It is used within the compiler to apply overall attributes to all operations
// within. For example, this is where the target device triple is stored.
module {

// This is the function operation as part of the "func" dialect. This dialect
// contains operations that have to do with defining and calling functions.
// FuncOps in MLIR have a name ("main" in this case), define arguments to the
// function, and define the output type. The function op contains one region
// containing ops which it will "execute" in order. The symantic of "executing"
// these ops in order comes from the abstraction of the FuncOp itself, not from
// MLIR's specifications.
// Here, we have a function named "main", it takes no arguments, and returns a
// tensor (which will be described later why).
func.func @main() -> tensor<256x1024xf32> {
    // This is the first occurence of a "Value" in this tutorial. Values are
    // what may get returned from operations. In MLIR, these are named using the
    // "%" symbol. Values in MLIR are Static-Single-Assignment (SSA). This means
    // that their value does not mutate during execution and they are only
    // assigned to once. This is a useful property since it makes the Data Flow
    // Graph (DFG) of functions directed and acyclic which enables many compiler
    // optimizations.
    //
    // Values in MLIR always have a "type". In this case, this value is a Tensor
    // type. The Tensor type specifies a multi-dimensional array; however, there
    // is NO concept of memory (i.e. how the data is stored in the device). This
    // is by design. Tensors only represent a "chunk" of data, thats it. This is
    // A useful abstraction since it allows us to deal with compute at a higher
    // level of abstraction (not caring about how the buffers are allocated).
    //
    // In demo1, the FC_INPUT and FC_WEIGHT matrices were just initialized with
    // random values (not specifically 0 initialized). The "tensor" dialect
    // provides ops to work with tensors. In this case, we use the TensorEmpty
    // op to create an empty tensor of the given shape and type.
    %FC_INPUT = tensor.empty() : tensor<256x512xf32>
    %FC_WEIGHT = tensor.empty() : tensor<512x1024xf32>

    // Here we perform our first high-level linear-algebra operation. At a high
    // level, all we care about is that FC_INPUT and FC_WEIGHT are multiplied
    // together. We do not care about the algorithm used to perform this matrix
    // multiplication. Luckily, as part of the Linalg dialect, a matmul op
    // exists which does exactly this! This MatMulOp takes 2 matrices as inputs
    // and produces one matrix as output. What you may notice is that I have to
    // create another tensor to act as my output. This is a quirk of the linalg
    // dialect. Most ops in the linalg dialect need to know what is in the output
    // tensor before the operation occured. Some operations, for example, may
    // not set every value in the tensor and the user may want to zero initialize
    // the output tensor. In this case, we must set the initial value of the
    // output tensor to all 0s since matrix multiplies use multiply-accumulate
    // instructions which accumulate into the output buffer.
    %c_init = arith.constant 0.0 : f32
    %matmul_init = tensor.splat %c_init : tensor<256x1024xf32>
    %FC_OUTPUT = linalg.matmul
                    ins(%FC_INPUT, %FC_WEIGHT : tensor<256x512xf32>, tensor<512x1024xf32>)
                    outs(%matmul_init : tensor<256x1024xf32>) -> tensor<256x1024xf32>

    // Our second high-level linear algebra operation that we wish to perform is
    // an elementwise ReLU operation. Currently, the linalg dialect does not
    // contain the ReLU activation function. MLIR generally contains ops for all
    // basic operations people may need, but ReLU may just not be common enough.
    // Luckily, in the linalg dialect there is a way to specify a "generic"
    // linear algebra operation. Just like matmul before, we need to specify
    // inputs and outputs; but now we also need to specify the function of this
    // operation. We start by specifying how the matrices will be indexed and
    // iterated over. The indexing maps I provided basically just say the we
    // index the matrices without transposing. The iteration type I provided
    // basically just say that we can iterate in any order (there is no data
    // dependencies between iterations). Next, we specify the function of this
    // operation. In this case, I used the "arithmetic" dialect to describe a
    // compare and select that will set the input value to zero if it is negative.
    // Since all values for this relu are being written into, and the %out is
    // not being used, we can allocate the init tensor without setting it to some
    // number.
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

    // Return the final tensor result. This differs from the original main
    // function since MLIR is often smart enough to realize that this tensor is
    // never used and will optimize everything in this kernel away. To keep that
    // from happening for this tutorial, I just returned the result.
    func.return %OUT : tensor<256x1024xf32>
}

}

