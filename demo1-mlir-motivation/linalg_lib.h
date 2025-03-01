/**
 * @file
 * @author  Alex Singer
 * @date    March 2025
 * @brief   A fake linear algebra library for demo 1 of the beginner-friendly
 *          MLIR tutorial.
 *
 * This is a basic linear algebra library that one may write. This is not
 * optimized in any way, but it is just to demonstrate that these kernels can
 * be optimized by very skilled experts to allow users to achieve excellent
 * performance without expertise.
 */

#pragma once

#include "tensor.h"

/**
 * @brief Perform a matrix-multiplication on two, fixed-shape tensors.
 */
template<typename T, size_t M, size_t K, size_t N>
Tensor<T, M, N> matmul(Tensor<T, M, K>& A, Tensor<T, K, N>& B) {
    /**
     * This is the most basic implementation of a matmul of two fixed-shape
     * tensors.
     *
     * This can be made more efficient by doing the following high-level
     * optimization techniques:
     *  - Tiling: If you know the cache heirarchy of the device this will run
     *            on, you can tile the matrices (localize which regions of the
     *            matrices are being read/written from/to). This maximizes the
     *            reuse of elements in the cache which can improve the hit rate.
     *  - Packing: By allocating "scratch-pad" memory, one can further improve
     *             the performance by storing tiled memory into continuous
     *             arrays.
     *  - Vectorization: One can convert these matrix multiplications into
     *                   vector multiplies and adds.
     *  - Arch-Specific Instructions: If the target device has special
     *                                instructions, specific for matrix multiply
     *                                (like MAC or even a small MatMul), one can
     *                                use these instructions.
     *
     * However, notice that all of these optimizations require knowledge about
     * the underlying device that this kernel will run on. This is why this
     * library must be rewritten for different targets.
     *
     * There are other libraries, like Halide or TVM, which use schedules to try
     * and improve this process; however, this still requires an expert to
     * choose appropriate schedules and tune them.
     */
    Tensor<T, M, N> C;
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            C.get(i, j) = 0.f;
            for (size_t k = 0; k < K; k++) {
                C.get(i, j) += A.get(i, k) * B.get(k, j);
            }
        }
    }
    return C;
}

/**
 * @brief Perform an element-wise ReLu on all elements of the given fixed-shape
 *        tensor.
 */
template<typename T, size_t W, size_t H>
Tensor<T, W, H> relu(Tensor<T, W, H>& IN) {
    /**
     * Similar to the matmul kernel above, this is the most basic implementation
     * of this kernel.
     *
     * The performance of this kernel can be improved using similar techniques
     * to the matmul kernel above.
     */
    Tensor<T, W, H> OUT;
    for (size_t i = 0; i < W; i++) {
        for (size_t j = 0; j < H; j++) {
            OUT.get(i, j) = std::max(IN.get(i, j), 0.f);
        }
    }
    return OUT;
}
