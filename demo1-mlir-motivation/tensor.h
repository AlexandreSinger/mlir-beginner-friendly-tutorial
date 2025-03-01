/**
 * @file
 * @author  Alex Singer
 * @date    March 2025
 * @brief   A fake tensor class made for demo1 of the beginner-friendly tutorial
 *          to MLIR.
 *
 * This is a basic multi-dimensional tensor class used to abstract the data
 * allocations required to perform linalg operations. This is often controlled
 * by the linalg library and is very careful with how memory is managed to
 * achieve the best performance.
 */

#pragma once

#include <vector>

/**
 * @brief Basic 2D static-shaped tensor class.
 *
 * This can trivially be extended to dynamic-shaped, but for this demo I wanted
 * to keep things simple.
 *
 * NOTE: This type of class is necessary to define for a couple of reasons.
 *       First, C++ (and many programming languages) do not provide a continuous
 *       2D array of values. It is trivial to make one by linearizing the
 *       offset indices; however this is usually abstracted from the user.
 *       Second, high-performance libraries may use special classes to represent
 *       tensors to store more information on the data being stored in order to
 *       make their kernels even higher performance.
 */
template<typename T, size_t W, size_t H>
struct Tensor {
    Tensor() : data(W * H) {}

    T& get(size_t x, size_t y) {
        return data[(x * H) + y];
    }

    std::vector<T> data;
};

