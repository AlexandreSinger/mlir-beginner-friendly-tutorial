/**
 * @file
 * @author  Alex Singer
 * @date    March 2025
 * @brief   Demo 1 for the beginner friendly tutorial of MLIR.
 *
 * This demo will motivate the use of MLIR. This file writes, at a high-level,
 * what the programmer would like to do. That is, performing high-level matrix
 * operations using high-performance libraries provided externally.
 *
 * We assume that the tensor and linalg_lib libraries are out of our control
 * and were optimized for this device.
 */

#include "tensor.h"
#include "linalg_lib.h"

int main(void) {
    /**
     * This is some high-level code which acts basically as a FC activation
     * layer with batch size 256, input size 512, and output size 1024. This
     * layer uses the ReLu activation function.
     *
     * The user writes this code at a high level and relies on optimized code
     * written specifically for the chip they plan to run this on. The user can
     * expect to achieve better performance using these libraries than if they
     * tried to optimize it themself. This also makes the code more portable.
     *
     * NOTE: Often C++ is not the language of choice for this style of coding
     *       since it is so low-level (with many programmers using Python);
     *       however, this example works without loss of generality.
     */

    // Initialize the FC input and weight matrices. Here we assume they were
    // intiialized elsewhere for this demo.
    Tensor<float, 256, 512> FC_INPUT;
    Tensor<float, 512, 1024> FC_WEIGHT;

    // Perform the matrix-multiply and relu function from the FC layer.
    Tensor<float, 256, 1024> FC_OUTPUT = matmul(FC_INPUT, FC_WEIGHT);
    Tensor<float, 256, 1024> OUT = relu(FC_OUTPUT);
}

