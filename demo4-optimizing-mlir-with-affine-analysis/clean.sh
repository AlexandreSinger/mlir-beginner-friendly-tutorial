#!/bin/bash

set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Remove the MLIR files generated by the lowering script.
rm -f $SCRIPT_DIR/*.mlir

# Remove the LLVMIR files generated by the lowering script.
rm -f $SCRIPT_DIR/*.ll

