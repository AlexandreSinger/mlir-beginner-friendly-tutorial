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

