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

