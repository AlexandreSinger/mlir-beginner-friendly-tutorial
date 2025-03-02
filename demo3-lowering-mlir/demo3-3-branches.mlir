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
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
    %1 = arith.cmpi slt, %0, %c256 : index
    cf.cond_br %1, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
    %3 = arith.cmpi slt, %2, %c1024 : index
    cf.cond_br %3, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    memref.store %cst, %alloc_1[%0, %2] : memref<256x1024xf32>
    %4 = arith.addi %2, %c1 : index
    cf.br ^bb3(%4 : index)
  ^bb5:  // pred: ^bb3
    %5 = arith.addi %0, %c1 : index
    cf.br ^bb1(%5 : index)
  ^bb6:  // pred: ^bb1
    cf.br ^bb7(%c0 : index)
  ^bb7(%6: index):  // 2 preds: ^bb6, ^bb14
    %7 = arith.cmpi slt, %6, %c256 : index
    cf.cond_br %7, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    cf.br ^bb9(%c0 : index)
  ^bb9(%8: index):  // 2 preds: ^bb8, ^bb13
    %9 = arith.cmpi slt, %8, %c1024 : index
    cf.cond_br %9, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    cf.br ^bb11(%c0 : index)
  ^bb11(%10: index):  // 2 preds: ^bb10, ^bb12
    %11 = arith.cmpi slt, %10, %c512 : index
    cf.cond_br %11, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %12 = memref.load %alloc[%6, %10] : memref<256x512xf32>
    %13 = memref.load %alloc_0[%10, %8] : memref<512x1024xf32>
    %14 = memref.load %alloc_1[%6, %8] : memref<256x1024xf32>
    %15 = arith.mulf %12, %13 : f32
    %16 = arith.addf %14, %15 : f32
    memref.store %16, %alloc_1[%6, %8] : memref<256x1024xf32>
    %17 = arith.addi %10, %c1 : index
    cf.br ^bb11(%17 : index)
  ^bb13:  // pred: ^bb11
    %18 = arith.addi %8, %c1 : index
    cf.br ^bb9(%18 : index)
  ^bb14:  // pred: ^bb9
    %19 = arith.addi %6, %c1 : index
    cf.br ^bb7(%19 : index)
  ^bb15:  // pred: ^bb7
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    cf.br ^bb16(%c0 : index)
  ^bb16(%20: index):  // 2 preds: ^bb15, ^bb20
    %21 = arith.cmpi slt, %20, %c256 : index
    cf.cond_br %21, ^bb17, ^bb21
  ^bb17:  // pred: ^bb16
    cf.br ^bb18(%c0 : index)
  ^bb18(%22: index):  // 2 preds: ^bb17, ^bb19
    %23 = arith.cmpi slt, %22, %c1024 : index
    cf.cond_br %23, ^bb19, ^bb20
  ^bb19:  // pred: ^bb18
    %24 = memref.load %alloc_1[%20, %22] : memref<256x1024xf32>
    %25 = arith.cmpf ugt, %24, %cst : f32
    %26 = arith.select %25, %24, %cst : f32
    memref.store %26, %alloc_2[%20, %22] : memref<256x1024xf32>
    %27 = arith.addi %22, %c1 : index
    cf.br ^bb18(%27 : index)
  ^bb20:  // pred: ^bb18
    %28 = arith.addi %20, %c1 : index
    cf.br ^bb16(%28 : index)
  ^bb21:  // pred: ^bb16
    return %alloc_2 : memref<256x1024xf32>
  }
}

