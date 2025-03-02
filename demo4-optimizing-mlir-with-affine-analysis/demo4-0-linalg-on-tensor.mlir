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

