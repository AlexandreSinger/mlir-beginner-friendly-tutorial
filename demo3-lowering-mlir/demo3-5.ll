; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @main() {
  %1 = call ptr @malloc(i64 524352)
  %2 = ptrtoint ptr %1 to i64
  %3 = add i64 %2, 63
  %4 = urem i64 %3, 64
  %5 = sub i64 %3, %4
  %6 = inttoptr i64 %5 to ptr
  %7 = call ptr @malloc(i64 2097216)
  %8 = ptrtoint ptr %7 to i64
  %9 = add i64 %8, 63
  %10 = urem i64 %9, 64
  %11 = sub i64 %9, %10
  %12 = inttoptr i64 %11 to ptr
  %13 = call ptr @malloc(i64 1048640)
  %14 = ptrtoint ptr %13 to i64
  %15 = add i64 %14, 63
  %16 = urem i64 %15, 64
  %17 = sub i64 %15, %16
  %18 = inttoptr i64 %17 to ptr
  br label %19

19:                                               ; preds = %31, %0
  %20 = phi i64 [ %32, %31 ], [ 0, %0 ]
  %21 = icmp slt i64 %20, 256
  br i1 %21, label %22, label %33

22:                                               ; preds = %19
  br label %23

23:                                               ; preds = %26, %22
  %24 = phi i64 [ %30, %26 ], [ 0, %22 ]
  %25 = icmp slt i64 %24, 1024
  br i1 %25, label %26, label %31

26:                                               ; preds = %23
  %27 = mul i64 %20, 1024
  %28 = add i64 %27, %24
  %29 = getelementptr float, ptr %18, i64 %28
  store float 0.000000e+00, ptr %29, align 4
  %30 = add i64 %24, 1
  br label %23

31:                                               ; preds = %23
  %32 = add i64 %20, 1
  br label %19

33:                                               ; preds = %19
  br label %34

34:                                               ; preds = %66, %33
  %35 = phi i64 [ %67, %66 ], [ 0, %33 ]
  %36 = icmp slt i64 %35, 256
  br i1 %36, label %37, label %68

37:                                               ; preds = %34
  br label %38

38:                                               ; preds = %64, %37
  %39 = phi i64 [ %65, %64 ], [ 0, %37 ]
  %40 = icmp slt i64 %39, 1024
  br i1 %40, label %41, label %66

41:                                               ; preds = %38
  br label %42

42:                                               ; preds = %45, %41
  %43 = phi i64 [ %63, %45 ], [ 0, %41 ]
  %44 = icmp slt i64 %43, 512
  br i1 %44, label %45, label %64

45:                                               ; preds = %42
  %46 = mul i64 %35, 512
  %47 = add i64 %46, %43
  %48 = getelementptr float, ptr %6, i64 %47
  %49 = load float, ptr %48, align 4
  %50 = mul i64 %43, 1024
  %51 = add i64 %50, %39
  %52 = getelementptr float, ptr %12, i64 %51
  %53 = load float, ptr %52, align 4
  %54 = mul i64 %35, 1024
  %55 = add i64 %54, %39
  %56 = getelementptr float, ptr %18, i64 %55
  %57 = load float, ptr %56, align 4
  %58 = fmul float %49, %53
  %59 = fadd float %57, %58
  %60 = mul i64 %35, 1024
  %61 = add i64 %60, %39
  %62 = getelementptr float, ptr %18, i64 %61
  store float %59, ptr %62, align 4
  %63 = add i64 %43, 1
  br label %42

64:                                               ; preds = %42
  %65 = add i64 %39, 1
  br label %38

66:                                               ; preds = %38
  %67 = add i64 %35, 1
  br label %34

68:                                               ; preds = %34
  %69 = call ptr @malloc(i64 1048640)
  %70 = ptrtoint ptr %69 to i64
  %71 = add i64 %70, 63
  %72 = urem i64 %71, 64
  %73 = sub i64 %71, %72
  %74 = inttoptr i64 %73 to ptr
  %75 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %69, 0
  %76 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %75, ptr %74, 1
  %77 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %76, i64 0, 2
  %78 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %77, i64 256, 3, 0
  %79 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %78, i64 1024, 3, 1
  %80 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %79, i64 1024, 4, 0
  %81 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %80, i64 1, 4, 1
  br label %82

82:                                               ; preds = %100, %68
  %83 = phi i64 [ %101, %100 ], [ 0, %68 ]
  %84 = icmp slt i64 %83, 256
  br i1 %84, label %85, label %102

85:                                               ; preds = %82
  br label %86

86:                                               ; preds = %89, %85
  %87 = phi i64 [ %99, %89 ], [ 0, %85 ]
  %88 = icmp slt i64 %87, 1024
  br i1 %88, label %89, label %100

89:                                               ; preds = %86
  %90 = mul i64 %83, 1024
  %91 = add i64 %90, %87
  %92 = getelementptr float, ptr %18, i64 %91
  %93 = load float, ptr %92, align 4
  %94 = fcmp ugt float %93, 0.000000e+00
  %95 = select i1 %94, float %93, float 0.000000e+00
  %96 = mul i64 %83, 1024
  %97 = add i64 %96, %87
  %98 = getelementptr float, ptr %74, i64 %97
  store float %95, ptr %98, align 4
  %99 = add i64 %87, 1
  br label %86

100:                                              ; preds = %86
  %101 = add i64 %83, 1
  br label %82

102:                                              ; preds = %82
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %81
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
