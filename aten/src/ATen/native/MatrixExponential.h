#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using matrix_exp_fn = void(*)(Tensor&, const Tensor&);

DECLARE_DISPATCH(matrix_exp_fn, matrix_exp_stub);

}} // namespace at::native
