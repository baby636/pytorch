#include <ATen/native/MatrixExponential.h>

namespace at { namespace native {

DEFINE_DISPATCH(matrix_exp_stub);

// Computes the matrix exponential for a given batch of squared matrices.
// The implementaion is based on:
//
// Bader, P.; Blanes, S.; Casas, F.
// Computing the Matrix Exponential with an Optimized Taylor Polynomial Approximation.
// Mathematics 2019, 7, 1174.
//
Tensor matrix_exp(const Tensor& a) {
  TORCH_CHECK(a.dim() >= 2 
          && (at::isFloatingType(a.scalar_type()) 
           || at::isComplexType(a.scalar_type())),
              "matrix_exp(", a.scalar_type(), "{", a.sizes(), "}): expected a tensor "
              "of floating types with dim at least 2");
  TORCH_CHECK(a.size(-1) == a.size(-2),
              "matrix_exp(", a.scalar_type(), "{", a.sizes(), "}): expected a tensor "
              "of squared matrices");

  // Case of 1x1 matrices
  if (a.size(-1) == 1) {
    return a.exp();
  }

  // Create input for kernels by squashing batch dimensions,
  // i.e. make an n-dim tensor a 3-dim tensor
  auto mexp = at::empty(a.sizes(), a.options());
  auto mexp_squashed_batch_dims = mexp.view(
    {-1, mexp.size(-2), mexp.size(-1)});
  auto a_squashed_batch_dims = a.view(
    {-1, a.size(-2), a.size(-1)});

  matrix_exp_stub(
    a.device().type(),
    mexp_squashed_batch_dims,
    a_squashed_batch_dims
  );

  return mexp;
}

namespace {

// Based on:
//
// Mathias, Roy. 
// “A Chain Rule for Matrix Functions and Applications.”
// SIAM J. Matrix Anal. Appl. 17 (1996): 610-620.
//
template <typename func_t>
Tensor backward_analytic_function_of_a_matrix(
    const Tensor& self, const Tensor& grad,
    const func_t& function_of_a_matrix
  ) {
  auto self_transposed = self.transpose(-2, -1);
  auto self_transposed_sizes = self_transposed.sizes().vec();
  self_transposed_sizes[self.dim() - 2] <<= 1;
  self_transposed_sizes[self.dim() - 1] <<= 1;

  auto n = self_transposed.size(-1);
  auto meta_grad = at::zeros(self_transposed_sizes, grad.options());
  meta_grad.narrow(-2, 0, n).narrow(-1, 0, n).copy_(self_transposed);
  meta_grad.narrow(-2, n, n).narrow(-1, n, n).copy_(self_transposed);
  meta_grad.narrow(-2, 0, n).narrow(-1, n, n).copy_(grad);

  auto grad_input = function_of_a_matrix(meta_grad)
    .narrow(-2, 0, n).narrow(-1, n, n);
  return grad_input;
}

} // anonymous namespace

Tensor matrix_exp_backward(const Tensor& self, const Tensor& grad) {
  return backward_analytic_function_of_a_matrix(
    self, grad,
    [](const Tensor& a) {
      return a.matrix_exp();
    }
  );
}

}} // namespace at::native
