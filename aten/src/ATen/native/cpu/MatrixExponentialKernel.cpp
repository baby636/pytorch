#include <ATen/native/MatrixExponential.h>
#include <ATen/Parallel.h>

namespace at { namespace native {

namespace {

template <typename scalar_t, int ROW, int COL>
using array2d = std::array<std::array<scalar_t, COL>, ROW>;

// we consider 6 Taylor expansions of degree
// 1, 2, 4, 8, 12, 18
constexpr int total_n_degs = 6;

// 0! to 4!
// Introduced to avoid 'magic number's clang-tidy complaints about
constexpr int fact_array_size = 5;
constexpr std::array<float, fact_array_size> fact = {1., 1., 2., 6., 24.};

Tensor matrix_power(const Tensor& matrices, const Tensor& powers) {
  if (matrices.dim() > 2) {
    auto res = at::empty(matrices.sizes(), matrices.options());
    auto num_matrices = matrices.size(0);
    at::parallel_for(0, num_matrices, 0,
      [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; ++i) {
          auto res_ith_matrix = res.select(0, i);
          auto matrices_ith_matrix = matrices.select(0, i);
          auto powers_ith_power = powers.select(0, i)
            .template item<int64_t>();
          res_ith_matrix.copy_(
            at::matrix_power(
              matrices_ith_matrix,
              powers_ith_power
            )
          );
        }
      }
    );
    return res;
  }
  else {
    int64_t n = powers.template item<int64_t>();
    return at::matrix_power(matrices, n);
  }
}

Tensor operator_1_norm(const Tensor& tensor) {
  return std::get<0>(tensor.abs().sum(-2).max(-1));
}

Tensor compute_T1(const Tensor& A) {
  const auto& I = at::eye(A.size(-1), A.options()).expand_as(A);
  return I + A;
}

Tensor compute_T2(const Tensor& A) {
  const auto& A2 = at::matmul(A, A);
  return compute_T1(A) + A2 / fact[2];
}

Tensor compute_T4(const Tensor& A) {
  const auto& I = at::eye(A.size(-1), A.options()).expand_as(A);
  const auto& A2 = at::matmul(A, A);
  return I + A + at::matmul(A2, I / fact[2] + A / fact[3] + A2 / fact[4]);
}

template <typename scalar_t>
Tensor compute_T8(const Tensor& A) {
  constexpr scalar_t sqrt_177 = 0.1330413469565007072504e+2;
  constexpr scalar_t x3 = 2. / 3.;
  constexpr scalar_t x1 = x3 * ((1. + sqrt_177) / 88.);
  constexpr scalar_t x2 = x3 * ((1. + sqrt_177) / 352.);
  constexpr scalar_t x4 = (-271. + 29. * sqrt_177) / (315. * x3);
  constexpr scalar_t x5 = (-11. + 11. * sqrt_177) / (1260. * x3);
  constexpr scalar_t x6 = (-99. + 11. * sqrt_177) / (5040. * x3);
  constexpr scalar_t x7 = (89. - sqrt_177) / (5040. * x3);
  constexpr scalar_t y2 = (857. - 58. * sqrt_177) / 630.;

  const auto& I = at::eye(A.size(-1), A.options()).expand_as(A);
  const auto& A2 = at::matmul(A, A);
  const auto& A4 = at::matmul(A2, x1 * A + x2 * A2);
  const auto& A8 = at::matmul(x3 * A2 + A4,
    x4 * I + x5 * A + x6 * A2 + x7 * A4);
  return I + A + y2 * A2 + A8;
}

template <typename scalar_t>
Tensor compute_T12(const Tensor& A) {
  constexpr int num_prods = 4;
  constexpr array2d<scalar_t, num_prods, num_prods> b = {{
    {
      9.0198e-16,
      0.46932117595418237389,
      -0.20099424927047284052,
      -0.04623946134063071740
    },
    {
      5.31597895759871264183,
      1.19926790417132231573,
      0.01179296240992997031,
      0.01108844528519167989
    },
    {
      0.18188869982170434744,
      0.05502798439925399070,
      0.09351590770535414968,
      0.00610700528898058230
    },
    {
      -2.0861320e-13,
      -0.13181061013830184015,
      -0.02027855540589259079,
      -0.00675951846863086359
    }
  }};

  const auto& I = at::eye(A.size(-1), A.options()).expand_as(A);
  const auto& A2 = at::matmul(A, A);
  const auto& A3 = at::matmul(A2, A);
  std::array<
    std::reference_wrapper<const Tensor>,
    num_prods> As = {I, A, A2, A3};

  std::array<Tensor, num_prods> Bs;
  for (int i = 0; i < num_prods; ++i) {
    Bs[i] = at::zeros(A.sizes(), A.options());
  }

  for (int i = 0; i < num_prods; ++i) {
    for (int j = 0; j < num_prods; ++j) {
      Bs[i] += b[i][j] * As[j];
    }
  }

  const auto& A6 = Bs[2] + at::matmul(Bs[3], Bs[3]);
  const auto& res = Bs[0] + at::matmul(Bs[1] + A6, A6);

  return res;
}

template <typename scalar_t>
Tensor compute_T18(const Tensor& A) {
  constexpr int num_prods = 5;
  constexpr array2d<scalar_t, num_prods, num_prods> b = {{
    {
      0.,
      -1.00365581030144618291e-01,
      -8.02924648241156932449e-03,
      -8.92138498045729985177e-04,
      0.
    },
    {
      0.,
      3.97849749499645077844e-01,
      1.36783778460411720168e+00,
      4.98289622525382669416e-01,
      -6.37898194594723280150e-04
    },
    {
      -1.09676396052962061844e+01,
      1.68015813878906206114e+00,
      5.71779846478865511061e-02,
      -6.98210122488052056106e-03,
      3.34975017086070470649e-05
    },
    {
      -9.04316832390810593223e-02,
      -6.76404519071381882256e-02,
      6.75961301770459654925e-02,
      2.95552570429315521194e-02,
      -1.39180257516060693404e-05
    },
    {
      0.,
      0.,
      -9.23364619367118555360e-02,
      -1.69364939002081722752e-02,
      -1.40086798182036094347e-05
    }
  }};

  const auto& I = at::eye(A.size(-1), A.options()).expand_as(A);
  const auto& A2 = at::matmul(A, A);
  const auto& A3 = at::matmul(A2, A);
  const auto& A6 = at::matmul(A3, A3);
  std::array<
    std::reference_wrapper<const Tensor>,
    num_prods> As = {I, A, A2, A3, A6};

  std::array<Tensor, num_prods> Bs;
  for (int i = 0; i < num_prods; ++i) {
    Bs[i] = at::zeros(A.sizes(), A.options());
  }

  for (int i = 0; i < num_prods; ++i) {
    for (int j = 0; j < num_prods; ++j) {
      Bs[i] += b[i][j] * As[j];
    }
  }

  const auto& A9 = at::matmul(Bs[0], Bs[4]) + Bs[3];
  const auto& res = Bs[1] + at::matmul(Bs[2] + A9, A9);

  return res;
}

template <typename scalar_t>
Tensor mexp_impl(const Tensor& a, std::array<scalar_t, total_n_degs> thetas) {
  auto norm = operator_1_norm(a);

  constexpr std::array<
    Tensor(*)(const Tensor&),
    total_n_degs - 1> 
  compute_Ts = {
    compute_T1, compute_T2, compute_T4,
    compute_T8<scalar_t>, compute_T12<scalar_t>
  };

  for (int i = 0; i < total_n_degs - 1; ++i) {
    if ((norm <= thetas[i]).all().template item<bool>()) {
      return compute_Ts[i](a);
    }
  }

  // Scale
  auto s = at::max(at::zeros_like(norm),
              at::ceil(at::log2(norm / thetas[total_n_degs - 1]))).to(at::kLong);
  auto pow2s = at::pow(2, s);
  auto a_scaled = a / pow2s.unsqueeze(-1).unsqueeze(-1);

  // Square
  return matrix_power(compute_T18<scalar_t>(a_scaled), pow2s);
}

// matrix exponential
Tensor mexp(const Tensor& a) {
  if (a.scalar_type() == at::ScalarType::Float) {
    constexpr std::array<float, total_n_degs> thetas_float = {
      1.192092800768788e-07, // deg 1
      5.978858893805233e-04, // deg 2
      5.116619363445086e-02, // deg 4
      5.800524627688768e-01, // deg 8
      1.461661507209034e+00, // deg 12
      3.010066362817634e+00  // deg 18
    };
    return mexp_impl<float>(a, thetas_float);
  }
  else { // if Double
    constexpr std::array<double, total_n_degs> thetas_double = {
      2.220446049250313e-16, // deg 1
      2.580956802971767e-08, // deg 2
      3.397168839976962e-04, // deg 4
      4.991228871115323e-02, // deg 8
      2.996158913811580e-01, // deg 12
      1.090863719290036e+00  // deg 18
    };
    return mexp_impl<double>(a, thetas_double);
  }
}

void matrix_exp_cpu_kernel(Tensor& res, const Tensor& a) {
  if (a.dim() <= 2) {
    res.copy_(mexp(a));
  }
  else {
    auto num_matrices = a.size(0);
    at::parallel_for(0, num_matrices, 0,
      [&](int64_t start, int64_t end) {
        for (auto i = start; i < end; ++i) {
          auto res_ith_matrix = res.select(0, i);
          auto a_ith_matrix = a.select(0, i);
          res_ith_matrix.copy_(mexp(a_ith_matrix));
        }
      }
    );
  }
}

} // anonymous namespace

REGISTER_DISPATCH(matrix_exp_stub, &matrix_exp_cpu_kernel);

}} // namespace at::native
