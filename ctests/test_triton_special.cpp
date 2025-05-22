#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(special_op_test, rotary_embedding) {
  torch::manual_seed(0);
  const torch::Device device(torch::kCUDA, 0);

  auto rotary_interleaved = true;
  torch::Tensor q =
      torch::randn({1, 1024, 8, 64}, device);
  torch::Tensor k =
      torch::randn({1, 1024, 1, 64}, device);

  torch::Tensor position_ids =
      torch::randint(0, 1024, {1, 1024}, device);

  torch::Tensor inv_frep =
      1.0 / torch::pow(10000, (torch::arange(0, 64, 2, device) / 64));
  auto t = torch::arange(1024, device);
  auto freqs = torch::outer(t, inv_frep);
  torch::Tensor cos = torch::cos(freqs);
  torch::Tensor sin = torch::sin(freqs);

  auto compute_ref =
      [&](torch::Tensor q,
          torch::Tensor k,
          torch::Tensor cos,
          torch::Tensor sin,
          torch::Tensor position_ids,
          bool rotary_interleaved) {
        cos = cos[position_ids].unsqueeze(-2);
        sin = sin[position_ids].unsqueeze(-2);
        cos = torch::repeat_interleave(cos, 2, -1);
        sin = torch::repeat_interleave(sin, 2, -1);

        auto rotate_fn = [&](torch::Tensor x) {
          auto x1 = at::slice(x, dim=-1, step=2);
          auto x2 = at::slice(x, dim=-1, start=1, step=2);
          // auto x1 = x[..., ::2];
          // auto x2 = x[..., 1 ::2];
          return torch::stack({-x2, x1}, -1).flatten(-2);
        };

        auto q_embed = rotate_fn(q) * sin + (q * cos);
        auto k_embed = rotate_fn(k) * sin + (k * cos);
        return std::make_tuple(q_embed, k_embed);
      };

  std::tuple<torch::Tensor, torch::Tensor> out_torch = compute_ref(q, k, cos, sin, position_ids, rotary_interleaved);
  std::tuple<torch::Tensor, torch::Tensor> out_triton = flag_gems::rotary_embedding(q, k, cos, sin, position_ids, rotary_interleaved);

  EXPECT_TRUE(torch::allclose(std::get<0>(out_torch), std::get<0>(out_triton), 1e-2, 1e-3));
  EXPECT_TRUE(torch::allclose(std::get<1>(out_torch), std::get<1>(out_triton), 1e-2, 1e-3));
}
