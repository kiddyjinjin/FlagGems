#include <gtest/gtest.h>
#include "flag_gems/operators.h"
#include "torch/torch.h"

TEST(special_op_test, rotary_embedding) {
  torch::manual_seed(0);
  const torch::Device device(torch::kCUDA, 0);
  auto dtype = GetParam();

  auto max_seq_len = 1024;
  auto head_dim = 64;
  auto rotary_interleaved = true;

  auto seq_len = torch::randint(1, max_seq_len, {1}, torch::TensorOptions().device(device)).item();
  torch::Tensor q =
      torch::randn({1, seq_len, 8, head_dim}, torch::TensorOptions().device(device).dtype(dtype));
  torch::Tensor k =
      torch::randn({1, seq_len, 1, head_dim}, torch::TensorOptions().device(device).dtype(dtype));

  torch::Tensor position_ids =
      torch::randint(0, max_seq_len, {1, seq_len}, torch::TensorOptions().device(device));

  torch::Tensor inv_frep =
      1.0 / (10000 *
             *(torch::arange(0, head_dim, 2, torch::TensorOptions().device(device).dtype(dtype)) / head_dim));
  auto t = torch::arange(max_seq_len, torch::TensorOptions().device(device).dtype(dtype));
  auto freqs = torch::outer(t, inv_frep);
  torch::Tensor cos = torch::cos(freqs).to(dtype);
  torch::Tensor sin = torch::sin(freqs).to(dtype);

  auto compute_ref =
      [&](torch::Tensor q,
          torch::Tensor k,
          torch::Tensor cos,
          torch::Tensor sin,
          torch::Tensor position_ids,
          bool rotary_interleaved) {
        if (position_ids.defined()) {
          cos = cos[position_ids].unsqueeze(-2);
          sin = sin[position_ids].unsqueeze(-2);
        } else {
          cos = cos[:q.size(-3), :].view({1, -1, 1, head_dim});
          sin = sin[:q.size(-3), :].view({1, -1, 1, head_dim});
        }
        cos = torch::repeat_interleave(cos, 2, -1);
        sin = torch::repeat_interleave(sin, 2, -1);

        torch::Tensor rotate_fn(torch::Tensor x) {
          auto x1 = x[..., ::2];
          auto x2 = x[..., 1 ::2];
          return torch::stack({-x2, x1}, -1).flatten(-2);
        }

        auto q_embed = rotate_fn(q) * sin + (q * cos);
        auto k_embed = rotate_fn(k) * sin + (k * cos);
        return std::make_tuple(q_embed, k_embed);
      }

  std::tuple<torch::Tensor, torch::Tensor>
      out_torch = compute_ref(q, k, cos, sin, position_ids, rotary_interleaved);
  std::tuple<torch::Tensor, torch::Tensor> out_triton =
      flag_gems::rotary_embedding(q, k, cos, sin, position_ids, rotary_interleaved);

  EXPECT_TRUE(torch::allclose(out_torch[0], out_triton[0], 1e-2, 1e-3));
  EXPECT_TRUE(torch::allclose(out_torch[1], out_triton[1], 1e-2, 1e-3));
}
