#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_functions.h"

namespace flag_gems {
using namespace triton_jit;

std::tuple<at::Tensor, at::Tensor> rotary_embedding(const at::Tensor& q,
                                                    const at::Tensor& k,
                                                    const at::Tensor& cos,
                                                    const at::Tensor& sin,
                                                    std::optional<at::Tensor> position_ids,
                                                    std::optional<bool> rotary_interleaved, ) {
  TORCH_CHECK(k.sizes(-1) == q.sizes(-1),
              "q and k must have the same size in the last dimension, got ",
              q.sizes(-1),
              " and ",
              k.sizes(-1));
  TORCH_CHECK(cos.sizes(-1) == sin.sizes(-1),
              "cos and sin must have the same size in the last dimension, got ",
              cos.sizes(-1),
              " and ",
              sin.sizes(-1));
  TORCH_CHECK(cos.sizes(-1) * 2 == q.sizes(-1),
              "cos and sin must have half the size of q in the last dimension, got ",
              cos.sizes(-1),
              " and ",
              q.sizes(-1));
  TORCH_CHECK(cos.strides(-1) == 1, "cos must be contiguous in the last dimension");
  TORCH_CHECK(sin.strides(-1) == 1, "sin must be contiguous in the last dimension");

  auto q_shape = q.sizes();
  auto k_shape = k.sizes();

  TORCH_CHECK(q_shape[:-2] == k_shape[:-2], "q and k must have the same length, got ", q_shape[:-2],
                                                                                                   " and ",
                                                                                                   k_shape
              [:-2]);

  auto seq_len = 0;
  auto position_ids_stride = 0;

  if (position_ids.has_value()) {
    TORCH_CHECK(
        position_ids.value().sizes() == q_shape[:-2],
                                                    "position_ids must have the same length as q, got ",
                                                    position_ids.value().sizes(),
                                                    " and ",
                                                    q_shape
        [:-2]);
    position_ids = position_ids.view(-1);
    position_ids_stride = position_ids.value().strides(0);
  } else {
    TORCH_CHECK(q_shape.length() == 4,
                "q must have 4 dimensions if position_ids is not provided, got ",
                q_shape);
    seq_len = q_shape[-3];
  }

  q = at::view(q, [ -1, q_shape[-2], q_shape[-1] ]);
  k = at::view(k, [ -1, k_shape[-2], k_shape[-1] ]);
  auto q_stride = q.strides();
  auto k_stride = k.strides();

  auto q_embed = at::empty_like(q);
  auto k_embed = at::empty_like(k);
  auto q_embed_stride = q_embed.strides();
  auto k_embed_stride = k_embed.strides();

  auto n_tokens = q.shape(0);
  auto n_heads = q.shape(1);
  auto head_dim = q.shape(2);

  auto padded_head_dim = std::max(utils::next_power_of_2(head_dim), 16);

  const TritonJITFunction& f =
      TritonJITFunction::getInstance(std::string(utils::get_triton_src_path() / "rotary_embedding.py"),
                                     "apply_rotary_pos_emb_kernel");

  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  f(raw_stream,
    n_tokens,
    1,
    1,
    num_warps,
    num_stages,
    q_embed,
    k_embed,
    q,
    k,
    cos,
    sin,
    position_ids,
    q_stride[0],
    q_stride[1],
    q_stride[2],
    k_stride[0],
    k_stride[1],
    k_stride[2],
    q_embed_stride[0],
    q_embed_stride[1],
    q_embed_stride[2],
    k_embed_stride[0],
    k_embed_stride[1],
    k_embed_stride[2],
    position_ids_stride,
    cos.strides(0),
    sin.strides(0),
    seq_len,
    q_shape[-2],
    k_shape[-2],
    head_dim,
    padded_head_dim,
    rotary_interleaved,
    MAX_POSITION_EMBEDDINGS = cos.sizes(0), ) q_embed = at::view(q_embed, q_shape);
  k_embed = at::view(k_embed, k_shape);
  return {q_embed, k_embed};
}
}  // namespace flag_gems