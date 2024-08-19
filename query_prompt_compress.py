from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, Tuple
from transformers import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import math
from torch import nn

model_name = "HuggingFaceTB/SmolLM-135M-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation='sdpa').cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

query_window = 64
template_window = 64
max_tokens = 2048

def sdpa_forward( # SDPA Forward with Attention Scores included
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # print("EHLLOOO")

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # print("Value states: ", value_states, kv_seq_len)
    
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # print("Key states shape: ", key_states.shape)
        B, H, T, D = key_states.shape
        eye_matrix = torch.eye(T, dtype=key_states.dtype).to(key_states.device)
        value_eye = eye_matrix.unsqueeze(0).unsqueeze(0).expand(B, H, T, T)

        # print("Value eye shape: ", value_eye.shape)
        # From SnapKV
        attn_scores = torch.matmul(query_states[..., -query_window:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_scores = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

        print("Attention scores shape: ", attn_scores.shape)
    
        # Trying to implement https://arxiv.org/html/2406.12335v1
        # value_norms = torch.linalg.norm(value_states[:-query_window], dim=-1, ord=1)
        # value_norms = value_norms.unsqueeze(2)

        # print("Value norms shape: ", value_norms.shape)
    
        # attn_scores = attn_scores * value_norms

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # print("Attention output shape: ", attn_output.shape)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # if not(output_attentions):
        #     attn_scores = None
    
        return attn_output, attn_scores, past_key_value

for i in range(len(model.model.layers)):
    model.model.layers[i].forward = sdpa_forward.__get__(model.model.layers[i].self_attn, type(model.model.layers[i].self_attn))

sample_text = open("snapkv.txt", "r").read()
encoded_tokens = tokenizer(sample_text, return_tensors="pt")
for key in encoded_tokens:
    encoded_tokens[key] = encoded_tokens[key].cuda()
print(encoded_tokens.input_ids.shape)

from einops import repeat
import time

start_time = time.perf_counter()
top_tokens = []

encoded_tokens = tokenizer.encode(sample_text, return_tensors="pt")[0]  
print("Encoding tokens took: ", time.perf_counter() - start_time)

chunk_count = math.ceil((encoded_tokens.shape[-1] - template_window - query_window) / (max_tokens - query_window - template_window))
print(encoded_tokens[template_window:-query_window])
split_tensors = encoded_tokens[template_window:-query_window].chunk(chunks=chunk_count, dim=-1)
# split_tensors = torch.split(encoded_tokens[template_adjustment:-query_window], max_tokens - query_window - template_adjustment)
max_split_tokens = 0
for split_tensor in split_tensors:
    max_split_tokens = max(max_split_tokens, split_tensor.shape[-1])
padded_tensors = []
attention_masks = []

template_tokens = encoded_tokens[:template_window]
query_tokens = encoded_tokens[-query_window:]

print("Query tokens shape: ", query_tokens.shape)
print("Template tokens shape: ", template_tokens.shape)

for i, chunk_tensor in enumerate(split_tensors):
    joined_tensor = torch.cat((template_tokens, chunk_tensor, query_tokens), dim=-1)
    pad_tensor = torch.tensor(tokenizer.pad_token_id).expand(max_tokens - joined_tensor.shape[-1])
    chunk_attention_mask = torch.IntTensor([0] * (max_tokens - chunk_tensor.shape[-1]) + [1] * (chunk_tensor.shape[-1]))
    padded_tensors.append(torch.cat((pad_tensor, joined_tensor), dim=-1))
    attention_masks.append(chunk_attention_mask)

attention_masks = torch.stack(attention_masks, dim=0)
padded_tensors = torch.stack(padded_tensors, dim=0)
print("Generated padded tensors: ", padded_tensors.shape, " and attention masks: ", attention_masks.shape, " in ", time.perf_counter() - start_time)

padded_tensors = padded_tensors.to(model.device)
attention_masks = attention_masks.to(model.device)

# batch run 
outputs = model(padded_tensors, attention_mask=attention_masks, output_attentions=True)#, return_dict_in_generate=True, output_attentions=True)
for key in outputs:
    print("Output key: ", key)
print("Forward pass completed in: ", time.perf_counter() - start_time)

output_attentions = torch.stack(outputs.attentions)
print("Proce

# get the top tokens attended to by the query tokens
# create one single list of "most important tokens"
# return this string.
end_time = time.perf_counter()
print("Time taken: ", end_time - start_time)