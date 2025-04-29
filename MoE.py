# basic imports we need
import torch
import torch.nn as nn
import torch.nn.functional as F

# this stuff is for the layerskip recipe which helps with training
from torchtune.modules.early_exit_loss import early_exit_loss, linear_l_loss_scale,RotationalEarlyExitCurriculum
from torchtune.modules.layer_dropout import LayerDropout, prepare_layer_dropout
from torchtune.modules.common_utils import slice_str_to_array


# each expert is just a simple MLP
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.mlp(x)


# this is the main MoE layer that routes tokens to different experts
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, k=2, capacity_factor=1.0, layer_idx=0):
        super().__init__()
        # how many expert networks we have
        self.num_experts = num_experts
        # how many experts each token goes to
        self.k = k
        # helps control overflow
        self.capacity_factor = capacity_factor
        self.layer_idx = layer_idx
        # router network decides which experts to use
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # create all the experts
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)
        num_tokens = batch_size * seq_len

        # figure out which experts to use
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, indices = torch.topk(routing_weights, self.k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # calculate load balancing loss so all experts get used evenly
        assignment_counts = torch.zeros(self.num_experts, device=x.device)
        for expert_idx in range(self.num_experts):
            assignment_counts[expert_idx] = (indices == expert_idx).sum()

        ideal_count_per_expert = num_tokens * self.k / self.num_experts
        load_balancing_loss = torch.sum((assignment_counts - ideal_count_per_expert)**2) / (num_tokens**2)

        # figure out how many tokens each expert can handle
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)

        # actually route tokens to experts and collect results
        final_output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            expert_mask = (indices == expert_idx)
            if not expert_mask.any():
                continue

            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            if token_indices.shape[0] == 0:
                continue

            # don't overflow capacity
            token_indices = token_indices[:capacity]
            expert_weights = routing_weights[token_indices, expert_mask[token_indices].nonzero(as_tuple=True)[1]]

            # run the expert network and combine outputs
            expert_input = x_flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)

            # put expert outputs back in right place
            final_output.index_add_(0, token_indices, weighted_output)

        final_output = final_output.reshape(batch_size, seq_len, d_model)
        return final_output, load_balancing_loss


# custom layer dropout that works with moe layers
class MoELayerDropout(nn.Module):
    def __init__(self, prob=0.0, dim=0, disable_on_eval=True, seed=None):
        super().__init__()
        self.prob = prob
        self.dim = dim
        self.disable_on_eval = disable_on_eval
        self.generator = torch.Generator(device="cpu")
        self.inferred = None
        if seed is not None:
            self.generator.manual_seed(seed)

    def forward(self, function, input, *args, **kwargs):
        n = input.shape[self.dim]
        if self.prob == 0 or (self.disable_on_eval and self.training is False):
            self.inferred = 1.0
            return function(input, *args, **kwargs)

        skip = (
            torch.bernoulli(torch.Tensor((n) * [self.prob]), generator=self.generator)
            .to(input.device)
            .to(input.dtype)
        )
        self.inferred = 1 - torch.mean(skip)
        ind_selected = (skip == 0).nonzero().squeeze()

        if ind_selected.numel() > 0:
            x_selected = torch.index_select(input, self.dim, ind_selected)
            out_selected = function(x_selected, *args, **kwargs)

        out = input

        # handle MoE layers since return tuple
        if ind_selected.numel() > 0:
            if isinstance(out_selected, tuple):
                if not isinstance(out, tuple):
                    out = (out, torch.tensor(0.0, device=out.device))

                out_tensor, out_aux_loss = out
                out_selected_tensor, out_selected_aux_loss = out_selected

                out_tensor = out_tensor.clone()
                out_tensor[ind_selected] = out_selected_tensor

                return out_tensor, out_selected_aux_loss
            else:
                out = out.clone()
                out[ind_selected] = out_selected
                return out

        return out


# wrapper for moe layers with dropout
class MoELayerDropoutWrapper(nn.Module):
    def __init__(self, module, dropout):
        super().__init__()
        self.module = module
        self.dropout = dropout

    def forward(self, x, *args, **kwargs):
        return self.dropout(self.module, x, *args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# transformer block that can use MoE
class MoETransformer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, use_moe=False, num_experts=4, k=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.use_moe = use_moe

        if use_moe:
            self.moe = MoELayer(d_model, d_ff, num_experts=num_experts, k=k)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + residual

        residual = x
        x = self.norm2(x)

        if self.use_moe:
            moe_output, aux_loss = self.moe(x)
            x = moe_output + residual
            return x, aux_loss
        else:
            x = self.mlp(x)
            x = x + residual
            return x


# helper function to add dropout to layers
def prepare_moe_layer_dropout(layers, prob_max=0.2, prob_layer_scale="exp", moe_layers=None, disable_on_eval=True):
    """Apply layer dropout to a mix of standard and MoE layers"""
    num_layers = len(layers)

    for idx, layer in enumerate(layers):
        # higher layers get higher dropout prob
        depth = idx / max(1, num_layers - 1)
        if prob_layer_scale == "linear":
            prob = prob_max * depth
        elif prob_layer_scale == "exp":
            prob = prob_max * (depth ** 2)  # less dropout early, more later
        else:
            prob = prob_max

        is_moe_layer = moe_layers is not None and idx in moe_layers
        if is_moe_layer:
            dropout = MoELayerDropout(prob=prob, disable_on_eval=disable_on_eval, seed=idx)
            layers[idx] = MoELayerDropoutWrapper(layers[idx], dropout)
        else:
            # regular layers get standard dropout
            dropout = LayerDropout(prob=prob, disable_on_eval=disable_on_eval, seed=idx)
            layers[idx] = MoELayerDropoutWrapper(layers[idx], dropout)


# our main model, inspired by GPT2 and the switch transformers paper
class MoELayerSkipModel(nn.Module):
    def __init__(self, vocab_size, n_layers=12, n_heads=12, d_model=768, d_ff=3072,
                 max_seq_len=128, moe_layers=None, num_experts=4, k=2):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # create all transformer layers
        # use MoE every other layer (like switch transformers)
        self.layers = nn.ModuleList([
            MoETransformer(
                d_model=d_model,
                nhead=n_heads,
                d_ff=d_ff,
                use_moe=(i % 2 == 1),
                num_experts=num_experts,
                k=k
            )
            for i in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.output_hidden_states = None

        # for early exit curriculum
        self.curriculum = None
        self.n_layers = n_layers
        self.moe_layers = moe_layers
        self.moe_losses = None

    def forward(self, input_ids):
        x = self.tok_embeddings(input_ids)
        pos_embeds = self.pos_embeddings[:, :x.size(1), :]
        x = x + pos_embeds

        hidden_states = {}
        moe_losses = []

        for idx, layer in enumerate(self.layers):
            if idx in self.moe_layers:
                layer_output = layer(x)
                if isinstance(layer_output, tuple):
                    x, aux_loss = layer_output
                    moe_losses.append(aux_loss)
                else:
                    x = layer_output
            else:
                x = layer(x)

            if self.output_hidden_states is not None and idx in self.output_hidden_states:
                hidden_states[idx] = x

        self.moe_losses = moe_losses
        x = self.norm(x)
        logits = self.output(x)

        if hidden_states:
            return list(hidden_states.values()) + [logits]
        else:
            return logits

    def unembed(self, hidden):
        hidden = self.norm(hidden)
        return self.output(hidden)

