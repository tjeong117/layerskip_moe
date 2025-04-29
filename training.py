import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

# import from our MoE module
from layerskip_moe.MoE import MoELayerSkipModel, prepare_moe_layer_dropout
from torchtune.modules.early_exit_loss import early_exit_loss, linear_l_loss_scale, RotationalEarlyExitCurriculum

# dataset for wikitext
class WikiTextDataset(Dataset):
    def __init__(self, tokenizer, seq_len, sample_fraction):
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split='train')
        total_samples = len(dataset)
        num_samples = int(total_samples * sample_fraction)

        # sample random entries amounting to a fixed fraction of the data
        random_batch = random.sample(range(total_samples), num_samples)

        # collect text chunks
        text_chunks = []
        for i in tqdm(random_batch, desc=f"Collecting data"):
            text_chunks.append(dataset[i]["text"])
        text = "\n\n".join(text_chunks)
        print(f"Using {len(text_chunks)} examples")

        # tokenize the text in chunks to avoid the max length issue
        print(f"Tokenizing data:")
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        all_tokens = []
        for i, chunk in enumerate(tqdm(chunks, desc="Tokenizing chunks")):
            # manually trunking to prevent annoying token error
            chunk_tokens = tokenizer(chunk, return_tensors='pt')["input_ids"].squeeze(0)
            all_tokens.append(chunk_tokens)

        # concatenate all tokens
        tokens = torch.cat(all_tokens)

        self.seq_len = seq_len
        self.tokens = tokens
        print()
        print(f"Finished tokenizing")

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        return self.tokens[start:end]

# main training loop
def train_model(vocab_size, seq_len=128, batch_size=16, n_layers=12, n_heads=8, d_model=512, d_ff=2048, epochs=10, learning_rate=5e-4, sample_fraction=0.1, num_experts=8, k=2, moe_loss_weight=0.01):

    # makes sure it's fast
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        print("USING CPU!!! THIS IS BAD!")

    # setup moe layers, for speed does every other layer
    moe_layers = [i for i in range(n_layers) if i % 2 == 1]

    # initialize tokenizer if needed
    if 'tokenizer' not in globals():
        global tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # create dataset
    dataset = WikiTextDataset(tokenizer, seq_len=seq_len, sample_fraction=sample_fraction)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize model
    model = MoELayerSkipModel(
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        max_seq_len=seq_len,
        moe_layers=moe_layers,
        num_experts=num_experts,
        k=k
    ).to(device)

    # apply custom layer dropout for MoE
    prepare_moe_layer_dropout(
        model.layers,
        prob_max=0.2,
        prob_layer_scale="exp",
        moe_layers=moe_layers,
        disable_on_eval=True
    )

    # setup curriculum
    do_output_hidden_states = [False] * n_layers
    curriculum = RotationalEarlyExitCurriculum(
        do_output_hidden_states=do_output_hidden_states,
        max_steps=len(dataset) * epochs // batch_size,
        train_last_layer=True,
        verbose=False,
        last_step=0
    )
    model.curriculum = curriculum

    # setup optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # training stats for printing and visualization
    training_stats = {
        "epochs_completed": 0,
        "global_step": 0,
        "losses": [],
        "moe_losses": [],
        "perplexities": [],
        "exit_layer_stats": [],
        "model_config": {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_experts": num_experts,
            "k": k,
            "moe_layers": moe_layers
        }
    }

    print(f"Begin training")
    # helps prints model params out nicely
    print(f"Model has { (sum(p.numel() for p in model.parameters()) / 1e6):.2f}M params")

    # sample prompt for generation
    sample_prompt = "My favorite car is a"

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()

        # get the active layers from the curriculum
        if hasattr(model, 'curriculum') and model.curriculum is not None:
            active_layers = model.curriculum.get()
            model.output_hidden_states = [i for i, is_active in enumerate(active_layers) if is_active]

        epoch_losses = []
        epoch_moe_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(pbar):
            input_ids = batch.to(device)
            labels = input_ids.clone()
            optimizer.zero_grad()

            outputs = model(input_ids)

            # handle MoE load balancing losses
            moe_aux_losses = model.moe_losses if hasattr(model, 'moe_losses') else None
            moe_loss = sum(moe_aux_losses) if moe_aux_losses else 0

            if isinstance(outputs, list):
                # map hidden states to their layer indices
                hidden_states = {
                    model.output_hidden_states[i]: outputs[i]
                    for i in range(len(outputs) - 1)
                }
                logits = outputs[-1]
            else:
                hidden_states = {}
                logits = outputs

            # shift labels for causal lm loss
            labels = F.pad(labels[:, 1:], (0, 1), value=-1)

            if hidden_states:
                # use early exit loss from TorchTune
                lm_loss = early_exit_loss(
                    model, hidden_states, labels, loss_fn,
                    e_scale=1.0, loss_scale_fn=linear_l_loss_scale
                )
            else:
                lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            # combine losses
            loss = lm_loss + moe_loss_weight * moe_loss
            loss.backward()
            optimizer.step()

            # update the curriculum
            if hasattr(model, 'curriculum') and model.curriculum is not None:
                model.curriculum.step()
                active_layers = model.curriculum.get()
                model.output_hidden_states = [i for i, is_active in enumerate(active_layers) if is_active]

            epoch_losses.append(lm_loss.item())
            if moe_aux_losses:
                epoch_moe_losses.append(moe_loss.item())

            # update progress bar
            pbar.set_postfix(loss=lm_loss.item(), moe_loss=(moe_loss.item() if moe_aux_losses else 0))

        # end of epoch statistics
        training_stats['global_step'] += len(dataloader)
        training_stats['epochs_completed'] += 1
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_moe_loss = sum(epoch_moe_losses) / len(epoch_moe_losses) if epoch_moe_losses else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        training_stats['losses'].append(avg_loss)
        training_stats['moe_losses'].append(avg_moe_loss)
        training_stats['perplexities'].append(perplexity)

        # generate sample text with temperature
        text, exit_layers = generate_tokens(model, tokenizer, sample_prompt)
        avg_exit_layer = sum(exit_layers) / len(exit_layers) if exit_layers else 0
        training_stats['exit_layer_stats'].append(avg_exit_layer)

        # calculate epoch time and print summary
        epoch_time = time.time() - epoch_start_time
        print(f"\n---------- Epoch {epoch + 1}/{epochs} ----------")
        print(f"Average train loss: {avg_loss:.4f}")
        print(f"Average MoE loss: {avg_moe_loss:.6f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Average exit layer: {avg_exit_layer:.2f}")
        print(f"Epoch took {epoch_time:.2f} seconds")
        print(f"Generated text: {text}")
        print("-----------------------------------")


    print()
    print()
    print("Finished training")
    return model, training_stats

# generates tokens, not speculative decoding :(
def generate_tokens(model, tokenizer, prompt, max_length=20, temperature=0.8, top_k=40):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        # tokenize the prompt
        input_tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        generated = input_tokens[0].tolist()
        input_ids = input_tokens
        exit_layers = []

        # generate tokens
        for _ in range(max_length):
            hidden_states_per_layer = []

            # process through all layers to get hidden states
            x = model.tok_embeddings(input_ids)
            pos_embeds = model.pos_embeddings[:, :x.size(1), :]
            x = x + pos_embeds

            for idx, layer in enumerate(model.layers):
                if idx in model.moe_layers:
                    layer_output = layer(x)
                    if isinstance(layer_output, tuple):
                        x, _ = layer_output
                    else:
                        x = layer_output
                else:
                    x = layer(x)
                hidden_states_per_layer.append(x)

            # determine best layer based on max confidence
            best_layer = 0
            best_confidence = -float('inf')
            for idx, hidden in enumerate(hidden_states_per_layer):
                logits = model.unembed(hidden)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                confidence, _ = probs.max(dim=-1)
                if confidence.item() > best_confidence:
                    best_confidence = confidence.item()
                    best_layer = idx

            # use the best layer's hidden state for generation
            next_token_logits = model.unembed(hidden_states_per_layer[best_layer])[:, -1, :]

            # apply temperature scaling
            scaled_logits = next_token_logits / temperature

            # apply top-k sampling
            top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)

            # sample from the filtered distribution
            next_token_idx = torch.multinomial(probs[0], num_samples=1).item()
            next_token = top_k_indices[0][next_token_idx].item()

            # store the token and the exit layer used
            generated.append(next_token)
            exit_layers.append(best_layer)

            # update input for next iteration
            input_ids = torch.tensor([generated], dtype=torch.long).to(device)

        # decode the generated tokens
        decoded_text = tokenizer.decode(generated)

        return decoded_text, exit_layers
