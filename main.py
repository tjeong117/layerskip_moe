# !pip install datasets torchtune torchao
# !git clone https://github.com/nickpapciak/layerskip_moe.git

import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2TokenizerFast
import sys
from layerskip_moe.training import train_model, generate_tokens


def visualize_training_stats(stats):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(stats['losses'], 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(stats['perplexities'], 'r-')
    plt.title('Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(stats['moe_losses'], 'g-')
    plt.title('MoE Load Balancing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MoE Loss')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(stats['exit_layer_stats'], 'purple')
    plt.title('Average Exit Layer')
    plt.xlabel('Epoch')
    plt.ylabel('Layer Index')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('model_stats.png')
    print("Model statistics visualization saved as 'model_stats.png'")
    plt.show()

def test_generation(model, tokenizer):
    """Test text generation with different prompts"""
    prompts = [
        "My favorite class at Georgia Tech is definitely not",
        "Abraham Lincoln's middle name is",
        "The weather is currently",
        "Hey there!"
    ]

    print("\n Testing: ")

    results = []

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        text, exit_layers = generate_tokens(
            model,
            tokenizer,
            prompt,
            max_length=30,
            temperature=0.9
        )
        print(f"Generated: {text}")
        print(f"Exit layers: {exit_layers}")
        print("-" * 40)
        results.append((prompt, exit_layers))

    plt.figure(figsize=(15, 5))
    for i, (prompt, exit_layers) in enumerate(results):
        plt.subplot(1, len(results), i + 1)
        plt.hist(exit_layers, bins=range(max(exit_layers) + 2), align='left', rwidth=0.8)
        plt.title(f"Prompt {i+1}")
        plt.xlabel('Exit Layer')
        plt.ylabel('Frequency')
        plt.xticks(range(max(exit_layers) + 1))
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('exit_layer_distribution.png')
    plt.show()

# hyperparameters
seq_len = 128
batch_size = 16
n_layers = 12
n_heads = 8
d_model = 512
d_ff = 2048
epochs = 5
learning_rate = 5e-4
sample_fraction = 0.01

# MoE parameters
num_experts = 8
k = 2  # experts per token
moe_loss_weight = 0.01

print("Loading tokenizer")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size

# train model with all hyperparameters
print("Starting training...")
model, stats = train_model(
    vocab_size=vocab_size,
    seq_len=seq_len,
    batch_size=batch_size,
    n_layers=n_layers,
    n_heads=n_heads,
    d_model=d_model,
    d_ff=d_ff,
    epochs=epochs,
    learning_rate=learning_rate,
    sample_fraction=sample_fraction,
    num_experts=num_experts,
    k=k,
    moe_loss_weight=moe_loss_weight
)

# print model statistics
visualize_training_stats(stats)

# test generation
test_generation(model, tokenizer)
