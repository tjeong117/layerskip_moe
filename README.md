# LayerSkip for Mixture of Experts
Implementing facebookresearch/LayerSkip with a Mixture of Experts (MoE) model


To try out running the code yourself, the easiest approach is to copy the code from `main.py` into a Google Colab notebook. Then, run

```
!pip install datasets torchtune torchao
!git clone https://github.com/nickpapciak/layerskip_moe.git
```

in one of the cells, to install the project correctly. Then you can tune the hyperparameters

```
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

num_experts = 8
k = 2
moe_loss_weight = 0.01
```

and try training! After training it will generate results about the training session.


## Abstract

> *Large language models (LLMs) have demonstrated remarkable capabilities but remain computationally expensive to deploy and operate. Mixture of Experts (MoE) architectures have emerged as a promising approach for scaling LLMs efficiently by selectively activating only a subset of expert parameters for each forward pass. While MoE provides width-wise sparsity (activating only a portion of the network horizontally), we identify an opportunity to integrate LayerSkip to provide complementary depth-wise sparsity, enabling dynamic computation paths based on input complexity.*

## Introduction

Recent advances in large language models (LLMs) have demonstrated incredible capabilities and revolutionary computational abilities. However, to support the large increase in the accuracy and capability of these models, the size of the models has had to increase drastically. This has underscored a clear need for efficient, scalable, sparsity techniques that can allow models to get larger without as much computational overhead. In recent years, Mixture of Experts (MoE) models have made themselves known as such a promising solution. They make LLMs more efficient by routing data through the network so that only a sparse subset of "experts" is activated at each step. Thus far, the approach has seen some great success. Deepseek-V3 and GPT4 which both heavily rely on the MoE architecture have achieved state-of-the-art results using this technique, with gargantuanly sized models. However, despite the massive model sizes, they are able to keep inference costs feasible because of (1) the inherent scalability of MoE and transformer architectures, and (2) the sparsity introduced by the MoE models. They are starting to reach a potential limit, however. The recent Deepseek-V3 model has a staggering 256 experts, for example, which is wildly larger from the 16 experts of the precursor GPT4 model. This represents a dramatic shift in the way MoE architecture will be used, and emphasizes the importance of model size when trying to improve LLMs. This is especially worrying as model sizes have been increasing faster than computational resources to support them have. Thus, there is a desperate need for further sparsity techniques to make inference cheaper and more efficient.

We propose implementing Meta Research's LayerSkip as such a sparsity technique. LayerSkip allows the selective bypassing of expert layers based on input complexity, to add even more sparsity to an MoE model.

LayerSkip has been previously only been implemented in traditional dense LLama-7B models. The LayerSkip architecture works in three main stages:

1. **Layer Dropout**: Layer dropout is applied during training with one of two different curriculums and increasing rates as the network deepens.
2. **Early Exit**: The model learns to leave certain layers early during inference. The model is trained to try to "make up its mind" early in inference rather than going back and forth (as traditional models currently demonstrate). Each layer shares a single exit layer (called the LM head), which is in contrast to previous approaches which tried to use separate exit layers for each individual layer.
3. **Self-Speculative Decoding**: Because we exit at early layers, we can verify the results using the layer layers of the network. We perform this operation in parallel and go back and "fix" the predictions if we realize they are wrong. This allows us to get a significant accuracy boost and speeds up the model.
