# Decoder-Only Transformer for Autoregressive Language Modeling

A 109.5M-parameter decoder-only Transformer implemented from scratch in TensorFlow / Keras, with a custom data pipeline, training engine, and evaluation harness. This document gives the model, objective, and optimization procedure in full mathematical form.

> Math is written in LaTeX and renders on GitHub.

## Contents

1. [Notation](#1-notation)
2. [Model](#2-model)
3. [Training Objective](#3-training-objective)
4. [Optimization](#4-optimization)
5. [Parameter and Compute Accounting](#5-parameter-and-compute-accounting)
6. [Evaluation](#6-evaluation)
7. [Parameter-Efficient Fine-Tuning](#7-parameter-efficient-fine-tuning)
8. [Usage](#8-usage)

---

## 1. Notation

| Symbol | Meaning | Value |
|---|---|---|
| $V$ | Vocabulary size (Byte-Level BPE) | 32,000 |
| $d$ | Model (hidden) dimension | 768 |
| $L$ | Number of decoder blocks | 12 |
| $H$ | Attention heads | 12 |
| $d_h = d/H$ | Head dimension | 64 |
| $d_{\mathrm{ff}}$ | SwiGLU intermediate dimension | 2048 |
| $T$ | Context length | 256 |
| $\theta_{\mathrm{rope}}$ | RoPE base | 10,000 |

A token sequence is $x = (x_1, \dots, x_T) \in \lbrace 1, \dots, V \rbrace^T$. Hidden states are $X^{(\ell)} \in \mathbb{R}^{T \times d}$.

---

## 2. Model

### 2.1 Embedding and residual stream

With embedding matrix $E \in \mathbb{R}^{V \times d}$, the input state is

```math
X^{(0)}_{t,:} = E_{x_t,:}
```

Each pre-norm block $\ell = 1, \dots, L$ updates the residual stream as

```math
\tilde{X}^{(\ell)} = X^{(\ell-1)} + \operatorname{MHA}^{(\ell)}\!\Big(\operatorname{RMSNorm}\big(X^{(\ell-1)}\big)\Big)
```

```math
X^{(\ell)} = \tilde{X}^{(\ell)} + \operatorname{FFN}^{(\ell)}\!\Big(\operatorname{RMSNorm}\big(\tilde{X}^{(\ell)}\big)\Big)
```

Unrolling the recursion shows that the final state is the embedding plus a sum of block contributions, which gives gradients a direct path to every layer:

```math
X^{(L)} = X^{(0)} + \sum_{\ell=1}^{L} \Big( \operatorname{MHA}^{(\ell)}(\cdot) + \operatorname{FFN}^{(\ell)}(\cdot) \Big)
```

### 2.2 RMSNorm

For a row vector $x \in \mathbb{R}^d$ with learned scale $g \in \mathbb{R}^d$:

```math
\operatorname{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\dfrac{1}{d} \displaystyle\sum_{j=1}^{d} x_j^2 + \varepsilon}} \; g_i
```

Unlike LayerNorm, there is no mean subtraction and no bias, which removes $2d$ parameters per norm and one reduction per call.

### 2.3 Causal multi-head attention with RoPE

For head $h \in \lbrace 1, \dots, H \rbrace$, with bias-free projections $W_Q^{h}, W_K^{h}, W_V^{h} \in \mathbb{R}^{d \times d_h}$ applied to $Z = \operatorname{RMSNorm}(X)$:

```math
Q^h = Z\, W_Q^h, \qquad K^h = Z\, W_K^h, \qquad V^h = Z\, W_V^h
```

**Rotary position embedding.** Define the frequencies

```math
\omega_j = \theta_{\mathrm{rope}}^{-2(j-1)/d_h}, \qquad j = 1, \dots, \tfrac{d_h}{2}
```

and, for position $m$, the block-diagonal rotation

```math
R_m = \operatorname{diag}\!\left(
\begin{pmatrix} \cos m\omega_1 & -\sin m\omega_1 \\ \sin m\omega_1 & \cos m\omega_1 \end{pmatrix},
\ \dots,\
\begin{pmatrix} \cos m\omega_{d_h/2} & -\sin m\omega_{d_h/2} \\ \sin m\omega_{d_h/2} & \cos m\omega_{d_h/2} \end{pmatrix}
\right) \in SO(d_h)
```

Queries and keys are rotated by their own positions, $\hat q_m = R_m q_m$ and $\hat k_n = R_n k_n$. Because rotations are orthogonal and compose additively, $R_m^{\top} R_n = R_{n-m}$, so

```math
\hat q_m^{\top} \hat k_n = q_m^{\top} R_{n-m}\, k_n
```

The attention logit therefore depends on positions only through the **relative offset** $n-m$, with no learned positional table. In complex form, writing $q^{(j)}, k^{(j)} \in \mathbb{C}$ for the $j$-th coordinate pair,

```math
\hat q_m^{\top} \hat k_n = \operatorname{Re}\!\left[ \sum_{j=1}^{d_h/2} q^{(j)} \, \overline{k^{(j)}} \; e^{\, i (m-n)\omega_j} \right]
```

**Causal attention.** With mask $M_{mn} = 0$ for $n \le m$ and $M_{mn} = -\infty$ otherwise,

```math
A^h = \operatorname{softmax}\!\left( \frac{\hat Q^h \, (\hat K^h)^{\top}}{\sqrt{d_h}} + M \right), \qquad
\operatorname{head}_h = A^h V^h
```

The factor $\sqrt{d_h}$ keeps the logit variance near 1 for unit-variance inputs. Heads are concatenated and projected:

```math
\operatorname{MHA}(X) = \big[ \operatorname{head}_1 \,\Vert\, \cdots \,\Vert\, \operatorname{head}_H \big]\, W_O, \qquad W_O \in \mathbb{R}^{d \times d}
```

Because $A^h_{mn} = 0$ for $n > m$, the output at position $m$ is a function of $x_{\le m}$ only. This is what permits the shifted-target objective below without leakage.

### 2.4 SwiGLU feed-forward network

With $W_1, W_3 \in \mathbb{R}^{d \times d_{\mathrm{ff}}}$ and $W_2 \in \mathbb{R}^{d_{\mathrm{ff}} \times d}$:

```math
\operatorname{FFN}(z) = \Big( \operatorname{SiLU}(z W_1) \odot (z W_3) \Big) W_2,
\qquad
\operatorname{SiLU}(u) = u\,\sigma(u) = \frac{u}{1 + e^{-u}}
```

The gate $\operatorname{SiLU}(z W_1)$ multiplicatively modulates the linear branch $z W_3$.

### 2.5 Output distribution

After a final RMSNorm, logits are produced by the **tied** embedding matrix:

```math
\mathrm{logits} = \operatorname{RMSNorm}\big(X^{(L)}\big)\, E^{\top} \in \mathbb{R}^{T \times V},
\qquad
p_\theta(\cdot \mid x_{\le t}) = \operatorname{softmax}\big(\mathrm{logits}_t\big)
```

---

## 3. Training Objective

The model factorizes sequence probability autoregressively:

```math
p_\theta(x_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})
```

Training minimizes the empirical negative log-likelihood. For a packed chunk of $T+1$ tokens, the inputs are $x_{1:T}$ and the targets are $x_{2:T+1}$ (a one-token shift), giving

```math
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T} \log p_\theta\!\left( x^{(n)}_{t+1} \,\middle|\, x^{(n)}_{\le t} \right)
```

This is the cross-entropy between the empirical next-token distribution and the model. Minimizing it is equivalent to minimizing $D_{\mathrm{KL}}(p_{\mathrm{data}} \Vert p_\theta)$, up to the constant data entropy $H(p_{\mathrm{data}})$.

**Gradient at the logits.** For a one-hot target $y_t$ and $p_t = \operatorname{softmax}(z_t)$,

```math
\frac{\partial \ell_t}{\partial z_t} = p_t - y_t
```

so the per-token gradient is bounded in norm by $\sqrt{2}$ regardless of scale.

**Perplexity and chance level.**

```math
\mathrm{PPL} = \exp(\mathcal{L})
```

A uniform predictor attains $\mathcal{L} = \ln V = \ln 32000 \approx 10.37$ (that is, $\mathrm{PPL} = V$), which is the baseline against which training progress is measured.

---

## 4. Optimization

### 4.1 AdamW

With gradient $g_s = \nabla_\theta \mathcal{L}$ at step $s$, learning rate $\eta_s$, and decoupled weight decay $\lambda$:

```math
m_s = \beta_1 m_{s-1} + (1 - \beta_1)\, g_s, \qquad
v_s = \beta_2 v_{s-1} + (1 - \beta_2)\, g_s^{\odot 2}
```

```math
\hat m_s = \frac{m_s}{1 - \beta_1^{\,s}}, \qquad \hat v_s = \frac{v_s}{1 - \beta_2^{\,s}}
```

```math
\theta_s = \theta_{s-1} - \eta_s \left( \frac{\hat m_s}{\sqrt{\hat v_s} + \varepsilon} + \lambda\, \theta_{s-1} \right)
```

Reference defaults: $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\varepsilon = 10^{-8}$, $\lambda = 0.1$.

### 4.2 Learning-rate schedule (linear warmup, cosine decay)

With warmup length $S_w$ and total steps $S$:

```math
\eta_s =
\begin{cases}
\eta_{\max}\, \dfrac{s}{S_w}, & s < S_w, \\[2ex]
\eta_{\min} + \dfrac{1}{2} \big( \eta_{\max} - \eta_{\min} \big) \left( 1 + \cos\!\left( \pi\, \dfrac{s - S_w}{S - S_w} \right) \right), & s \ge S_w
\end{cases}
```

Reference defaults: $\eta_{\max} = 3 \times 10^{-4}$, $\eta_{\min} = 3 \times 10^{-5}$, $S_w = 2000$.

### 4.3 Gradient accumulation and clipping

With $K$ micro-batches of size $B$ (default $K = 16$, $B = 1$), the accumulated gradient and the effective batch size are

```math
\bar g = \frac{1}{K} \sum_{k=1}^{K} g^{(k)}, \qquad B_{\mathrm{eff}} = K\, B\, T \ \text{ tokens per update}
```

Global-norm clipping with threshold $c = 1$:

```math
\bar g \leftarrow \bar g \cdot \min\!\left( 1,\ \frac{c}{\lVert \bar g \rVert_2} \right)
```

Mixed precision (`float16`) uses dynamic loss scaling: the loss is multiplied by a scale $\kappa$ before backpropagation, and the gradients are divided by $\kappa$ before the update.

> Values above are archived engine defaults; the `run_config.json` written with each run is authoritative.

---

## 5. Parameter and Compute Accounting

### 5.1 Parameter count

All projections are bias-free and the output head is tied, so

```math
N_{\mathrm{params}} = \underbrace{V d}_{\text{embedding}}
+ L \Big( \underbrace{4 d^2}_{\text{attention}} + \underbrace{3\, d\, d_{\mathrm{ff}}}_{\text{SwiGLU}} + \underbrace{2 d}_{\text{RMSNorm}} \Big)
+ \underbrace{d}_{\text{final norm}}
```

Substituting the configuration:

```math
\begin{aligned}
V d &= 32000 \cdot 768 = 24{,}576{,}000 \\
4 d^2 &= 2{,}359{,}296, \qquad 3\, d\, d_{\mathrm{ff}} = 4{,}718{,}592, \qquad 2d = 1{,}536 \\
L \cdot (\cdots) &= 12 \cdot 7{,}079{,}424 = 84{,}953{,}088 \\
N_{\mathrm{params}} &= 24{,}576{,}000 + 84{,}953{,}088 + 768 = \mathbf{109{,}529{,}856}
\end{aligned}
```

Tying removes a second $V d = 24{,}576{,}000$ parameters that an untied output head would add.

### 5.2 Compute

The standard dense-Transformer approximation for training FLOPs over $D$ tokens is

```math
C_{\mathrm{train}} \approx 6\, N_{\mathrm{params}}\, D
```

Attention adds a context-dependent term. Per layer, the score and value products cost $O(T^2 d)$ FLOPs and the score tensor needs $O(H T^2)$ memory:

```math
C_{\mathrm{attn}} \sim 4\, L\, T^2\, d \ \text{ FLOPs per sequence (forward)}
```

For example, training on $D = 10^{9}$ tokens costs $C_{\mathrm{train}} \approx 6 \times 1.095 \times 10^{8} \times 10^{9} \approx 6.6 \times 10^{17}$ FLOPs.

---

## 6. Evaluation

### 6.1 Held-out metrics

For a held-out set with $N$ scored target tokens:

```math
\mathcal{L}_{\mathrm{val}} = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x_i \mid x_{<i}),
\qquad
\mathrm{Acc}_{\mathrm{top1}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\!\left[ \arg\max_{v}\, p_\theta(v \mid x_{<i}) = x_i \right]
```

### 6.2 Sliding-window scoring

For a book of $n_b$ tokens, windows of length $W = 256$ advance with stride $S = 128$. Each target is scored exactly once, using the maximum available context in its window; the first token of each book is skipped, and context never crosses book boundaries:

```math
N = \sum_{b} \, (n_b - 1)
```

### 6.3 Tokenizer-independent comparison

Perplexity and top-1 accuracy are not comparable across tokenizers. Normalizing by raw bytes gives bits per byte:

```math
\mathrm{BPB} = \frac{N_{\mathrm{tok}}}{N_{\mathrm{bytes}}} \cdot \frac{\mathcal{L}_{\mathrm{val}}}{\ln 2}
```

---

## 7. Parameter-Efficient Fine-Tuning

### 7.1 LoRA

For a frozen weight $W_0 \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}$, a low-rank update is learned:

```math
W = W_0 + \frac{\alpha}{r}\, A B,
\qquad A \in \mathbb{R}^{d_{\mathrm{in}} \times r}, \quad B \in \mathbb{R}^{r \times d_{\mathrm{out}}},
\qquad r \ll \min(d_{\mathrm{in}}, d_{\mathrm{out}})
```

with $r = 8$ and $\alpha = 16$ (scale $\alpha / r = 2$). The number of trainable parameters per adapted layer is $r\,(d_{\mathrm{in}} + d_{\mathrm{out}})$.

If adapters are enabled on all seven Dense projections per block (four attention, three SwiGLU):

```math
N_{\mathrm{LoRA}} = L \cdot r \Big[ 4\,(2d) + 3\,(d + d_{\mathrm{ff}}) \Big] = 12 \cdot 8 \cdot 14{,}592 = 1{,}400{,}832 \approx 1.28\% \ \text{ of } N_{\mathrm{params}}
```

### 7.2 INT4 weight quantization

A representative symmetric per-channel scheme maps weights to 4-bit integers:

```math
s = \frac{\max_i |w_i|}{7}, \qquad
q_i = \operatorname{clip}\Big( \operatorname{round}\big( w_i / s \big),\ -8,\ 7 \Big), \qquad
\hat w_i = s\, q_i
```

with unclipped error $|w_i - \hat w_i| \le s/2$. Base weights are stored at 4 bits and only the adapters are trained at higher precision. The package is *QLoRA-style* (Keras-native INT4), not bitsandbytes NF4 with double quantization.

### 7.3 Response-masked instruction loss

With mask $\mu_t = 1$ on response tokens and $\mu_t = 0$ on prompt tokens:

```math
\mathcal{L}_{\mathrm{IFT}} = -\frac{\sum_{t} \mu_t \log p_\theta(x_t \mid x_{<t})}{\sum_{t} \mu_t}
```

---

## 8. Usage

```bash
conda env create -f environment.yml
conda activate <environment-name>

# Run from the project root
python -m src.<pretraining_module>
python -m src.finetuning.qlora_ift
```

## License

This project is distributed under the terms of the [MIT License](LICENSE), which permits use, modification, and redistribution, in source and binary form, subject to preservation of the copyright and permission notice. The software is provided "as is", without warranty of any kind. Training corpora and third-party datasets referenced by this project remain subject to their respective licenses and terms of use.

## Author

**Amir Salar Sagharisaz**  
[GitHub](https://github.com/AmirSalarSagharisaz) · [LinkedIn](https://www.linkedin.com/in/amirsalarsagharisaz)
