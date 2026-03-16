 # Addendum: Core Elhage 2021 Concepts
*Keep with Memorize Sheet*

## 1. The Path Expansion Trick
Attention-only models can be written as a sum of interpretable end-to-end functions mapping tokens to changes in logits. 
- **0-Layer (Direct) Path:** The model's baseline bigram statistics. Embeddings multiply directly with unembeddings.
  - $\text{Logits}_{\text{direct}} = W_E W_U$
- **1-Layer Path:** The sum of the effects of every individual attention head.
  - $\text{Logits}_{\text{1-layer}} = \sum_{h} (W_E W_{OV}^{h} W_U)$
- **Total Logits:** $\text{Logits} = W_E W_U + \sum_{h} (W_E W_{OV}^{h} W_U)$

## 2. QK vs. OV Asymmetry
Attention heads contain two largely independent computations:
- **QK Circuit ($W_{QK} = W_Q W_K^T$):** Dictates *where* to move information. It calculates the attention pattern (the weights).
- **OV Circuit ($W_{OV} = W_V W_O$):** Dictates *what* information to move. It governs which subspace of the residual stream the head reads from, and which subspace it writes to. 
- *Interpretability Rule of Thumb:* If you are looking at attention heatmaps, you are analyzing the QK circuit. If you are doing SVD to see what features a head represents, you are analyzing the OV circuit.



---

You're completely right. Let's make this a single, unified block of Markdown with zero conversational filler inside it, and we will use proper LaTeX for the math so there is absolute clarity on the calculus before you write the code.

You are absolutely right. I skipped the explicit definition of the Jacobian matrix itself and jumped straight to the final chain-rule result. That is sloppy, especially since you need the exact theoretical steps for your sheet.

According to your `memorize_sheet.md`, the Jacobian of the softmax is exactly $\frac{\partial \pi_i}{\partial z_j} = \pi_i(\delta_{ij} - \pi_j)$.

Here is the corrected, complete `.md` block with $\frac{\partial A}{\partial S}$ explicitly defined and derived. Copy and paste this:

```markdown
# Emergency Backprop & Attribution Reference

## 1. Variable Mapping

| Math Derivative | Code Variable | Shape | Description |
| :--- | :--- | :--- | :--- |
| $\frac{\partial L}{\partial O}$ | `gO` | `(b, i, m)` | Gradient of Loss wrt Attention Output |
| $\frac{\partial L}{\partial A}$ | `gA` | `(b, h, i, j)` | Gradient of Loss wrt Attention Probabilities |
| $\frac{\partial A}{\partial S}$ | *(Implicit)* | `(j, k)` per query | The Softmax Jacobian Matrix |
| $\frac{\partial L}{\partial S}$ | `gS` | `(b, h, i, j)` | Gradient of Loss wrt Raw Attention Scores |
| $\frac{\partial L}{\partial Q}$ | `gQ` | `(b, h, i, a)` | Gradient of Loss wrt Queries |

---

## 2. Phase 1: Through the Value Matrix
To find how the probabilities $A$ affected the output $O = AV$:

**Math:**
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} V^T$$

**Code (NumPy):**
```python
# Assuming gO has been projected back through W_O into head-space (b, h, i, a)
# V is (b, h, j, a)
gA = np.einsum('bhia, bhja -> bhij', gO, V)

```

## 3. Phase 2: The Softmax Jacobian (dA/dS)

For a fixed query $i$, the softmax is taken over the source keys $j$. The Jacobian matrix $\frac{\partial A}{\partial S}$ defines how a change in score $S_{ij}$ affects probability $A_{ik}$.

**Math (The Jacobian):**


$$ \frac{\partial A_{ik}}{\partial S_{ij}} = A_{ik}(\delta_{jk} - A_{ij}) $$


*(where $\delta_{jk}$ is 1 if $j=k$, and 0 otherwise)*

**Math (The Chain Rule):**
To get the loss with respect to the scores, we multiply the incoming gradient by the Jacobian and sum over $k$:


$$ \frac{\partial L}{\partial S_{ij}} = \sum_k \frac{\partial L}{\partial A_{ik}} \frac{\partial A_{ik}}{\partial S_{ij}} $$

$$ \frac{\partial L}{\partial S_{ij}} = \sum_k \frac{\partial L}{\partial A_{ik}} A_{ik} (\delta_{jk} - A_{ij}) $$

$$ \frac{\partial L}{\partial S_{ij}} = A_{ij} \left( \frac{\partial L}{\partial A_{ij}} - \sum_{k} A_{ik} \frac{\partial L}{\partial A_{ik}} \right) $$

**Code (NumPy):**

```python
# We compute the final chain rule result directly using the Jacobian-Vector Product shortcut.
# Sum weighted gradients across the SOURCE (j/k) dimension
weighted_sum = np.sum(A * gA, axis=-1, keepdims=True)
gS = A * (gA - weighted_sum)

```

## 4. Phase 3: Through the Query-Key Dot Product

To find how the Queries and Keys affected the raw scores $S = \frac{Q K^T}{\sqrt{d_k}}$.

**Math:**


$$ \frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S} K $$

$$ \frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \left( \frac{\partial L}{\partial S} \right)^T Q $$

**Code (NumPy):**

```python
# dk is the head dimension (alpha)
scale = 1.0 / np.sqrt(dk)
gQ = scale * np.einsum('bhij, bhja -> bhia', gS, K)
gK = scale * np.einsum('bhij, bhia -> bhja', gS, Q)

```

## 5. Phase 4: To the Weight Matrices

To find how the weight matrices affected the Queries (e.g., $Q = X W_Q$).

**Math:**


$$ \frac{\partial L}{\partial W_Q} = X^T \frac{\partial L}{\partial Q} $$

**Code (NumPy):**

```python
# X is (b, i, m), gQ is (b, h, i, a)
# Summing over batch (b) and sequence position (i)
gWq = np.einsum('bim, bhia -> hma', X, gQ)

```

```

That bridges the gap exactly from your `memorize_sheet.md` all the way through the chain rule to the final NumPy code. 

Good catch. Now you have the complete theoretical picture. Ready to start?

```


# Emergency Backprop & Attribution Reference

## 1. Variable Mapping

| Math Derivative | Code Variable | Shape | Description |
| :--- | :--- | :--- | :--- |
| $\frac{\partial L}{\partial O}$ | `gO` | `(b, i, m)` | Gradient of Loss wrt Attention Output |
| $\frac{\partial L}{\partial A}$ | `gA` | `(b, h, i, j)` | Gradient of Loss wrt Attention Probabilities |
| $\frac{\partial L}{\partial S}$ | `gS` | `(b, h, i, j)` | Gradient of Loss wrt Raw Attention Scores |
| $\frac{\partial L}{\partial Q}$ | `gQ` | `(b, h, i, a)` | Gradient of Loss wrt Queries |

---

## 2. Phase 1: Through the Value Matrix
To find how the probabilities $A$ affected the output $O = AV$:

**Math:**
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} V^T$$

**Code (NumPy):**
```python
# Assuming gO has been projected back through W_O into head-space (b, h, i, a)
# V is (b, h, j, a)
gA = np.einsum('bhia, bhja -> bhij', gO, V)

```

## 3. Phase 2: The Softmax Jacobian

To find how the raw scores $S$ affected the probabilities $A$, accounting for the sum-to-one constraint across the source/key dimension $j$.

**Math:**


$$ \frac{\partial L}{\partial S_{ij}} = A_{ij} \left( \frac{\partial L}{\partial A_{ij}} - \sum_{k} A_{ik} \frac{\partial L}{\partial A_{ik}} \right) $$

**Code (NumPy):**

```python
# Sum weighted gradients across the SOURCE (j) dimension
weighted_sum = np.sum(A * gA, axis=-1, keepdims=True)
gS = A * (gA - weighted_sum)

```

## 4. Phase 3: Through the Query-Key Dot Product

To find how the Queries and Keys affected the raw scores $S = \frac{Q K^T}{\sqrt{d_k}}$.

**Math:**


$$ \frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S} K $$

$$ \frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \left( \frac{\partial L}{\partial S} \right)^T Q $$

**Code (NumPy):**

```python
# dk is the head dimension (alpha)
scale = 1.0 / np.sqrt(dk)
gQ = scale * np.einsum('bhij, bhja -> bhia', gS, K)
gK = scale * np.einsum('bhij, bhia -> bhja', gS, Q)

```

## 5. Phase 4: To the Weight Matrices

To find how the weight matrices affected the Queries (e.g., $Q = X W_Q$).

**Math:**


$$ \frac{\partial L}{\partial W_Q} = X^T \frac{\partial L}{\partial Q} $$

**Code (NumPy):**

```python
# X is (b, i, m), gQ is (b, h, i, a)
# Summing over batch (b) and sequence position (i)
gWq = np.einsum('bim, bhia -> hma', X, gQ)

```

```

Copy that block directly. Now that the math is rigorously defined alongside your `einsum` index notation, you have a complete map to survive any gradient question they throw at you.

Are you ready to start your 120-minute timer for Practice Test 1, or is there one last edge case you want to lock down?

```



You are exactly right to pivot here. In Mechanistic Interpretability, you aren't training the model—you are dissecting it. You use gradients not to update weights, but to answer the question: *"Which part of this network is responsible for this specific behavior?"*

Here are the three specific attribution techniques that *do* show up in interpretability research and Anthropic-style assessments.

Copy and paste this directly into your reference sheet.

```markdown
# Interpretability & Attribution Techniques

In interpretability, we use gradients and projections to trace model behavior back to specific components (like attention heads or neurons).

## 1. Gradient $\times$ Activation (First-Order Attribution)
Used to answer: *"If I ablated (zeroed out) this specific activation, how much would my loss change?"* Instead of actually running the model thousands of times to ablate every node, we use the first-order Taylor approximation.

**Math:**
$$\text{Attribution} = X \odot \frac{\partial L}{\partial X}$$

**Code (NumPy):**
```python
# X is the activation (e.g., the residual stream before a layer)
# grad_X is the gradient of the loss wrt X (provided by autograd)
# A large negative value means this activation is working hard to DECREASE the loss.
attribution_scores = X * grad_X 

# Often, we sum over the feature dimension to get a single importance score per token
token_importance = np.sum(attribution_scores, axis=-1)

```

---

## 2. Attention Score Attribution

Used to answer: *"Which source token $j$ was actually responsible for lowering the loss for target token $i$?"*
Just looking at the raw attention weight $A_{ij}$ is misleading (a head might attend to a token, but pass useless information). We multiply the raw score by its gradient.

**Math:**


$$\text{Attribution}_{ij} = S_{ij} \odot \frac{\partial L}{\partial S_{ij}}$$

**Code (NumPy):**

```python
# Assuming you have gS (the Softmax Jacobian output from the earlier sheet)
# S is the pre-softmax attention scores
score_attribution = S * gS

```

---

## 3. Direct Logit Attribution (The "Logit Lens")

Used to answer: *"Is Attention Head 3 directly predicting the correct next word, or is it just passing context to later layers?"*
This is a **forward-pass** technique. We take the output of a single head, project it through the output weights, and multiply it directly by the unembedding matrix ($W_U$) to see its vocabulary logits.

**Math:**


$$\text{Logits}_{\text{head}} = (O_{\text{head}} W_O) W_U$$

**Code (NumPy):**

```python
# O is a specific head's output: (batch, seq, d_v)
# W_O_head is that head's specific slice of W_O: (d_v, d_model)
# W_U is the unembedding matrix: (d_model, vocab_size)

# Project the head's output directly into vocabulary space
head_logits = np.einsum('bia, am, mv -> biv', O, W_O_head, W_U)

# To find its attribution to the CORRECT target token:
# (Assuming targets is an array of shape (batch, seq) containing vocab indices)
batch_indices = np.arange(batch)[:, None]
seq_indices = np.arange(seq)[None, :]
direct_attribution = head_logits[batch_indices, seq_indices, targets]

```

```

### Why these three matter:
1. **Grad $\times$ Act** is the fastest way to find the most important components in a circuit.
2. **Attention Attribution** tells you *why* a circuit is firing (what information it's moving).
3. **Logit Lens** proves that the circuit is actually responsible for the final output, not just an intermediate step.

This is the actual math of Mechanistic Interpretability. Keep this next to your `einsum` strings, and you are officially ready for anything they throw at you. Go start that timer!

```