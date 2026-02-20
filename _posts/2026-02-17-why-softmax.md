---
title: "Why Softmax? A Hypothesis Testing Perspective on Attention Weights"
layout: single
author_profile: false
toc: true
toc_label: "Contents"
toc_sticky: true
mathjax: true
sidebar:
  - title: "Notation"
    text: |
      **Tokens & Embeddings**
      - \\(x_i\\) = token at position \\(i\\)
      - \\(h_i\\) = contextualized embedding
      - \\(d_{\text{model}}\\) = embedding dimension
      - \\(T\\) = sequence length
      
      **Attention Components**
      - \\(q_t\\) = query vector
      - \\(k_i\\) = key vector
      - \\(v_i\\) = value vector
      - \\(d_k\\) = key/query dimension
      - \\(z_i = k_i \cdot q_t\\) = query-key score
      - \\(\mathcal{O}_t\\) = attention output
      - \\(W_K, W_Q, W_V\\) = learned matrices
      
      **Distributions**
      - \\(\pi_i\\) = attention weight (softmax)
      - \\(u\\) = uniform distribution
      - \\(n\\) = number of scores
      
      **Diagnostics**
      - \\(\hat W_K, \hat W_Q\\) = trained matrices
      - \\(\hat \pi_t\\) = trained attention weights
      - \\(\hat \rho_{\text{eff},t}\\) = effective KL budget
      - \\(\partial \hat \rho\\) = temperature susceptibility

      **Other**
      - \\(\beta\\) = inverse temperature
      - \\(\rho\\) = KL budget
      - \\(\text{KL}(\pi \| u)\\) = KL divergence
      - \\(H(\pi)\\) = Shannon entropy
tags: [attention, softmax, hypothesis testing, KL divergence, machine learning, deep learning]
excerpt: "Softmax is ubiquitous in transformers, yet its role in attention can feel more heuristic than inevitable. In this post, I try to make it feel more natural and show how this interpretation suggests useful diagnostics for the often circuit-like behavior of attention heads."
---

Softmax is ubiquitous in transformers, yet its role in attention can feel more heuristic than inevitable (at least to me). In this post, I try to make it feel more natural and show how this interpretation suggests useful diagnostics for the often circuit-like behavior of attention heads.

<!--more-->

## Introduction: the attention mechanism

Consider a stream of tokens (e.g. words from an LLM) to be embedded:

$$
x=\{x_1,x_2,...,x_i,...,x_T \}.
$$

After embedding (and potentially many passes through MLPs and attention heads) we have the contextualized tokens

$$
h_i \in \mathbb R^{d_{\text{model}}}.
$$

The attention mechanism updates this *residual stream* (as it is also called) by computing three quantities from learned parameters \\(W_K, W_Q\\) and \\( W_V \\). 

Given the most recent embedded token in the stream, \\(h_t\\), and all tokens before it \\( \\{ h_i :i < t \\} \\), these three quantities are the keys, query, and values---and are defined as

$$
k_i=W_Kh_i
$$

$$
q_t=W_Qh_t
$$

$$
v_i=W_Vh_i
$$

where \\(q,k \\in \\mathbb{R}^{d_k}\\) and typically \\( d_k<d_{\text{model}} \\). 

The update to the residual stream at position \\( t \\) is calculated as[^2]

$$
h_t^{\text{(new)}} =\mathcal{O}_t+ h_t^{\text{(old)}}
$$

with

$$
\label{eq:O_t}
\mathcal{O}_t=\sum_i\pi_{i,t} v_i 
$$

$$
\pi_{i,t} = \frac{e^{\beta k_i\cdot q_t}}{\sum_j e^{\beta k_j\cdot q_t}},
$$

 where we identify \\( \pi_i \\) with the \\(\mathrm{softmax}\\) function:

$$
\pi_{i,t}=\mathrm{softmax}(\beta k_i\cdot q_t),
$$

and we have introduced the scalar \\( \beta\\) for later use.

This lends itself to the following interpretation: for any given token at position \\( t \\), the query vector, \\(q_t\\), defines what \\(h_t\\) is "looking for" from previous tokens, and the keys, \\( k_i \\), determine which of the previous tokens get "advertised".[^3] 

The query-key pairs define the distribution \\( \pi_i \\) over which the values, \\( v_i \\), are averaged. We can see that this distribution determines what values the attention head should "focus" on. 

This post explores the question: **why softmax and not something else?**

I emphasize that this is just the way I like to think about it... not *the* way it should be understood.

## Softmax as hypothesis testing

For notational simplicity we fix a destination position and drop the index $t$. We define the query-key score for a given key as

$$ \label{eq:z}
z_i=k_i\cdot q.
$$

We let \\(n\\) denote the number of scores. 

Leaving \\(z_i\\) alone for the moment, let us imagine that we had no good reason to prefer one index over another when calculating \\( \mathcal{O}_t\\) from Eq. \eqref{eq:O_t}. The only distribution invariant under permutation of the indices (which is the symmetry that reflects our ignorance) is the uniform distribution, which we denote by \\(u\\).

Of course, we do have reason to prefer some indices over others in our distribution \\( \pi \\), namely the scores \\( z_i \\). We have two competing objectives: create a distribution that maximizes the expected score, \\( \sum_i \pi_i z_i \\) (thus properly weighting the evidence afforded us), but also do not overcommit to any particular index's score beyond what we believe is justifiable given our prior ignorance.

In hypothesis testing, the Kullback-Leibler (KL) divergence is a natural measure of distinguishability from a null hypothesis. The number of samples required to determine that said null hypothesis is false is proportional to \\( \frac{1}{\mathrm{KL}(\pi\\|u)}\\)[^1]. Loosely speaking, if we are given some "budget" \\( \rho \\), and if the KL exceeds it, then we may say that we have enough evidence to reject the null (uniform) hypothesis. This defines our notion of overcommitment. Our competing objectives are thus defined by constructing \\( \pi_i \\) such that the average score, \\( \langle z_i\rangle\\), is maximal (our commitment to our evidence is maximal), while remaining within our commitment "budget" defined by \\(\rho \\) and the KL-divergence. This defines the constrained optimization problem

$$
\max_{\pi \in \Delta}
\sum_i \pi_i z_i \quad \text{s.t.} \quad \mathrm{KL}(\pi \| u) \leq \rho,
$$

where \\(\Delta =\\{ \pi \in \mathbb{R}^n : \pi_i \geq 0,\; \sum_i \pi_i = 1 \\}\\) is the probability simplex.

Rather than solve this directly, we can relax the hard constraint into a penalty, yielding the equivalent unconstrained problem

$$
\max_{\pi \in \Delta}
\sum_i \pi_i z_i - \frac{1}{\beta} \mathrm{KL}(\pi \| u),
$$

where \\( \frac{1}{\beta} \\) controls the trade-off between maximizing expected score and staying close to uniform. Each value of \\( \beta \\) corresponds to a particular budget \\( \rho \\): large \\( \beta \\) (loose budget) allows sharper distributions, while small \\( \beta \\) (tight budget) keeps \\( \pi \\) near uniform. Introducing a Lagrange multiplier for normalization and taking first-order conditions, one finds that the solution is

$$
\pi_i^\star \propto e^{\beta z_i},
$$

which recovers softmax, defining the attention weights used in transformers. 

## Interpretation

I should reiterate that this is merely my *interpretation* of the softmax function. In modern commercial transformer architectures the above optimization problems are not explicitly written into the training objective and play no role at inference time.

That being said, the very fact that the softmax function is used in each attention head lends credence to the preceding interpretation. Each trained head in real, deployed transformers could be interpreted as instantiating some solution to a commitment-to-evidence-versus-ignorance optimization problem. Of course, the training objective does not explicitly enforce this constrained problem; the point is that the resulting functional form admits this interpretation.

The parameters \\(\beta\\) and \\(\rho\\) are not to be found in any such real-world transformer in the likes of Claude or ChatGPT, but each does indeed have their own weight matrices \\(\hat W_K, \hat W_Q\\) and \\( \hat W_V \\). So, for a given residual stream \\( \\{h_i\\} \\) for some context \\(\\{x_i\\}\\), there is nothing stopping us from interrogating an attention head by examining the quantity 

$$
\hat \rho_{\text{eff}, t}:=\mathrm{KL}(\hat \pi_t \|u )
\label{eq:rho_eff}
$$

for

$$
\hat\pi_{t,i}=\mathrm{softmax}(h_i^{\top}\hat W_K^\top\hat W_Qh_t).
$$

\\(\hat \rho_{\text{eff},t}\\) is an interesting quantity---we can think of it as measuring the "commitment to evidence" in a given attention head for given learned parameters and a given context. This last point is worth repeating: *it is a context dependent quantity*.

\\(\hat \rho_{\text{eff},t}\\) will be large when the evidence to focus on certain past tokens while building \\(\hat \pi_t\\) is large. It will be small when the evidence is "flimsy." This is not necessarily bad, however; one can imagine certain heads operate well by considering evidence from many tokens, instead of only a few (to be very hand-wavy, think of a head that considers general themes and the tone of a context, instead of particular grammatical rules or other minutiae).

This quantity, \\(\hat \rho_{\text{eff},t}\\), is therefore a proxy for how selectively information is routed through a given head. 

Equation \eqref{eq:rho_eff} can also be written as 

$$
\hat \rho_{\text{eff},t} = \log n - H(\hat \pi_t)
$$

where we see that \\(\hat \rho_{\text{eff},t}\\) is dependent upon the length of the context window. Since \\(\hat \rho_{\text{eff},t} = \log n - H(\hat \pi_t)\\), if it grows logarithmically with \\(n\\) then \\(H(\hat \pi_t)\\) must remain \\(O(1)\\)---meaning the head continues to focus on a fixed number of tokens regardless of how long the context becomes. Such a head can be seen as being robust. 

## Further implications: circuits & interpretability

Recall that in our derivation, the parameter \\(\beta\\) controlled the trade-off between evidence and ignorance. Though it does not appear explicitly in a trained transformer, we can still ask: how sensitive is \\(\hat \rho_{\text{eff}}\\) to perturbations in an artificial temperature parameter \\(\beta\\), evaluated at \\(\beta=1\\) (which recovers the actual attention weights)?

If we define the quantity

$$
\partial \hat \rho := \partial _\beta \hat \rho_{\mathrm{eff}}\Big |_{\beta=1} =\mathrm{Var}_{\hat \pi}( z_i),
\label{eq:d_beta}
$$

(where we have dropped \\(t \\) for simplicity of notation). Using standard exponential-family identities, we can see that this corresponds to a susceptibility to perturbations in temperature, similar to stat mech[^4]. Seeing the behavior of \\(\partial \hat \rho \\) in different contexts can allow one to further characterize a particular head.

This is particularly true when considering work in interpretability and circuits in transformer architectures[^5][^6]. Both Eqs. \eqref{eq:rho_eff} and \eqref{eq:d_beta} would be interesting to track over many different contexts. 

Think of a circuit that is activated in particular contexts, such as identifying which noun in a sentence is the indirect object. One can imagine each context string for that head mapping to a certain point in the \\( (\hat \rho, \partial \hat \rho) \\) plane. When the circuit is activated, the contexts would cluster towards high \\( \hat \rho \\) and low \\( \partial \hat \rho \\) (certain and stable). When not activated it would show low \\( \hat \rho \\) and low \\( \partial \hat \rho \\) (no preference for any past tokens and stable).

The next post shall explore the behaviors of these quantities in the indirect object identification (IOI) circuit in GPT-2. 

## References and Footnotes
[^1]: Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.), Section 11.8 (Chernoff-Stein Lemma). Wiley-Interscience.
[^2]: In practice there are quite a few more bells and whistles when considering multiple attention heads, LayerNorm, etc., but we shall skip over those for simplicity.
[^3]: The mapping from literal token \\(x_i\\) to embedded token \\(h_i\\) is not one to one—as one goes through more attention/MLP layers the information between positions can become more and more mixed.
[^4]: See, e.g., Kardar, M. (2007). *Statistical Physics of Particles*, Ch. 4. Cambridge University Press. In the canonical ensemble, the derivative of a thermodynamic average with respect to temperature yields the variance of the conjugate quantity (the fluctuation-dissipation relation).
[^5]: Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned." *Proceedings of ACL*, 5797--5808. Identifies specialized vs. redundant heads via a confidence metric (average max attention weight).
[^6]: Zhai, S., Likhomanenko, T., Littwin, E., Busbridge, D., Ramapuram, J., Zhang, Y., Gu, J., & Susskind, J. M. (2023). "Stabilizing Transformer Training by Preventing Attention Entropy Collapse." *Proceedings of ICML*. Tracks attention entropy during training and identifies pathological entropy collapse.
