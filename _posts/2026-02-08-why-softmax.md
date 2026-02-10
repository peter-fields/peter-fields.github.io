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
      - \\(z_i = k_i \cdot q\\) = query-key score
      - \\(\mathcal{O}_t\\) = attention output
      - \\(W_K, W_Q, W_V\\) = learned matrices
      
      **Distributions**
      - \\(\pi_i\\) = attention weight (softmax)
      - \\(u\\) = uniform distribution
      - \\(n\\) = number of scores
      
      **Other**
      - \\(\beta\\) = inverse temperature
      - \\(\rho\\) = KL budget
      - \\(\text{KL}(\pi \| u)\\) = KL divergence
tags: [attention, softmax, hypothesis testing, KL divergence, machine learning, deep learning]
---

## Introduction: the attention mechanism

<!-- The attention mechanism used in large langauge models dynamically updates vecotrized encodings of tokens by allowing for context dependent weighted sums of values to be added to these contextualized tokens.  -->

Consider a stream of tokens (e.g. words from an LLM) to be embedded:

$$
x=\{x_1,x_2,...,x_i,...,x_T \}.
$$

After embedding (and potentially many passes through MLPs and attention heads) we have the contextualized tokens

$$
h_i \in \mathbb R^{d_{\text{model}}}.
$$

The attention mechanism updates this *residual stream* (as it is also called) by calculating three different quantities from learned parameters \\(W_K, W_Q\\) and \\( W_V \\). 

Given the most recent embedded token in the stream, \\(h_t\\), and all tokens before it \\( \\{ h_i :i < t \\} \\), the keys, query, and values are defined as

$$
\begin{equation}
k_i=W_Kh_i
\end{equation}
$$

$$
\begin{equation}
q_t=W_Qh_t
\end{equation}
$$

$$
\begin{equation}
v_i=W_Vh_i
\end{equation}
$$

where \\(q,k \\in \\mathbb{R}^{d_k}\\) and \\( d_k<d_{\text{model}} \\). 

The update to the residual stream at position \\( t \\) (in practice there are quite a few more bells and whistles when considers multiple attention heads, LayerNorm, etc., but we shall skip over those for simplicity) are caluclated as 

$$
h_t^{\text{(new)}} =\mathcal{O}_t+ h_t^{\text{(old)}}
$$

with

$$
\begin{equation}\label{eq:O_t}
\mathcal{O}_t=\sum_i\pi_{i,t} v_i
\end{equation}
$$

$$
\begin{equation}
\pi_{i,t} = \frac{e^{\beta k_i\cdot q_t}}{\sum_j e^{\beta k_j\cdot q_t}},
\end{equation}
$$

 where we identify \\( \pi_i \\) with the \\(\mathrm{softmax}\\) function:

$$
\begin{equation}
\pi_{i,t}=\mathrm{softmax}(\beta k_i\cdot q_t),
\end{equation}
$$

and we have introduced the scalar \\( \beta\\) for later use.

This lends itself to the following interpretation: for any given token at position \\( t \\), the query vector, \\(q_t\\), defines what \\(h_t\\) is "looking for" from previous tokens, and the keys, \\( k_i \\), determine which of the previous tokens get "advertised". 

(Again, the mapping from literal token \\(x_i\\) to embedded token \\(h_i\\) is not one to one—as one goes through more attention/MLP layers the information between positions can become more and more mixed). 

In any case, the query-key pairs define the distribution \\( \pi_i \\) over which the values, \\( v_i \\), are averaged. We can see that this distribution determines what values the attention head should "focus" on. 

This blog post is mainly concerned with a few thoughts I've been having around this question: **why the softmax distribution and not something else?**

I emphasize that this is just the way I like to think about it... not *the* way it should be understood.

## Softmax as hypothesis testing

For simplicity of notation let us drop $t$ and define the query-key score for a given key as 

\begin{equation} \label{eq:z}
z_i=k_i\cdot q.
\end{equation}

We let \\(n\\) denote the numbers of scores. 

Leaving \\(z_i\\) alone for the moment, let us imagine that we had no good reason to prefer one index over another when calculating \\( \mathcal{O}_t\\) from Eq. \eqref{eq:O_t}. The only distribution invariant under permutation of the indices (which is the symmetry that reflects our ignorance) is the uniform distribution, which we denote by \\(u\\).

Of course, we do have reason to prefer some indices over others in our distribution \\( \pi \\), namely the scores \\( z_i \\). We have two competing objectives: create a distribution that maximizes the expected score, \\( \sum_i \pi_i z_i \\) (thus properly weighting our evidence afforded us), but also do not overcommit to any particular index's score beyond what we believe is justifiable given our prior ignorance.

In hypothesis testing, the Kullback-Leibler (KL) divergence is a natural measure of distinguishability from a null hypothesis. The number of samples required to determine that said null hypothesis is false is proportional to \\( \frac{1}{\mathrm{KL}(\pi\\|u)}\\)[^1]. If we are given some "budget" \\( \rho \\), and if the KL exceeds it, then we may say that we have enough evidence to reject the null (uniform) hypothesis. This defines our notion of overcommitment. We write down the hard-constraint optimization problem as 

\begin{equation}
\max_{\pi \in \Delta} 
\sum_i \pi_i z_i - \frac{1}{\beta} \mathrm{KL}(\pi \| u)
\end{equation}

The limit \( \beta \to \infty \) produces hard routing.

$$
\pi_i^\star
= \arg\max_{\pi}
\left[
\sum_i \pi_i z_i
- \frac{1}{\beta}\mathrm{KL}(\pi\|u)
\right].
$$

$$
\pi_i = \frac{e^{\beta z_i}}{\sum_j e^{\beta z_j}}
$$

## Motivation

Attention weights are usually written as

$$
\pi_i = \frac{e^{\beta z_i}}{\sum_j e^{\beta z_j}}.
$$

But why *this* functional form?

## A KL-constrained view

Let \(u\) be the uniform distribution over admissible indices.
Consider the optimization problem

$$

$$

The solution is

\[
\pi_i^\star \propto e^{\beta z_i},
\]

which recovers softmax.

## Interpretation

The KL term bounds distinguishability from a null hypothesis.
Large \(\beta\) allows sharper routing; small \(\beta\) enforces caution.

## References
[^1]: Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.

