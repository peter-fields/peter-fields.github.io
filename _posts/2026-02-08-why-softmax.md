---
title: "Why Softmax?"
layout: single
toc: true
toc_label: "Contents"
toc_sticky: true
mathjax: true
---

## Introduction

The attention mechanism used in large langauge models dynamically updates vecotrized encodings of tokens by allowing for context dependent weighted sums of values to be added to these contextualized tokens. 

Consider a stream of tokens (before embedding):
$$
x=\{x_1,x_2,...,x_i,...,x_T \}.
$$
After embedding (and potentially many passes through MLPs and attention heads) we have the contextualized tokens
$$
h_i \in \mathbb R^{d_{\text{model}}}.
$$
The attention mechanism updates this *residual stream* (as it is also called) by calculating three different quantities from learned parameters $W_K, W_Q$ and $W_V$. 

Consider the most recent embedded token in the stream, $h_t$, and all tokens before it $\{ h_i :i < t\}$. The keys, query, and values are defined as
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


where $q,k \in \mathbb{R}^{d_k}$ and $d_k<d_{\text{model}}$. 

The update to the residual stream at position $t$ (in practice there are quite a few more bells and whistles when considers multiple attention heads, LayerNorm, etc., but we shall skip over those for simplicity) are caluclated as 
$$
h_t^{\text{(new)}} =\mathcal{O}_t+ h_t^{\text{(old)}}
$$
with
$$
\begin{equation}
\mathcal{O}_t=\sum_i\pi_{i,t} v_i
\end{equation}
$$
$$
\begin{equation}
\pi_{i,t} = \frac{e^{\beta k_i\cdot q_t}}{\sum_j e^{\beta k_j\cdot q_t}},
\end{equation}
$$
 where we identify $\pi_i$ with the $\mathrm{softmax}$ function:
$$
\begin{equation}
\pi_{i,t}=\mathrm{softmax}(\beta k_i\cdot q_t),
\end{equation}
$$
and we have introduced the scalar $\beta$ for later use.

This lends itself to the following interpretation: for any given token at position $t$, the query vector, $q_t$, defines what $h_t$ is "looking for" from previous tokens, and the keys, $k_i$, determine which of the previous tokens get "advertised". (Again, the mapping from literal token $x_i$ to embedded token $h_i$ is not one to one—as one goes through more attention/MLP layers the information between positions can become more and more mixed). 

In any case, the query-key pairs define the distribution $\pi_i$ over which the values, $v_i$, are averaged. We can see that this distribution determines what values the attention head should "focus" on. 

This blog post is mainly concerned with a few thoughts I've been having around this question: **why the softmax distribution and not something else?**

I emphasize that this is just the way I like to think about it... not *the* way it should be understood.

## Softmax as hypothesis testing

For simplicity of notation let us drop $t$ and define the query-key score for a given key as 

$$
\begin{equation}
z_i=k_i\cdot q.
\end{equation}
$$ (eq)

We let $n$ denote the numbers of scores. 

Leaving $z_i$ alone for the moment, let us imagine that we had no good reason to prefer one index over another when calculating $   \mathcal{O}_t$ from Eq. \@ref{O_t}

See Eq.[](#eq)

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
\max_{\pi \in \Delta} \left[
\sum_i \pi_i z_i
- \frac{1}{\beta} \mathrm{KL}(\pi \| u)
\right].
$$

The solution is

\[
\pi_i^\star \propto e^{\beta z_i},
\]

which recovers softmax.

## Interpretation

The KL term bounds distinguishability from a null hypothesis.
Large \(\beta\) allows sharper routing; small \(\beta\) enforces caution.
