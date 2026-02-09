---
title: "Why Softmax?"
layout: single
toc: true
toc_label: "Contents"
toc_sticky: true
mathjax: true
---

<div class="mathjax">

\[
\pi_i = \frac{e^{\beta z_i}}{\sum_j e^{\beta z_j}}
\]

</div>

Inline test: \( a^2 + b^2 = c^2 \).

Display test:

\[
\pi_i = \frac{e^{\beta z_i}}{\sum_j e^{\beta z_j}}.
\]

## Motivation

Attention weights are usually written as

\[
\pi_i = \frac{e^{\beta z_i}}{\sum_j e^{\beta z_j}}.
\]

But why *this* functional form?

## A KL-constrained view

Let \(u\) be the uniform distribution over admissible indices.
Consider the optimization problem

\[
\max_{\pi \in \Delta} \left[
\sum_i \pi_i z_i
- \frac{1}{\beta} \mathrm{KL}(\pi \| u)
\right].
\]

The solution is

\[
\pi_i^\star \propto e^{\beta z_i},
\]

which recovers softmax.

## Interpretation

The KL term bounds distinguishability from a null hypothesis.
Large \(\beta\) allows sharper routing; small \(\beta\) enforces caution.
