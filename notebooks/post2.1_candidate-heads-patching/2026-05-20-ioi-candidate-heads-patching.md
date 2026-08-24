---
title: "Follow-up: Patching the Candidate Heads (L7H11, L8H1, L8H11)"
layout: single
author_profile: false
toc: true
toc_label: "Contents"
toc_sticky: true
mathjax: true
tags: [mechanistic-interpretability, attention, transformers, IOI-circuit, activation-patching]
excerpt: "Brief follow-up to the attention-diagnostics post: causal verification of three heads (L7H11, L8H1, L8H11) flagged by the C_diff analysis as candidate unlabeled circuit elements."
---

## Motivation

In the [previous post]({{ site.baseurl }}{% post_url 2026-02-24-attention-diagnostics %}), the contrast \\(C\_{\text{IOI}} - C\_{\text{non-IOI}}\\) between KL correlation matrices flagged three layer-7/8 heads — **L7H11**, **L8H1**, **L8H11** — as anti-correlated with the core name movers (L9H6, L9H9) specifically on IOI prompts. None appear in the Wang et al. (2022) circuit; their swings are among the largest in the model:

| Head pair | r (non-IOI) | r (IOI) | Δ |
|-----------|------------:|--------:|--:|
| L8H1  ↔ L9H6 | +0.543 | −0.731 | **−1.274** |
| L8H11 ↔ L9H6 | +0.526 | −0.636 | −1.162 |
| L7H11 ↔ L9H6 | +0.679 | −0.477 | −1.156 |
| L8H1  ↔ L9H9 | +0.437 | −0.725 | −1.162 |
| L8H11 ↔ L9H9 | +0.384 | −0.681 | −1.065 |
| L7H11 ↔ L9H9 | +0.521 | −0.410 | −0.932 |

Observational statistics can flag candidates but can't prove necessity. This post does the causal check via activation patching.

## Setup

*(to be filled in: prompt pairs, patching protocol, metric — logit diff)*

## Results

*(to be filled in)*

## Interpretation

*(to be filled in)*

## Limitations

*(to be filled in)*
