---
title: "Attention Diagnostics: Testing KL and Susceptibility on the IOI Circuit"
layout: single
author_profile: false
toc: true
toc_label: "Contents"
toc_sticky: true
mathjax: true
sidebar:
  - title: "Notation"
    text: |
      **Attention**<br>
      \\(\pi\\) — attention weights<br>
      \\(z\\) — pre-softmax scores<br>
      \\(n\\) — sequence length<br>
      \\(u = 1/n\\) — uniform distribution<br>
      <br>
      **Diagnostics**<br>
      \\(\hat{\rho}\_{\text{eff}}\\) — \\(\text{KL}(\hat{\pi} \| u)\\)<br>
      \\(\chi\\) — \\(\text{Var}\_{\hat{\pi}}(\log \hat{\pi}) / (\log n)^2\\)<br>
      \\(\Delta\text{KL}\\) — shift between conditions<br>
      \\(\Delta\chi\\) — shift between conditions<br>
tags: [mechanistic-interpretability, attention, transformers, IOI-circuit, diagnostics]
excerpt: ""
---