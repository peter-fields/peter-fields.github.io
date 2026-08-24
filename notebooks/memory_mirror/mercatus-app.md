---
name: mercatus-app
description: "Mercatus Center 'Future of Scientific Discovery' Emerging Scholar application — program details + submitted answers (research interests, classical liberalism, why applying). The humanities/policy lane from the ZOË/Cluny/Luke Burgis intro."
metadata: 
  node_type: memory
  type: project
  originSessionId: 24e0f58e-a194-4c1e-a0de-757c991eef02
  modified: 2026-08-20T03:55:37.001Z
---

# Mercatus Center — Future of Scientific Discovery Emerging Scholar

**Status:** application prepared (v2 answers below) — shared for records 2026-08-19.
**Origin / lane:** this is the **Mercatus opening from the ZOË/Cluny conference** (Luke Burgis introduced Peter to Mercatus folks; Peter ticked "yes" to a possible DC fellowship). The "AI + public-engagement / humanities / policy" lane realized — see [[zoe-scholarship-app]].

## Position
- **Future of Scientific Discovery Emerging Scholar** (2 positions), Mercatus Center @ George Mason.
- Full-time, **1 year, Arlington VA (in person)**, **Sept 8 2026 – Sept 8 2027**. Salary **$80K–$150K**.
- Part of the **Emerging Scholars Program** (identifies "rising classical liberal thinkers," makes them stronger public communicators). Led by **Rebecca Lowe** (Director of Emerging Scholars). Two assigned mentors each.
- Structure: substantial research project + regular public-facing writing/multimedia; weekly work-in-progress seminar; policy discussion group; 1:1s; **10-week communications training** (writing, branding, public speaking, media, podcasting, design).
- Track themes: institutions/norms around science (funding, regulation, incentives); effects of AI/emerging tools on research; communicating new scientific frontiers; moral/civic implications of accelerating discovery.
- Classical-liberal / market-oriented framing (Mercatus = the well-known GMU free-market think tank). Cover letter must state interest in Mercatus's mission.

## ⚠️ Form character restriction
Answers must NOT contain: `! @ $ ^ & - _ = + { [ } ] \ | / ? ; " ' < >` — **no hyphens, no em-dashes, no special chars.** (This is why the answers below read without Peter's usual em-dashes; he complied.)

## Submitted answers (v2, verbatim — no-special-char constraint applied)

**Q1 — Current research interests and strengths?**
> I am interested in researching fundamentals of AI systems, particularly Large Language Models, what they are and how they work. I recently received my PhD from the University of Chicago, where I studied statistical physics, with applications to machine learning and biology.
>
> Statistical physics was developed to understand how the microscopic properties of particles related to macroscopic thermodynamic observables. Think, for example, of how air molecule velocities may be related to the temperature and pressure in a room.
>
> More recently, the mathematical apparatus of statistical physics has been applied to other areas concerned with collective behaviors, such as protein science and artificial neural networks. I am well acquainted with this work, and used these ideas to study machine learning techniques for protein modeling and design: specifically how current models generalize beyond patterns not seen in training data, allowing for synthesis of functional proteins never seen in nature.
>
> Two related questions I am interested in with regards to AI: first is the Computation in Superposition (CiS) hypothesis (arxiv:2408.05451), second is how to characterize the geometry of token embedding space. LLMs embed each token into a vector space, turning each token into a list of numbers. The vector associated with each token is updated and compared against other vectors as the network processes information. CiS is a proposed mechanism for part of this information processing. Understanding the geometry of the vector space formalizes a notion of distance and orientation between vectors, further elucidating how the LLM utilizes and processes information.

**Q2 — Are you a classical liberal?**
> I am very much aligned with classical liberalism's commitment to liberty and the importance of preserving this right when considering how a state ought to be formed. Such a commitment resonates with my high esteem for America and Americans, my fascination with our past, and my optimism as regards our future.
>
> America is a nation "conceived in Liberty." I do not take this for granted. Such a commitment is implicated in many aspects of America I cherish deeply. The optimism as regards one's ability to improve one's condition in life. The willingness to work hard. The consideration to keep state power diffuse, thereby protecting against tyranny. The dedication to ensuring personal freedom. The guarantee of due process and the rule of law. The ability to steer one's nation through democracy.

**Q3 — Why are you applying for this position?**
> Participating in the Emerging Scholars program at the Mercatus Center would allow me to further my technical research while pursuing public engagement on AI related issues. I am currently interested in understanding what AI is and is not. To bring this understanding to the public is greatly appealing to me. I enjoy explaining and teaching, and developing my writing skills beyond the merely technical is a long term goal.
>
> The interdisciplinary environment at Mercatus is a further draw. AI elicits more than simply technical questions. With it comes philosophical inquiry, economic impacts, and policy debates. I expect to broaden the horizon of my thoughts on such matters as I embed myself in the multifaceted intellectual environment at Mercatus. And I expect to bring technical expertise to bear on such questions beyond my own specialties.

## Research Interests one-pager (submitted, verbatim — reflowed from PDF wrapping)

*(Accessible / public-facing version of his research agenda: CiS + Compressed Computation + residual-stream geometry. Reusable for other public-engagement contexts.)*

> In Large Language Models, each word (or token) is encoded as a high-dimensional vector, or more simply, a list of numbers. Each list associated with each token is updated and modified as it travels through the neural network, and is known as the residual stream. The final residual stream vector at the end of the neural network is used to predict the next token.
>
> Residual stream vectors carry information and transfer it among themselves. The Computation in Superposition hypothesis (CiS) is concerned with how this information is processed as it goes through the network (arxiv:2408.05451). Think, for example, if there may be a clever way to design a circuit that can compute more logical operations (e.g. AND, OR) then there are functional operators in the network. Currently, researchers think LLMs might be utilizing CiS, but it is unclear how or if at all.
>
> An interesting alternate hypothesis I would like to explore is Compressed Computation (lesswrong.com/posts/ZxFchCFJFcgysYsT9). Under this hypothesis, the appearance of CiS may arise due to correlations among the variables being processed. Essentially, when correlated, the effective number of variables that need to be processed is smaller. Furthermore, and more speculatively, this current line of research could benefit from understanding this problem through the lens of information theory, which has been historically successful at elucidating how bits of information are optimally encoded and passed through noisy channels.
>
> As the residual stream is processed, each vector is repeatedly compared against others. How they are compared may create a notion of a geometric space within which these vectors live. Most current research of the residual stream regards this space as Euclidean, that is, amenable to a simple notion of distance between each vector. However, LLM architectures invoke a much richer notion of distance and geometry to measure vectors against each other, one that is learned from data. Understanding this geometry may lead to a better understanding of how LLMs process language.
>
> As an example, persona vectors seek to find directions in the residual stream vector space that correlate with certain behaviors (arXiv:2507.21509). One direction is known as the "evil vector," as it has been shown to correlate with bad, undesirable text from the LLM. What if an embedded token vector is far from an "evil" persona vector in a Euclidean sense, but actually close in a model-native geometric sense? This calls into question the utility of such probes to understand how the residual stream vector space correlates with different genres of text.
>
> The actual residual stream vector comparisons and computations are done by what is known as the attention mechanism. In any given LLM, hundreds to thousands of attention heads invoke this mechanism as the residual stream is processed layer-by-layer. I would like to investigate whether these attention heads have any key similarities (or differences) in how they make their vector comparisons, and if so, how this may inform a notion of a model-native geometry for the residual stream vector space.

*(Note: this one-pager is the public-facing framing of the [[idea_qk_metric]] W_QK=G+B work — same "residual stream isn't Euclidean, the model uses a learned geometry" thesis, made accessible. The persona-vector / "evil vector" example is a nice reusable hook.)*

Related: [[zoe-scholarship-app]], [[index-apps-and-projects]], [[user_profile_faith_philosophy]], [[idea_qk_metric]], [[compressed-computation-project]]
