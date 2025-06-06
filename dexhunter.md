---
timezone: UTC+8
---

> 请在上边的 timezone 添加你的当地时区(UTC)，这会有助于你的打卡状态的自动化更新，如果没有添加，默认为北京时间 UTC+8 时区


# 你的名字

1. 自我介绍 dex
2. 你认为你会完成本次残酷学习吗？ 会
3. 你的联系方式（推荐 Telegram） @dexhunt3r

## Notes

<!-- Content_START -->

# 2025.06.01

Reading paper *How far are we from AGI* [1] Section 2-3

* preception
* reasoning
* memory
* **metacognition**
    * self improvement

> the agent’s iterative adaptation via task execution (Le, 2019;
> ang et al., 2023e), 
> code execution (Gao et al., 2020), or feedback from physical simulations (Qian et al.,
> 2024; Xu et al., 2023a).
> Other strategies for self-evolution 
> include prompt adaptation and optimization
> (Wang et al., 2023h; Aksitov et al., 2023), 
> continuous improvement through error identification and selfreflection, 
> and memory retrieval as a mechanism for short- or long-term learning.


[1]: https://openreview.net/pdf?id=H2ZKqfNd0U

# 2025.06.02

Reading paper *How far are we from AGI* [1] Section 4-7

evaluation

* comprehensiveness
* fairness
* efficiency

“What” Do We Evaluate: Existing Evaluation Aspects 

* Open-ended conversation -> multi turn 

## Usecases (sec 7)

* AI for Science Discovery and Research
* Generative Visual Intelligence
    * vllm
* World Models for AGI
* Decentralized AI
    * edge devices
* AI for Coding
    * copilot
* Embodied AI: AI for Robotics
* Human-AI Collaboration

# 2025.06.03

## Reasoning

* [Tree of Thoughts](https://arxiv.org/pdf/2305.10601)

Current shortcoming of LM solving general problems
1. Locally, they do not explore different continuations within a thought process (the branches of the tree)
2. Globally, they do not incorporate any type of planning, lookahead, or backtracking to help evaluate these different
options (the kind of heuristic-guided search that seems characteristic of human problem-solving)

Structure of ToT

1. How to **decompose** the intermediate process into thought steps; 
2. How to **generate** potential thoughts from each state; 
3. How to heuristically **evaluate** states; 
4. What **search** algorithm to use.

Search Algorithm

1. ToT-BFS
2. ToT-DFS

> ToT is like giving GPT-4 a whiteboard 
> and saying “think out loud, explore branches, 
> score them yourself, then pick the best line of reasoning.”


# 2025.06.04

[REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS](https://arxiv.org/pdf/2210.03629)

* reason + act


# 2025.06.05

[LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models](https://arxiv.org/abs/2404.05221)




<!-- Content_END -->
