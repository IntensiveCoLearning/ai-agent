---
timezone: UTC+8
---


# fffuuuming

1. 自我介绍：資工碩班，想踏入web3
2. 你认为你会完成本次残酷学习吗？盡力
3. 你的联系方式（推荐 Telegram）@fffuuuming

## Notes

<!-- Content_START -->

### 2025.06.01
- Agent: anything that can be viewed as 
    - **perceiving its environment** through sensors
    - **acting upon that environment** through actuators
- LLM agent: use **language**, e.g. **generating tokens** to do the above, although perception of other input modalities is possible, language is still important in **reasoning** and **communication**

LLM agent workflow:
![截圖 2025-06-02 凌晨12.34.35](https://hackmd.io/_uploads/rkMVbZqzlg.png)
Example:
![截圖 2025-06-02 凌晨12.33.59](https://hackmd.io/_uploads/SksB-Z5Mee.png)
There still remain challenges for LLM agent such as **self-reflection (meta reasoning), state inferences, replanning, synthetic data, internalized search...**, but thanks to the use of language, both of its expressiveness and adaptivity are high, and language-based inferences makes its reasoning flexible (but fuzzy)

### 2025.06.02
---
**ReACT**: A Prompting strategy that **mimic human thinking which alternates between thinking and doing**, by interleaving **reasoning traces** and **action steps** in LLMs, general workflow includes:
1. Reason about a problem (via thoughts)
2. Take actions (e.g., interact with tools, environments)
3. Use the observations to guide further reasoning.

By doing this, it can
- Reduce hallucinations by grounding reasoning in environment feedback
- Good interpretability (just like human-thinking)
- Good Error Recovery: when encounter poor search results, it can reason about alternate paths and adapte its plan dynamically
- Good sample efficiency

Limitation:
- Repetitive reasoning loops if decoding gets stuck.
- Sensitive to search failures, especially with weak APIs.
- Prompt design and trajectory annotation can be labor-intensive.
- Performance drops in smaller models or zero-shot settings compared to fine-tuned baselines.

Simple comparison


| Feature | CoT |  act-only   | ReAct |
| -------- | -------- | --- | -------- |
|  Input        | Natural language query         |  Environment observation   | Query + observation         |
|  Output        | Thought sequences only         |Action commands only     | Alternating thought & action         |
| Memory/Plan Tracking         |Internal memory only          | Observation-based only    | Thought + observation         |
| Sample Efficiency         | Few-shot, but limited generalization         |Depends on demonstrations     |      Few-shot effective, adaptable    |
| Error Recovery         | Poor (error propagates)         | Poor (no reflection)    | Good (reason over feedback)         |
---
**Question**
1. what does sub-optimal greedy decoding procedure mean ? and how does decoding method affect the reasoning

### 2025.06.04
---
Motivation:
- CoT: Linear sequence — from input → [thought₁, thought₂, …, thoughtₙ] → answer.
    - left-to-right generation without exploration.
    - If any step fails early (bad thought), the rest of the chain collapses.
- CoT-SC: Samples multiple CoT chains and chooses the most frequent answer
    - Still linear: No branching or backtracking -> **non-strategic**

[**Tree of Thought**](https://arxiv.org/abs/2305.10601):  generalizes CoT by introducing **structured exploration** and **planning over thoughts**.
- Workflow
    1.	**Thought Decomposition**: Define meaningful intermediate steps.
	2.	**Thought Generation**: Generate multiple candidate thoughts at each step.
	3.	**Evaluation**: Heuristically rate or vote on thoughts using the LM itself.
	4.	Search Strategy:
	    - **BFS**: Explore top-k best thoughts at each level.
	    - **DFS**: Follow promising path, backtrack if stuck.
	5.	**Final Answer**: Choose best solution from the search tree.
- Tree structure: 
    - Nodes: states, which is a partial solution (input + current thoughts).
    - Branches: possible next thoughts.

<!-- Content_END -->
