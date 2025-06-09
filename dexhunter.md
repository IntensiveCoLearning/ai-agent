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

Question

why it's innovative?

Answer

That's a fair and important question. It's easy to look at the `Thought -> Action -> Observation` loop and think, "Isn't this just a simple script?"

The innovation of ReAct isn't in creating a complex new algorithm, but in demonstrating that a **simple, human-intuitive prompting structure can unlock sophisticated, dynamic behaviors in large language models.**

Here’s a breakdown of why it was considered innovative at the time it was published:

### 1. It Solved the Grounding Problem for Chain-of-Thought (CoT)

Before ReAct, the big innovation was **Chain-of-Thought (CoT) prompting**. You'd ask an LLM a complex question, and by simply adding "Let's think step-by-step," the model would generate a reasoning trace and arrive at a more accurate answer.

**The Problem with CoT:** The reasoning was entirely "internal." The LLM was thinking in a closed box, relying only on the knowledge baked into its parameters during training. This led to two major flaws:
*   **Fact Hallucination:** The model would confidently state incorrect facts as part of its reasoning (e.g., "The Eiffel Tower is in Rome...").
*   **Error Propagation:** One small mistake early in the reasoning chain would derail the entire process, leading to a wrong final answer.

**ReAct's Innovation:** ReAct directly addressed this by **grounding the reasoning in an external reality**. By interleaving `Action` and `Observation` steps, the model is forced to:
*   **Verify its own thoughts:** Instead of just *assuming* a fact, it generates an action to *look it up*.
*   **React to new information:** The observations from the environment become part of the prompt history, allowing the model to dynamically correct its course. It's no longer a static, one-shot reasoning process.

This was a major conceptual leap from "let's have the model think" to "let's have the model think, act, and learn from what happens."

### 2. It Made LLM-based Agents More Robust and Strategic

Before ReAct, early work on LLM-based agents often prompted the model to generate a **full plan or a sequence of actions upfront**.

**The Problem with this approach:** The world is unpredictable. An action might fail, or the environment might not be in the state the model expected. A static, pre-generated plan is brittle and cannot handle exceptions. Other methods relied on complex reinforcement learning or imitation learning setups requiring massive amounts of training data.

**ReAct's Innovation:** ReAct enables **dynamic, reactive planning**. Because the agent re-evaluates its strategy after *every single action*, it can:
*   **Decompose complex goals on the fly:** It only needs to figure out the very next logical step, not the entire solution.
*   **Handle errors gracefully:** If a search for "Front Row" fails, the ReAct agent can *think*, "Hmm, that didn't work. The search results suggest 'Front Row (software)'. I should try that instead." An agent with a static plan would have been stuck.
*   **Achieve impressive performance with almost no training:** The most stunning result from the paper is that ReAct achieved state-of-the-art performance on complex benchmarks like ALFWorld and WebShop with just **one or two examples** in the prompt. This demonstrated that the latent planning and reasoning capabilities of the LLM could be effectively channeled through this simple interactive loop, bypassing the need for huge, task-specific datasets.

### 3. It Created a Highly Interpretable and Controllable Agent Framework

Why does an agent do what it does? This is a fundamental question in AI.

**The Problem with other methods:**
*   **Deep Reinforcement Learning:** The policy is often a black box. It's hard to know *why* the agent chose a specific action.
*   **Simple Action-Generation LLMs:** The model just spits out an action. You don't see the reasoning behind it.

**ReAct's Innovation:** The `Thought` step provides a **natural language audit trail** for the agent's behavior. A human can read the thoughts and understand the agent's "intentions" and "beliefs" at every step. This is huge for:
*   **Debugging:** You can see exactly where the agent's reasoning went wrong.
*   **Trust:** The system is no longer a black box.
*   **Human-in-the-loop control:** As shown in the paper's appendix, a human can literally **edit the agent's thoughts** to correct its behavior mid-task. This is a powerful and intuitive way to collaborate with an AI agent.


 the **Act** is a structured command that the agent's external code can parse and execute.

# 2025.06.05

[LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models](https://arxiv.org/abs/2404.05221)

This paper introduces two primary innovations that work together to advance the study of step-by-step reasoning in Large Language Models (LLMs).

Here’s a breakdown of what's new and why the **LLM Reasoners** library, in particular, is an innovative approach.

---

### The Two Core Problems and Their Novel Solutions

The paper identifies two major bottlenecks holding back research in LLM reasoning:

1.  **The Evaluation Problem:** How do we know if a reasoning chain is actually correct? Just checking the final answer is unreliable, as LLMs can arrive at the right answer through flawed logic (a "false positive").
2.  **The Unification Problem:** Researchers have proposed many reasoning methods (Chain-of-Thought, Tree-of-Thoughts, RAP), but they are all described and implemented differently, making it hard to compare them systematically or build upon them.

The paper's novelty lies in its two corresponding solutions:

#### 1. AutoRace: A New, Automated Evaluation Method

Previous methods for evaluating the *process* of reasoning were either expensive (requiring human annotators) or rigid (using fixed, human-written prompts that don't adapt well to different tasks).

**What's New about AutoRace?**
*   **Fully Automated:** It requires no human effort (like writing few-shot examples or detailed prompts) for a new task.
*   **Adaptive and Task-Specific:** Instead of using a generic checklist, AutoRace *learns* what to look for. It automatically generates a list of evaluation criteria tailored to a specific domain (e.g., math, commonsense). It does this by:
    1.  **Collecting Errors:** It finds examples where an LLM produces a wrong final answer.
    2.  **Summarizing Mistakes:** It uses a powerful LLM (like GPT-4) to analyze these incorrect reasoning chains and summarize the common failure patterns into a general criteria list (e.g., "Ensure calculations are correct," "Comprehend all details of the problem statement").
    3.  **Evaluating:** It then uses this auto-generated, task-specific list to guide its evaluation of new reasoning chains.

This makes evaluation more accurate, scalable, and adaptable than previous approaches.

---

#### 2. LLM Reasoners: A New, Unified Library and Formulation

This is arguably the most significant innovation for the research community. Before this paper, comparing different reasoning algorithms was like comparing apples and oranges. Each was a monolithic piece of code with its own logic.

**What's New and Innovative about LLM Reasoners?**

The core innovation is **reframing all step-by-step reasoning algorithms as a single, unified search problem.**

As shown in Figure 1(d) of the paper, they formulate any reasoning process as an attempt to find the best sequence of steps by optimizing an objective composed of three modular, interchangeable components:

1.  **World Model:** Defines the "state" of the problem and how it changes after each reasoning step (an "action").
    *   *Simple case (CoT):* The state is just the text generated so far.
    *   *Complex case (RAP):* The state is a structured set of known facts and variables.
    *   *Innovation:* This component explicitly separates tracking the problem's state from the search process.

2.  **Reward Function:** Assigns a score to each reasoning step, guiding the search toward good paths.
    *   *Simple case (CoT):* The reward is implicitly the likelihood of the next token.
    *   *Complex case (ToT, RAP):* The reward can be a score from the LLM evaluating the step ("is this step promising?") or a heuristic.
    *   *Innovation:* It makes the "guiding signal" for reasoning an explicit, pluggable module.

3.  **Search Algorithm:** The strategy used to explore the space of possible reasoning chains.
    *   *Simple case (CoT):* A simple **Greedy Search** (just takes the most likely next step).
    *   *Complex case (ToT):* **Breadth-First or Depth-First Search** to build a tree.
    *   *Complex case (RAP):* **Monte-Carlo Tree Search (MCTS)** for more efficient exploration.
    *   *Innovation:* It treats the exploration strategy as an independent choice, separate from the reward or world model.

**Why is this so Innovative?**

*   **Conceptual Clarity:** It provides a common vocabulary. Now, we can describe Chain-of-Thought not just as "prompting" but as **"Greedy Search using a likelihood-based reward and a simple text-based world model."** This allows for precise, systematic comparisons.
*   **Modularity and Reusability:** It turns reasoning algorithm design into playing with building blocks. A researcher can now easily ask and implement new ideas:
    *   "What if I use the Tree-of-Thoughts search algorithm but with the reward function from RAP?"
    *   "I have a new idea for a world model for chemistry problems. I can just write that class and plug it into the existing MCTS search algorithm."
*   **Lowers the Barrier to Entry:** Instead of writing a complex reasoning system from scratch, researchers can now inherit from the library's base classes and only implement the part they want to innovate on. This drastically speeds up development and experimentation.
*   **Enables Systematic Analysis:** By using this library, the authors were able to conduct a fair, controlled analysis of different components, leading to insights like "search breadth is often more important than depth" and "an explicit world model is crucial for embodied tasks."

In summary, **LLM Reasoners is innovative because it moves the field from a collection of ad-hoc, monolithic algorithms to a principled, modular framework.** This transforms the process of creating and analyzing reasoning methods from an art into a more systematic science, paving the way for faster and more rigorous progress.

# 2025.06.06

[HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831)

# 2025.06.07

[Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)

# 2025.06.08

[LLM+P: Empowering Large Language Models with Optimal Planning Proficiency](https://arxiv.org/abs/2304.11477)

# 2025.06.09

[Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/abs/2310.04406)

<!-- Content_END -->
