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

### 2025.06.05
LLM+P
---
Motivation:

- LLMs are excellent at **linguistic competence**: generating plausible, fluent text, but struggle with **functional competence**, especially on tasks requiring **long-horizon reasoning** like symbolic planning.
- LLMs do not internally reason with formal models of the world (e.g., they don’t “know” physics, constraints, or state transitions).
- Instead of retraining or fine-tuning LLMs LLM+P **leverages classical planners**, which are built to solve such problems optimally and reliably—and **using LLMs for natural language translation between humans and planners**
    > LLM is an **interface** with the classical planner.

Mechanism:
![截圖 2025-06-06 凌晨2.20.07](https://hackmd.io/_uploads/S1sygP17eg.png)
- **Context**: Example problem & PDDL for in-context learning
- **Domain PDDL**: Provides a lifted representation of the **underlying rules of the world**, including a set of predicates that define the **state space S** and the **actions (i.e., A )** with their **preconditions** and **effects (i.e., the transition function f )**
- **Problem PDDL**: provides a list of objects to ground the domain, the problem’s initial state and goal conditions which comes from original natural language prompt (Problem)

### 2025.06.06
[LATS: general framework unifying reasoning, acting, and planning in LLMs](https://arxiv.org/abs/2310.04406)

A combination of ReAct + ToT, with advanced **Monte Carlo Tree Search (MCTS)**
- Searches over combinations of actions and reasoning.
- Uses feedback from the environment.
- **Self-reflects** on failures to improve future attempts.
- Deliberately plans using **MCTS** with **LM value scoring**.

### 2025.06.08
[Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)
---
Language Models as **Probabilistic Production Systems**:
- Traditional production systems use rule like `"X Y Z → X W Z"` -> **fixed**, **rule based**
- LLM defines a **probability distribution** over which productions to select when presented with input X, yielding a distribution `P(Yi|X)` over possible completions -> probabilistic **production systems that sample a possible completion each time they are called**, e.g., `X ∼∼▸ X Y`
    - opaqueness: LLMs consist of billions of uninterpretable parameters with inherent randomness from their probabilistic formulation
    - advantage: scale and pre-training

Prompt Engineering as **Control Flow**
- In traditional production system, control flow chooses **which rule to apply**
- LLMs use prompt engineering to influence the behavior of the model.

Towards **cognitive language agents**
Cognitive agents **place LLMs in a loop with memory and world interaction** — not just one-way text generation.

---
Cognitive Language Agents using a structured framework with 
1. Memory
2. Action Space
3. Decision making

| Memory Type | What it stores | Example Use |
| -------- | -------- | -------- |
| Working  | Temporary task state         | Goals, latest inputs, LLM outputs         |
| Episodic | Past experiences         | What happened before         |
| Semantic | General knowledge         | World facts, inferred rules         |
| Procedural| How to do things     | Code, prompt templates, LLM weights     |

**Action Space**
- **External** (Grounding): Interact with outside world (E.g., move robot arm, click webpage button, talk to a user)
    - Physical environments
    - Dialogue with humans or other agents
    - Digital environments
- **Internal**
    - Retrieval: Read from long-term memory
	- Reasoning: Use LLM to think and update working memory
	- Learning: process the contents of working memory to generate new information and writes back to working memory, reasoning can be used to support learning (by writing the results into long-term memory) or decision-making (by using the results as additional context for subsequent LLM calls).

**Decision making**
In each cycle, program code defines a sequence of reasoning and retrieval actions to propose and evaluate alternatives (**planning stage**), then executes the selected action (**execution stage**) – then the cycle loops again.
- Planning stage:
    - Proposal: use reasoning / retrieval to sample one / more external grounding actions from the LLM
    - Evaluation: assigns a value to proposed actions, based on heuristic rules / LLM values / LLM reasoning...
    - Selection: selects one to execute or rejects them and loops back to the proposal step
- Execution: executing the relevant procedures from the agent’s source code, it can be either external grounding action (e.g., an API call) or an internal learning action (e.g. write to episodic memory). an observation can be made from the environment, providing feedback from the agent’s action, and the cycle loops again.
---
A simple comparison of previous work based on CoALA framework

🔍 Agent Summary Table
| Agent               | Memory Used                                 | Internal Actions                  | External Actions       | Decision-Making Strategy                          | Notable Characteristics                     |
|---------------------|----------------------------------------------|-----------------------------------|------------------------|---------------------------------------------------|----------------------------------------------|
| **SayCan**          | ✅ Procedural (LLM + value function)         | ❌ None                           | ✅ Physical robot actions | Evaluate all possible actions using LLM + value net | LLM used as evaluator, not planner           |
| **ReAct**           | ❌ None                                       | ✅ Reasoning                      | ✅ Digital (e.g., API)   | Single LLM step: think → act                       | Simplest reasoning+acting loop               |
| **Voyager**         | ✅ Procedural + Episodic + Semantic          | ✅ Reason, Retrieve, Learn        | ✅ Digital (Minecraft)  | Multi-step loop with goal checking & code updates | Learns new skills over time                  |
| **Generative Agents** | ✅ Episodic + Semantic                      | ✅ Reason, Retrieve, Learn        | ✅ Social / digital env | Plans daily schedule, adapts plans on the fly     | Social simulation + memory reflection        |
| **Tree of Thoughts** | ❌ None                                       | ✅ Reasoning                      | ✅ Final output only     | Tree search (propose → evaluate → select)         | Structured deliberation, no memory           |

---
🧭 Feature Comparison Table

| Dimension           | SayCan | ReAct | Voyager | Generative Agents | Tree of Thoughts |
|--------------------|--------|-------|---------|-------------------|------------------|
| Long-term Memory   | ✅     | ❌    | ✅      | ✅                | ❌               |
| Internal Actions   | ❌     | ✅    | ✅      | ✅                | ✅               |
| External Actions   | ✅     | ✅    | ✅      | ✅                | ✅ (final output)|
| Learning           | ❌     | ❌    | ✅      | ✅                | ❌               |
| Structured Planning| ❌     | ❌    | ✅      | ✅                | ✅               |

### 2025.06.09
[TOOLLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789)
---
**Motivation**
- Current open source LLMs (e.g., LLaMA) limited using external tools (APIs) to fulfill human instructions. The reason is that **current instruction tuning largely focuses on basic language tasks but ignores the tool-use domain**.
- Even with previous work on building instruction tuning data for tool use, they fail to fully stimulate the tool-use capabilities within LLMs and have inherent limitations: **limitTeodol APIs**, **constrained scenario** (instructions that only involve one single tool), and **inferior planning and reasoning:** (error propagation, exploration)

**ToolLLM: a general tool-use framework**
- Data construction: **ToolBench**
- Model training: **ToolLLaMA**
- Evaluation: **ToolEval**

Framework overview
![截圖 2025-06-10 凌晨12.16.46](https://hackmd.io/_uploads/HkSZFK47xe.png)

**Dataset Construction**
1. **API collection**: Using **RapidAPI**
    - All APIs inside can be classified into 49 **coarse-grained categories** or **fine-grained categorization collections**
    - Since each tool may be composed of multiple APIs, for each tools, record its important metadata and all of its available APIs, then for each API, collect all info of the API, which serves as a valuable resource for LLMs to understand and effectively use the APIs
    - Perform filtering to retain high-quality tool
2. **Instruction Generation**
Maintain two properties: **diversity** & **multi-tool usage** by sampling different combinations of APIs and craft various instructions that involve them.
    - Given total API set $\mathbb{S}_{API}$, sample a subset $\mathbb{S}^{sub}_{N}$ = $\{API_1 , · · · , API_N\}$
    - Prompt ChatGPT to understand the functionalities
of these APIs and then generate **possible instructions** and **relevant APIs** -> **(instruction, relevant API) pair** will be used for **training the API retriever**
        - **Inst∗** involve APIs in $\mathbb{S}^{sub}_{N}$
        -  $\mathbb{S}^{rel}_∗ \subset\mathbb{S}^{sub}_N$) -> $\{[\mathbb{S}^{rel}_1, Inst_1],· · ·, [\mathbb{S}^{rel}_{N^{'}}, Inst_{N^{'}}]\}$
    - Sampling Strategies:
        - **Single-tookl instruction (I1)**: iterate over each tool and generate instructions for its APIs
        - **multi-tool instructions**: randomly select 2-5 tools from the same category / collection and sample at most 3 APIs from each tool to generate the instructions -> **intra-category (I2)**, **intra-collection (I3)**
    - filter those with the hallucinated relevant APIs by assessing whether they exist in $\mathbb{S}^{sub}_{N}$
3. **Solution Path Annotation**
    - **Multi-round conversation**:
Given **Inst∗**, we prompt ChatGPT to search for a valid action sequence $\{a_1 , · · · , a_N\}$ -> $ChatGPT(a_t|{a_1, r_1, · · · , a_{t−1}, r_{t−1}}, Inst_∗)$
        - $r_∗$: real API response.
        - $a_t$: `“Thought: · · · , API Name: · · · , Parameters: · · · ”`
    - **GPT's function call feature**: Treat each API as a special function and feed its API documentation into ChatGPT’s function field
        - For each **Inst∗**, feed all the sampled APIs $\mathbb{S}^{sub}_{N}$ to ChatGPT’s as available N functions
        - Define two functions `Finish with Final Answer` and `Finish by Giving Up` to finish an action sequence
    - **DFDST**: To solve the **error propagation** & **limited exploration**
        - allows the model to assess different reasoning paths and choose to either **(1) proceed along a promising path** or **(2) abandon an existing node by calling the “Finish by Giving Up” function and expand a new node**
        - During node expansion, to diversify the child nodes and expand the search space, we **prompt ChatGPT with the information of the previously generated nodes and explicitly encourage the model to generate a distinct node**

### 2025.06.10
[Gorilla: Large Language Model Connected with Massive APIs](https://openreview.net/forum?id=tBRNC6YemY)
---
General Framework
![截圖 2025-06-10 下午5.59.48](https://hackmd.io/_uploads/HJ6QGKrXgx.png)

**APIBench**: using HuggingFace, Torch Hub, and TensorFlow Hub
1. **API Documentation**: Aftere collection, convert the model cards for each of these 1,645 API calls into a JSON object for **better generalization** beyond API calls within the ML domain, to other domains, fields described below:
    ```
    {domain, framework, functionality, api_name, api_call, api_arguments, environment_requirements, example_code, performance, description}
    ```            
2. **Instruction Generation**: self-instruct paradigm
    -  6 hand-written example pairs (**instruction + API call**) for each of the 3 API sources for **in-context learning**
    -  For each of the above 1,645 real API entries, ask GPT-4 to generate 10 new instruction + API pairs, using the examples as templates.
    -  No using actual name of the API while writing instructions
3. **API Call with Constraints**
    - Incorporate instructions that have constraints (e.g. `parameter size`, ` lower bound on accurac` for ML-based API calls) in training dataset.
    - ex: Invoke an image classification model that uses less than 10M parameters, but maintains an ImageNet accuracy of at least 70%
    -  Not only must the LLM understand the user’s functional description, but it also needs to **reason about the various constraints embedded within the request**

**Gorilla**
Use above {instruction, API} pairs
- **Retriever-Aware training (RAT)**
Since retrieved documentation is not necessarily accurate -> **RAT teach the LLM to ‘judge’ the retriever at inference time**, if retrieved API document
    - relevant to the user’s prompt: use the API documentation to respond to the user’s question
    - isn't relevant: trained model isn't distracted by irrelevant context. LLM then **relies on the domain-specific knowledge baked-in during RAT training**, to provide the user with the relevant API
- **Gorilla Inference**
In retrieval mode, the retriever first retrieves the most up-to-date API documentation stored in the API Database and **concatenat it to the user prompt** along with the message “Use this API documentation for reference, **no further prompt tuning**
Ex :
    ```
    “Use this API documentation for reference: <retrieved_API_doc_JSON>”
    ```
- **AST Sub-Tree Matching**
![截圖 2025-06-10 晚上9.35.28](https://hackmd.io/_uploads/rkGhEnBXle.png)

### 2025.06.11
[Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403)
---

**Motivation**:
In real world, queries complexity may vary, so it's not suitable to just use either non-retrieval, single-step, or multi-steps retrieval retrieval-augmented LLMs

**Solution**:
Propose **Adaptive-RAG**: Adaptive QA framework that can dynamically select the most suitable strategy for (retrieval-augmented) LLMs from the simplest to the most sophisticated ones based on the query complexity

**Mechanism Overview**:
Train a classifier which is a smaller LM, to predict the complexity level of incoming queries with automatically collected labels obtained from **actual predicted outcomes of models** and **inherent inductive biases in datasets**

---
**Previous Method**
1. Non Retrieval for QA: $\mathbf{\bar{a}} = \mathrm{LLM}(\mathbf{q})$
    - $\mathbf{x} = [x_1, x_2, ..., x_n]$, $\mathbf{y} = [y_1, y_2, ..., y_n]$
    - $\mathbf{y} = \mathrm{LLM}(\mathbf{x})$
    - In problem setup for QA: $\mathbf{q} = \mathbf{x}$, $\mathbf{\bar{a}} = \mathbf{y}$
2. Single-step Approach for QA: $\mathbf{\bar{a}} = \mathrm{LLM}(\mathbf{q}, \mathbf{d})$
    - Incorporate external knowledge $\mathbf{d}$ from the external knowledge source $\mathrm{D}$ via retriever
    - $\mathbf{d} = \mathrm{Retriever}(\mathbf{q}; \mathrm{D})$
3. Multi-steps Approach for QA: $\mathbf{\bar{a_i}} = \mathrm{LLM}(\mathbf{q}, \mathbf{d_i}, \mathbf{c_i})$
    - The process begins with the initial query $\mathbf{q}$, and at every retrieval step i, new documents $\mathbf{d_i}$ are retrieved from $\mathrm{D}$ and then incorporated into the input of LLMs
    - $\mathbf{d_i} = \mathrm{Retriever}(\mathbf{q}, \mathbf{c_i}; \mathrm{D})$
    - $\mathbf{c_i}$: Additional context, can be composed of previous documents and outcomes: $(\mathbf{d_1}, \mathbf{d_2}, ..., \mathbf{d_{i-1}}, \mathbf{\bar{a_1}}, \mathbf{\bar{a_2}}, ..., \mathbf{\bar{a_{i-1}}})$ (depends on the implementation of LLMs)
    - By interacting with Retriever like this in several rounds, LLMs progressively refining its understanding of $\mathbf{q}$, until it formulates the final answer from findings accumulated across these multiple steps
---
**Adaptive-RAG**: $\mathrm{o} = \mathrm{Classifier}(\mathbf{q})$


Train a classifier to choose among 3 modes above: $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$ labels for non-retrieval, single-step and multi-steps

How to construct dataset ?
1. **Predicted outcomes**: 
Labeling the query complexity based on the results from three different retrieval-augmented LLM strategies, e.g. assign the label to the approach which correctly generates the answer among $\{\mathbf{A}, \mathbf{B}, \mathbf{C}\}$, breaking tie with a higher priority to a simpler model
2. **Inherent inductive biases in datasets**
For those queries that remain `unlabeled` after the first labeling step, we assign $\mathbf{B}$, to queries in single-hop datasets and $\mathbf{C}$, to queries in multi-hop datasets
    - The three retrieval-augmented approaches may all fail to generate the correct answer
    - The benchmark datasets may already have **meaningful inductive biases** about the most appropriate retrieval-augmented LLM strategies for their queries
    - 
<!-- Content_END -->
