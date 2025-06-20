---
timezone: UTC+8
---


# fffuuuming

1. 自我介绍：資工碩班，想踏入web3
2. 你认为你会完成本次残酷学习吗？盡力
3. 你的联系方式（推荐 Telegram）@fffuuuming

[HACKMD LINK](https://hackmd.io/@BKuxYlL0T6iT9rROj2QldQ/B1HNk87Mle)
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
Labeling the query complexity based on the results from three different retrieval-augmented LLM strategies, e.g. assign the label to the approach which correctly generates the answer among {$\mathbf{A}, \mathbf{B}, \mathbf{C}$}, breaking tie with a higher priority to a simpler model
2. **Inherent inductive biases in datasets**
For those queries that remain `unlabeled` after the first labeling step, we assign $\mathbf{B}$, to queries in single-hop datasets and $\mathbf{C}$, to queries in multi-hop datasets
    - The three retrieval-augmented approaches may all fail to generate the correct answer
    - The benchmark datasets may already have **meaningful inductive biases** about the most appropriate retrieval-augmented LLM strategies for their queries

### 2025.06.12
[Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884)
---
**Motivation**:
Current LLMs have **hallucinations** occur since either
- The accuracy of generated texts cannot be secured solely by the parametric knowledge they encapsulate
- Even with RAG, they rely heavily on the **relevance of retrieved documents**

**Solution**: **C**orrective **R**etrieval **A**ugmented **G**eneration (**CRAG**)
Self-correct the results of retriever and improve the utilization of documents for augmenting generation

How ?
Use a light weight **retrieval evaluator** to assess the overall quality of retrieved documents for a query, returning **confidence degree** based on which different knowledge retrieval actions can be triggered, with downstream **decompose-then-recompose** refinement & **large-scale web searches**

**Advantage**:
- plug-and-play, which can be integrated into other off-the-shell models such as **RAG**, **self-RAG**

**Method Overview**
![截圖 2025-06-12 晚上9.32.39](https://hackmd.io/_uploads/SyBfvLd7gx.png)

----
**Retrieval Evaluator: Is this retrieved document relevant to the question?**
Fine-tuned a lightweight model **T5-large** to score how relevant a document is to a question.

Contrastive learning:
- **Positive examples (relevant)**: Use high-quality passages linked to questions (e.g., from PopQA dataset).
- **Negative examples (irrelevant)**: Use similar-looking but unrelated documents.

For each questions $x$, there are generally 10 documents retrieved $\{d_1, d_2, ..., d_k\}$, $\mathrm{score_i}$ = retrieval evaluator evaluates the relevance of each pair $(x, d_i)$, $d_i ∈ D$
$\mathrm{Confidence}$ = Calculate and give a final judgment based on $\{\mathrm{score_1}, \mathrm{score_2}, ...\mathrm{score_k}\}$


---
**Action Trigger**
Based on the aforementioned confidence score for each retrieved document, three types of actions are designed and triggered accordingly where the upper and lower thresholds are set.
- **Correct**: The confidence score of at least one retrieved document is higher than the upper threshold
    - Even if a relevant document can be found, there is inevitably **some noisy knowledge strips** in this document
    - To extract the most critical knowledge strips within this document, a **knowledge refinement**
- **InCorrect**: The confidence scores of all retrieved documents are below the lower threshold.
    - Seek new sources of knowledge for correction -> **web search**
- **Ambiguous**: The accuracy of the retrieval is hard to distinguish and the evaluator gives an intermediate score.
    - Both types of processed knowledge in `Correct` and `Incorrect` are combined to complement each other

**Knowledge Refinement**
- If a retrieved result is as short as one or two sentences: it is regarded as an individual strip
- Otherwise:
    - retrieval documents are required to be split into smaller units
    - The retrieval evaluator finetuned is employed to calculate the relevance score of each knowledge strip
    - Irrelevant knowledge strips are filtered out, while relevant ones are recomposed via concatenation in order -> **internal knowledge**

**Web Search**
The inputs are rewritten into queries composed of keywords by ChatGPT to mimic the daily usage of search engine. utilize the URL links to navigate web pages, transcribe their content, and employ the same knowledge refinement method

### 2025.06.13
[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://huggingface.co/papers/2308.08155)
---
**Motivation**: 

Prior work suggests that multiple agents can help encourage divergent thinking, improve factuality and reasoning, and provide validation
-> **how to facilitate the development of LLM applications based on the multi-agent approach ?**

**Idea: multi-agent conversations**

There are 3 reasons:
1. LLMs' ability to incorporate feedback
    - LLM agents can cooperate through conversations with each other or human(s)
2. single LLM can exhibit a broad range of capabilities
    - Conversations between differently configured agents can help combine broad LLM capabilities in a **modular** and **complementary** manner
3. LLM's ability to solve complex tasks when the tasks are broken into simpler subtasks

But, there's still some problems to be solved:
1. How to **design individual agents** that are capable, reusable, customizable, and effective in multi-agent collaboration
2. How to develop a straightforward, **unified interface** that can accommodate a wide range of agent conversation patterns
    - Applications of varying complexities may need distinct sets of agents with 
        - specific capabilities
        - different conversation patterns (single- or multi-turn dialogs)
        - different human involvement modes
        - static vs. dynamic conversation
    - developers may prefer the flexibility to program agent interactions in natural language or code

To address this, it proposes:
**AutoGen: generalized multi-agent conversation framework**



With two key concepts
1. **Customizable and conversable agents**:
Leverages the strong capability of the most advanced LLMs in taking feedback and making progress via chat and also allows combining capabilities of LLMs in a modular fashion
2. **Conversation programming**:
Simplify and unify complex LLM application workflows as multi-agent conversations
    - Can be achieved via a fusion of natural and pro- gramming languages
    - Easy extension and experimentation

Framework Overview:
![截圖 2025-06-13 晚上10.42.19](https://hackmd.io/_uploads/B1_lFnK7ll.png)

---
**The $\mathrm{AutoGen}$ Framework**
![截圖 2025-06-13 晚上10.45.03](https://hackmd.io/_uploads/SJCOFhKQle.png)
#### Conversable Agents:  entity with a specific role & capabilities
Agent capabilities can be powered by
- LLMs
- Humans
- Tools

Moreover, agnets can be **customized** and **cooperated**
- `ConversableAgent`: The highest-level agent abstraction and, by default, can use LLMs, humans, and tools
    - `AssistantAgent`, `UserProxyAgent`: two pre-configured `ConversableAgent` subclasses, each representing a common usage mode, (i.e., acting as an AI assistant (backed by LLMs) and acting as a human proxy to solicit human input or execute code/function calls (backed by humans and/or tools))
- LLM-backed assistant agent and a tool- and human-backed user proxy agent are deployed together to tackle a task.
    - Assistant agent generates a solution with the help of LLMs and passes the solution to the user proxy agent
    - User proxy agent solicits human inputs or executes the assistant’s code and passes the results as feedback back to the assistant.
#### Conversation Programming: make developers able to specify and mold these multi-agent conversations
- **Computation** = what each agent does in response to a message (e.g., runs code, calls LLM, replies)
    - conversation-centric:
- **Control Flow** = when and how these actions happen, and who talks to whom next.
    - conversation-driven: functions of the inter-agent conversation

1. Unified interfaces and auto-reply mechanisms for automated agent chat
2. Control by fusion of programming and natural language.
### 2025.06.14
[Autonomous Evaluation and Refinement of Digital Agents](https://arxiv.org/abs/2404.06474)
---
**Motivation**
It's important to evaluate agents' performance and refine based on it

**Solution**
Propose a to domain-general neural models to automatically evaluate user instructions and arbitrary agent trajectories, with two approaches offering trade-offs in inference cost, modularity of design, performance and transparency.

1. Modular caption-then-reason approach"
    - vision-language model (**VLM**) first captions the screenshots and generate a language-based description
    - Language model (LM) is used to reason about if an agent succeeds based on textual information
2. End-to-end approach:
Prompt an advanced VLM like GPT-4V to directly evaluate a trajectory

**Advantage**
Domain-general evaluation models can support the **autonomous evaluation and refinement of digital agents**, without requiring access to human demonstrations or oracle evaluation metrics

**Method Overview**
![截圖 2025-06-14 晚上8.46.02](https://hackmd.io/_uploads/H1_Mylsmlg.png)
- Evaluation of a digital agent’s trajectory
- Reward function for [Reflexion](https://arxiv.org/abs/2303.11366)
- Filtered behavior cloning to enhance model performance
---
**Domain-General Evaluators**

Given a user instruction $x$ and an initial environment state $s_0$, an agent generates and executes a sequence of actions $\bar{a} = ⟨a_0,a_1,...,a_n⟩$, resulting in a sequence of state visits $\bar{s} = ⟨s_0,s_1,s_2,...,s_{n+1}⟩$.
- assume $a$ and $x$ are in text form
- $s$ is represented as a screenshot image

How do it evaluate ? Given  $x$, $\bar{a}$, $\bar{s}$ as input, the model produces a scalar evaluation $\bar{r} = ⟨r_0, r_1, . . . , r_n⟩$ corresponding to each step of the trajectory. It can hence provide either trajectory-level or per-step evaluations

1. **End-to-End Approach**
Provide an instruction-tuned VLM (GPT-4V) with $x$, $\bar{a}$ and $\bar{s}$, prompt it to first produce a **text-based reasoning process** then output its **evaluation result**
    - expensive
2. **Modular Caption-then-Reason Approach**
First use a VLM to produce a description of the agent’s observations given as $\bar{s}$ , then feed these descriptions, along with actions $\bar{a}$ and the user’s instruction $x$ to an LM to produce a final evaluation  


    **Captioner**
    To deal with potential information loss, it constructs a dataset and use it to fine-tune QWen-VL-chat (Bai et al., 2023) model by
    - acquires screenshots from a variety web and device control domains
    - use GPT-4V to provide an initial detailed description for each screenshot
    - manually filter out or fix apparent errors in GPT-4V’s output, resulting a total of 1,263 data points.

    **Reasoner**
 Provide the actions, generated descriptions, and the original user instruction to a language-only instruction-tuned model to produce a text-based thought and reasoning process as well as the final evaluation.
### 2025.06.15
[ResearchTown: Simulator of Human Research Community](https://github.com/ulab-uiuc/research-town)
---
**Intention**: Can we simulate human research communities with LLMs ?

**Proposed RESEARCHTOWN**: A multi-agent, graph-based framework works as a simulator for research community
- **Agent-data graph**
    - 2 types of nodes: `agent`, `data` (papers, reviews, blogs...). An `agent` node can be considered a function over `data` nodes
    - 3 types of edges: `agent-agent` (reviewing expertise), `data-data` (paper citations), `agent-data` (authorship)
- **TextGNN**: Text-based inference framework that models various research activities (e.g., paper reading, paper writing, and review writing)
    - message-passing processes are defined based on text-form information processing with LLMs
- **Node-masking prediction task** for evaluation: similarity-based, how closely simulator's outputs align with those of the real-world research community

a research community can be regarded as a special form of agent-data graph, called community graph, with research agents and research papers as two types of nodes, and we consider three types of edges (review, au- thor, and cite) in the graph. Different community activities, such as paper writing and review writing, can be modeled as special message-passing processes on the community graph

Community Graph
![截圖 2025-06-15 下午2.46.21](https://hackmd.io/_uploads/rJpS2JhXlx.png)

---
#### Agent-Data Graph for Multi-agent LLMs
$\mathcal{G} = (\mathcal{V}, \mathcal{E})$ , $\mathcal{V} = \mathcal{V}_a \cup \mathcal{V}_d$ , $\mathcal{E} = \mathcal{E}_{aa} \cup \mathcal{E}_{ad} \cup \mathcal{E}_{dd}$ (agent-agent, data-data, agent-data)
- Each `data` node $v$ ∈ $\mathcal{V}_d$ comes with attributes, e.g., **a piece of text**, $x_v$
- Each `agent` node $u$ is accompanied with an **agent function**: an $\mathrm{LLM} \space f_u(·)$ with its prompt template and the profile
    - for message generation and message aggregation
    - takes $x_v$ from $v$ ∈ $\mathcal{V}_d$ as the input, and output new data based on its profile prompt $\mathbf{x}_u$, e.g., $\mathbf{x_{uv}} = f_u([\mathbf{x}_u,\mathbf{x}_v])$, where [·] indicates filling the prompt template with $x_u$ and $x_v$
---
#### Building TextGNN on Agent-Data Graphs
We need to use LLMs to generate new data and interactions on the agent-data graph, and TextGNN serves as a **text-based message-passing mechanism** on an agent-data graph

What's the difference from standard vanilla GNN ?
- All hidden states are defined in the **text space**. e.g. $h_v ∈ \mathrm{Σ∗}$
- Heterogeneous nodes: `agent` & `data`
- Typed edges: `agent-agent`, `agent-data`, `data-data`

**Technical detail**

Initial hidden states: 
- `data` nodes: $h^{(0)}_v = x_v$,
- `agent` nodes: $h^{(0)}_u = ∅$

Forward Propagation:

`agent`: Update the `agent` node $u$’s hidden state using:
- Its own prior state $\mathbf{h}_u^{(k-1)}$
- Messages from neighboring `agents` $a$ and `data` $d$, where:
    - $f_a$: agent function for neighbor $a$ encodes the triple $(a,u,d)$,
    - All messages are concatenated and passed to $f_u$
        >the function representing the `agent` u’s attribute
- Analogous to $h_u^{(k)} = \text{AGG}u \left(h_u^{(k-1)}, \text{MSG}{a,d} \right)$, but in **text space**, using **LLMs as aggregation functions**


$\mathbf{h}_u^{(k)} = f_u\left(\left[\mathbf{h}_u^{(k-1)}, \left\{ f_a\left([\mathbf{h}_a^{(k-1)}, \mathbf{h}u^{(k-1)}, \mathbf{h}d^{(k-1)}]\right) \mid (u,a) \in \mathcal{E}{aa}, (u,d) \in \mathcal{E}{ad} \right\} \right] \right)$

`data`: Update the `data` node $v$’s hidden state using:
- Its prior state $\mathbf{h}_v^{(k-1)}$
- Messages from `agent` neighbors $a$ and other `data` nodes $d$
- Each message is processed by an agent function $f_a$,
- The global function $f_g$ (not personalized) performs final aggregation.
- Correspond to $h_v^{(k)} = \text{AGG}g \left(h_v^{(k-1)}, \text{MSG}{a,d} \right)$, where $f_g$ acts like a global aggregator without any specialization.


$\mathbf{h}_v^{(k)} = f_g\left(\left[\mathbf{h}_v^{(k-1)}, \left\{ f_a\left([\mathbf{h}_a^{(k-1)}, \mathbf{h}v^{(k-1)}, \mathbf{h}d^{(k-1)}]\right) \mid (v,a) \in \mathcal{E}{ad}, (v,d) \in \mathcal{E}{dd} \right\} \right] \right)$

Simple Comparison Summary:


| GNN Component | TextGNN Equivalent |
| -------- | -------- |
| Node hidden state         | Text string or summary via LLM         |
| MSG function         | $f_a$: agent-specific message constructor         |
| AGG function         | $f_u$, $f_g$: LLM-based aggregators         |
| Neighborhood types     | $\mathcal{E}_{aa} ,  \mathcal{E}_{ad} ,  \mathcal{E}_{dd}$    |

#### RESEARCHTOWN: Applying TextGNN to Community Graph
Overall, the simulation algorithm can be considered as a 2-layer GNN where the paper reading is the first layer of information aggregation. Both paper writing and review writing are the second layer of the GNN to generate the final simulation outputs, as the following shown:

Simulation Overview: Three stages proccess
![截圖 2025-06-15 下午4.01.39](https://hackmd.io/_uploads/SkIWCxnQlx.png)
1. **Paper reading**: 
2. **Paper writing**
3. **Review writing**

### 2025.06.16
[If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents](https://arxiv.org/abs/2401.00812)
---
一篇對於 "**how code assists LLMs and where code benefits LLMs as IAs**" 整理的非常詳盡的清單，架構如下圖：
![截圖 2025-06-16 晚上10.37.16](https://hackmd.io/_uploads/rJD4no6Qlx.png)
除了 related LLMs to Physical Ends 以外其他都還算有興趣，有空會看個大概
### 2025.06.17
[CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society](https://arxiv.org/abs/2303.17760)
---
**Motivation**: Current chat-based language models' success heavily relies on human input to guide the conversation

**Replace human intervention with an autonomous communicative agent !** 
- capable of steering the conversation toward task completion with minimal human supervision
- Several issues such as **role flipping, assistant repeating instructions, flake replies**, and **infinite loop of messages** ? 
- Need to align these models with human intentions and to explore means enabling their effective cooperation

**Solution**: 

Communicative cooperative agent framework using **role-playing** with **inception prompting** to autonomously guide the communicative agents toward task completion.
- Only a preliminary idea is needed from human to guide the conversations toward complex task-solving !
- Offers a scalable approach for studying the cooperative behaviors and capabilities of multi-agent systems
- Collected dataset such as `AI Society`, `Code`, `Misalignment`, demonstrate the significant emergence of LLM training abilities by utilizing the datasets we have collected from simulating four distinct agent collaboration scenarios
---
**Framework Overview**
![截圖 2025-06-17 下午2.24.25](https://hackmd.io/_uploads/r1jVqt0Qlx.png)

**Assistant-user** scenario: A preliminary idea is given at the start. Agents will conceptualize the idea into a specific task and complete it autonomously through conversations.

- `Human user`: 最初 idea 發想者 & role assignment
    - can be generated by prompting LLMs
- `Task Specifier`: 下 prompt 小能手: conversational agents usually require a concrete task prompt for realizing the task
- `AI user`, `AI Assistant`: 角色扮演
#### Role-playing Framework
- **AI user** (task planner, e.g. Python Programmer): Providing instructions
- **AI assistant** (task executor, e.g. Stock Trader): Respond with a solution that fulfills the instructions
- $\mathcal{P_{A}}$, $\mathcal{P_{U}}$ : Assistant/User system prompt/message
- $\mathcal{F_{1}}$, $\mathcal{F_{2}}$ : large-scale auto-regressive language models
- $\mathcal{A} \leftarrow \mathcal{F_{1}^{P_A}}$, $\mathcal{U} \leftarrow \mathcal{F_{2}^{P_U}}$ : pass system message to models as the assistant and user agents: 
- $\mathcal{I_t}$:  user instruction message obtained at time t
- $\mathcal{S_t}$: assistant's solution

The set of conversational messages obtained up until time t:
$$\mathcal{M_t} = \{(I_0, S_0),...,(I_t, S_t)\}$$
At the next time step, t + 1, the AI user $\mathcal{U}$ takes the historical conversation message set $\mathcal{M_t}$ and provides a new instruction $\mathcal{I_{t+1}}$
$$\mathcal{I_{t+1}} = \mathcal{U(M_t)}$$
The produced instruction message $\mathcal{I_{t+1}}$ is then passed along with message set $\mathcal{M_t}$ to  $\mathcal{A}$. The AI assistant will then respond with a solution $\mathcal{S_{t+1}}$
$$\mathcal{S_{t+1}} = \mathcal{A(M_t, I_{t+1})}$$
After obtaining the solution $\mathcal{S_{t+1}}$ to the instruction $\mathcal{I_{t+1}}$, the message set is updated to $\mathcal{M_{t+1}}$
$$\mathcal{M_{t+1}} \leftarrow \mathcal{M_t} \cup (\mathcal{I_{t+1}}, \mathcal{S_{t+1}})$$

This model of `AI-AI` communicative scenarios can be easily extended to model `human-AI` communication or communication between more than two agents
#### Inception Prompting
pPompt engineering occurs solely at the beginning of role-playing, for task specification and role assignment

Three prompts:
- $\mathcal{P_{T}}$: task specifier prompt, contains information about the roles of the AI assistant and AI user in the role-playing session
- $\mathcal{P_{A}}$, $\mathcal{P_{U}}$: contain info of assigned task and roles, communication protocols, termination conditions, and constraints or requirements to avoid unwanted behaviors
### 2025.06.19
[SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION](https://arxiv.org/pdf/2310.11511)
---
**Motivation**: 
Resolve the **Hallucinations** from vanilla LLMs and disadvantage of vanilla RAG (**indiscriminately** retrieving and incorporating a **fixed number** of retrieved passages, **regardless of whether retrieval is necessary, or passages are relevant**)

**Solution**: Self-Reflective Retrieval-Augmented Gen- eration (SELF-RAG)
Two key concept: **demanded retrieva**l & **self-reflection**, specifically, it trains a single arbitrary LM that **adaptively retrieves passages on-demand**, and generates and reflects on retrieved passages and its own generations using **reflection tokens**.

**Advantage**
- significant gains in improving factuality and citation accuracy for long-form generations
- Generating reflection tokens makes the LM controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements
- It leverages **Knowledge Distillation** by first prompting GPT-4 to predict an appropriate reflection token as training data, then train critic model $C$ on it, finally update the training corpus by inserting reflection tokens into task outputs offline, this offline approach reduce the training cost

**SELF-RAG Overview**
![截圖 2025-06-19 晚上9.11.30](https://hackmd.io/_uploads/rJLqnKZ4xe.png)

---
Type of reflection tokens
![截圖 2025-06-19 晚上10.31.29](https://hackmd.io/_uploads/SJoO1i-Nll.png)
Workflow
![截圖 2025-06-19 晚上10.39.26](https://hackmd.io/_uploads/r1OEZjbEgx.png)
### 2025.06.20
[VOYAGER: An Open-Ended Embodied Agent with Large Language Models](https://voyager.minedojo.org)
---
**Motivation** Build an embodied agent that:
- Explores, plans, and learns autonomously in an open-ended environment.
- Can accumulate and transfer knowledge over long time spans like a human.
- Performs lifelong learning: acquiring increasingly complex skills without human intervention.

**Limitations of Prior SOTA**
| Approach | Limitation |
| -------- | -------- |
| RL / Imitation learning         |Works on low-level actions; struggles with long-horizon planning and generalization.          |
| LLM Agents (ReAct, Reflexion, AutoGPT)         |   Not designed for lifelong learning; lack mechanisms for knowledge retention and reuse.       |
| Embodied Agents     | Typically lack compositional skill reuse, error recovery, and self-directed exploration     |

**Solution: LLM-powered lifelong embodied agent**

Three key components:
- **Automatic curriculum**: Explores autonomously, proposes its own tasks, and refines skills based on outcomes.
- **Skill library**: storing and retrieving complex behaviors
- **Iterative prompting mechanism**: generates executable code for embodied control

Other approaches:
- Uses **code** as action space: allowing for high-level, temporally extended behavior
- Interacts with GPT-4 as a black-box LLM: no fine-tuning or gradient updates needed.
---
**Framework & Workflow Overview**
![截圖 2025-06-20 晚上10.36.37](https://hackmd.io/_uploads/SyHMzlXEgl.png)
#### Automatic Curriculum
![截圖 2025-06-20 晚上11.03.40](https://hackmd.io/_uploads/rJc8OgQ4le.png)
#### Skill Library
![截圖 2025-06-20 晚上11.04.13](https://hackmd.io/_uploads/H1sddl7Vlx.png)
#### Iterative Prompting Mechanism
![截圖 2025-06-20 晚上11.05.12](https://hackmd.io/_uploads/BkS2uxm4xx.png)
![截圖 2025-06-20 晚上11.05.36](https://hackmd.io/_uploads/r1A6ueQVel.png)

**Summary Advantage**:
| Column 1 | Column 2 |
| -------- | -------- |
| Automatic Task Generation         | Curriculum is dynamically created based on agent state, not predefined         |
| Lifelong Learning        | Continuously accumulates skills without forgetting or retraining         |
| Skill Reuse & Composition        | Builds complex behavior from smaller verified programs.         |
| Self-Verification         | Automates success/failure assessment and correction logic.         |
| Code-as-Policy         | Code is modular, interpretable, composable, and temporally extended.         |
| Black-Box LLM Use     | No model finetuning required. Uses GPT-4 API only     |
<!-- Content_END -->
