---
timezone: UTC+8
---

> 请在上边的 timezone 添加你的当地时区(UTC)，这会有助于你的打卡状态的自动化更新，如果没有添加，默认为北京时间 UTC+8 时区


# Shawn

1. 自我介绍：web3从业者
2. 你认为你会完成本次残酷学习吗？ 会

## Notes

<!-- Content_START -->
### 2025.06.03

学习内容：Language Agents: Foundations, Prospects, and Risks (Slides) 与 How far are we from AGI? 第1-2节。

1. 两种竞争观点：
- LLM-first view的缺点：对非文本任务可能支持不足，比如对于视觉的理解和物理控制。
- Agent-first view的缺点：工程量太大，除了原本AI中所存在的传统问题，LLM的局限也需要一并解决。

2. AGI 系统的“大脑”也可以根本性地组织成四个主要组件：感知、记忆（AGI 与包含感知获取和行动执行的环境之间的交互将被保存为 AGI 的记忆）、推理能力和元认知（由记忆驱动）。


### 2025.06.04

学习内容：How far are we from AGI? 第3节。

长期记忆与短期记忆：
- 会从长期记忆中提取数据作为短期记忆。
- 更多的上下文窗口能提升记忆存储与利用效率。

学习ReAct：

论文提出的Reasoning and Acting方法，让LLM能够在同一次推理过程中同时输出“Thought”和“Action“，并通过对外部环境（例如Wikipedia API或模拟环境）的交互，持续获得信息辅助推理和决策。
- Thought：模型对问题进行推理（类似gpt正常模式）
- Action：模型在推理中得到要搜索的关键字，通过API进行搜索。
- Observation：根据环境（API)所返回的新信息，作为输入放入下一轮Thought中。

### 2025.06.05



### 2025.07.11

笔记内容

### 2025.07.12

<!-- Content_END -->
