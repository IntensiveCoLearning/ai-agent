---
timezone: UTC+8
---

> 请在上边的 timezone 添加你的当地时区(UTC)，这会有助于你的打卡状态的自动化更新，如果没有添加，默认为北京时间 UTC+8 时区


# 你的名字

1. 自我介绍 I am Univ. Let's learn ai agent together.
2. 你认为你会完成本次残酷学习吗？ Yes
3. 你的联系方式（推荐 Telegram） @Univ

## Notes

<!-- Content_START -->

### 2025.06.01
[Language Agents: Foundations, Prospects, and Risks (Slides)](https://language-agent-tutorial.github.io/slides/I-Introduction.pdf)
## 讀後心得
- ai agent  
  從簡單的「感知→行動」定義來看，AI agent 就像能在虛擬世界裡「動起來」的程式。以前我只覺得 ChatGPT 是回答問題的工具，但投影片提到「現代 agent = LLM + 外部環境」後，我突然意識到，當它擁有感測與執行 API 時，就能真的幫我完成任務，而不只是輸出文字。
---
- reasoning  
  投影片把「在內心生成 token」稱作一種新型態的行動。我以前覺得推理是數學題的步驟，現在懂了：對 LLM 來說，每一次 forward pass 都是在「想」。而自我反思（self-reflection）就是對自己的想法再想一次，讓 agent 能隨時停下來檢查方向，這對避免一直亂試非常重要。
---
- Language agents  
  - 「Language agent」它強調語言不只是輸出介面，而是一種在外部世界溝通、在內部做推理的萬能通道。因為語言夠通用，這些 agent 能在各種 App 之間當橋樑，學習成本也低很多。
  - 不過講者也提醒，光有 LLM 仍不足；要真正成為 agent，需要記憶、規劃、對多模態感知的整合，甚至要面對安全與社會影響等風險。這意味著未來不只要學 prompt，還要懂基礎 AI、倫理、以及怎麼把各種 API 串在一起，才能打造負責任的語言代理人。
---
### 2025.06.02
[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- **傳統方法問題**：
  - 「推理（Reasoning）」與「行動（Acting）」以前是**獨立處理**。
  - 只有推理：容易「幻覺」（捏造事實）且無外部驗證。
  - 只有行動：缺乏策略與目標導向的思考。

- **ReAct 要做的**：
  - 模擬人類的「邊想邊做」。
  - 模型能交錯進行**自然語言推理**與**任務操作**。
### 2025.06.03
[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- Tree of Thoughts
  - 一種**推理結構框架**，讓 LLM 像搜尋樹狀結構般展開推理。
  - 每個「節點」是一個 Thought（想法）。
  - 多個 Thought 組成一條思路（path）。
  - 可以設定評估策略來篩選、修剪或擴展思路。

### 2025.06.04
[Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403)
1. 提出 Adaptive-RAG 架構
- 根據查詢複雜度，自動選擇下列三種策略之一：
  - A 類（No Retrieval）→ LLM 直接回答。
  - B 類（Single-step Retrieval）→ 一次檢索後回答。
  - C 類（Multi-step Retrieval）→ 多次檢索並推理。

2. 設計輕量級查詢分類器（Classifier）
- 自動判斷查詢的複雜度（A/B/C）。
- 使用少量標註資料（結合模型預測結果與資料集偏差）自動生成訓練資料。

3. 兼顧準確率與效率
- 對於簡單問題節省資源，對於複雜問題保證答案正確性。
- 效能顯著優於現有方法（如 Self-RAG、Adaptive Retrieval 等）。

* 在 6 個 QA 資料集（SQuAD, NQ, TriviaQA, MuSiQue, HotpotQA, 2WikiMultiHopQA）上進行測試。
* 與 GPT-3.5、FLAN-T5-XL/XXL 等模型搭配使用，Adaptive-RAG 在 F1、EM、Acc 準確度上均表現最好，同時比多步推理快。
* 分類器準確率在 54% 左右，但仍能顯著提升整體 QA 系統效能。
* 若使用 Oracle 分類器（完美分類），效能可再上升。
### 2025.06.05
[LLM+P: Empowering Large Language Models with Optimal Planning Proficiency](https://arxiv.org/abs/2304.11477)
- 讓LLM學會「規劃」，例如分成幾步驟完成一個任務，而不是只會亂試或講一堆話。
- 抽象化（Abstraction）
  - 狀態（目前情況）
  - 行動（可以做的動作）
  - → 用 LLM 自動產出 PDDL 格式（像任務語言）。
- 規劃（Planning）
  - 把「任務格式」丟給一個專門會規劃的 AI 工具（Planner）
- 實體化（Instantiation）
  - 用 LLM 把 Planner 給的行動步驟變成自然語言，讓人或機器可以執行。
### 2025.06.06
[Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/abs/2310.04406)
- 結合三種能力
  - 推理（Reasoning）
  - 行動（Acting）
  - 規劃（Planning）
- LATS 就像是讓語言模型「思考後再行動」，流程如下：
  1. **產生行動選項**  
    - 模型先想出幾個可能的下一步（例如：要先查什麼資料？）
  2. **展開樹狀選擇（Tree Search）**
    - 像玩西洋棋一樣，預測每個選擇可能帶來的結果。
  3. **評估每個分支**  
    - 用語言模型判斷哪一條路最有可能成功完成任務。
  4️. **選擇最佳行動**  
    - 依照評估結果，選出最好的下一步。

### 2025.06.07

### 2025.06.08

### 2025.06.09

### 2025.06.10

<!-- Content_END -->
