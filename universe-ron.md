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
  1.  **產生行動選項**  
    - 模型先想出幾個可能的下一步（例如：要先查什麼資料？）
  2.  **展開樹狀選擇（Tree Search）**
    - 像玩西洋棋一樣，預測每個選擇可能帶來的結果。
  3.  **評估每個分支**  
    - 用語言模型判斷哪一條路最有可能成功完成任務。
  4.  **選擇最佳行動**  
    - 依照評估結果，選出最好的下一步。

### 2025.06.07
[Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)
#### 記憶系統（Memory）
語言代理人像人一樣，有：
- **短期記憶**：目前正在處理的事情。
- **長期記憶**：分成三種：
  - 程序記憶（如何做事）
  - 語意記憶（知道什麼）
  - 情節記憶（做過什麼）
#### 行動空間（Actions）
分成兩類：
- **外部行動**：與外界互動（查資料、控制機器人、說話）
- **內部行動**：像是
  - 推理（Reasoning）
  - 讀資料（Retrieval）
  - 學習（Learning）
#### 決策流程（Decision-Making）
代理人會進行一個「思考 → 選擇行動 → 執行 → 再觀察」的循環。  
在每一輪，它會：
- 從記憶中讀資料
- 進行推理
- 決定下一步做什麼


### 2025.06.08
[HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831)
- HippoRAG 是一種新的記憶架構，模仿人腦記憶系統中的「海馬迴（hippocampus）」來幫助大型語言模型（LLM）更有效記住與整合新知識。
#### 雙階段流程：

##### Offline Indexing（離線建立知識圖）
- 用語言模型從文本中抽出「主詞、動詞、受詞」的知識三元組。
- 建立一個知識圖譜（KG），像是「誰做了什麼事」。

##### Online Retrieval（線上查資料）
- 使用 Personalized PageRank（個人化網頁排名）演算法，根據使用者的問題去圖中找出最相關的知識節點。
- 這個步驟可以一次找出跨段落的關聯（也就是 multi-hop reasoning），更快更準確。

> HippoRAG 讓語言模型像人腦一樣能記住事物間的關聯，提升「長期記憶」與「多步推理」能力。

### 2025.06.09
[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models]
- 這篇在說，通過生成一系列中間推理步驟，來提高 LLM 做複雜推理的能力
- 通過思維鏈提示，來激發 LLM 的推理能力
- 思維鏈提示在算術、常識和符號推理任務上都有顯著的性能提升
- 使用思維鏈提示的PaLM 540B模型在GSM8K數學文字問題基準測試中達到了最先進的準確性，甚至超越了帶驗證器的微調GPT-3模型
- 允許模型將多步驟問題分解為中間步驟
- 提供可解釋的模型行為視窗
- 適用於多種需要人類通過語言解決的任務

### 2025.06.10
[Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](https://arxiv.org/abs/2401.06209)
- HippoRAG 是一種新的記憶架構，模仿人腦記憶系統中的「海馬迴（hippocampus）」來幫助 LLM 更有效記住與整合新知識
- 現在的語言模型記憶力差，只能靠查資料（RAG）來補充，但效率和準確度都不夠好
- 真正的記憶系統應該能「累積經驗」、「連結舊知識」，並在需要時快速提取
- **大腦的海馬迴**能幫助我們把不同知識連起來，從少量線索中找到答案
- HippoRAG 仿照這個結構，把資料轉成「知識圖譜（Knowledge Graph）」來記住知識之間的關係
- 用 LM 從文本中抽出「主詞、動詞、受詞」的知識三元組
- 建立一個知識圖譜（KG），像是「誰做了什麼事」
- 使用 Personalized PageRank（個人化網頁排名）演算法，根據使用者的問題去圖中找出最相關的知識節點
- 這個步驟可以一次找出跨段落的關聯（也就是 multi-hop reasoning），更快更準確

### 2025.06.11
[VisualWebArena: Evaluating Multimodal Agents on Realistic Visually Grounded Web Tasks](https://arxiv.org/html/2401.13649v2)
- 雖然 GPT-4V、Gemini 等多模態大語言模型（MLLM）能看圖說話，但作者發現它們在處理**簡單圖片問題時常犯錯**。
- 這篇論文想了解：為什麼這些模型「看圖不準」？
- 很多 MLLM 都依賴 **CLIP** 做圖像理解，但 CLIP 會把**長得很不一樣的圖片看成一樣**。  
- 這些錯誤被叫做 **CLIP-blind pairs**。
- 作者用這些錯誤圖片對製作了一個評估工具叫做：
- → **MMVP 基準測試**，包含許多簡單的視覺問答題（像是「狗是面向左還是右？」）
- 作者提出一種方法叫 **Mixture of Features (MoF)**：
  - 把 CLIP 的圖像特徵與另一種模型（DINOv2）的特徵混在一起用。
  - 有兩種方法：
    - Additive-MoF：直接把兩者加起來2
    - Interleaved-MoF：空間交錯混合

### 2025.06.12
[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework](https://huggingface.co/papers/2308.08155)
- AutoGen 是一個由微軟開源的框架，可以讓你建立「多個 AI 代理人（Agents）」互相聊天、合作來完成任務。  
- 這些代理人可以是 LLM（像 GPT）、人類、工具或它們的組合。
- 用「聊天流程」設計任務解法
- 可以用自然語言 + 程式語言混合控制對話流程
- 支援動態聊天，不用每次都固定對話順序

### 2025.06.13
[Corrective Retrieval-Augmented Generation](https://arxiv.org/pdf/2401.15884)
- 大語言模型（LLM）有時會「幻想」（捏造錯誤資訊），  
- 即使加了查資料（RAG）也不一定正確，因為查到的內容可能是錯的。
- 所以這篇論文提出一種改進方法叫 **CRAG**，  
- 可以**檢查資料是不是錯的，然後自己修正**。
- CRAG 增加了 3 個功能：
  - 評估器（Retrieval Evaluator）  
    - → 幫你看「查到的資料對不對」  
    - → 分成三種情況：✅正確 / ❌錯誤 / 🤔不確定
  - 自動修正策略  
    - 如果是正確的 → 精煉資料，只留下重點
    - 如果是錯的 → 去網路搜尋新資料（用 Google Search API）
    - 如果不確定 → 兩者都用（保險做法）
  - 精煉方法（Knowledge Refinement）  
    - → 把查到的長篇資料切成小段，只留下最有用的部分給模型看
- 不需要換模型，只要「外掛」CRAG 就能讓結果變準
- 還會自己去網路查資料，不再死守舊資料
- 可以和現有的 RAG 或 Self-RAG 模型搭配使用

### 2025.06.14
[AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework](https://huggingface.co/papers/2308.08155)


### 2025.06.15
[CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society](https://arxiv.org/abs/2303.17760)


### 2025.06.16
[Autonomous Evaluation and Refinement of Digital Agents](https://arxiv.org/abs/2404.06474)


### 2025.06.17
[Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132)


### 2025.06.18

### 2025.06.19

### 2025.06.20

### 2025.06.21

<!-- Content_END -->
