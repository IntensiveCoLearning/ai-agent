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
- Conversable Agents（可對話的代理人）
  - 每個代理人有特定的「角色」（如寫程式、執行程式、回答問題）
  - 可以透過聊天溝通：收訊息 → 回覆 → 再執行動作
  - 支援人類參與、程式執行、自動推理

- Conversation Programming（對話式編程）
  - 把「對話流程」當作寫程式的一部分
  - 可以用自然語言 + 程式碼來設定誰要說什麼、什麼時候講、遇到錯誤怎麼做
  - 
#### AutoGen 可以做什麼？
1. **自動解數學題**：代理人 A 解題 → 代理人 B 執行程式 → 回傳結果給 A
2. **問答系統（RAG）**：先查資料 → 再回答問題 → 如果沒查到就繼續查
3. **網頁互動任務**：在模擬世界中做決策（像找物品）
4. **多人討論聊天**：每個代理人扮演不同角色一起解決複雜問題
5. **AI 玩西洋棋**：AI 或人類下棋，第三個代理人負責檢查規則

- AutoGen 在多個任務上測試，比 GPT-4 或其他多代理人系統表現更好：
  - 數學題：比 ChatGPT + Plugin 更準
  - 查資料問答：可互動查資料，比傳統 RAG 更準
  - 多代理人決策：引入「常識代理人」可減少錯誤
  - 寫程式任務：加入「安全檢查代理人」讓結果更穩定

### 2025.06.15
[CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society](https://arxiv.org/abs/2303.17760)
- 一個 AI 扮演「使用者」（User），例如股票交易員
- 一個 AI 扮演「助手」（Assistant），例如 Python 工程師
- 它們透過多輪對話完成任務，例如打造一個交易機器人
- → 全程不需要人類介入，自己想辦法合作解題！

1. 人類只提供一個「初始點子」和兩個角色（例如股票機器人 + 程式設計師）
2. CAMEL 框架的「任務指定器」會把這個點子轉成明確任務
3. 兩個 AI 開始互動對話，一個下指令、一個解任務，直到任務完成

- 使用一種叫做 **Inception Prompting** 的方法設定角色，避免角色互換或失控
- 產生大量高品質對話資料集（AI Society, Code, Math, Science）
- 可以用來訓練、分析 LLM 的合作能力與行為模式

- **常見問題**：
  - AI 搞錯角色（assistant 指揮 user）
  - 無意義重複（你說我說一直重複）
  - 停不下來（無限回圈）

- **解法**：透過嚴謹的 prompt 工程來控制對話流程與結束時機

- 他們用 CAMEL 自動產的資料來訓練 LLaMA-7B，發現：
  - 加入更多對話資料（從 AI Society → Code → Math → Science）
  - 模型就能做出越來越強的推理與寫程式表現！

### 2025.06.16
[Autonomous Evaluation and Refinement of Digital Agents](https://arxiv.org/abs/2404.06474)
- 想解決兩個問題：
  1. 如何讓 AI 自動評估「自己有沒有做好任務」
  2. 如何根據這個評估結果「自我修正」變得更強

- 可以讓 AI 代理人：
  - 自動給自己評分（成功 or 失敗）
  - 再根據這個結果進行訓練或改進

- 兩種評估模型設計：
  1. **End-to-End 模型**（像 GPT-4V）  
     直接輸入指令、動作和畫面 → 給出任務成功與否的評估
  2. **Modular 模型（分階段）**  
     - 第一步：用 VLM 把畫面轉成「描述文字」
     - 第二步：用 LM 分析這些文字 + 使用者指令 + 動作 → 給出評估

- 這些自動評估器可以拿來：
  1. **Reflexion（反思強化）**  
     如果評估模型說任務失敗 → AI 重新思考並再試一次  
     → GPT-4 Agent 成功率提升 **29%**
  2. **Filtered Behavior Cloning（挑資料再訓練）**  
     收集大量操作記錄 → 留下表現比較好的那部分來進行再訓練  
     → iOS 控制任務提升 **75%**

### 2025.06.17
[Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132)
- **目的**：評估 LLMs 與人類偏好的契合度，解決傳統基準測試無法捕捉真實應用場景的問題。
- **運作方式**：用戶在網站上輸入問題，兩個匿名 LLMs 回答，然後用戶投票選出較佳回答，模型身份在投票後才揭曉。
- **特點**：
  - 免費、開源，任何人都可使用。
  - 收集來自全球用戶的 240,000+ 投票，涵蓋 100+ 語言。
  - 提供 50+ 主流模型（如 GPT-4、Llama、Mistral）。
#### 如何評估模型？
- **方法**：採用成對比較（pairwise comparison），用戶比較兩個模型的回答並投票。
- **排名系統**：
  - 使用 Bradley-Terry 模型計算模型勝率和排名。
  - 設計高效採樣算法，優先比較性能相近的模型，減少所需投票數。
- **數據分析**：
  - 用戶問題涵蓋編程、詩歌、數學等 600+ 主題，顯示高度多樣性。
  - 這些問題能有效區分模型在不同領域的強弱（如 GPT-4 在編程任務中勝率高達 97%）。
  - 投票質量經專家驗證，與專家評估的一致性達 72%-83%。
  - 
### 2025.06.18
[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/pdf/2310.11511)

- **SELF-RAG**（Self-Reflective Retrieval-Augmented Generation）是一種新的框架，旨在提升大型語言模型（LLMs）的回答質量和事實準確性，同時保持其靈活性。以下是其核心內容的簡單解釋：

- 什麼是 SELF-RAG？
  - **目的**：解決 LLMs 常因依賴內部知識而產生事實錯誤的問題，通過檢索外部資料和自我反思來提升回答的準確性與質量。
  - **運作方式**：
    1. **按需檢索**：模型根據問題判斷是否需要檢索外部資料，生成「檢索標記」（retrieval token）決定是否調用檢索器。
    2. **生成與評估**：處理多個檢索到的段落，生成回答，並用「反思標記」（reflection tokens）評估段落相關性、回答的事實支持度及整體質量。
    3. **選擇最佳回答**：根據反思標記的評分，選出最符合事實且質量最高的回答，並提供引用支持驗證。
  - **特點**：
    - 比傳統檢索增強生成（RAG）更靈活，僅在需要時檢索，避免不必要的資料干擾。
    - 使用特殊標記（reflection tokens）控制生成過程，適應不同任務需求。
    - 提供引用，方便用戶驗證回答的事實性。

- 與傳統 RAG 的區別
  - **傳統 RAG**：不管問題是否需要，總是檢索固定數量的段落，可能引入無關資料，降低回答質量，且不保證回答與檢索內容一致。
  - **SELF-RAG**：
    - 動態決定是否檢索，減少無用資料的影響。
    - 通過反思標記自我評估，確保回答與檢索內容一致並高質量。
    - 支持推理時調整檢索頻率或優先級，靈活應對不同任務（如事實性要求高或開放性任務）。

- 訓練過程
  - **模型**：訓練一個通用語言模型，擴展詞彙表加入反思標記。
  - **數據準備**：
    - 使用 GPT-4 生成反思標記，評估檢索需求、段落相關性和回答質量。
    - 將這些標記插入訓練數據，模擬推理過程。
  - **訓練方式**：
    - 訓練「批評模型」（critic model）生成反思標記。
    - 用標準語言模型目標訓練「生成模型」，使其能獨立生成文本和反思標記，無需推理時依賴批評模型。
  - **優勢**：訓練成本低於強化學習（RLHF），且推理時可控性強。

- SELF-RAG 通過按需檢索和自我反思，顯著提升了語言模型的事實準確性和生成質量。其靈活的控制機制和高效的訓練方法使其在多種任務中表現優異，特別適合需要高事實性的場景，同時保持創意任務的靈活性。

### 2025.06.19
[What Are Tools Anyway? A Survey from the Language Model Perspective](https://arxiv.org/pdf/2403.15452)

- 什麼是工具？
  - **定義**：工具是語言模型（LMs）使用的外部電腦程式，模型會生成函數調用和輸入參數來執行這些程式。
  - **工具類型**：
    - **感知（Perception）**：從環境獲取資訊，例如 `get_time()` 獲取當前時間。
    - **行動（Action）**：改變環境狀態，例如 `make_post()` 修改網站內容。
    - **計算（Computation）**：執行複雜計算，例如用計算器進行數學運算或翻譯語言。
    - **多功能**：一個工具可能同時具備多種功能，例如搜尋引擎既能計算也能感知。

- 為什麼需要工具？
  - **補足語言模型的不足**：
    - 語言模型在複雜任務（如數學計算、邏輯推理）或需要外部資訊（當前天氣、最新資料）時表現不佳。
    - 工具能擴展語言模型的能力，幫助解決這些問題。
  - **例子**：
    - 用計算器解決數學問題。
    - 用搜尋引擎獲取最新資訊。
    - 用 SQL 查詢資料庫。

- 工具如何使用？
  - **基本流程**（以 Toolformer 為例）：
    1. 用戶提問（例如「今天天氣如何？」）。
    2. 模型生成文字或工具調用（例如 `check_weather()`）。
    3. 執行工具，獲取結果（例如「晴天」）。
    4. 將結果融入回答，繼續生成文字（例如「今天是晴天」）。
  - **學習方式**：
    - **推理時提示**：通過上下文提供工具說明和範例，讓模型學會使用。
    - **訓練時學習**：用人工或大型模型生成的範例訓練模型，讓其掌握工具使用。

  - 工具適用的場景
  - **有用場景**：
    - **知識獲取**：查詢資料庫或網路（如 SQL、搜尋引擎）。
    - **計算任務**：數學運算、程式執行（如計算器、Python 解釋器）。
    - **與世界互動**：獲取天氣、位置或管理日曆、郵件。
    - **非文字模態**：處理圖片、音訊（例如圖片問答、播放音樂）。
    - **專業模型**：使用專門的語言模型作為工具，處理特定任務。
  - **不適用場景**：
    - 機器翻譯、文本摘要、情感分析等任務，語言模型本身已能高效處理，工具的幫助有限。

  - 進階工具使用
  - **多工具選擇**：
    - 若工具數量少，模型可直接從上下文中選擇。
    - 若工具數量多，需用檢索器篩選相關工具。
    - 複雜任務可能需要多個工具組合（例如巢狀或並行調用）。
  - **程式化任務**：
    - 使用程式語言內建函數、外部庫或專門設計的工具函數。
    - 例如用 `matplotlib` 畫圖或設計 `locate_objects` 檢測圖片中的物體。
  - **工具創建**：
    - 當現有工具不足時，模型可生成新工具。
    - 例如自動生成程式庫或為特定任務設計函數。

- 工具評估
  - **基準測試**：
    - **現有資料集**：如 BigBench（推理）、MATH（數學）、TabMWP（表格問答）。
    - **API 基準**：如 ToolBench、API-Bank，測試真實世界的 API 使用。
  - **評估指標**：
    - **任務完成度**：測量工具是否幫助完成任務。
    - **工具選擇準確性**：評估模型是否選對工具。
    - **工具可重用性**：檢查工具是否能通用於多個任務。
  - **缺失的評估面向**：
    - **效率**：工具整合的計算成本。
    - **工具質量**：執行速度、穩定性和安全性。
    - **可重現性**：處理隨機或不穩定輸出的挑戰。
    - **安全性**：確保工具使用安全且可驗證。

- 工具使用的權衡
  - **性能提升 vs 計算成本**：
    - 工具能顯著提升特定任務的表現（如數學、API 相關任務），但需要額外計算成本。
    - 例如，ToolFormer 在數學任務上提升 30.4%，但多語言任務幾乎無提升。
  - **高效方法**：
    - TroVE 在工具生成中效率最高，成本低且表現提升顯著。
    - API-Bank 和 ToolAlpaca 適合需要高效推理的場景。
  - **任務選擇**：
    - 數學、表格問答、API 任務受益最大。
    - 語言相關任務（如翻譯）工具幫助有限。

- 總結
  工具讓語言模型更強大，特別是在需要外部資訊或複雜計算的任務中。通過統一的定義和系統性分析，我們可以看到工具在不同場景中的應用及其效率。未來需更多研究聚焦於自然用例的基準、可執行的工具以及更全面的評估指標。


### 2025.06.20
[Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325)

- **定義**：讓多個語言模型（稱為「智能體」）針對問題各自提出答案，並通過多輪討論和批判彼此的回答，最終達成一致的答案。
- **過程**：
  1. 每個智能體獨立回答問題。
  2. 智能體閱讀並評價其他智能體的答案，更新自己的回答。
  3. 重複此過程數輪，直到答案趨於一致。

- 為什麼需要多智能體辯論？
  - **問題**：單一語言模型可能會自信地給出錯誤事實（幻覺）或推理錯誤。
  - **優勢**：
    - 通過討論，智能體可以修正錯誤，減少不確定的事實。
    - 多個智能體能同時探索不同推理路徑，提升答案準確性。
    - 適用於現有模型，無需改變模型內部結構。

- 應用場景
  - **推理任務**：
    - **數學問題**：如算術、GSM8K（小學數學推理）。
    - **策略推理**：如預測國際象棋最佳下一步。
  - **事實性任務**：
    - 撰寫人物傳記，減少錯誤事實（如出生地、日期）。
    - 新基準測試：評估計算機科學家傳記的事實正確性。

- 效果如何？
  - **實驗結果**（以3個智能體、2輪辯論為例）：
    - **算術**：準確率從67%提升至81.8%。
    - **小學數學**：從77%提升至85%。
    - **國際象棋**：棋步優勢分數從91.4提升至122.9。
  - **事實性**：減少了傳記中不一致的事實（如出生地從「西班牙」修正為「古巴」）。
  - **關鍵發現**：
    - 更多智能體和更多輪次能進一步提升表現。
    - 即使初始答案全錯，辯論也能引導出正確答案。

- 技術細節
  - **提示（Prompt）**：
    - 使用統一提示模板，兼容不同任務。
    - 可控制辯論長度：較「固執」的提示讓討論更長，但答案更準確。
  - **優化**：
    - 總結其他智能體的回答，減少上下文長度，提升效率。
    - 不同初始提示（如模擬教授、數學家）可進一步提升表現。
  - **模型**：主要使用ChatGPT，也測試了混合不同模型（如Bard）。

- 總結
  多智能體辯論是一種簡單但有效的方法，能顯著提升語言模型在推理和事實性上的表現。它模擬人類的多線程推理和多源事實檢查，特別適合數學、策略和傳記撰寫等任務。未來可探索更高效的辯論機制和更廣泛的應用。



### 2025.06.21

1.  **Agent Application (Agent应用):**
    * **Auto-research:** [ResearchTown: Simulator of Human Research Community](https://github.com/ulab-uiuc/research-town)

    * **Coding Agents:** [If LLM Is the Wizard, Then Code Is the Wand: A Survey on How Code Empowers Large Language Models to Serve as Intelligent Agents](https://arxiv.org/abs/2401.00812)
  
   
    * **Social Agents:** [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
  

    * **Gaming Agents:** [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://voyager.minedojo.org)
  

2.  **Challenges from Agents to AGI:**
    * **Data:** [BAGEL: Bootstrapping Agents by Guiding Exploration with Language](https://arxiv.org/abs/2403.08140)
  
    * **Safety:**
        * [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)
     
        * [DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models](https://arxiv.org/abs/2306.11698)
    * **Alignment:**
        * [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
     
        * [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)


3.  **Future Perspectives & Broader Impact:**
    * Section 4-7 of [How far are we from AGI?](https://openreview.net/pdf?id=H2ZKqfNd0U)



<!-- Content_END -->
