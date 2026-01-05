# 挑戰 - 多代理系統

## 介紹

多代理系統（Multi-Agent Systems，簡稱 MAS）由多個自主代理組成，每個代理都有不同的目標、行為和責任範圍。這些代理能獨立運作，根據自身的本地知識與環境做出決策。然而，它們也能彼此溝通與分享資訊，根據各自的目標進行合作或競爭。MAS 通常用於任務分散在多個實體，且系統透過去中心化運作能獲得效益的場景。常見應用包含交通管理、機器人團隊、分散式 AI 以及需要協調但不依賴中央控制器的網路系統。

在此挑戰中，你將建立一個多代理系統，該系統接收使用者的請求並透過一組各具特定角色與專長領域的代理來處理。各代理會獨立分析請求，並根據其責任範圍提供回應。最終輸出將是所有代理回應的綜合集合，協同回答使用者的問題，並反映每個角色獨特的觀點。

## 任務1 - Microsoft Foundry 模型部署與環境設定

1. **Azure OpenAI 服務部署：**

    - 建立一個 Azure OpenAI 服務實例，SKU 大小設定為 Standard `S0`。

        > **注意：** 請確保區域設定為 **East US**。

    - 部署於名稱前綴為 `openaiagents` 的資源群組中。

    - 取得 Azure OpenAI 金鑰與端點（Endpoint）。

1. **部署 Azure OpenAI 模型：**

    - Azure OpenAI 提供一個名為 **Microsoft Foundry Portal** 的網頁入口，供你部署、管理與探索模型。你將從使用 Microsoft Foundry 部署模型開始探索 Azure OpenAI。

    - 從總覽面板啟動 Microsoft Foundry Portal，部署一個 Azure OpenAI 模型，例如 `gpt-4o`。

        >- **注意：** 確保部署名稱為 **gpt-4o**。
        >- **注意：** 確保部署類型（Deployment Type）設定為 **Global Standard**，且模型版本使用 **2024-11-20**。

    - 取得模型的 **部署名稱（Deployment name）** 與 **API 版本（API version）**。

        >- **提示：** API 版本可從目標 URI 中擷取。

1. 更新 `.env` 檔案，填入 Microsoft Foundry 部署資訊：

    ```
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=Replace with your deployment name
    AZURE_OPENAI_ENDPOINT=Replace with your endpoint URL
    AZURE_OPENAI_API_KEY=Replace with your API key
    AZURE_OPENAI_API_VERSION=Replace with your API version
    ```
> **恭喜** 完成此任務！現在是時候驗證結果了，請按照以下步驟操作：
>
> - 如果您收到成功訊息，可以繼續進行下一個任務。
> - 如果沒有，請仔細閱讀錯誤訊息，依照實驗指南中的指示重新嘗試該步驟。
> - 如果您需要任何協助，請隨時透過 cloudlabs-support@spektrasystems.com 與我們聯繫，我們提供 24/7 全天候支援服務。

<validation step="fd3e58c5-f047-4b86-8177-2489ee812527" />

## 任務 2 - 設定多代理人工作流程並在核准後自動推送程式碼

1. 在您的 **Lab VM** 上開啟 **VS Code**。然後，從路徑 **C:\LabFiles\\** 中於 **VS Code** 中開啟 **CAPSTONE-PROJECT** 資料夾。

1. 打開 `multi_agent.py` 檔案。在此檔案中你將實作本挑戰所需的所有程式碼。

2. 為三個代理人建立角色，並依照以下指示設定：

    - **業務分析師角色**

        ```
        You are a Business Analyst which will take the requirements from the user (also known as a 'customer') and create a project plan for creating the requested app. The Business Analyst understands the user requirements and creates detailed documents with requirements and costing. The documents should be usable by the SoftwareEngineer as a reference for implementing the required features, and by the Product Owner for reference to determine if the application delivered by the Software Engineer meets all of the user's requirements.
        ```

    - **軟體工程師 Persona**

        ```
        You are a Software Engineer, and your goal is create a web app using HTML and JavaScript by taking into consideration all the requirements given by the Business Analyst. The application should implement all the requested features. Deliver the code to the Product Owner for review when completed. You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear.
        ```

    - **產品擁有者角色**

        ```
        You are the Product Owner which will review the software engineer's code to ensure all user  requirements are completed. You are the guardian of quality, ensuring the final product meets all specifications. IMPORTANT: Verify that the Software Engineer has shared the HTML code using the format ```html [code] ```. This format is required for the code to be saved and pushed to GitHub. Once all client requirements are completed and the code is properly formatted, reply with 'READY FOR USER APPROVAL'. If there are missing features or formatting issues, you will need to send a request back to the SoftwareEngineer or BusinessAnalyst with details of the defect.
        ```

1. 為上述三個角色各建立一個 `ChatCompletionAgent`。每個代理人應包含：
    - 指示內容（即角色提示語）
    - 唯一名稱（僅限英文字母，無空白或特殊字元）
    - 指向一個 `Kernel` 物件的參考

1. 建立一個 `AgentGroupChat` 物件，將三個代理人串接起來。傳入：
    - 三個代理人的陣列
    - `ExecutionSettings`，其中 `TerminationStrategy` 設為 `ApprovalTerminationStrategy` 的實例

1. 在 `ApprovalTerminationStrategy` 類別中實作 `should_agent_terminate` 方法。當使用者在聊天歷史中回傳 "APPROVED" 時，代理人應該終止運行。

1. 在實作 `should_agent_terminate` 方法以偵測 "APPROVED" 後，新增一個回呼或後處理步驟，在條件達成時執行。

1. 從聊天歷史中擷取軟體工程師代理人提供的 HTML 程式碼。

1. 將擷取出的程式碼保存至檔案（例如 `index.html`）。

1. 建立一個 Bash 腳本（例如 `push_to_git.sh`），用以將檔案加入暫存區、提交並推送到指定的 Git 儲存庫：

    ```bash
    #!/bin/bash
    git add index.html
    git commit -m "Auto-commit: update from Software Engineer agent"
    git push origin main
    ```

1. 在 Python 程式碼中，使用 `subprocess` 模組呼叫此腳本，當偵測到 "APPROVED" 時執行。

1. 確保執行環境已配置必要的 Git 憑證，以支援非互動式推送。

   此自動化流程確保當產品負責人（或使用者）傳送 "APPROVED" 後，最新程式碼能自動推送至你的 Git 儲存庫。

> **恭喜** 完成此任務！現在是時候驗證結果了，請按照以下步驟操作：
>
> - 如果您收到成功訊息，可以繼續進行下一個任務。
> - 如果沒有，請仔細閱讀錯誤訊息，依照實驗指南中的指示重新嘗試該步驟。
> - 如果您需要任何協助，請隨時透過 cloudlabs-support@spektrasystems.com 與我們聯繫，我們提供 24/7 全天候支援服務。

<validation step="3b86923d-f180-411e-b41d-de4ed650d29e" />

## 任務 3 - 執行多代理人對話並驗證工作流程

1. 實作程式碼，使用 `AgentGroupChat` 物件的 `add_chat_message` 方法將使用者訊息傳送給代理人群組。訊息應包含：
    - 作者角色為 `AuthorRole.User`
    - 使用者輸入的聊天內容

2. 使用非同步迴圈遍歷 `AgentGroupChat` 回應，並在每則訊息抵達時印出：

    ```python
    async for content in chat.invoke():
        print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
    ```

3. 執行你的應用程式，並輸入一個建立計算機應用程式的請求。觀察商業分析師、軟體工程師與產品負責人如何協作規劃、開發並核准解決方案。

## 任務 4 - 使用容器登錄庫和 Azure 應用服務部署應用程式到 Azure

為了在線上託管你的應用程式，請依照以下步驟將應用程式容器化，推送至 Azure 容器登錄服務 (ACR)，並使用 Azure App Service 進行部署：

1. 開啟終端機，使用以下指令登入 Azure Developer CLI：

    ```bash
    azd auth login
    ```

2. 執行以下指令以部署所需資源到 Azure：

    ```bash
    azd up
    ```

1. 執行 **azd up** 指令時，系統會互動式要求你輸入設定值。請依提示提供以下資訊：

   - **唯一環境名稱**：輸入 **CapstoneEnv**。
   - **使用的 Azure 訂閱**：選擇預設訂閱，然後按下 **Enter**。
   - **位置基礎設施參數**：從選項中選擇 **East US 2**，然後按下 **Enter**。
   - **資源群組名稱基礎設施參數**：輸入 **CapstoneEnv**，然後按下 **Enter**。
   - **使用的資源群組**：從選項中選擇 **CapstoneEnv**，然後按下 **Enter**。

2. 開啟 Azure 入口網站，導覽至資源群組 **CapstoneEnv**。
3. 找到已部署的容器應用程式資源。
4. 複製該容器應用程式的端點 URL。
5. 在瀏覽器中開啟此端點，確認應用程式正常運作。

> **恭喜您完成此任務！** 現在是時候驗證成果了。請按照以下步驟操作：
> - 如果收到成功訊息，您可以繼續進行下一個任務。
> - 如果沒有，請仔細閱讀錯誤訊息，依照實驗指南中的說明重新嘗試該步驟。
> - 如果您需要任何協助，請隨時透過 cloudlabs-support@spektrasystems.com 與我們聯繫。我們提供 24/7 全天候支援服務。

<validation step="7ce1ccbb-408c-4528-9ee9-5e5e52b7ced9" />

## 成功標準

- 你已成功實作多代理聊天系統，具備以下功能：
    - 生成完整的 HTML 和 JavaScript 應用程式原始碼。
    - 由使用者進行完整的程式碼審查與核准流程。
    - 完成應用程式自動化部署至 Azure。
    - 使用者核准後，自動將程式碼推送至 Git 儲存庫。

## 額外挑戰

- 將聊天歷史中的 Markdown 格式程式碼複製並存成對應的檔案。
- 將 HTML 內容存為 `index.html`，並在瀏覽器中開啟測試。
- 檢查應用程式是否如 AI 所描述般正常運作。
- 向 AI 要求讓應用程式具備響應式設計或加入新功能，提升應用體驗。
- 嘗試調整不同代理人的角色設定，以優化結果或增加功能性。


## 学习资源

- [Agent Group Chat with Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-chat?pivots=programming-language-python)
- [MetaGPT](https://github.com/geekan/MetaGPT)
- [AutoGen Multi-Agent Conversational Framework](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat/)
- [AutoGen with Semantic Kernel](https://devblogs.microsoft.com/semantic-kernel/autogen-agents-meet-semantic-kernel/)
- [管理你的个人访问令牌](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)


## 结论

本挑战展示了如何利用 Microsoft Foundry 和 Semantic Kernel 构建和协调多代理系统。通过为业务分析师、软件工程师和产品负责人设计不同的角色，并配置具备终止策略的群组聊天环境，你创建了一个协作式 AI 工作流程，能够收集需求、开发代码并执行代码审查。该任务结构允许使用自治且互动的代理，进行复杂问题的可扩展、去中心化处理。

