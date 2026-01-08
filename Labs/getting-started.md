
# 挑戰實驗室

# 專案總結

### 預估整體時長：4 小時

## 概述

在本挑戰中，你將使用基於 Chainlit 的對話應用程式，此應用程式利用 Dapr 的發布-訂閱（pub/sub）消息模型，透過智慧 AI 代理來協調客服升級流程。此解決方案無縫整合多項 Azure 服務——包括用於自然語言處理的 Azure OpenAI、用於資料持久化的 Cosmos DB，以及用於可靠消息傳遞的 Azure Service Bus。當 AI 代理達到其解決能力的極限時，系統會智慧地將案件升級至人工支援代理，並觸發 Logic Apps 工作流程，透過電子郵件發送批准請求。這個實作實驗室將提供你寶貴的經驗，了解如何將 AI 驅動的對話介面、事件驅動架構與工作流程自動化結合，打造出回應迅速、可擴展且具有人性化意識的客服支援系統。


## 目標

完成此實驗後，你將能夠：

- **多代理商業角色工作流程**：體驗一個包含三個不同角色——軟體工程師、產品負責人與使用者——的協作工作流程。軟體工程師撰寫並提交程式碼，產品負責人審查並批准更改，經過批准後，解決方案會自動推送到 GitHub 以進行部署與版本控制。

## 先決條件

參與者應具備：

- 基本了解 Azure 服務，例如 Azure OpenAI 及其模型。
- 有使用 Azure Developer CLI（AZD）部署應用程式的經驗。
- 熟悉對話式 AI 工具，如 Streamlit 或類似框架。

## 元件說明

- **Azure OpenAI**：一項雲端服務，提供先進語言模型的存取，支援自然語言處理、內容生成及對話式 AI 功能。它讓你能夠安全且大規模地將強大的 AI 驅動功能整合到應用程式中。
- **Azure Container Apps**：一種無伺服器容器託管服務，讓你能夠部署與擴展微服務及容器化應用程式，無需管理基礎設施。

## 實驗室開始指南

歡迎來到你的 Capstone Project 工作坊，讓我們開始充分利用這次體驗：

## 進入你的實驗環境

當你準備好開始時，你的虛擬機器和 **Guide** 將會隨時在你的網頁瀏覽器中供你使用。

![Access Your VM and Lab Guide](./media/Agg1.png)

## 實驗室指南縮放功能

若要調整環境頁面的縮放比例，請點擊位於計時器旁邊的 **A↕ : 100%** 圖示。

![](./media/Agg2.png)

## 虛擬機器與實驗室指南

你的虛擬機器將是你整個工作坊的主要工作平台，而實驗室指南則是你成功的路線圖。

## 探索你的實驗室資源

為了更清楚了解你的實驗室資源與憑證，請前往 **Environment** 標籤頁。

![Explore Lab Resources](./media/Agg3.png)


## 使用分割視窗功能

為了方便起見，你可以從右上角選擇 **Split Window** 按鈕，將實驗室指南在獨立視窗中開啟。

![Use the Split Window Feature](./media/Agg4.png)

## 管理你的虛擬機器

你可以隨時從 **Resources (1)** 標籤頁中根據需求 **啟動、停止或重新啟動 (2)** 你的虛擬機器。你的體驗掌握在自己手中！

![Manage Your Virtual Machine](./media/Agg5.png)

> **注意：** 請確保腳本持續運行，並且在進入環境後不被終止。

## 開始使用 Azure 入口網站

1. 在你的虛擬機器上，點擊 Azure Portal 圖示。
2. 你會看到 **Sign into Microsoft Azure** 頁籤，在此輸入你的憑證：

   - **Email/Username:** <inject key="AzureAdUserEmail"></inject>

     ![Enter Your Username](./media/gt-5.png)

3. 接著，輸入你的密碼：

   - **Password:** <inject key="AzureAdUserPassword"></inject>

     ![Enter Your Password](./media/gt-4.png)

4. 若出現 **Action required** 彈跳視窗，請點選 **Ask later**。
5. 如果系統詢問是否 **stay signed in**，你可以點選 **No**。
6. 若出現 **Welcome to Microsoft Azure** 彈跳視窗，請直接點選 **"Cancel"** 跳過導覽。

## 登入 GitHub 的步驟。

1. 開啟新的瀏覽器分頁，前往以下 URL：**[https://github.com/login](https://github.com/login)**

   ![Enter Your Password](./media/Git1.png)

1. 在「Sign in to GitHub」頁面，於 **Username（1）** 與 **Password（2）** 欄位輸入你的個人 GitHub 憑證，然後點擊 **「Sign in」（3）。**

   ![Enter Your Password](./media/Git2.png)

1. 若出現驗證碼提示，請輸入寄送到你已註冊的電子郵件或驗證應用程式中的 GitHub 驗證碼。

## 若未看到「稍後提醒（Ask Later）」選項，進行 MFA 設定的步驟

1. 在 **「需要更多資訊（More information required）」** 提示中，選擇 **Next**。

2. 在 **「保護你的帳戶安全（Keep your account secure）」** 頁面，連續選擇兩次 **Next**。

3. **注意：** 若你的行動裝置尚未安裝 Microsoft Authenticator 應用程式：

   - 開啟 **Google Play 商店**（Android）或 **App Store**（iOS）。
   - 搜尋 **Microsoft Authenticator**，並點選 **安裝**。
   - 開啟 **Microsoft Authenticator** 應用程式，選擇 **新增帳戶（Add account）**，然後選擇 **工作或學校帳戶（Work or school account）**。

4. 電腦畫面上會顯示一個 **QR 碼**。

5. 在 Authenticator 應用程式中，選擇 **掃描 QR 碼（Scan a QR code）**，並掃描畫面上的 QR 碼。

6. 掃描完成後，點選 **Next** 繼續。

7. 在手機上的 Authenticator 應用程式輸入電腦畫面顯示的數字，並選擇 **Next**。

8. 若系統提示是否保持登入狀態，可以點選 **No**。

9. 若出現 **Welcome to Microsoft Azure** 彈跳視窗，請點選 **Maybe Later** 跳過導覽。

## 支援聯絡方式

CloudLabs 支援團隊全年無休，24 小時隨時透過電子郵件與線上聊天提供無縫協助。我們針對學習者與講師提供專屬的支援管道，確保你的所有需求能被即時且有效地回應。

學習者支援聯絡方式：

- 電子郵件支援：[cloudlabs-support@spektrasystems.com](mailto:cloudlabs-support@spektrasystems.com)
- 線上聊天支援：https://cloudlabs.ai/labs-support

請從右下角點選 **Next**，開始你的實驗室旅程！

![Start Your Azure Journey](./media/Agg6.png)

現在你已準備好探索強大的科技世界，有任何問題隨時歡迎聯繫我們。祝你工作坊愉快！
