# 挑战实验

# 毕业项目

### 总预计时长：4小时

## 概述

在本次挑战中，你将使用基于 Chainlit 的对话应用程序，该应用程序利用 Dapr 的发布-订阅（pub/sub）消息模型，通过智能 AI 代理协调客户服务升级。该解决方案无缝集成了多个 Azure 服务——包括用于自然语言处理的 Azure OpenAI、用于数据持久化的 Cosmos DB，以及用于可靠消息传递的 Azure Service Bus。当 AI 代理达到解决上限时，系统会智能地将案例升级到人工支持代理，通过触发一个 Logic Apps 工作流，发送电子邮件审批请求。本动手实验将为你提供宝贵的见解，了解如何将 AI 驱动的对话界面、事件驱动架构和工作流自动化相结合，构建响应迅速、可扩展且具备人性化感知的客户支持系统。


## 目标

在本实验结束时，你将能够：

- **多代理业务角色工作流**：体验涉及三种不同角色的协作工作流——Software Engineer、Product Owner 和 User。Software Engineer 编写并提交代码，Product Owner 审核并批准更改，批准后，解决方案将自动推送到 GitHub 进行部署和版本控制。

## 前提条件

参与者应具备：

- 对 Azure 服务（如 Azure OpenAI 和模型）的基本了解。
- 使用 Azure Developer CLI (AZD) 部署应用程序的经验。
- 接触过对话式 AI 工具，如 Streamlit 或类似框架。


## 组件说明

- **Azure OpenAI**：基于云的服务，提供高级语言模型访问，支持自然语言处理、内容生成和对话式 AI 功能。它使你能够将强大的 AI 驱动功能安全且可扩展地集成到应用程序中。
- **Azure Container Apps**：无服务器容器托管服务，允许你部署和扩展微服务及容器化应用程序，而无需管理基础设施。

## 实验入门

欢迎参加你的 Capstone Project Workshop，让我们充分利用这次体验，开始吧：

## 访问你的实验环境

当你准备好开始时，你的虚拟机和 **lab guide** 将通过网页浏览器随时可用。

![访问你的虚拟机和实验指南](./media/agg1.png)

## 实验指南缩放

要调整环境页面的缩放级别，请点击实验环境中计时器旁边的 **A↕ : 100%** 图标。

![](./media/agg2.png)

## 虚拟机与实验指南

你的虚拟机将是整个 workshop 中的工作平台，而实验指南则是你成功的路线图。

## 查看你的实验资源

为了更好地了解实验资源和凭据，导航到 **Environment** 选项卡。

![查看实验资源](./media/agg3.png)

## 使用分屏窗口功能

为了方便起见，你可以通过点击右上角的 **Split Window** 按钮，将实验指南在单独窗口中打开。

![使用分屏窗口功能](./media/agg4.png)


## 管理你的虚拟机

你可以根据需要，在 **Resources (1)** 选项卡中**启动、停止或重启 (2)** 虚拟机。整个体验由你掌控！

![管理你的虚拟机](./media/agg5.png)

> **注意：** 请确保在访问环境后脚本持续运行，且未被终止。


## 开始使用 Azure 门户

1. 在你的虚拟机上，点击 Azure Portal 图标。  
2. 你会看到 **Sign into Microsoft Azure** 标签。在这里输入你的凭据：

   - **电子邮件/用户名：** <inject key="AzureAdUserEmail"></inject>

     ![输入你的用户名](./media/gt-5.png)

3. 接下来，输入你的密码：

   - **密码：** <inject key="AzureAdUserPassword"></inject>

     ![输入你的密码](./media/gt-4.png)

4. 如果出现 **Action required** 弹窗，点击 **Ask later**。  
5. 如果提示是否 **stay signed in**，你可以点击 **No**。  
6. 如果出现 **Welcome to Microsoft Azure** 弹窗，直接点击 **"Cancel"** 跳过介绍。

## 登录 GitHub 的步骤。

1. 打开新的浏览器标签页，并访问以下 URL：**[https://github.com/login](https://github.com/login)**

   ![输入你的密码](./media/Git1.png)

1. 在 “Sign in to GitHub” 页面中，在 **Username (1)** 和 **Password（2）** 处输入你的个人 GitHub 凭据，然后点击 **“Sign in”（3）。**

   ![输入你的密码](./media/Git2.png)

1. 如果系统提示输入验证码，请输入发送到你已注册邮箱或身份验证应用中的 GitHub 验证码。

## 如果没有看到“稍后提醒”选项，进行 MFA 设置的步骤

1. 在 **"需要更多信息"** 提示时，选择 **Next**。

2. 在 **"保护你的账户安全"** 页面上，连续选择两次 **Next**。

3. **注意：** 如果你的手机上没有安装 Microsoft Authenticator 应用：

   - 打开 **Google Play 商店**（Android）或 **App Store**（iOS）。  
   - 搜索 **Microsoft Authenticator** 并点击 **安装**。  
   - 打开 **Microsoft Authenticator** 应用，选择 **添加账户**，然后选择 **工作或学校账户**。

4. 电脑屏幕上会显示一个 **二维码**。

5. 在 Authenticator 应用中，选择 **扫描二维码** 并扫描屏幕上的二维码。

6. 扫描完成后，点击 **Next** 继续。

7. 在手机上，在 Authenticator 应用中输入电脑屏幕显示的数字，然后选择 **Next**。  
8. 如果提示是否保持登录状态，可以点击 **No**。

9. 如果出现 **Welcome to Microsoft Azure** 弹窗，点击 **Maybe Later** 跳过介绍。


## 支持联系方式

CloudLabs 支持团队全年无休 24/7 全天候通过电子邮件和实时聊天为你提供无缝帮助。我们为学习者和讲师分别设立了专门的支持渠道，确保你的所有需求都能得到及时高效的处理。

学习者支持联系方式：

- 邮件支持：[cloudlabs-support@spektrasystems.com](mailto:cloudlabs-support@spektrasystems.com)  
- 实时聊天支持：https://cloudlabs.ai/labs-support

点击右下角的 **Next**，开启你的实验之旅！

![开始你的 Azure 之旅](./media/agg6.png)

现在你已经准备好探索强大的技术世界。如果过程中有任何疑问，随时联系我们。祝你工作坊愉快！
