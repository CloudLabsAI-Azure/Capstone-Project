# 挑战 - 多智能体系统

## 介绍

多智能体系统 （MAS） 由多个自主智能体组成，每个智能体都有不同的目标、行为和责任范围。这些代理独立运作，根据他们的当地知识和环境做出决策。但是，他们也可以相互交流和共享信息，根据他们的目标进行合作或竞争。MAS 通常用于任务分布在多个实体中并且系统受益于去中心化的场景。常见应用包括流量管理、机器人团队、分布式 AI 和网络系统，这些应用需要协调而不依赖中央控制器。

在这项挑战中，您将创建一个多代理系统，该系统接受用户的请求并通过一组代理进行处理，每个代理都根据特定的角色和专业领域进行设计。座席将单独分析请求，并根据他们定义的职责提供他们的响应。最终输出将是来自所有代理的答案的整合集合，以反映每个角色独特视角的方式协作解决用户的查询。


## 任务 1 - Microsoft Foundry 模型部署与环境配置


1. **部署 Azure OpenAI 服务：**

    - 使用标准 SKU `S0` 设置一个 Azure OpenAI 服务实例。

        > **注意：** 请确保区域设置为 **East US**。

    - 在资源组中部署，资源组名称以 `openaiagents` 为前缀。

    - 获取 Azure OpenAI 的密钥和终结点。


1. **部署 Azure OpenAI 模型：**

    - Azure OpenAI 提供了一个名为 **Microsoft Foundry Portal** 的基于网页的门户，供你部署、管理和探索模型。你将通过使用 Microsoft Foundry 来部署模型，开始探索 Azure OpenAI。

    - 从概览面板启动 Microsoft Foundry Portal，并部署一个 Azure OpenAI 模型，例如 `gpt-4o`。

        >- **注意：** 确保部署名称为 **gpt-4o**。  
        >- **注意：** 确保部署类型设置为 **Global Standard**，模型版本使用 **2024-11-20**。

    - 获取模型的 **部署名称** 和 **API 版本**。

        >- **提示：** API 版本可以从目标 URI 中获取。

1. 使用 Microsoft Foundry 部署详情更新 `.env` 文件：

    ```
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=Azure OpenAI 聊天部署名称
    AZURE_OPENAI_ENDPOINT=Azure OpenAI 终结点 URL
    AZURE_OPENAI_API_KEY=Azure OpenAI API 密钥
    AZURE_OPENAI_API_VERSION=Azure OpenAI API 版本
    ```
> **恭喜** 完成此任务！现在是验证的时候了。以下是步骤：
> - 如果收到成功消息，可以继续进行下一任务。
> - 如果没有，请仔细阅读错误信息，按照实验指导中的说明重试该步骤。
> - 如果需要帮助，请通过 cloudlabs-support@spektrasystems.com 联系我们。我们提供全天候 24/7 支持。
  
<validation step="aec82833-a1ec-47c5-96bd-5453b504c139" />

## 任务 2 - 配置多智能体工作流并在批准后自动推送代码

1. 在你的 **Lab VM** 上打开 **VS Code**。然后，从路径 **C:\LabFiles\\** 中在 **VS Code** 中打开 **CAPSTONE-PROJECT** 文件夹。

1. 打开 `multi_agent.py` 文件。在这里你将实现本挑战所需的所有代码。

1. 根据以下说明为三个代理创建角色：


    - **业务分析师角色**

        ```
        You are a Business Analyst which will take the requirements from the user (also known as a 'customer') and create a project plan for creating the requested app. The Business Analyst understands the user requirements and creates detailed documents with requirements and costing. The documents should be usable by the SoftwareEngineer as a reference for implementing the required features, and by the Product Owner for reference to determine if the application delivered by the Software Engineer meets all of the user's requirements.
        ```

    - **软件工程师 Persona**

        ```
        You are a Software Engineer, and your goal is create a web app using HTML and JavaScript by taking into consideration all the requirements given by the Business Analyst. The application should implement all the requested features. Deliver the code to the Product Owner for review when completed. You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear.
        ```

    - **产品所有者角色**

        ```
        You are the Product Owner which will review the software engineer's code to ensure all user  requirements are completed. You are the guardian of quality, ensuring the final product meets all specifications. IMPORTANT: Verify that the Software Engineer has shared the HTML code using the format ```html [code] ```. This format is required for the code to be saved and pushed to GitHub. Once all client requirements are completed and the code is properly formatted, reply with 'READY FOR USER APPROVAL'. If there are missing features or formatting issues, you will need to send a request back to the SoftwareEngineer or BusinessAnalyst with details of the defect.
        ```

1. 为上述每个角色创建一个 `ChatCompletionAgent`。每个代理应包含：  
    - 说明（角色提示）  
    - 唯一名称（仅字母，不含空格或特殊字符）  
    - 对 `Kernel` 对象的引用  

1. 创建一个 `AgentGroupChat` 对象，将三个代理绑定在一起。传入：  
    - 包含三个代理的数组  
    - 带有 `TerminationStrategy` 的 `ExecutionSettings`，该策略设置为 `ApprovalTerminationStrategy` 的实例  

1. 在 `ApprovalTerminationStrategy` 类中实现 `should_agent_terminate` 方法。当用户在聊天记录中返回 "APPROVED" 时，代理应终止运行。  

1. 在实现 `should_agent_terminate` 方法以检测 "APPROVED" 后，添加一个回调或后处理步骤，在满足条件时执行。
1. 从聊天记录中提取软件工程师代理提供的 HTML 代码。
1. 将提取的代码保存到文件（例如 `index.html`）。
1. 创建一个 Bash 脚本（例如 `push_to_git.sh`），该脚本将文件加入暂存区、提交并推送到目标 Git 仓库。
1. 在 Python 代码中，使用 `subprocess` 模块调用此脚本，当检测到 "APPROVED" 时执行。
1. 确保你的环境配置了必要的 Git 凭据，以支持非交互式推送。

   此自动化流程确保一旦产品负责人（或用户）发送 "APPROVED"，最新代码即可自动推送到你的 Git 仓库。

> **恭喜** 完成此任务！现在是验证的时候了。以下是步骤：
> - 如果收到成功消息，可以继续进行下一任务。
> - 如果没有，请仔细阅读错误信息，按照实验指导中的说明重试该步骤。
> - 如果需要帮助，请通过 cloudlabs-support@spektrasystems.com 联系我们。我们提供全天候 24/7 支持。
  
<validation step="1b2ffbe3-7834-4b75-b925-d6a907385789" />

## 任务3 - 运行多代理对话并验证工作流程

1. 实现代码，将用户消息发送到代理组，使用 `AgentGroupChat` 对象的 `add_chat_message` 方法。消息应包含：
    - `AuthorRole.User` 作为作者
    - 来自用户输入的聊天消息内容

2. 使用异步循环遍历 `AgentGroupChat` 的响应，并在每条消息到达时打印：

    ```python
    async for content in chat.invoke():
        print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
    ```

3. 运行你的应用程序，提供构建计算器应用的请求。观察业务分析师、软件工程师和产品负责人如何协作进行规划、构建和审批解决方案。

## 任务4 - 使用容器注册表和 Azure 应用服务将应用部署到 Azure

要使用 Azure 在线托管你的应用，请按照以下步骤将应用容器化，推送到 Azure 容器注册表 (ACR)，并使用 Azure 应用服务进行部署：

1. 打开终端，使用以下命令登录 Azure 开发者 CLI：

    ```bash
    azd auth login
    ```

1. 通过运行以下命令将所需资源部署到 Azure：

    ```bash
    azd up
    ```

1. 运行 **azd up** 命令时，系统会交互式地要求你提供配置详情。请按提示输入以下值：

   - **唯一环境名称**：输入 **CapstoneEnv**。
   - **要使用的 Azure 订阅**：选择出现的默认订阅 ，然后按 **Enter**。
   - **基础设施位置参数**：从选项中选择 **East US 2**，然后按 **Enter**。
   - **ResourceGroupName 基础设施参数**：输入 **CapstoneEnv**，然后按 **Enter**。
   - **要使用的资源组**：从选项中选择 **CapstoneEnv**，然后按 **Enter**。

1. 打开 Azure 门户，导航到资源组 **CapstoneEnv**。
1. 找到已部署的容器应用资源。
1. 复制容器应用的端点 URL。
1. 在浏览器中访问该端点，验证应用程序是否按预期运行。

> **恭喜** 完成此任务！现在是验证的时候了。以下是步骤：
> - 如果收到成功消息，可以继续进行下一任务。
> - 如果没有，请仔细阅读错误信息，按照实验指导中的说明重试该步骤。
> - 如果需要帮助，请通过 cloudlabs-support@spektrasystems.com 联系我们。我们提供全天候 24/7 支持。
  
<validation step="06da7bc8-2a60-45f4-866c-cb4b9de80b5c" />

## 成功标准

- 您已实现多智能体聊天系统，能够完成以下任务：
    - 生成请求应用的完整 HTML 和 JavaScript 源代码
    - 由用户进行详尽的代码审查和审批流程
    - 自动将应用部署到 Azure
    - 在用户审批后自动将代码推送到 Git 仓库


## 额外加分项

- 从聊天记录的 Markdown 中复制代码到文件系统中对应的文件。
- 将 HTML 内容保存为 `index.html` 并在浏览器中打开。
- 测试应用是否如 AI 描述般正常运行。
- 通过让 AI 添加响应式设计或新功能来增强应用。
- 试验修改不同的角色设定以提升结果或功能。



## 学习资源

- [使用 Semantic Kernel 的代理群聊](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-chat?pivots=programming-language-python)
- [MetaGPT](https://github.com/geekan/MetaGPT)
- [AutoGen 多智能体对话框架](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat/)
- [AutoGen 与 Semantic Kernel 结合](https://devblogs.microsoft.com/semantic-kernel/autogen-agents-meet-semantic-kernel/)
- [管理你的个人访问令牌](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)


## 结论

本挑战演示了如何使用 Microsoft Foundry 和 Semantic Kernel 构建及协调多智能体系统。通过为业务分析师、软件工程师和产品负责人设计不同的角色，并配置具有终止策略的群聊环境，您创建了一个能够协作完成需求收集、代码开发和代码审查的 AI 工作流。该任务结构支持使用自治且交互的智能体对复杂问题进行可扩展和去中心化的处理。
