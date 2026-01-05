# Challenge - Multi-Agent Systems

## Introduction

Multi-Agent Systems (MAS) consist of multiple autonomous agents, each with distinct goals, behaviors, and areas of responsibility. These agents operate independently, making decisions based on their local knowledge and environment. However, they can also communicate and share information with one another, either cooperating or competing depending on their objectives. MAS is typically used in scenarios where tasks are distributed across multiple entities and the system benefits from decentralization. Common applications include traffic management, robotic teams, distributed AI, and networked systems where coordination is required without relying on a central controller.

In this challenge, you will create a Multi-Agent System that accepts a user’s request and processes it through a collection of agents, each designed with a specific persona and area of expertise. The agents will individually analyze the request and contribute their responses based on their defined responsibilities. The final output will be a consolidated collection of answers from all agents, collaboratively addressing the user’s query in a way that reflects the unique perspective of each persona.


## Task 1 - Microsoft Foundry Model Deployment & Environment Configuration

1. **Azure OpenAI Service Deployment:**

    - Set up an Azure OpenAI Service instance with SKU size Standard `S0`.

        > **Note:** Ensure the region is set to **East US**.

    - Deploy it in the resource group named `openaiagents`.

    - Obtain the Azure OpenAI Key and Endpoint. 

1. **Deploy Azure OpenAI Models:**
   
    - Azure OpenAI provides a web-based portal named **Microsoft Foundry Portal** that you can use to deploy, manage, and explore models. You'll start your exploration of Azure OpenAI by using Microsoft Foundry portal to deploy a model.
    
    - Launch Microsoft Foundry Portal from the overview pane and deploy an Azure OpenAI Model, i.e., `gpt-4o`.

        >- **Note:** Make sure the deployments are named **gpt-4o**.
        >- **Note:** Ensure the Deployment Type is set to **Global Standard** and use **2024-11-20** for the model version.

    - Fetch the **Deployment name** and the **API version** of the model.

        >- **Hint:** API version can be fetched from the Target URI.

> **Congratulations** on completing the task! Now, it's time to validate it. Here are the steps:
> - If you receive a success message, you can proceed to the next task.
> - If not, carefully read the error message and retry the step, following the instructions in the lab guide. 
> - If you need any assistance, please contact us at cloudlabs-support@spektrasystems.com. We are available 24/7 to help you out.

<validation step="d6519c92-19e6-4dae-bdbe-3638f8d8db43" />

## Task 2 - Configure Multi-Agent Workflow and Automate Code Push on Approval

1. Open **VS Code** on your Lab VM. Then, open the **CAPSTONE-PROJECT** folder in VS Code from the path `C:\LabFiles\`

1. Update the `.env` file with the Microsoft Foundry deployment details:

    ```
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=Replace with your deployment name
    AZURE_OPENAI_ENDPOINT=Replace with your endpoint URL
    AZURE_OPENAI_API_KEY=Replace with your API key
    AZURE_OPENAI_API_VERSION=Replace with your API version
    ```

1. Open the `src>multi_agent.py` file. This is where you will implement all necessary code for this challenge.

1. Create personas for the three agents with the following instructions:

    - **Business Analyst Persona**

        ```
        You are a Business Analyst which will take the requirements from the user (also known as a 'customer') and create a project plan for creating the requested app. The Business Analyst understands the user requirements and creates detailed documents with requirements and costing. The documents should be usable by the SoftwareEngineer as a reference for implementing the required features, and by the Product Owner for reference to determine if the application delivered by the Software Engineer meets all of the user's requirements.
        ```

    - **Software Engineer Persona**

        ```
        You are a Software Engineer, and your goal is create a web app using HTML and JavaScript by taking into consideration all the requirements given by the Business Analyst. The application should implement all the requested features. Deliver the code to the Product Owner for review when completed. You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear.
        ```

    - **Product Owner Persona**

        ```
        You are the Product Owner which will review the software engineer's code to ensure all user  requirements are completed. You are the guardian of quality, ensuring the final product meets all specifications. IMPORTANT: Verify that the Software Engineer has shared the HTML code using the format ```html [code] ```. This format is required for the code to be saved and pushed to GitHub. Once all client requirements are completed and the code is properly formatted, reply with 'READY FOR USER APPROVAL'. If there are missing features or formatting issues, you will need to send a request back to the SoftwareEngineer or BusinessAnalyst with details of the defect.
        ```

1. Create a `ChatCompletionAgent` for each of the above personas. Each agent should have:
   - Instructions (the persona prompt)
   - A unique Name (letters only, no spaces or special characters)
   - A reference to a `Kernel` object

1. Create an `AgentGroupChat` object to tie together the three agents. Pass:
   - An array of the three agents
   - `ExecutionSettings` with a `TerminationStrategy` set to an instance of `ApprovalTerminationStrategy`

1. Implement the `should_agent_terminate` method in the `ApprovalTerminationStrategy` class. The agents should terminate when the User returns **"APPROVED"** in the chat history.

1. After implementing the `should_agent_terminate` method to detect **"APPROVED"**, add a callback or post-processing step that executes when this condition is met.

1. Extract the HTML code provided by the Software Engineer agent from the chat history.

1. Save the extracted code to a file (e.g., `index.html`).

1. Create a Bash script (`push_to_github.sh`) that stages, commits, and pushes the file to your desired Git repository.

1. In your Python code, use the `subprocess` module to call this script when **"APPROVED"** is detected.

1. Ensure your environment has the necessary Git credentials configured for non-interactive pushes.

   This automation ensures that once the user sends "APPROVED", the latest code is automatically pushed to your Git repository.

> **Congratulations** on completing the task! Now, it's time to validate it. Here are the steps:
> - If you receive a success message, you can proceed to the next task.
> - If not, carefully read the error message and retry the step, following the instructions in the lab guide. 
> - If you need any assistance, please contact us at cloudlabs-support@spektrasystems.com. We are available 24/7 to help you out.

<validation step="86730b76-da41-429e-9a9b-35b6ecd8bd79" />

## Task 3 - Run the Multi-Agent Conversation and Validate Workflow

1. Implement the code to send a user message to the agent group using `add_chat_message` on the `AgentGroupChat` object. The message should include:
    - `AuthorRole.User` as the author
    - The chat message contents from the user's input

1. Iterate through the responses from the `AgentGroupChat` using an asynchronous loop, and print each message as it arrives:

    ```python
    async for content in chat.invoke():
        print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
    ```

1. Run your application and provide a request to build a calculator app. Observe how the Business Analyst, Software Engineer, and Product Owner collaborate to plan, build, and approve the solution.


## Task 4 - Deploy the App to Azure Using Container Registry and Azure App Service

To host your app online using Azure, follow these steps to containerize your application, push it to Azure Container Registry (ACR), and deploy it using Azure App Service:

1. Open a terminal and sign in to the Azure Developer CLI using the following command:

    ```bash
    azd auth login
    ```

1. Deploy the required resources to Azure by running:

    ```bash
    azd up
    ```

1. When running the **azd up** command, you'll be asked to provide configuration details interactively. Provide the following values when prompted:

   - **Unique Environment Name**: Enter **CapstoneEnv**.
   - **Azure Subscription to use**: Choose the default subscription that appears and press **Enter**.
   - **Location Infrastructure Parameter**: Select **East US 2** from the options and press **Enter**.
   - **ResourceGroupName Infrastructure Parameter**: Type **CapstoneEnv** and press **Enter**.
   - **Resource Group to use**: Select **CapstoneEnv** from the options and press **Enter**.

1. Open the Azure portal and navigate to the resource group **CapstoneEnv**.
1. Locate the deployed container app resource.
1. Copy the endpoint URL of the container app.
1. Access the web app by visiting this endpoint in your browser and verify that the application functions as expected.

> **Congratulations** on completing the task! Now, it's time to validate it. Here are the steps:
> - If you receive a success message, you can proceed to the next task.
> - If not, carefully read the error message and retry the step, following the instructions in the lab guide. 
> - If you need any assistance, please contact us at cloudlabs-support@spektrasystems.com. We are available 24/7 to help you out.

<validation step="14625f2c-4adb-4d11-969d-74eb6be92a21" />

## Success Criteria

You have implemented the Multi-Agent Chat system that produces:
 - Generation of complete source code in HTML and JavaScript for the requested application
 - Thorough code review and approval process by User
 - Automated deployment of the application to Azure
 - Automated code push to a Git repository upon user approval


## Bonus

- Copy the code from the chat history markdown into matching files on your file system.
- Save HTML content as `index.html` and launch it in your web browser.
- Test if the application functions as the AI described.
- Enhance the app by asking the AI to make it responsive or add new features.
- Experiment with modifying personas to improve results or functionality.

## Learning Resources

- [Agent Group Chat with Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-chat?pivots=programming-language-python)
- [MetaGPT](https://github.com/geekan/MetaGPT)
- [AutoGen Multi-Agent Conversational Framework](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat/)
- [AutoGen with Semantic Kernel](https://devblogs.microsoft.com/semantic-kernel/autogen-agents-meet-semantic-kernel/)
- [Managing your personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

---

## Conclusion

This challenge demonstrated how to build and coordinate a Multi-Agent System using Microsoft Foundry and Semantic Kernel. By designing distinct personas for Business Analyst, Software Engineer, and Product Owner, and configuring a group chat environment with a termination strategy, you created a collaborative AI workflow capable of gathering requirements, developing code, and performing code reviews. The task structure allows for scalable, decentralized handling of complex problems using autonomous, interactive agents.
