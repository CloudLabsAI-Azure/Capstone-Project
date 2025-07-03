## Solution Guide
# Multi-Agent Systems - Solution Guide

## Introduction

Multi-Agent Systems (MAS) consist of multiple autonomous agents, each with distinct goals, behaviors, and areas of responsibility. These agents can interact with each other, either cooperating or competing, depending on the objectives they are designed to achieve. In MAS, each agent operates independently, making decisions based on its local knowledge and the environment, but they can communicate and share information to solve complex problems collectively.

MAS is often used in scenarios where tasks are distributed across different entities, and the overall system benefits from decentralization. Examples include simulations of real-world systems like traffic management, robotic teams, distributed AI applications, or networked systems where agents need to coordinate actions without a central controller. MAS allows for flexibility, scalability, and adaptability in solving dynamic and complex problems where a single agent or centralized system might be less efficient or incapable of handling the complexity on its own.

In this challenge, you will create a multi-agent system that takes the user's request and feeds it to a collection of agents. Each agent will have its own persona and responsibility. The final response will be a collection of answers from all agents that together will satisfy the user's request based on each persona's area of expertise.

## Task 1 - Azure AI Foundry Model Deployment & Environment Configuration

1. Navigate to `https://portal.azure.com` and log in with your Azure credentials.

    - **Email/Username**: <inject key="AzureAdUserEmail"></inject>
    - **Password**: <inject key="AzureAdUserPassword"></inject>

1. Search and Select Open AI. 

   ![](./Images/Image1.png)

1. On the **Azure Open AI (1)** content page, click on **+ create(2)**.

   ![](./Images/Image2.png)

1. Provide the following details and click on **Next**:

    - Subscription: Keep the default subscription **(1)**.

    - Resource Group: select **openaiagents**.

    - Region: **East US 2 (3)**

    - Name: **OpenAI-<inject key="Deployment ID" enableCopy="false"/>** **(4)**

    - Pricing Tier: **Standard SO (5)**

   ![](./Images/Image3.png)

1. Click on Next twice and click on **Review + Submit**.

1. Review all the values and click on **Create**.

1. Once the deployment is complete, click on **Go to resource**

1. In the Azure OpenAI resource pane, click on **Go to Azure AI Foundry portal**, it will navaigate to Azure AI Foundry portal.

   ![](./Images/Image4.png)

1. On the left panel select **Deployments (1)**. Click on **+ Deploy Model (2)** and select **Deploy Base Model (3)**.

   ![](./Images/Image5.png)

1. Search for **gpt-4o (1)**, **select it (2)**, and click on **Confirm (3)**.

   ![](./Images/Image6.png)

1. Click on **Customize** and provide the following details to deploy a gpt-4o model:

    - Deployment name: **gpt4-o (1)**
    - Deployment type: **Global Standard (2)**
    - Model Version: **2024-11-20 (3)**
    - Set the **Tokens per Minute Rate Limit** to **200k (4)**.
    - Leave the other values to default and click on **Deploy (5)**.

   ![](./Images/Image7a.png)

1. Once the gpt-4o deployment gets completed, copy the **Target URI (1)** and **Key (2)**. **Paste** these values in a notepad for further use. 

   ![](./Images/Image8.png)

1. Open VS Code on your Lab VM. Click on **File (1)** and select **Open Folder (2)**.

   ![](./Images/Image9.png)

1. Navigate to the path `C:\LabFiles\` **(1)**, select **CAPSTONE-PROJECT (2)** and click on **Select Folder (3)**.

   ![](./Images/Image10a.png)

1. Select the **checkbox (1)** and click **Yes,I trust the authors(2)** to proceed.

   ![](./Images/Image11a.png)

1. Expand the **src/ui** folder, rename the file from **Sample.env** to **.env**.

   ![](./Images/Image12a.png)

1. Update the `.env` file with the Azure AI Foundry deployment details and save the file:

    ```
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=Replace with your deployment name
    AZURE_OPENAI_ENDPOINT=Replace with your endpoint URL
    AZURE_OPENAI_API_KEY=Replace with your API key
    ```
   ![](./Images/Image13a.png)

## Task 2 - Create a GitHub Repository and Generate a PAT Token

1. Sign in to GitHub at [https://github.com](https://github.com).  

1. Create a new repository named **Capstone-Project-<inject key="Deployment ID" enableCopy="false"/>** **(1)**. Set the repository visibility to **Public (2)**, then click **Create Repository (3)**.

   ![](./Images/Image16.png)

1. Once the new repository is created, **copy the URL** of your repo and paste it into a notepad for future use.
 
   ![](./Images/Image25.png)

1. Click your **profile picture (1)** at the top-right corner and select **Settings (2)** from the dropdown menu.
 
   ![](./Images/Image17.png)

1. In the left sidebar, click **<> Developer settings**.  

   ![](./Images/Image18.png)

1. Expand **Personal access tokens** from the left panel. Select **Fine-grained tokens(1)** and click on **Generate new token (2)**.

   ![](./Images/Image19.png)
  
1. Enter **<inject key="Deployment ID" enableCopy="false"/>-PAT-RepoAccess** **(1)** as the name for your token. Set an expiration date to **30 days (2)**.  

   ![](./Images/Image20.png)

1. Scroll down and under **Repository Access**, click on **Only select repositories (1)**. Search for the **Capstone-Project-<inject key="Deployment ID" enableCopy="false"/> repository** **(2)** and **select it (3)**.

   ![](./Images/Image21.png)  

1. Under **Permissions**, expand **Repository Permissions (1)**. Provide **Read and Write (3)** access for **Contents (2)** under Repository permissions.

   ![](./Images/Image22.png)  

1. Scroll down to the bottom of the page, click **Generate token (1)**, and in the popup, review the permissions and click **Generate token (2)**.

   ![](./Images/Image23.png)  

1. **Copy (1)** the generated token and **paste** it into a notepad for future use.

   ![](./Images/Image24.png)
   
## Task 3 - Define Agent Personas and Configure Multi-Agent Chat

1. Open the `multi_agent.py` file. This is where you will implement all necessary code for this challenge.
   
1. Replace the code in the **multi_agent.py** file with the code from the link below and save the file.

    ```
    https://docs-api.cloudlabs.ai/repos/raw.githubusercontent.com/CloudLabsAI-Azure/Capstone-Project/refs/heads/soln-guide/src/ui/multi_agent.py
    ```

    ![](./Images/Image14a.png)

1. Create a file named `push_to_github.sh` under the `src/ui` directory. Paste the code from the link below and save the file.

    ```
    https://docs-api.cloudlabs.ai/repos/raw.githubusercontent.com/CloudLabsAI-Azure/Capstone-Project/refs/heads/soln-guide/src/ui/push_to_github.sh
    ```

    ![](./Images/Image15.png)

1. Update the following env variables in `.env` file with the values you copied in Task 2 and save the file.
    ```
    GITHUB_REPO_URL=Replace with your Github Repo
    GITHUB_PAT=Replace with your Github pat token
    GIT_USER_EMAIL=Replace with your Github email
    GITHUB_USERNAME=Replace with your Github username
    ```
    ![](./Images/Image27.png)

1. In the .env file, click on **CRLF (1)** in the bottom Status Bar and change it to **LF (2)** by selecting it. Save the file after making this change.

    ![](./Images/Image35.png)

1. Verify that **LF** is selected in the push_to_github.sh file as well.

    ![](./Images/Image36.png)

1. Click on the **ellipses (1)**. Select **Terminal (2)** and choose **New Terminal (3)**.

    ![](./Images/Image26.png)

1.  Run the following command:-

    ```
    azd auth login
    ```

    ![](./Images/Image28.png)

1. Sign in using the following credentials:-
    - **Email/Username**: <inject key="AzureAdUserEmail"></inject>
    - **Password**: <inject key="AzureAdUserPassword"></inject>

1. Run the following command to provision the web app and required resources to azure:-

    ```
    azd up
    ```
1. When running the **azd up** command, you'll be asked to provide configuration details interactively. Provide the following values when prompted:

   - **Unique Environment Name**: Enter **CapstoneEnv** **(1)**.
   - **Azure Subscription to use**: Choose the default subscription **(2)** that appears and press **Enter**.
   - **Location Infrastructure Parameter**: Select **East US 2** **(3)** from the options and press **Enter**.
   - **ResourceGroupName Infrastructure Parameter**: Type **CapstoneEnv** **(4)** and press **Enter**.
   - **Resource Group to use**: Select **CapstoneEnv** **(5)** from the options and press **Enter**.


   ![](./Images/Image38.png)

   - **Note:** Wait for 5 minutes until the command runs completely. 

## Task 4 - Multi-Agent Code Generation and Repository Integration

1. Navigate to the azure portal, and select the newly created Resource group named **rg-CapstoneEnv<inject key="Deployment ID" enableCopy="false"/>**.

1. Open the container app with prefix **dev-ui-**.

    ![](./Images/Image30.png)

1. Click on the **Application URL** present on the Overview page of the Container app.

    ![](./Images/Image31.png)

1. The Streamlit chat application will open. Try providing the **below prompt (1)** in the chat and click on **send**.

    ```
    Create code for simple calculator
    ```
   - **Note:** Wait until the agents collaborate and provide a reply.

    ![](./Images/Image32.png)

1. Once it runs and provides the code and other details, type **approved (1)** and select **send (2)** to approve the code. At the end of the chat, you can observe that the code is being pushed to the repository after approval.

    ![](./Images/Image33.png)

    ![](./Images/Image34.png)

1. Navigate to the repository **Capstone-Project-<inject key="Deployment ID" enableCopy="false"/>** and verify that the `generated_app.html` file has been created, containing the code for your simple calculator.

    ![](./Images/Image37.png)

## Success Criteria

- You have implemented the Multi-Agent Chat system that produces:
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

## Conclusion

This challenge demonstrated how to build and coordinate a Multi-Agent System using Azure AI Foundry and Semantic Kernel. By designing distinct personas for Business Analyst, Software Engineer, and Product Owner, and configuring a group chat environment with a termination strategy, you created a collaborative AI workflow capable of gathering requirements, developing code, and performing code reviews. The task structure allows for scalable, decentralized handling of complex problems using autonomous, interactive agents.

# You have successfully completed the Lab !!
