# Desafio - Sistemas Multiagentes

## Introdução

Os Sistemas Multiagentes (MAS) consistem em vários agentes autônomos, cada um com objetivos, comportamentos e áreas de responsabilidade distintos. Estes agentes operam de forma independente, tomando decisões com base no seu conhecimento local e ambiente. No entanto, também podem comunicar e partilhar informações entre si, cooperando ou competindo em função dos seus objetivos. O MAS é normalmente usado em cenários em que as tarefas são distribuídas por várias entidades e o sistema se beneficia da descentralização. As aplicações comuns incluem gestão de tráfego, equipas robóticas, IA distribuída e sistemas em rede onde a coordenação é necessária sem depender de um controlador central.

Neste desafio, você criará um Sistema Multiagente que aceita a solicitação de um usuário e a processa por meio de uma coleção de agentes, cada um projetado com uma persona específica e área de especialização. Os agentes analisarão individualmente o pedido e contribuirão com as suas respostas com base nas suas responsabilidades definidas. O resultado final será uma coleção consolidada de respostas de todos os agentes, abordando colaborativamente a consulta do usuário de uma forma que reflita a perspetiva única de cada persona.

## Tarefa 1 - Implantação do Modelo Microsoft Foundry e Configuração do Ambiente

1. **Implantação do Serviço Azure OpenAI:**

    - Configure uma instância do Serviço Azure OpenAI com SKU padrão `S0`.

        > **Nota:** Certifique-se de que a região esteja definida para **East US**.

    - Implante-o em um grupo de recursos com prefixo `openaiagents`.

    - Obtenha a Chave e o Endpoint do Azure OpenAI.

2. **Implantar Modelos Azure OpenAI:**
   
    - O Azure OpenAI disponibiliza um portal web chamado **Microsoft Foundry Portal** que pode ser usado para implantar, gerir e explorar modelos. Você começará sua exploração do Azure OpenAI utilizando o Microsoft Foundry para implantar um modelo.
    
    - Inicie o Microsoft Foundry Portal a partir do painel de visão geral e implante um Modelo Azure OpenAI, ou seja, `gpt-4o`.

        >- **Nota:** Certifique-se de que as implantações sejam nomeadas **gpt-4o**.
        >- **Nota:** Garanta que o Tipo de Implantação esteja definido como **Global Standard** e use a versão do modelo **2024-11-20**.

    - Obtenha o **Nome da Implantação** e a **Versão da API** do modelo.

        >- **Dica:** A versão da API pode ser obtida a partir do URI de Destino.

1. Atualize o arquivo `.env` com os detalhes da implantação do Microsoft Foundry:

    ```
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=Replace with your deployment name
    AZURE_OPENAI_ENDPOINT=Replace with your endpoint URL
    AZURE_OPENAI_API_KEY=Replace with your API key
    AZURE_OPENAI_API_VERSION=Replace with your API version
    ```
    
> **Parabéns** por completar a tarefa! Agora, é hora de validá-la. Aqui estão os passos:
> - Se você receber uma mensagem de sucesso, pode prosseguir para a próxima tarefa.
> - Se não, leia cuidadosamente a mensagem de erro e tente novamente o passo, seguindo as instruções do guia do laboratório.
> - Se precisar de ajuda, entre em contato conosco pelo email cloudlabs-support@spektrasystems.com. Estamos disponíveis 24 horas por dia, 7 dias por semana para ajudar você.
  
<validation step="8d299641-fbeb-4957-a43d-2227a0f87de6" />

## Tarefa 2 - Configurar Fluxo de Trabalho Multi-Agente e Automatizar o Envio de Código Após Aprovação

1. Abra o **VS Code** na sua **Lab VM**. Em seguida, abra a pasta **CAPSTONE-PROJECT** no **VS Code** a partir do caminho **C:\LabFiles\\**

1. Abra o arquivo `multi_agent.py`. É aqui que você vai implementar todo o código necessário para este desafio.

1. Crie personas para os três agentes com as seguintes instruções:


    - **Analista de Negócios Persona**

        ```
        You are a Business Analyst which will take the requirements from the user (also known as a 'customer') and create a project plan for creating the requested app. The Business Analyst understands the user requirements and creates detailed documents with requirements and costing. The documents should be usable by the SoftwareEngineer as a reference for implementing the required features, and by the Product Owner for reference to determine if the application delivered by the Software Engineer meets all of the user's requirements.
        ```

    - **Engenheiro de Software Persona**

        ```
        You are a Software Engineer, and your goal is create a web app using HTML and JavaScript by taking into consideration all the requirements given by the Business Analyst. The application should implement all the requested features. Deliver the code to the Product Owner for review when completed. You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear.
        ```

    - **Persona de Product Owner**

        ```
        You are the Product Owner which will review the software engineer's code to ensure all user  requirements are completed. You are the guardian of quality, ensuring the final product meets all specifications. IMPORTANT: Verify that the Software Engineer has shared the HTML code using the format ```html [code] ```. This format is required for the code to be saved and pushed to GitHub. Once all client requirements are completed and the code is properly formatted, reply with 'READY FOR USER APPROVAL'. If there are missing features or formatting issues, you will need to send a request back to the SoftwareEngineer or BusinessAnalyst with details of the defect.
        ```

1. Crie um `ChatCompletionAgent` para cada uma das personas acima. Cada agente deve ter:
    - Instruções (o prompt da persona)
    - Um Nome único (apenas letras, sem espaços ou caracteres especiais)
    - Uma referência a um objeto `Kernel`

1. Crie um objeto `AgentGroupChat` para unir os três agentes. Passe:
    - Um array com os três agentes
    - `ExecutionSettings` com uma `TerminationStrategy` configurada como uma instância de `ApprovalTerminationStrategy`

1. Implemente o método `should_agent_terminate` na classe `ApprovalTerminationStrategy`. Os agentes devem terminar quando o Usuário retornar "APPROVED" no histórico do chat.

1. Após implementar o método `should_agent_terminate` para detectar "APPROVED", adicione um callback ou etapa de pós-processamento que execute quando esta condição for satisfeita.
1. Extraia o código HTML fornecido pelo agente Software Engineer do histórico do chat.
1. Salve o código extraído em um arquivo (por exemplo, `index.html`).
1. Crie um script Bash (por exemplo, `push_to_git.sh`) que faça stage, commit e push do arquivo para o repositório Git desejado:
1. No seu código Python, utilize o módulo `subprocess` para chamar este script quando "APPROVED" for detectado:
1. Garanta que seu ambiente possua as credenciais Git necessárias configuradas para pushes não interativos.

   Esta automação assegura que, uma vez que o Product Owner (ou usuário) envie "APPROVED", o código mais recente seja automaticamente enviado para o seu repositório Git.

> **Parabéns** por completar a tarefa! Agora, é hora de validá-la. Aqui estão os passos:
> - Se você receber uma mensagem de sucesso, pode prosseguir para a próxima tarefa.
> - Se não, leia cuidadosamente a mensagem de erro e tente novamente o passo, seguindo as instruções do guia do laboratório.
> - Se precisar de ajuda, entre em contato conosco pelo email cloudlabs-support@spektrasystems.com. Estamos disponíveis 24 horas por dia, 7 dias por semana para ajudar você.

<validation step="45b2dc3e-4ea3-4170-8002-9ede8397a481" />

## Tarefa 3 - Executar a Conversa Multi-Agente e Validar o Fluxo de Trabalho

1. Implemente o código para enviar uma mensagem do usuário ao grupo de agentes usando `add_chat_message` no objeto `AgentGroupChat`. A mensagem deve incluir:
    - `AuthorRole.User` como autor
    - O conteúdo da mensagem do chat com base na entrada do usuário

1. Itere pelas respostas do `AgentGroupChat` utilizando um loop assíncrono, e imprima cada mensagem conforme ela chegar:

    ```python
    async for content in chat.invoke():
        print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
    ```

1. Execute sua aplicação e envie uma solicitação para construir uma aplicação de calculadora. Observe como o Business Analyst, Software Engineer e Product Owner colaboram para planejar, construir e aprovar a solução.

## Tarefa 4 - Implantar App no Azure com Registro de Contêiner e App Service

Para hospedar sua aplicação online usando o Azure, siga estes passos para conteinerizar sua aplicação, enviá-la ao Azure Container Registry (ACR) e implantá-la usando o Azure App Service:

1. Abra um terminal e faça login no Azure Developer CLI usando o seguinte comando:

    ```bash
    azd auth login
    ```

1. Implemente os recursos necessários no Azure executando:

    ```bash
    azd up
    ```

1. Ao executar o comando **azd up**, será solicitado que você forneça detalhes de configuração de forma interativa. Forneça os seguintes valores quando solicitado:

   - **Nome Único do Ambiente**: Insira **CapstoneEnv**.
   - **Subscrição Azure a utilizar**: Escolha a subscrição padrão que aparecer e pressione **Enter**.
   - **Parâmetro de Localização da Infraestrutura**: Selecione **East US 2** nas opções e pressione **Enter**.
   - **Parâmetro ResourceGroupName da Infraestrutura**: Digite **CapstoneEnv** e pressione **Enter**.
   - **Grupo de Recursos a utilizar**: Selecione **CapstoneEnv** nas opções e pressione **Enter**.

1. Abra o portal Azure e navegue até o grupo de recursos **CapstoneEnv**.
2. Localize o recurso do container app implantado.
3. Copie a URL do endpoint do container app.
4. Acesse a aplicação web visitando esse endpoint no seu navegador e verifique se a aplicação funciona conforme esperado.

> **Parabéns** por completar a tarefa! Agora, é hora de validá-la. Aqui estão os passos:
> - Se você receber uma mensagem de sucesso, pode prosseguir para a próxima tarefa.
> - Se não, leia cuidadosamente a mensagem de erro e tente novamente o passo, seguindo as instruções do guia do laboratório.
> - Se precisar de ajuda, entre em contato conosco pelo email cloudlabs-support@spektrasystems.com. Estamos disponíveis 24 horas por dia, 7 dias por semana para ajudar você.

<validation step="be861c77-e528-4b89-842a-156752ef20a1" />

## Critérios de Sucesso

- Você implementou o sistema Multi-Agente de Chat que produz:
    - Geração de código-fonte completo em HTML e JavaScript para a aplicação solicitada
    - Processo rigoroso de revisão e aprovação de código pelo Usuário
    - Implantação automatizada da aplicação no Azure
    - Envio automatizado do código para um repositório Git após aprovação do usuário

## Bónus

- Copie o código do histórico do chat em markdown para os ficheiros correspondentes no seu sistema de ficheiros.
- Guarde o conteúdo HTML como `index.html` e abra-o no seu navegador web.
- Teste se a aplicação funciona conforme descrito pela IA.
- Melhore a aplicação pedindo à IA para torná-la responsiva ou adicionar novas funcionalidades.
- Experimente modificar as personas para melhorar os resultados ou a funcionalidade.

## Recursos de aprendizagem

- [Agent Group Chat with Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-chat?pivots=programming-language-python)
- [MetaGPT](https://github.com/geekan/MetaGPT)
- [AutoGen Multi-Agent Conversational Framework](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat/)
- [AutoGen with Semantic Kernel](https://devblogs.microsoft.com/semantic-kernel/autogen-agents-meet-semantic-kernel/)
- [Managing your personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)


## Conclusão

Este desafio demonstrou como construir e coordenar um Sistema Multi-Agente usando Microsoft Foundry e Semantic Kernel. Ao desenhar personas distintas para Business Analyst, Software Engineer e Product Owner, e configurar um ambiente de chat em grupo com uma estratégia de terminação, você criou um fluxo de trabalho colaborativo de IA capaz de recolher requisitos, desenvolver código e realizar revisões de código. A estrutura da tarefa permite um tratamento escalável e descentralizado de problemas complexos utilizando agentes autónomos e interativos.

