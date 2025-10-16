
# Laboratório de Desafios

# Projeto Capstone

### Overall Estimated Duration: 4 Hours

## Visão Geral

Neste desafio, você trabalhará com uma aplicação conversacional baseada em Chainlit que utiliza o modelo de mensagens publish-subscribe (pub/sub) do Dapr para orquestrar escalonamentos de atendimento ao cliente através de agentes de IA inteligentes. A solução integra perfeitamente vários serviços do Azure — incluindo Azure OpenAI para processamento de linguagem natural, Cosmos DB para persistência de dados e Azure Service Bus para mensagens confiáveis. Quando os agentes de IA atingem seus limites de resolução, o sistema escala inteligentemente o caso para um agente de suporte humano, acionando um fluxo de trabalho do Logic Apps que envia uma solicitação de aprovação por email. Este laboratório prático oferece um insight valioso sobre como interfaces conversacionais impulsionadas por IA, arquitetura orientada a eventos e automação de fluxos de trabalho podem ser combinadas para criar sistemas de suporte ao cliente responsivos, escaláveis e conscientes da interação humana.


## Objetivo

Ao final deste laboratório, você será capaz de:

- **Fluxo de Trabalho Multi-Agente com Personas de Negócio**: Experimentar um fluxo de trabalho colaborativo envolvendo três personas distintas — Software Engineer, Product Owner e User. O Software Engineer escreve e submete o código, o Product Owner revisa e aprova as alterações e, após a aprovação, a solução é automaticamente enviada para o GitHub para implantação e controle de versão.


## Pré-requisitos

Os participantes devem possuir:

- Conhecimentos básicos dos serviços Azure, como Azure OpenAI e seus modelos.
- Experiência com implantação de aplicações usando o Azure Developer CLI (AZD).
- Familiaridade com ferramentas de IA conversacional como Streamlit ou frameworks semelhantes.


## Explicação dos Componentes

- **Azure OpenAI**: Um serviço na nuvem que fornece acesso a modelos avançados de linguagem, permitindo processamento de linguagem natural, geração de conteúdo e capacidades de IA conversacional. Permite integrar funcionalidades poderosas baseadas em IA em suas aplicações de forma segura e em escala.

- **Azure Container Apps**: Um serviço de hospedagem de containers serverless que permite implantar e escalar microsserviços e aplicações conteinerizadas sem a necessidade de gerenciar infraestrutura.


## Iniciando o laboratório

Bem-vindo ao seu Workshop Azure Agentic AI. Vamos começar a tirar o máximo proveito desta experiência:

## Acesso ao seu Ambiente de Laboratório

Quando estiver pronto para começar, sua máquina virtual e o **guia do laboratório** estarão disponíveis diretamente no seu navegador web.

![Acesse sua VM e o Guia do Laboratório](./media/Agg1.png)

## Zoom In/Zoom Out do Guia do Laboratório

Para ajustar o nível de zoom na página do ambiente, clique no ícone **A↕ : 100%** localizado ao lado do temporizador no ambiente do laboratório

![](./media/Agg2.png)

## Máquina Virtual & Guia do Laboratório

Sua máquina virtual será sua ferramenta principal durante todo o workshop. O guia do laboratório é o seu roteiro para o sucesso.

## Explorando os Recursos do Seu Laboratório

Para compreender melhor os recursos e credenciais do seu laboratório, navegue até a aba **Environment**.

![Explore os Recursos do Laboratório](./media/Agg3.png)

## Utilizando o Recurso de Janela Dividida

Para maior conveniência, você pode abrir o guia do laboratório em uma janela separada selecionando o botão **Split Window** no canto superior direito.

![Use o Recurso de Janela Dividida](./media/Agg4.png)

## Gerenciando sua Máquina Virtual

Sinta-se à vontade para **iniciar, parar ou reiniciar (2)** sua máquina virtual conforme necessário na aba **Resources (1)**. A experiência está em suas mãos!

![Gerencie sua Máquina Virtual](./media/Agg5.png)

> **Observação:** Por favor, certifique-se de que o script continue rodando e não seja encerrado após acessar o ambiente.


## Vamos Começar com o Azure Portal

1. Na sua máquina virtual, clique no ícone do Azure Portal.  
2. Você verá a aba **Sign into Microsoft Azure**. Aqui, insira suas credenciais:

   - **Email/Username:** <inject key="AzureAdUserEmail"></inject>

     ![Digite seu nome de usuário](./media/gt-5.png)

3. Em seguida, forneça sua senha:

   - **Password:** <inject key="AzureAdUserPassword"></inject>

     ![Digite sua senha](./media/gt-4.png)


4. Se a janela pop-up **Action required** aparecer, clique em **Ask later**.  
5. Se for solicitado para **manter-se conectado**, você pode clicar em **No**.  
6. Se a janela pop-up **Welcome to Microsoft Azure** aparecer, simplesmente clique em **"Cancel"** para pular o tour.

## Passos para entrar no GitHub

1. Abra uma nova guia do navegador e acesse a seguinte URL: **[https://github.com/login](https://github.com/login)**

   ![Digite sua senha](./media/Git1.png)

1. Na página Sign in to GitHub, insira suas credenciais pessoais do GitHub em **Username (1)** e **Password (2)** e, em seguida, clique em **Sign in (3)**.

   ![Digite sua senha](./media/Git2.png)

1. Se for solicitado um código de verificação, insira o código de verificação do GitHub recebido no seu e-mail registrado ou no aplicativo de autenticação.


## Passos para Configurar o MFA Caso a Opção "Ask Later" Não Esteja Visível

1. No prompt **"More information required"**, selecione **Next**.

1. Na página **"Keep your account secure"**, selecione **Next** duas vezes.

1. **Observação:** Se você não tiver o aplicativo Microsoft Authenticator instalado no seu dispositivo móvel:

   - Abra a **Google Play Store** (Android) ou a **App Store** (iOS).
   - Procure por **Microsoft Authenticator** e toque em **Install**.
   - Abra o aplicativo **Microsoft Authenticator**, selecione **Add account** e escolha **Work or school account**.

1. Um **QR code** será exibido na tela do seu computador.

1. No aplicativo Authenticator, selecione **Scan a QR code** e escaneie o código exibido na sua tela.

1. Após escanear, clique em **Next** para continuar.

1. No seu telefone, insira o número exibido na tela do computador no aplicativo Authenticator e selecione **Next**.

1. Se for solicitado para manter-se conectado, você pode clicar em **No**.

1. Se a janela pop-up **Welcome to Microsoft Azure** aparecer, simplesmente clique em **Maybe Later** para pular o tour.


## Support Contact

A equipe de suporte do CloudLabs está disponível 24 horas por dia, 7 dias por semana, 365 dias por ano, via email e chat ao vivo para garantir assistência contínua a qualquer momento. Oferecemos canais de suporte dedicados, específicos para aprendizes e instrutores, garantindo que todas as suas necessidades sejam atendidas rápida e eficientemente.

Contatos de Suporte para Aprendizes:

- Suporte por Email: [cloudlabs-support@spektrasystems.com](mailto:cloudlabs-support@spektrasystems.com)
- Suporte via Chat ao Vivo: https://cloudlabs.ai/labs-support

Clique em **Next** no canto inferior direito para iniciar sua jornada no Lab!

![Start Your Azure Journey](./media/Agg6.png)

Agora você está pronto para explorar o poderoso mundo da tecnologia. Fique à vontade para entrar em contato se tiver alguma dúvida durante o caminho. Aproveite seu workshop!
