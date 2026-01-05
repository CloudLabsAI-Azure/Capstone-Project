# 도전과제 - 다중 에이전트 시스템

## 소개

MAS(Multi-Agent System)는 각각 고유한 목표, 동작 및 책임 영역을 가진 여러 자율 에이전트로 구성됩니다. 이러한 에이전트는 독립적으로 운영되며 현지 지식과 환경에 따라 결정을 내립니다. 그러나 그들은 또한 목표에 따라 협력하거나 경쟁하면서 서로 소통하고 정보를 공유할 수 있습니다. MAS는 일반적으로 작업이 여러 엔터티에 분산되어 있고 시스템이 분산화의 이점을 누리는 시나리오에서 사용됩니다. 일반적인 애플리케이션에는 교통 관리, 로봇 팀, 분산된 AI 및 중앙 컨트롤러에 의존하지 않고 조정이 필요한 네트워크 시스템이 포함됩니다.

이 챌린지에서는 사용자의 요청을 수락하고 각각 특정 페르소나와 전문 분야로 설계된 에이전트 컬렉션을 통해 처리하는 다중 에이전트 시스템을 만듭니다. 상담원은 개별적으로 요청을 분석하고 정의된 책임에 따라 응답을 제공합니다. 최종 출력은 모든 상담원의 답변을 통합한 모음으로, 각 페르소나의 고유한 관점을 반영하는 방식으로 사용자의 쿼리를 공동으로 해결합니다.

## 작업 1 - Microsoft Foundry 모델 배포 및 환경 구성

1. **Azure OpenAI 서비스 배포:**

    - SKU 크기 **Standard S0**로 Azure OpenAI 서비스 인스턴스를 설정하세요.

        > **참고:** 지역은 반드시 **East US**로 설정하세요.

    - `openaiagents` 접두사가 붙은 리소스 그룹에 배포하세요.

    - Azure OpenAI 키와 엔드포인트를 가져오세요.

1. **Azure OpenAI 모델 배포:**
   
    - Azure OpenAI에서는 **Microsoft Foundry 포털**이라는 웹 기반 포털을 제공하여 모델을 배포, 관리, 탐색할 수 있습니다. Microsoft Foundry를 사용하여 모델을 배포하는 것으로 시작하세요.
    
    - 개요 창에서 Microsoft Foundry 포털을 실행하고, `gpt-4o` 모델을 배포하세요.

        >- **참고:** 배포 이름은 반드시 **gpt-4o**로 지정하세요.
        >- **참고:** 배포 유형은 **Global Standard**로 설정하고, 모델 버전은 **2024-11-20**을 사용하세요.

    - **배포 이름**과 **API 버전**을 가져오세요.

        >- **힌트:** API 버전은 Target URI에서 확인할 수 있습니다.

1. Microsoft Foundry 배포 정보를 사용하여 `.env` 파일을 다음과 같이 업데이트하세요:

    ```
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=본인의 배포 이름으로 교체
    AZURE_OPENAI_ENDPOINT=엔드포인트 URL로 교체
    AZURE_OPENAI_API_KEY=API 키로 교체
    AZURE_OPENAI_API_VERSION=API 버전으로 교체
    ```

> **축하합니다** 작업을 완료하셨습니다! 이제 검증할 시간입니다. 다음 단계에 따라 진행하세요:
> - 성공 메시지가 표시되면 다음 작업으로 진행할 수 있습니다.
> - 그렇지 않으면 오류 메시지를 주의 깊게 읽고 실습 가이드의 지침에 따라 단계를 다시 시도하세요.
> - 도움이 필요하시면 언제든지 cloudlabs-support@spektrasystems.com 으로 연락 주세요. 24시간 연중무휴로 지원합니다.

<validation step="0785507e-e2d1-445e-8120-2116c5c4ddfc" />

## 작업 2 - 다중 에이전트 워크플로우 구성 및 승인 시 코드 자동 푸시 설정

1. `multi_agent.py` 파일을 엽니다. 이 파일에서 이번 과제에 필요한 모든 코드를 구현하게 됩니다.

1. 다음 지침에 따라 세 명의 에이전트에 대한 페르소나를 생성합니다:

    - **Business Analyst 페르소나**

        ```
        You are a Business Analyst which will take the requirements from the user (also known as a 'customer') and create a project plan for creating the requested app. The Business Analyst understands the user requirements and creates detailed documents with requirements and costing. The documents should be usable by the SoftwareEngineer as a reference for implementing the required features, and by the Product Owner for reference to determine if the application delivered by the Software Engineer meets all of the user's requirements.
        ```

    - **소프트웨어 엔지니어 페르소나**

        ```
        You are a Software Engineer, and your goal is create a web app using HTML and JavaScript by taking into consideration all the requirements given by the Business Analyst. The application should implement all the requested features. Deliver the code to the Product Owner for review when completed. You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear.
        ```

    - **제품 소유자 페르소나**

        ```
        You are the Product Owner which will review the software engineer's code to ensure all user  requirements are completed. You are the guardian of quality, ensuring the final product meets all specifications. IMPORTANT: Verify that the Software Engineer has shared the HTML code using the format ```html [code] ```. This format is required for the code to be saved and pushed to GitHub. Once all client requirements are completed and the code is properly formatted, reply with 'READY FOR USER APPROVAL'. If there are missing features or formatting issues, you will need to send a request back to the SoftwareEngineer or BusinessAnalyst with details of the defect.
        ```

3. 위에서 정의한 각 페르소나에 대해 `ChatCompletionAgent`를 생성합니다. 각 에이전트는 다음을 포함해야 합니다:
    - Instructions (페르소나 프롬프트)
    - 고유한 Name (영문자만 사용, 공백이나 특수문자 불가)
    - `Kernel` 객체에 대한 참조

4. 세 개의 에이전트를 함께 연결하기 위해 `AgentGroupChat` 객체를 생성합니다. 다음을 전달합니다:
    - 세 개의 에이전트가 담긴 배열
    - `ExecutionSettings`와 `TerminationStrategy`를 `ApprovalTerminationStrategy`의 인스턴스로 설정

5. `ApprovalTerminationStrategy` 클래스에서 `should_agent_terminate` 메서드를 구현합니다. 사용자 채팅 기록에 "APPROVED"가 반환되었을 때 에이전트가 종료되도록 설정합니다.

6. "APPROVED"를 감지하는 `should_agent_terminate` 메소드를 구현한 후 이 조건이 충족될 때 실행되는 콜백 또는 사후 처리 단계를 추가합니다.
7. 채팅 기록에서 소프트웨어 엔지니어 에이전트가 제공한 HTML 코드를 추출합니다.
8. 추출된 코드를 파일(예: index.html)에 저장합니다.
9. 파일을 스테이징, 커밋 및 원하는 Git 리포지토리로 푸시하는 Bash 스크립트(예: push_to_git.sh)를 만듭니다.
10. Python 코드에서 `subprocess` 모듈을 사용하여 "APPROVED"가 감지될 때 이 스크립트를 호출합니다.
11. 환경에 비대화형 푸시에 대해 구성된 필요한 Git 자격 증명이 있는지 확인합니다.

이 자동화를 통해 제품 소유자(또는 사용자)가 "APPROVED"를 전송하면 최신 코드가 Git 리포지토리에 자동으로 푸시됩니다.

> **축하합니다** 작업을 완료하셨습니다! 이제 검증할 차례입니다. 다음 단계를 따라주세요:
> - 성공 메시지가 표시되면 다음 작업으로 진행할 수 있습니다.
> - 그렇지 않은 경우, 오류 메시지를 주의 깊게 읽고 실습 가이드의 지침에 따라 해당 단계를 다시 시도하세요.
> - 도움이 필요하시면 언제든지 cloudlabs-support@spektrasystems.com 으로 연락해 주세요. 24시간 연중무휴 지원합니다.

<validation step="5c335ab0-2d34-4343-abfb-a995c9f945db" />


## 작업 3 - 다중 에이전트 대화 실행 및 워크플로우 검증

1. 사용자 메시지를 `AgentGroupChat` 객체에 전달하기 위해 `add_chat_message` 코드를 구현합니다. 메시지에는 다음을 포함해야 합니다:
    - 작성자(author)로 `AuthorRole.User`
    - 사용자 입력에서 가져온 채팅 메시지 내용

1. `AgentGroupChat`의 응답을 비동기 반복문을 사용하여 순회하고, 각 메시지가 도착할 때마다 출력합니다:

    ```python
    async for content in chat.invoke():
        print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
    ```

1. 애플리케이션을 실행하고 계산기 앱을 만들어달라는 요청을 전달합니다. Business Analyst, Software Engineer, Product Owner가 협업하여 어떻게 계획하고, 개발하고, 솔루션을 승인하는지 확인해보세요.


## 작업 4 - 컨테이너 레지스트리와 Azure App Service를 사용하여 앱 배포

Azure를 사용하여 온라인으로 앱을 호스트하려면 다음 단계에 따라 애플리케이션을 컨테이너화하고, ACR(Azure Container Registry)에 푸시하고, Azure App Service를 사용하여 배포합니다.

1. 터미널을 열고 다음 명령을 사용하여 Azure Developer CLI에 로그인합니다.

    ```bash
    azd auth login
    ```

1. 다음을 실행하여 필요한 리소스를 Azure에 배포합니다.

    ```bash
    azd up
    ```

1. azd up 명령을 실행할 때 구성 세부 정보를 대화형으로 제공하라는 메시지가 표시됩니다. 메시지가 표시되면 다음 값을 제공합니다.

   - **고유 환경 이름**: **CapstoneEnv** 입력하세요.  
   - **사용할 Azure 구독**: 기본 구독  을 선택하고 **Enter** 키를 누르세요.  
   - **위치 인프라 매개변수**: 옵션에서 **East US 2**  를 선택하고 **Enter** 키를 누르세요.  
   - **ResourceGroupName 인프라 매개변수**: **CapstoneEnv** 입력하세요.  
   - **사용할 리소스 그룹**: 옵션에서 **CapstoneEnv** 를 선택하고 **Enter** 키를 누르세요.  

1. Azure 포털을 열고 리소스 그룹 **CapstoneEnv** 으로 이동합니다.  
1. 배포된 컨테이너 앱 리소스를 찾습니다.  
1. 컨테이너 앱의 엔드포인트 URL을 복사합니다.  
1. 이 엔드포인트를 브라우저에서 열어 웹 앱에 접속하고, 애플리케이션이 정상적으로 작동하는지 확인합니다.  

> **축하합니다** 작업을 완료하셨습니다! 이제 검증할 시간입니다. 다음 단계들을 따라 주세요:
> - 성공 메시지가 표시되면 다음 작업으로 진행할 수 있습니다.
> - 그렇지 않은 경우, 오류 메시지를 주의 깊게 읽고 실습 가이드의 지침에 따라 해당 단계를 다시 시도하세요.
> - 도움이 필요하시면 언제든지 cloudlabs-support@spektrasystems.com 으로 연락해 주세요. 24시간 365일 지원해 드립니다.

<validation step="a59f082a-271a-40c7-8ff1-bd4c7b536c5d" />

## 성공 기준

- 다중 에이전트 채팅 시스템을 구현하여 다음을 수행합니다:  
    - 요청된 애플리케이션에 대한 완전한 HTML 및 JavaScript 소스 코드 생성  
    - 사용자에 의한 철저한 코드 검토 및 승인 과정  
    - 애플리케이션의 Azure 자동 배포  
    - 사용자 승인 시 Git 저장소로 자동 코드 푸시  

## 보너스

- 채팅 기록의 마크다운에서 코드를 복사하여 파일 시스템의 해당 파일에 붙여넣기 합니다.  
- HTML 내용을 **index.html** 파일로 저장하고 웹 브라우저에서 실행합니다.  
- 애플리케이션이 AI가 설명한 대로 작동하는지 테스트합니다.  
- AI에게 반응형 디자인을 적용하거나 새로운 기능을 추가하도록 요청하여 앱을 향상시킵니다.  
- 페르소나를 수정하여 결과나 기능을 개선하는 실험을 해봅니다.  

## 학습 자료

- [Agent Group Chat with Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-chat?pivots=programming-language-python)
- [MetaGPT](https://github.com/geekan/MetaGPT)
- [AutoGen Multi-Agent Conversational Framework](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat/)
- [AutoGen with Semantic Kernel](https://devblogs.microsoft.com/semantic-kernel/autogen-agents-meet-semantic-kernel/)
- [Managing your personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

## 결론

이 챌린지에서는 Microsoft Foundry 및 Semantic Kernel을 사용하여 다중 에이전트 시스템을 빌드하고 조정하는 방법을 보여 주었습니다. 비즈니스 분석가, 소프트웨어 엔지니어 및 제품 소유자를 위한 고유한 페르소나를 디자인하고 종료 전략으로 그룹 채팅 환경을 구성하여 요구 사항을 수집하고, 코드를 개발하고, 코드 검토를 수행할 수 있는 협업 AI 워크플로를 만들었습니다. 작업 구조는 자율적인 대화형 에이전트를 사용하여 복잡한 문제를 확장 가능하고 분산적으로 처리할 수 있도록 합니다.
