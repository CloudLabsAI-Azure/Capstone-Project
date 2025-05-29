# 課題 - マルチエージェントシステム

## 紹介

マルチエージェントシステム(MAS)は、それぞれが異なる目標、動作、および責任範囲を持つ複数の自律エージェントで構成されています。これらのエージェントは独立して動作し、地域の知識と環境に基づいて意思決定を行います。しかし、彼らはまた、彼らの目的に応じて協力したり競争したりして、互いにコミュニケーションを取り、情報を共有することもできます。MAS は通常、タスクが複数のエンティティに分散され、システムが分散化の恩恵を受けるシナリオで使用されます。一般的なアプリケーションには、トラフィック管理、ロボットチーム、分散型AI、中央コントローラーに依存しない調整が必要なネットワークシステムなどがあります。

この課題では、ユーザーのリクエストを受け入れ、それぞれが特定のペルソナと専門分野で設計されたエージェントのコレクションを通じて処理するマルチエージェントシステムを作成します。エージェントは個別にリクエストを分析し、定義された責任に基づいて応答を提供します。最終的な出力は、すべてのエージェントからの回答の統合コレクションであり、各ペルソナの独自の視点を反映する方法でユーザーのクエリに協力して対処します。


## タスク 1 - Azure AI Foundry モデルのデプロイと環境の構成

1. **Azure OpenAI Service のデプロイ:**

    - SKU サイズ Standard S0 の Azure OpenAI Service インスタンスを設定します。

        > **手記:** リージョンが [米国東部] に設定されていることを確認します。

    - `openaiagents` というプレフィックスが付いたリソース グループにデプロイします。

    - Azure OpenAI キーとエンドポイントを取得します。

1. **Azure OpenAI モデルをデプロイします。**
   
    - Azure OpenAI には、モデルのデプロイ、管理、探索に使用できる Azure AI Foundry Portal という名前の Web ベースのポータルが用意されています。Azure OpenAI の調査を開始するには、Azure AI Foundry を使用してモデルをデプロイします。
    
    - 概要ウィンドウから Azure AI Foundry ポータルを起動し、Azure OpenAI モデル (gpt-4o) をデプロイします。

        >- **手記:** デプロイの名前が gpt-4o であることを確認します。
        >- **手記:** [デプロイの種類] が [グローバル標準] に設定されていることを確認し、モデル バージョンに 2024-11-20 を使用します。

    - モデルのデプロイ名と API バージョンをフェッチします。

        >- **ヒント:** API バージョンは、ターゲット URI から取得できます。

1. Azure AI Foundry デプロイの詳細で .env ファイルを更新します。

    ```
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=Replace with your deployment name
    AZURE_OPENAI_ENDPOINT=Replace with your endpoint URL
    AZURE_OPENAI_API_KEY=Replace with your API key
    AZURE_OPENAI_API_VERSION=Replace with your API version
    ```


## タスク 2 - エージェントのペルソナの定義とマルチエージェント チャットの構成

1. `multi_agent.py` ファイルを開きます。ここで、このチャレンジに必要なすべてのコードを実装します。

1. 次の手順で、3 人のエージェントのペルソナを作成します。

    - **ビジネスアナリストのペルソナ**

        ```
        You are a Business Analyst which will take the requirements from the user (also known as a 'customer') and create a project plan for creating the requested app. The Business Analyst understands the user requirements and creates detailed documents with requirements and costing. The documents should be usable by the SoftwareEngineer as a reference for implementing the required features, and by the Product Owner for reference to determine if the application delivered by the Software Engineer meets all of the user's requirements.
        ```

    - **ソフトウェアエンジニアのペルソナ**

        ```
        You are a Software Engineer, and your goal is create a web app using HTML and JavaScript by taking into consideration all the requirements given by the Business Analyst. The application should implement all the requested features. Deliver the code to the Product Owner for review when completed. You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear.
        ```

    - **プロダクトオーナーペルソナ**

        ```
        You are the Product Owner which will review the software engineer's code to ensure all user  requirements are completed. You are the guardian of quality, ensuring the final product meets all specifications. IMPORTANT: Verify that the Software Engineer has shared the HTML code using the format ```html [code] ```. This format is required for the code to be saved and pushed to GitHub. Once all client requirements are completed and the code is properly formatted, reply with 'READY FOR USER APPROVAL'. If there are missing features or formatting issues, you will need to send a request back to the SoftwareEngineer or BusinessAnalyst with details of the defect.
        ```

1. 上記の各ペルソナに対して `ChatCompletionAgent` を作成します。各エージェントには、次のものが必要です。
    - 指示 (ペルソナ プロンプト)
    - 意の名前 (文字のみ、スペースや特殊文字は使用不可)
    - `Kernel` オブジェクトへの参照

1. `AgentGroupChat` オブジェクトを作成して、3 人のエージェントを結び付けます。通る：
    - 3 つのエージェントの配列
    - `ExecutionSettings` と `TerminationStrategy` のインスタンスに設定された `TerminationStrategy`
1. `should_agent_terminate` メソッドを `ApprovalTerminationStrategy` クラスに実装します。エージェントは、ユーザーがチャット履歴で `APPROVED` を返したときに終了する必要があります。

## タスク 3 - ユーザー承認時の Git プッシュのトリガー

ユーザーがチャットで「APPROVED」を送信すると、Bashスクリプトがトリガーされ、ソフトウェアエンジニアエージェントが記述したコードをGitリポジトリにプッシュするロジックを追加します。

1. `APPROVED` を検出する `should_agent_terminate` メソッドを実装した後、この条件が満たされたときに実行されるコールバックまたは後処理ステップを追加します。
2. ソフトウェアエンジニアエージェントから提供されたHTMLコードをチャット履歴から抽出します。
3. 抽出したコードをファイル(index.htmlなど)に保存します。
4. Bashスクリプト(push_to_git.shなど)を作成し、ファイルをステージング、コミット、目的のGitリポジトリにプッシュします。
5. Python コードで、subprocess モジュールを使用して、"APPROVED" が検出されたときにこのスクリプトを呼び出します。
6. お使いの環境で、非対話型プッシュに必要な Git 資格情報が構成されていることを確認します。

この自動化により、ユーザーが `APPROVED` を送信すると、最新のコードが自動的にGitリポジトリにプッシュされます。

## タスク 4 - マルチエージェントの会話の実行とワークフローの検証

1. `AgentGroupChat` オブジェクトの `add_chat_message` を使用して、エージェント グループにユーザー メッセージを送信するコードを実装します。メッセージには、次のものを含める必要があります。
    - 著者としての `AuthorRole.User`
    - ユーザーの入力からのチャット メッセージの内容

1. 非同期ループを使用して `AgentGroupChat` からの応答を反復処理し、各メッセージが到着したときに出力します。

    ```python
    async for content in chat.invoke():
        print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
    ```

1. アプリケーションを実行し、電卓アプリの構築要求を提供します。ビジネス アナリスト、ソフトウェア エンジニア、およびプロダクト オーナーが協力してソリューションを計画、構築、承認する方法を観察します。

## タスク 5 - タスク 5 - アプリを Azure にデプロイする
### Container Registry と Azure App Service を使用した Azure へのアプリのデプロイ

Azure を使用してアプリをオンラインでホストするには、次の手順に従ってアプリケーションをコンテナー化し、Azure Container Registry (ACR) にプッシュして、Azure App Service を使用してデプロイします。

1. ターミナルを開き、次のコマンドを使用して Azure Developer CLI にサインインします。

    ```bash
    azd auth login
    ```

1. 次のコマンドを実行して、必要なリソースを Azure にデプロイします。

    ```bash
    azd up
    ```

1. コマンドを azd up 実行すると、構成の詳細を対話形式で提供するように求められます。プロンプトが表示されたら、次の値を指定します。

   - **Unique Environment Name**: Enter **CapstoneEnv-<inject key="Deployment ID" enableCopy="false"/>** **(1)** .
   - **Azure Subscription to use**: Choose the default subscription **(2)** that appears and press **Enter**.
   - **Location Infrastructure Parameter**: Select **East US 2** **(3)** from the options and press **Enter**.
   - **ResourceGroupName Infrastructure Parameter**: Type **CapstoneEnv-<inject key="Deployment ID" enableCopy="false"/>** **(4)** and press **Enter**.
   - **Resource Group to use**: Select **CapstoneEnv-<inject key="Deployment ID" enableCopy="false"/>** **(5)** from the options and press **Enter**.

1. Azure portal を開き、リソース グループ **CapstoneEnv-<inject key="Deployment ID" enableCopy="false"/>** に移動します。.
2. デプロイされたコンテナ アプリ リソースを見つけます。
3. コンテナアプリのエンドポイントURLをコピーします。
4. ブラウザでこのエンドポイントにアクセスして Web アプリにアクセスし、アプリケーションが期待どおりに機能することを確認します。

## 成功基準

- マルチエージェントチャットシステムを実装し、以下を実現しました。
    - 要求されたアプリケーションのHTMLおよび `JavaScript` による完全なソースコードの生成
    - ユーザーによる徹底的なコードレビューと承認プロセス
    - Azure へのアプリケーションの自動デプロイ
    - ユーザーの承認時に Git リポジトリにコードを自動的にプッシュ

---

## ボーナス

- チャット履歴のマークダウンから、ファイル システム上の一致するファイルにコードをコピーします。
- HTMLコンテンツを `index.html` として保存し、Webブラウザで起動します。
- アプリケーションが AI で説明されているとおりに機能するかどうかをテストします。
- AI に応答性を持たせたり、新機能を追加したりするように依頼して、アプリを強化します。
- ペルソナを変更して、結果や機能を改善してみてください。

---

## 学習リソース

- [Agent Group Chat with Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-chat?pivots=programming-language-python)
- [MetaGPT](https://github.com/geekan/MetaGPT)
- [AutoGen Multi-Agent Conversational Framework](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat/)
- [AutoGen with Semantic Kernel](https://devblogs.microsoft.com/semantic-kernel/autogen-agents-meet-semantic-kernel/)
- [Managing your personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

---

## 結論

この課題では、Azure AI Foundry とセマンティック カーネルを使用してマルチエージェント システムを構築し、調整する方法を示しました。ビジネス アナリスト、ソフトウェア エンジニア、プロダクト オーナーの個別のペルソナを設計し、終了戦略を使用してグループ チャット環境を構成することで、要件の収集、コードの開発、コード レビューの実行が可能な協調的な AI ワークフローを作成しました。タスク構造により、自律的でインタラクティブなエージェントを使用して、複雑な問題をスケーラブルかつ分散的に処理できます。
