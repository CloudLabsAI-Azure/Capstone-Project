param uniqueId string
param prefix string
param containerRegistry string
param location string
param uiAppExists bool
param emptyContainerImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
param userAssignedIdentityName string = '${prefix}-ui-identity-${uniqueId}'
param azureOpenAIEndpoint string = ''
param azureOpenAIKey string = ''
param azureOpenAIChatDeploymentName string = ''
param azureOpenAIApiVersion string = '2024-02-01'
param githubRepoUrl string = ''
param githubPat string = ''
param githubUsername string = ''
param gitUserEmail string = ''

resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-11-02-preview' = {
  name: '${prefix}-containerAppEnv-${uniqueId}'
  location: location
  properties: {}
}

module fetchLatestImageUI './fetch-container-image.bicep' = {
  name: 'ui-app-image'
  params: {
    exists: uiAppExists
    name: '${prefix}-ui-${uniqueId}'
  }
}

resource userAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2018-11-30' = {
  name: userAssignedIdentityName
  location: location
}

resource acr 'Microsoft.ContainerRegistry/registries@2021-06-01-preview' existing = {
  name: containerRegistry
}

resource acrPullRoleAssignment 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = {
  name: guid(userAssignedIdentity.id, containerRegistry, 'acrpull')
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d') // AcrPull
    principalId: userAssignedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource uiContainerApp 'Microsoft.App/containerApps@2023-11-02-preview' = {
  name: '${prefix}-ui-${uniqueId}'
  location: location
  tags: {
    'azd-service-name': 'ui'
  }
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${userAssignedIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 80
        transport: 'auto'
      }
      registries: [
        {
          server: '${containerRegistry}.azurecr.io'
          identity: userAssignedIdentity.id
        }
      ]
    }
    template: {
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
      containers: [
        {
          name: 'ui'
          image: uiAppExists ? fetchLatestImageUI.outputs.containers[0].image : emptyContainerImage
          resources: {
            cpu: 1
            memory: '2Gi'
          }
          env: [
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: azureOpenAIEndpoint
            }
            {
              name: 'AZURE_OPENAI_API_KEY'
              value: azureOpenAIKey
            }
            {
              name: 'AZURE_OPENAI_CHAT_DEPLOYMENT_NAME'
              value: azureOpenAIChatDeploymentName
            }
            {
              name: 'AZURE_OPENAI_API_VERSION'
              value: azureOpenAIApiVersion
            }
            {
              name: 'GITHUB_REPO_URL'
              value: githubRepoUrl
            }
            {
              name: 'GITHUB_PAT'
              value: githubPat
            }
            {
              name: 'GITHUB_USERNAME'
              value: githubUsername
            }
            {
              name: 'GIT_USER_EMAIL'
              value: gitUserEmail
            }
          ]
        }
      ]
    }
  }
}
