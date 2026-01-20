targetScope = 'resourceGroup'

@minLength(1)
@maxLength(64)
@description('Name of the environment that can be used as part of naming resource convention')
param environmentName string

@description('Primary location for all resources')
param location string

@description('Name of the resource group to deploy resources into')
param resourceGroupName string

param prefix string = 'dev'
param uiAppExists bool = false
param openAiName string = ''
param openAiResourceGroupName string = ''
param openAIModel string = 'gpt-4o'
param openAIApiVersion string = '2024-02-01'

// Optional GitHub configuration for pushing generated HTML to GitHub
@secure()
param githubPat string = ''
param githubRepoUrl string = ''
param githubUsername string = ''
param gitUserEmail string = ''

var tags = {
  'azd-env-name': environmentName
}

var uniqueId = uniqueString(resourceGroup().id)

// Reference to existing Azure OpenAI resource if provided
resource openAi 'Microsoft.CognitiveServices/accounts@2023-05-01' existing = if (!empty(openAiName)) {
  name: openAiName
  scope: resourceGroup(openAiResourceGroupName)
}

module acrModule './acr.bicep' = {
  name: 'acr'
  scope: resourceGroup()
  params: {
    uniqueId: uniqueId
    prefix: prefix
    location: location
  }
}

module aca './aca.bicep' = {
  name: 'aca'
  scope: resourceGroup()
  params: {
    uniqueId: uniqueId
    prefix: prefix
    containerRegistry: acrModule.outputs.acrName
    location: location
    uiAppExists: uiAppExists
    azureOpenAIEndpoint: !empty(openAiName) ? openAi.properties.endpoint : ''
    azureOpenAIKey: !empty(openAiName) ? openAi.listKeys().key1 : ''
    azureOpenAIChatDeploymentName: openAIModel
    azureOpenAIApiVersion: openAIApiVersion
    githubRepoUrl: githubRepoUrl
    githubPat: githubPat
    githubUsername: githubUsername
    gitUserEmail: gitUserEmail
  }
}

// These outputs are copied by azd to .azure/<env name>/.env file
// post provision script will use these values, too
output AZURE_RESOURCE_GROUP string = resourceGroupName
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = acrModule.outputs.acrEndpoint
