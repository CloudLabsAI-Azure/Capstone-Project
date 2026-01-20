import logging
import os
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from typing import List, Dict

# Add Logger
logger = logging.getLogger(__name__)

load_dotenv(override=True)

chat_history: List[Dict[str, str]] = []

def initialize_client():
    """Initialize Azure OpenAI client"""
    client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    return client


async def process_message(user_input):
    """Process user message using Azure OpenAI"""
    global chat_history
    
    client = initialize_client()
    
    # Add user message to history
    chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    #Challenge 03 and 04 - Services Required
    #Challenge 03 - Create Prompt Execution Settings
    
    # Challenge 03 - Add Time Plugin
    # Placeholder for Time plugin

    # Challenge 04 - Import OpenAPI Spec
    # Placeholder for OpenAPI plugin

    # Challenge 05 - Add Search Plugin

    # Challenge 06- Semantic kernel filters

    # Challenge 07 - Text To Image Plugin
    # Placeholder for Text To Image plugin

    # Start Challenge 02 - Sending a message to the chat completion service
    try:
        response = await client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            messages=chat_history,
            temperature=0.7,
            max_tokens=800
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        chat_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return f"Error: {str(e)}"

def reset_chat_history():
    """Reset chat history"""
    global chat_history
    chat_history = []