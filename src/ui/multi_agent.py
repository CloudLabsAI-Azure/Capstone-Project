import os
import asyncio
import subprocess
import semantic_kernel as sk
import logging

# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv() 

# Add Logger
logger = logging.getLogger(__name__)

from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.contents import ChatHistory

chat_history = ChatHistory()


class ApprovalTerminationStrategy(TerminationStrategy):
    
    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        # Check if any message in the chat history contains "APPROVED" from the user
        if history:
            for message in history:
                # Check if the message is from the user and contains "APPROVED"
                if (message.role == AuthorRole.USER and 
                    "APPROVED" in message.content.upper()):
                    # Execute approval callback if provided
                    await on_approval_callback(agent, history, "APPROVED")
                    
                    return True
        
        # Fallback: terminate after a reasonable number of messages to prevent infinite loops
        if len(history) >= 20:
            return True
        
        return False
    
async def on_approval_callback(agent, history, reason):
    """Callback function executed when APPROVED termination condition is met."""
    
    # Extract HTML code from Software Engineer messages
    html_content = extract_html_from_software_engineer(history)
    if html_content:
        save_html_to_file(html_content, "index.html")
        print(f"   ✅ HTML code saved to index.html")
        
    else:
        print(f"   ⚠️ No HTML code found in Software Engineer messages")
    
     # Execute push_to_github.sh script
    await execute_push_to_github()
    
async def execute_push_to_github():
    """Execute the push_to_github.sh script using subprocess."""
    try:
        # Get location of current file
        # use __file___
        # Determine the script path and command based on OS
        script_name = "push_to_github.sh"
        script_path = os.path.join(os.path.dirname(__file__), script_name)

        print(f"Calling Script 📁 File located at: {script_path}")
               
        # Execute the script
        result = subprocess.run(
                ["bash", script_path], 
                capture_output=True, 
                text=True, 
                timeout=300
                )        
        print(f"Script run: {result.stdout}")
        print(f"script errors: {result.stderr}")
    except Exception as e:
        print(f"Error executing script: {e}")


def extract_html_from_software_engineer(history):
    """Extract HTML code from Software Engineer agent messages."""
    html_content = ""
    
    for message in history:
        # Check if the message is from the Software Engineer
        if hasattr(message, 'name') and message.name == "SoftwareEngineer":
            content = message.content
            
            # Look for HTML code blocks (```html or ```HTML)
            import re
            html_blocks = re.findall(r'```(?:html|HTML)\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
            
            for block in html_blocks:
                html_content += block.strip() + "\n"
            
            # Also look for standalone HTML tags (less strict pattern)
            if not html_blocks:
                # Look for content that contains HTML-like structures
                html_patterns = [
                    r'<!DOCTYPE html.*?>.*?</html>',
                    r'<html.*?>.*?</html>',
                    r'<div.*?>.*?</div>',
                    r'<body.*?>.*?</body>'
                ]
                
                for pattern in html_patterns:
                    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                    for match in matches:
                        html_content += match.strip() + "\n"
    
    return html_content.strip()


def save_html_to_file(html_content, filename):
    """Save HTML content to a file."""
    try:
        # Get the current working directory (workspace root)
        file_path = os.path.join(os.getcwd(), filename)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(html_content)
        
        print(f"   📁 File saved at: {file_path}")
        
    except Exception as e:
        print(f"   ❌ Error saving HTML file: {e}")


async def run_multi_agent(user_input: str):
    """implement the multi-agent system."""
    
    # Initialize kernel
    kernel = sk.Kernel()
    
    # Add Azure OpenAI chat completion service
    azure_openai_service = AzureChatCompletion(
        deployment_name=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )
    kernel.add_service(azure_openai_service)

    # Create Business Analyst Agent
    business_analyst = ChatCompletionAgent(
        kernel=kernel,
        name="BusinessAnalyst",
        instructions="You are a Business Analyst which will take the requirements from the user (also known as a 'customer') and create a project plan for creating the requested app. The Business Analyst understands the user requirements and creates detailed documents with requirements and costing. The documents should be usable by the SoftwareEngineer as a reference for implementing the required features, and by the Product Owner for reference to determine if the application delivered by the Software Engineer meets all of the user's requirements.",
        description="Analyzes business requirements and translates them into specifications",
    )

    # Create Software Engineer Agent  
    software_engineer = ChatCompletionAgent(
        kernel=kernel,
        name="SoftwareEngineer",
        instructions="You are a Software Engineer, and your goal is create a web app using HTML and JavaScript by taking into consideration all the requirements given by the Business Analyst. The application should implement all the requested features. Deliver the code to the Product Owner for review when completed. You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear.",
        description="Designs and implements technical solutions",
    )

    # Create Product Owner Agent
    product_owner = ChatCompletionAgent(
        kernel=kernel,
        name="ProductOwner", 
        instructions="You are the Product Owner which will review the software engineer's code to ensure all user  requirements are completed. You are the guardian of quality, ensuring the final product meets all specifications. IMPORTANT: Verify that the Software Engineer has shared the HTML code using the format ```html [code] ```. This format is required for the code to be saved and pushed to GitHub. Once all client requirements are completed and the code is properly formatted, reply with 'READY FOR USER APPROVAL'. If there are missing features or formatting issues, you will need to send a request back to the SoftwareEngineer or BusinessAnalyst with details of the defect.",
        description="Defines product vision and prioritizes features",
    )

    # Create agent group chat with termination strategy and callbacks
    termination_strategy = ApprovalTerminationStrategy()    
    
    # Create agent group chat
    agent_group_chat = AgentGroupChat(
        agents=[business_analyst, software_engineer, product_owner],
        termination_strategy=termination_strategy,
        chat_history=chat_history
    )

    message = ChatMessageContent(role=AuthorRole.USER, content=user_input)

    await agent_group_chat.add_chat_message(message)

    try:              
        result = []
        # Get responses from the multi-agent conversation
        async for content in agent_group_chat.invoke():
            print(f"# {content.role} - {content.name or '*'}: '{content.content}'") 
            result.append(ChatMessageContent(role=content.role, name=content.name, content=content.content))

        return result 

    except Exception as e:
        print(f"Error during chat invocation: {e}")

    return

def reset_multi_agent_chat_history():
    global chat_history
    chat_history = ChatHistory()