import os
import json
import re
import subprocess
import asyncio
from typing import List, Dict, Any

from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MultiAgentOrchestrator:
    def __init__(self):
        self.business_analyst = None
        self.software_engineer = None
        self.product_owner = None
        self.credential = None
        
    async def initialize(self):
        """Initialize the Microsoft Agent Framework agents."""
        # Get Azure OpenAI configuration from environment
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        
        if not all([api_key, endpoint, deployment_name]):
            raise ValueError("Missing required Azure OpenAI environment variables")
        
        # Remove trailing slash from endpoint if present
        endpoint = endpoint.rstrip('/')
        
        # Create Azure OpenAI client with API key authentication
        from openai import AsyncAzureOpenAI
        
        print(f"Initializing Azure OpenAI with endpoint: {endpoint}")
        print(f"Deployment: {deployment_name}, API Version: {api_version}")
        
        # Use OpenAI client directly since AzureOpenAIChatClient needs proper config
        openai_client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        
        # Create a simple wrapper for the chat client
        class SimpleAzureChatClient:
            def __init__(self, client, deployment):
                self.client = client
                self.deployment = deployment
            
            async def complete(self, messages):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.deployment,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"Error in Azure OpenAI call: {e}")
                    raise
        
        chat_client = SimpleAzureChatClient(openai_client, deployment_name)
        
        # Create Business Analyst Agent
        self.business_analyst = {
            "name": "BusinessAnalyst",
            "instructions": """
            You are a Business Analyst which will take the requirements from the user (also known as a 'customer')
            and create a project plan for creating the requested app. The Business Analyst understands the user
            requirements and creates detailed documents with requirements and costing. The documents should be 
            usable by the SoftwareEngineer as a reference for implementing the required features, and by the 
            Product Owner for reference to determine if the application delivered by the Software Engineer meets
            all of the user's requirements.
            """,
            "client": chat_client
        }
        
        # Create Software Engineer Agent
        self.software_engineer = {
            "name": "SoftwareEngineer",
            "instructions": """
            You are a Software Engineer, and your goal is create a web app using HTML and JavaScript
            by taking into consideration all the requirements given by the Business Analyst. The application should
            implement all the requested features. Deliver the code to the Product Owner for review when completed.
            
            IMPORTANT: Share ONLY the raw HTML code without any markdown formatting. 
            Start directly with <!DOCTYPE html> or <html> tag.
            Do NOT wrap it in ```html code blocks.
            
            You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear.
            """,
            "client": chat_client
        }
        
        # Create Product Owner Agent
        self.product_owner = {
            "name": "ProductOwner",
            "instructions": """
            You are the Product Owner which will review the software engineer's code to ensure all user 
            requirements are completed. You are the guardian of quality, ensuring the final product meets
            all specifications. 
            
            IMPORTANT: When the Software Engineer shares HTML code, verify it is complete and functional.
            The HTML should start with <!DOCTYPE html> or <html> tags.
            
            Once all client requirements are completed and the code is properly formatted, reply with 'READY FOR USER APPROVAL'.
            If there are missing features or issues, you will need to send a request back to the SoftwareEngineer 
            or BusinessAnalyst with details of the defect.
            """,
            "client": chat_client
        }
    
    async def orchestrate(self, user_input: str, max_iterations: int = 5) -> List[Dict[str, Any]]:
        """Orchestrate the multi-agent conversation using Microsoft Agent Framework."""
        results = []
        
        # Start with user input
        results.append({
            "role": "user",
            "agent": "User",
            "content": user_input
        })
        
        for iteration in range(max_iterations):
            # Business Analyst processes requirements
            ba_message = user_input if iteration == 0 else f"Review the current progress and continue: {results[-1]['content']}"
            ba_messages = [
                {"role": "system", "content": self.business_analyst["instructions"]},
                {"role": "user", "content": ba_message}
            ]
            ba_response = await self.business_analyst["client"].complete(ba_messages)
            
            results.append({
                "role": "assistant",
                "agent": "BusinessAnalyst",
                "content": ba_response
            })
            
            # Software Engineer implements
            se_message = f"Based on the Business Analyst's requirements, implement the solution: {ba_response}"
            se_messages = [
                {"role": "system", "content": self.software_engineer["instructions"]},
                {"role": "user", "content": se_message}
            ]
            se_response = await self.software_engineer["client"].complete(se_messages)
            
            results.append({
                "role": "assistant",
                "agent": "SoftwareEngineer",
                "content": se_response
            })
            
            # Product Owner reviews
            po_message = f"Review the Software Engineer's implementation: {se_response}"
            po_messages = [
                {"role": "system", "content": self.product_owner["instructions"]},
                {"role": "user", "content": po_message}
            ]
            po_response = await self.product_owner["client"].complete(po_messages)
            
            results.append({
                "role": "assistant",
                "agent": "ProductOwner",
                "content": po_response
            })
            
            # Check for approval readiness
            if "READY FOR USER APPROVAL" in po_response.upper():
                results.append({
                    "role": "system",
                    "agent": "system",
                    "content": "The Product Owner has indicated that the work is ready for your approval. Please respond with 'APPROVED' to push the code to GitHub or 'NOT APPROVED' to have the agents continue working on it."
                })
                break
            
            # Add small delay between iterations
            await asyncio.sleep(1)
        
        return results
    
    async def cleanup(self):
        """Clean up resources."""
        pass


async def run_multi_agent(input: str):
    """Main entry point for multi-agent orchestration."""
    
    # Define the output HTML filename
    html_filename = "index.html"
    html_path = os.path.join(os.path.dirname(__file__), html_filename)
    
    # Handle approval/rejection
    if input.strip().upper() == "APPROVED":
        if os.path.exists(html_path):
            try:
                # Get GitHub configuration from environment
                github_repo_url = os.getenv("GITHUB_REPO_URL")
                github_pat = os.getenv("GITHUB_PAT")
                git_user_email = os.getenv("GIT_USER_EMAIL")
                github_username = os.getenv("GITHUB_USERNAME")
                
                if not github_repo_url:
                    return {
                        "messages": [{
                            "role": "system",
                            "agent": "system",
                            "content": f"✅ HTML file saved locally at {html_path}\n\nℹ️ GitHub integration is not configured. To enable automatic pushing to GitHub, please set the following environment variables:\n- GITHUB_REPO_URL\n- GITHUB_PAT\n- GITHUB_USERNAME\n- GIT_USER_EMAIL\n\nSee GITHUB_SETUP.md for detailed instructions."
                        }]
                    }
                
                # Create a temporary directory for the repo
                import tempfile
                import shutil
                temp_dir = tempfile.mkdtemp()
                
                try:
                    # Configure git credentials using PAT if provided
                    if github_pat and github_username:
                        # Replace https:// with https://username:token@
                        auth_url = github_repo_url.replace("https://", f"https://{github_username}:{github_pat}@")
                    else:
                        auth_url = github_repo_url
                    
                    # Clone the repository
                    clone_result = subprocess.run(
                        ["git", "clone", auth_url, temp_dir],
                        capture_output=True,
                        text=True
                    )
                    
                    if clone_result.returncode != 0:
                        error_msg = clone_result.stderr.strip() if clone_result.stderr else "Unknown error"
                        return {
                            "messages": [{
                                "role": "system",
                                "agent": "system",
                                "content": f"❌ Failed to clone repository.\n\n📋 Error details:\n{error_msg}\n\n✅ HTML file saved locally at {html_path}\n\nℹ️ Please check:\n1. GITHUB_REPO_URL is correct\n2. GITHUB_PAT has 'repo' scope\n3. GITHUB_USERNAME matches the PAT owner\n4. You have write access to the repository\n\nSee GITHUB_SETUP.md for troubleshooting."
                            }]
                        }
                    
                    # Copy the generated HTML file to the repo
                    dest_path = os.path.join(temp_dir, html_filename)
                    shutil.copy2(html_path, dest_path)
                    
                    # Configure git user if provided
                    if git_user_email:
                        subprocess.run(
                            ["git", "config", "user.email", git_user_email],
                            cwd=temp_dir
                        )
                    if github_username:
                        subprocess.run(
                            ["git", "config", "user.name", github_username],
                            cwd=temp_dir
                        )
                    
                    # Git add
                    add_result = subprocess.run(
                        ["git", "add", html_filename],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True
                    )
                    
                    # Git commit
                    commit_result = subprocess.run(
                        ["git", "commit", "-m", "Update generated app from multi-agent system"],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True
                    )
                    
                    # Check if there's anything to commit
                    if commit_result.returncode != 0:
                        if "nothing to commit" in commit_result.stdout or "nothing to commit" in commit_result.stderr:
                            return {
                                "messages": [{
                                    "role": "system",
                                    "agent": "system",
                                    "content": "No changes to commit. The file may already exist in the repository with the same content."
                                }]
                            }
                        else:
                            return {
                                "messages": [{
                                    "role": "system",
                                    "agent": "system",
                                    "content": f"Commit failed.\nError: {commit_result.stderr}\n\nHTML file saved locally at {html_path}"
                                }]
                            }
                    
                    # Git push
                    push_result = subprocess.run(
                        ["git", "push", "origin", "main"],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True
                    )
                    
                    # Try master branch if main fails
                    if push_result.returncode != 0 and "main" in push_result.stderr:
                        push_result = subprocess.run(
                            ["git", "push", "origin", "master"],
                            cwd=temp_dir,
                            capture_output=True,
                            text=True
                        )
                    
                    if push_result.returncode == 0:
                        return {
                            "messages": [{
                                "role": "system",
                                "agent": "system",
                                "content": f"✅ Success! {html_filename} has been pushed to {github_repo_url}\n\nYou can view it at your repository."
                            }]
                        }
                    else:
                        return {
                            "messages": [{
                                "role": "system",
                                "agent": "system",
                                "content": f"Push failed.\nError: {push_result.stderr}\n\nThe file was committed locally. Please check your GitHub credentials and permissions.\n\nHTML file saved at {html_path}"
                            }]
                        }
                        
                finally:
                    # Clean up temp directory
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass
                    
            except Exception as e:
                return {
                    "messages": [{
                        "role": "system",
                        "agent": "system",
                        "content": f"HTML file saved at {html_path}. Error during GitHub push: {str(e)}"
                    }]
                }
        else:
            return {
                "messages": [{
                    "role": "system",
                    "agent": "system",
                    "content": "No HTML content was found to push to GitHub. Please make sure the agents have generated HTML content between ```html and ``` markers."
                }]
            }
    
    if input.strip().upper() == "NOT APPROVED":
        return {
            "messages": [{
                "role": "system",
                "agent": "system",
                "content": "Your feedback has been received. The agents will continue working on improving the code."
            }]
        }
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    try:
        # Initialize agents
        await orchestrator.initialize()
        
        # Run orchestration
        results = await orchestrator.orchestrate(input)
        
        # Save chat history
        chat_history_path = os.path.join(os.path.dirname(__file__), "chat_history.json")
        with open(chat_history_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Extract HTML content - try multiple patterns
        html_content = None
        
        for message in results:
            content = message.get("content", "")
            if isinstance(content, str):
                # Pattern 1: Code block with html marker
                html_matches = re.findall(r"```html\s*\n([\s\S]*?)\n\s*```", content, re.MULTILINE)
                if html_matches:
                    html_content = html_matches[-1].strip()
                    continue
                
                # Pattern 2: Code block without newlines
                html_matches = re.findall(r"```html\s*([\s\S]*?)\s*```", content)
                if html_matches:
                    html_content = html_matches[-1].strip()
                    continue
                
                # Pattern 3: Direct HTML (starts with <!DOCTYPE or <html>)
                if "<!DOCTYPE html>" in content or content.strip().startswith("<html"):
                    # Extract from DOCTYPE or <html> to closing </html>
                    if "<!DOCTYPE html>" in content:
                        start = content.find("<!DOCTYPE html>")
                        end = content.rfind("</html>")
                        if end != -1:
                            html_content = content[start:end+7].strip()
                    elif content.strip().startswith("<html"):
                        start = content.find("<html")
                        end = content.rfind("</html>")
                        if end != -1:
                            html_content = content[start:end+7].strip()
        
        if html_content:
            # Remove any remaining markdown artifacts
            html_content = html_content.replace("```html", "").replace("```", "").strip()
            
            html_path = os.path.join(os.path.dirname(__file__), "index.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            # Log success
            print(f"✅ HTML content saved to {html_path}")
            print(f"Content preview: {html_content[:100]}...")
        else:
            print("⚠️ Warning: No HTML content found in agent responses")
        
        return {
            "messages": results
        }
        
    finally:
        # Clean up resources
        await orchestrator.cleanup()


# For compatibility with asyncio
import asyncio