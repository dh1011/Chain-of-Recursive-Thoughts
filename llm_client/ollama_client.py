import os
from typing import List, Dict, Optional, Union
import openai
from .llm_client_base import LLMClientBase

class OllamaClient(LLMClientBase):
    """Ollama client implementation for LLM interactions using OpenAI API format."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "dolphin3:latest"):
        """Initialize the Ollama client.
        
        Args:
            api_key: Not used for Ollama as it runs locally
            model: Model identifier to use
        """
        self.model = model
        self.client = openai.OpenAI(
            api_key="#no-key",
            base_url="http://localhost:11434/v1"
        )
    
    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = True,
        max_tokens: Optional[int] = None
    ) -> Union[str, None]:
        """Generate a completion from Ollama using OpenAI API format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness in the output (0.0 to 1.0)
            stream: Whether to stream the response
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text or None if there was an error
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=stream,
                max_tokens=max_tokens
            )
            
            if stream:
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        print(content, end="", flush=True)
                print()  # New line after streaming
                return full_response
            else:
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Ollama API Error: {e}")
            return None
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama.
        
        Returns:
            List of model identifiers
        """
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            print(f"Error getting available models: {e}")
            return []
    
    def validate_api_key(self) -> bool:
        """Validate the API key.
        
        Since Ollama runs locally, we'll just check if the server is running.
        
        Returns:
            True if the server is accessible, False otherwise
        """
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

def main():
    # Initialize client with dolphin3 model
    client = OllamaClient(model="dolphin3:latest")
    
    # Test if server is running
    if not client.validate_api_key():
        print("Error: Ollama server is not running")
        return
        
    # Example messages
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    # Generate completion
    response = client.generate_completion(
        messages=messages,
        temperature=0.7
    )
    print("Response:", response)

if __name__ == "__main__":
    main()
