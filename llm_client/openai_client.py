import os
from typing import List, Dict, Optional, Union
import openai
from .llm_client_base import LLMClientBase

class OpenAIClient(LLMClientBase):
    """OpenAI implementation of the LLM client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model identifier to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = True,
        max_tokens: Optional[int] = None
    ) -> Union[str, None]:
        """Generate a completion using OpenAI's API.
        
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
            print(f"OpenAI API Error: {e}")
            return None
    
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models.
        
        Returns:
            List of model identifiers
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error getting available models: {e}")
            return []
    
    def validate_api_key(self) -> bool:
        """Validate the OpenAI API key.
        
        Returns:
            True if the API key is valid, False otherwise
        """
        try:
            # Try to list models as a simple validation
            self.client.models.list()
            return True
        except Exception:
            return False 