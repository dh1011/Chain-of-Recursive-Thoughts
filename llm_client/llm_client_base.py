from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union

class LLMClientBase(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, model: str = "default-model"):
        """Initialize the LLM client.
        
        Args:
            api_key: API key for the LLM service
            model: Model identifier to use
        """
        pass
    
    @abstractmethod
    def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = True,
        max_tokens: Optional[int] = None
    ) -> Union[str, None]:
        """Generate a completion from the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness in the output (0.0 to 1.0)
            stream: Whether to stream the response
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text or None if there was an error
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this client.
        
        Returns:
            List of model identifiers
        """
        pass
    
    @abstractmethod
    def validate_api_key(self) -> bool:
        """Validate the API key.
        
        Returns:
            True if the API key is valid, False otherwise
        """
        pass 