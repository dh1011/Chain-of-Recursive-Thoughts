import os
from typing import List, Dict
from llm_client.openai_client import OpenAIClient
from llm_client.ollama_client import OllamaClient
from memory import Memory

class EnhancedRecursiveThinkingChat:
    def __init__(self, client_type: str = "openai", model: str = "gpt-4o"):
        """Initialize with either OpenAI or Ollama client."""
        if client_type.lower() == "openai":
            self.client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model=model)
        else:
            self.client = OllamaClient(model='dolphin3:latest')
        self.conversation_history = []
        self.full_thinking_log = []
        self.memory = Memory()
    
    def _determine_thinking_rounds(self, prompt: str) -> int:
        """Let the model decide how many rounds of thinking are needed."""
        meta_prompt = f"""Given this message: "{prompt}"
        
How many rounds of iterative thinking (1-5) would be optimal to generate the best response?
Consider the complexity and nuance required.
Respond with just a number between 1 and 5."""
        
        messages = [{"role": "user", "content": meta_prompt}]
        
        print("\n=== DETERMINING THINKING ROUNDS ===")
        response = self.client.generate_completion(messages, temperature=0.3, stream=True)
        print("=" * 50 + "\n")
        
        # Extract digits from response
        digits = ''.join(filter(str.isdigit, response))
        
        # If no digits found, default to 3 rounds
        if not digits:
            print("Warning: No valid number found in response. Defaulting to 3 thinking rounds.")
            return 3
            
        rounds = int(digits)
        # Ensure rounds is between 1 and 5
        return min(max(rounds, 1), 5)
    
    def _generate_alternatives(self, base_response: str, prompt: str, num_alternatives: int = 3) -> List[str]:
        """Generate alternative responses."""
        alternatives = []
        
        for i in range(num_alternatives):
            print(f"\n=== GENERATING ALTERNATIVE {i+1} ===")
            alt_prompt = f"""Original message: {prompt}
            
Current response: {base_response}

Generate an alternative response that might be better. Be creative and consider different approaches.
Alternative response:"""
            
            messages = self.conversation_history + [{"role": "user", "content": alt_prompt}]
            alternative = self.client.generate_completion(messages, temperature=0.7 + i * 0.1, stream=True)
            alternatives.append(alternative)
            print("=" * 50)
        
        return alternatives
    
    def _evaluate_responses(self, prompt: str, current_best: str, alternatives: List[str]) -> tuple[str, str]:
        """Evaluate responses and select the best one."""
        print("\n=== EVALUATING RESPONSES ===")
        eval_prompt = f"""Original message: {prompt}

Evaluate these responses and choose the best one:

Current best: {current_best}

Alternatives:
{chr(10).join([f"{i+1}. {alt}" for i, alt in enumerate(alternatives)])}

Which response best addresses the original message? Consider accuracy, clarity, and completeness.
First, respond with ONLY 'current' or a number (1-{len(alternatives)}).
Then on a new line, explain your choice in one sentence."""
        
        messages = [{"role": "user", "content": eval_prompt}]
        evaluation = self.client.generate_completion(messages, temperature=0.2, stream=True)
        print("=" * 50)
        
        # Better parsing
        lines = [line.strip() for line in evaluation.split('\n') if line.strip()]
        
        choice = 'current'
        explanation = "No explanation provided"
        
        if lines:
            first_line = lines[0].lower()
            if 'current' in first_line:
                choice = 'current'
            else:
                for char in first_line:
                    if char.isdigit():
                        choice = char
                        break
            
            if len(lines) > 1:
                explanation = ' '.join(lines[1:])
        
        if choice == 'current':
            return current_best, explanation
        
        index = int(choice) - 1
        if 0 <= index < len(alternatives):
            return alternatives[index], explanation
        
        return current_best, explanation
    
    def think_and_respond(self, user_input: str, verbose: bool = True) -> Dict:
        """Process user input with recursive thinking."""
        print("\n" + "=" * 50)
        print("🤔 RECURSIVE THINKING PROCESS STARTING")
        print("=" * 50)
        
        thinking_rounds = self._determine_thinking_rounds(user_input)
        
        if verbose:
            print(f"\n🤔 Thinking... ({thinking_rounds} rounds needed)")
        
        # Initial response
        print("\n=== GENERATING INITIAL RESPONSE ===")
        messages = self.conversation_history + [{"role": "user", "content": user_input}]
        current_best = self.client.generate_completion(messages, stream=True)
        print("=" * 50)
        
        thinking_history = [{"round": 0, "response": current_best, "selected": True}]
        
        # Iterative improvement
        for round_num in range(1, thinking_rounds + 1):
            if verbose:
                print(f"\n=== ROUND {round_num}/{thinking_rounds} ===")
            
            # Generate alternatives
            alternatives = self._generate_alternatives(current_best, user_input)
            
            # Store alternatives in history
            for i, alt in enumerate(alternatives):
                thinking_history.append({
                    "round": round_num,
                    "response": alt,
                    "selected": False,
                    "alternative_number": i + 1
                })
            
            # Evaluate and select best
            new_best, explanation = self._evaluate_responses(user_input, current_best, alternatives)
            
            # Update selection in history
            if new_best != current_best:
                for item in thinking_history:
                    if item["round"] == round_num and item["response"] == new_best:
                        item["selected"] = True
                        item["explanation"] = explanation
                current_best = new_best
                
                if verbose:
                    print(f"\n    ✓ Selected alternative: {explanation}")
            else:
                for item in thinking_history:
                    if item["selected"] and item["response"] == current_best:
                        item["explanation"] = explanation
                        break
                
                if verbose:
                    print(f"\n    ✓ Kept current response: {explanation}")
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Add the final selected response to conversation history
        self.conversation_history.append({"role": "assistant", "content": current_best})
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Save the full thinking process including all rounds and alternatives
        self.full_thinking_log.append({
            "user_input": user_input,
            "thinking_rounds": thinking_rounds,
            "thinking_history": thinking_history,
            "final_response": current_best
        })
        
        # Automatically save both conversation and full thinking log
        self.save_conversation()
        self.save_full_log()
        
        print("\n" + "=" * 50)
        print("🎯 FINAL RESPONSE SELECTED")
        print("=" * 50)
        
        return {
            "response": current_best,
            "thinking_rounds": thinking_rounds,
            "thinking_history": thinking_history
        }
    
    def save_full_log(self, filename: str = None):
        """Save the full thinking process log using the memory module."""
        self.memory.remember(self.conversation_history, self.full_thinking_log)
        print("Full thinking log saved to history directory")
    
    def save_conversation(self, filename: str = None):
        """Save the conversation using the memory module."""
        self.memory.remember(self.conversation_history)
        print("Conversation saved to history directory")

def main():
    print("🤖 Enhanced Recursive Thinking Chat")
    print("=" * 50)
    
    # Choose client type
    while True:
        client_type = input("Choose LLM client (openai/ollama): ").strip().lower()
        if client_type in ["openai", "ollama"]:
            break
        print("Please enter either 'openai' or 'ollama'")
    
    # Initialize chat
    chat = EnhancedRecursiveThinkingChat(client_type=client_type)
    
    print("\nChat initialized! Type 'exit' to quit.")
    print("The AI will think recursively before each response.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            break
        elif not user_input:
            continue
        
        # Get response with thinking process
        result = chat.think_and_respond(user_input)
        
        print(f"\n🤖 AI FINAL RESPONSE: \n{result['response']}\n")

    print("Goodbye! 👋")

if __name__ == "__main__":
    main()
