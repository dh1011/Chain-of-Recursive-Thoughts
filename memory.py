import os
import json
from datetime import datetime
from typing import Dict, List

class Memory:
    def __init__(self):
        """Initialize the memory module and ensure history directory exists."""
        self.history_dir = "history"
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
    
    def remember(self, conversation_history: List[Dict], thinking_history: List[Dict] = None) -> str:
        """
        Save chat history to a JSON file with timestamp.
        
        Args:
            conversation_history: List of conversation messages
            thinking_history: Optional list of thinking process history
            
        Returns:
            str: Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"
        filepath = os.path.join(self.history_dir, filename)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "conversation": conversation_history,
            "thinking_history": thinking_history or []
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def recall(self, filename: str = None) -> None:
        """
        Display chat history from a JSON file in a nicely formatted way.
        
        Args:
            filename: Optional specific file to recall. If None, shows the most recent file.
        """
        if filename is None:
            # Get the most recent file
            files = [f for f in os.listdir(self.history_dir) if f.endswith('.json')]
            if not files:
                print("No chat history found.")
                return
            filename = max(files, key=lambda x: os.path.getctime(os.path.join(self.history_dir, x)))
        
        filepath = os.path.join(self.history_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print("\n=== CHAT HISTORY ===")
            print(f"Timestamp: {data['timestamp']}")
            print("\n--- Conversation ---")
            
            for msg in data['conversation']:
                role = msg['role'].upper()
                content = msg['content']
                print(f"\n{role}: {content}")
            
            if data['thinking_history']:
                print("\n--- Thinking Process ---")
                for item in data['thinking_history']:
                    round_num = item['round']
                    selected = "[SELECTED]" if item.get('selected', False) else "[ALTERNATIVE]"
                    print(f"\nRound {round_num} {selected}:")
                    print(f"Response: {item['response']}")
                    if 'explanation' in item and item['selected']:
                        print(f"Reason: {item['explanation']}")
            
            print("\n" + "=" * 20)
            
        except FileNotFoundError:
            print(f"File {filename} not found in history directory.")
        except json.JSONDecodeError:
            print(f"Error reading {filename}. File may be corrupted.")
