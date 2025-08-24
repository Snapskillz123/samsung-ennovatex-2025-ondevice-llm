"""
WhatsApp Chat Parser for Samsung EnnovateX 2025 AI Challenge
Converts exported WhatsApp chats to supervised fine-tuning format
"""

import re
import json
import sys
import argparse
from typing import List, Dict, Any
from pathlib import Path

# WhatsApp message regex pattern
# Matches both formats:
# - "[12/08/2025, 10:34 PM] Alice: Sure, I'll send it tomorrow."
# - "12/08/2025, 10:34 pm - Alice: Sure, I'll send it tomorrow."
MSG_PATTERN = re.compile(
    r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2})[\u202f ]?(AM|PM)?\]?\s*[-:]?\s*([^:]+?): (.+)$', 
    re.IGNORECASE
)

def parse_whatsapp_line(line: str) -> Dict[str, str] | None:
    """Parse a single WhatsApp line into structured data."""
    match = MSG_PATTERN.match(line.strip())
    if not match:
        return None
    
    date, time, period, sender, text = match.groups()
    
    # Skip media messages
    if ("<Media omitted" in text or "image omitted" in text.lower() or 
        "<attached:" in text or "‎<attached:" in text):
        return None
    
    # Skip system messages
    if (any(phrase in text.lower() for phrase in [
        "joined using this group's invite link", "left", "created group",
        "added you", "messages and calls are end-to-end encrypted",
        "you're now an admin", "was added", "pinned a message",
        "this message was deleted", "this message was edited"
    ])):
        return None
        
    return {
        "date": date,
        "time": time + (" " + period if period else ""),
        "sender": sender.strip(),
        "text": text.strip()
    }

def create_conversation_pair(conversation_buffer: List[Dict], max_history: int = 6) -> Dict[str, str] | None:
    """Create a training pair from conversation history."""
    if len(conversation_buffer) < 2:
        return None
    
    # Last message is the target response
    target = conversation_buffer[-1]
    
    # Previous messages form the context (limit to max_history)
    history = conversation_buffer[:-1][-max_history:]
    
    # Build conversation history
    history_text = "\n".join([f"{msg['sender']}: {msg['text']}" for msg in history])
    
    return {
        "instruction": "Given the chat history, reply naturally in the user's personal communication style.",
        "input": history_text,
        "output": f"{target['sender']}: {target['text']}"
    }

def process_whatsapp_file(input_path: Path, output_path: Path, max_pairs: int = None) -> int:
    """Process WhatsApp export file and create training pairs."""
    conversations = []
    current_conversation = []
    pairs_created = 0
    
    print(f"Processing WhatsApp file: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000 == 0:
                print(f"Processed {line_num} lines, created {pairs_created} pairs")
            
            parsed_msg = parse_whatsapp_line(line)
            if not parsed_msg:
                continue
            
            current_conversation.append(parsed_msg)
            
            # Create training pairs when we have enough context
            if len(current_conversation) >= 8:
                pair = create_conversation_pair(current_conversation)
                if pair:
                    conversations.append(pair)
                    pairs_created += 1
                    
                    if max_pairs and pairs_created >= max_pairs:
                        break
                
                # Keep sliding window of conversation
                current_conversation = current_conversation[-7:]
    
    # Process remaining conversation
    if len(current_conversation) >= 2:
        pair = create_conversation_pair(current_conversation)
        if pair:
            conversations.append(pair)
            pairs_created += 1
    
    # Save to JSONL format
    print(f"Saving {len(conversations)} training pairs to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for conversation in conversations:
            f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
    
    return len(conversations)

def main():
    parser = argparse.ArgumentParser(
        description="Convert WhatsApp chat export to training format"
    )
    parser.add_argument(
        "--input", 
        required=True, 
        type=Path,
        help="Path to WhatsApp export .txt file"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        type=Path,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--max-pairs", 
        type=int,
        help="Maximum number of training pairs to create"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        num_pairs = process_whatsapp_file(args.input, args.output, args.max_pairs)
        print(f"Successfully created {num_pairs} training pairs")
        print(f"Output saved to: {args.output}")
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
