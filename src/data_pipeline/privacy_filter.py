"""
Privacy Filter for Samsung EnnovateX 2025 AI Challenge
Removes PII and sensitive information from training data
"""

import re
import json
import argparse
from typing import List, Dict, Any, Set
from pathlib import Path

# PII patterns
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
CREDIT_CARD_PATTERN = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
SSN_PATTERN = re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b')
URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# Common sensitive keywords
SENSITIVE_KEYWORDS = {
    'password', 'pwd', 'pin', 'ssn', 'social security', 'credit card', 
    'bank account', 'routing number', 'passport', 'license plate',
    'home address', 'work address', 'salary', 'income'
}

class PrivacyFilter:
    """Filter PII and sensitive information from text data."""
    
    def __init__(self, 
                 mask_emails: bool = True,
                 mask_phones: bool = True, 
                 mask_urls: bool = True,
                 mask_numbers: bool = True,
                 preserve_style: bool = True):
        self.mask_emails = mask_emails
        self.mask_phones = mask_phones
        self.mask_urls = mask_urls
        self.mask_numbers = mask_numbers
        self.preserve_style = preserve_style
        
        # Keep counters for consistent replacement
        self.email_counter = 0
        self.phone_counter = 0
        self.url_counter = 0
        self.number_counter = 0
        
    def _mask_emails(self, text: str) -> str:
        """Replace emails with placeholder tokens."""
        def replace_email(match):
            self.email_counter += 1
            return f"<EMAIL_{self.email_counter}>"
        
        return EMAIL_PATTERN.sub(replace_email, text)
    
    def _mask_phones(self, text: str) -> str:
        """Replace phone numbers with placeholder tokens."""
        def replace_phone(match):
            self.phone_counter += 1
            return f"<PHONE_{self.phone_counter}>"
        
        return PHONE_PATTERN.sub(replace_phone, text)
    
    def _mask_urls(self, text: str) -> str:
        """Replace URLs with placeholder tokens."""
        def replace_url(match):
            self.url_counter += 1
            return f"<URL_{self.url_counter}>"
        
        return URL_PATTERN.sub(replace_url, text)
    
    def _mask_sensitive_numbers(self, text: str) -> str:
        """Replace credit cards and SSNs with placeholders."""
        # Credit cards
        text = CREDIT_CARD_PATTERN.sub('<CREDIT_CARD>', text)
        # SSNs  
        text = SSN_PATTERN.sub('<SSN>', text)
        return text
    
    def _contains_sensitive_keywords(self, text: str) -> bool:
        """Check if text contains sensitive keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in SENSITIVE_KEYWORDS)
    
    def filter_text(self, text: str) -> tuple[str, bool]:
        """
        Filter text and return (filtered_text, should_keep).
        
        Returns:
            tuple: (filtered_text, should_keep_example)
        """
        original_text = text
        
        # Check for sensitive keywords - might reject entire example
        if self._contains_sensitive_keywords(text):
            return text, False
        
        # Apply masking
        if self.mask_emails:
            text = self._mask_emails(text)
        
        if self.mask_phones:
            text = self._mask_phones(text)
            
        if self.mask_urls:
            text = self._mask_urls(text)
            
        if self.mask_numbers:
            text = self._mask_sensitive_numbers(text)
        
        # If too much was masked, might want to reject
        if self.preserve_style:
            # Don't keep examples where >30% was masked
            masked_tokens = len(re.findall(r'<[A-Z_0-9]+>', text))
            total_tokens = len(text.split())
            if total_tokens > 0 and (masked_tokens / total_tokens) > 0.3:
                return original_text, False
        
        return text, True
    
    def filter_conversation_pair(self, pair: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        """Filter a conversation pair and return (filtered_pair, should_keep)."""
        # Filter each text field
        filtered_pair = pair.copy()
        should_keep = True
        
        # Filter instruction
        if 'instruction' in pair:
            filtered_instruction, keep_instruction = self.filter_text(pair['instruction'])
            filtered_pair['instruction'] = filtered_instruction
            should_keep = should_keep and keep_instruction
        
        # Filter input
        if 'input' in pair:
            filtered_input, keep_input = self.filter_text(pair['input'])
            filtered_pair['input'] = filtered_input
            should_keep = should_keep and keep_input
        
        # Filter output
        if 'output' in pair:
            filtered_output, keep_output = self.filter_text(pair['output'])
            filtered_pair['output'] = filtered_output
            should_keep = should_keep and keep_output
        
        return filtered_pair, should_keep

def process_jsonl_file(input_path: Path, 
                      output_path: Path, 
                      privacy_filter: PrivacyFilter) -> Dict[str, int]:
    """Process JSONL file through privacy filter."""
    
    stats = {
        'total_examples': 0,
        'filtered_examples': 0,
        'rejected_examples': 0
    }
    
    print(f"Processing: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if line_num % 100 == 0:
                print(f"Processed {line_num} examples, "
                     f"kept {stats['filtered_examples']}, "
                     f"rejected {stats['rejected_examples']}")
            
            try:
                example = json.loads(line.strip())
                stats['total_examples'] += 1
                
                filtered_example, should_keep = privacy_filter.filter_conversation_pair(example)
                
                if should_keep:
                    outfile.write(json.dumps(filtered_example, ensure_ascii=False) + '\n')
                    stats['filtered_examples'] += 1
                else:
                    stats['rejected_examples'] += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Apply privacy filtering to training data"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        required=True, 
        type=Path,
        help="Output filtered JSONL file"
    )
    parser.add_argument(
        "--no-email-masking",
        action="store_true",
        help="Disable email masking"
    )
    parser.add_argument(
        "--no-phone-masking", 
        action="store_true",
        help="Disable phone number masking"
    )
    parser.add_argument(
        "--no-url-masking",
        action="store_true", 
        help="Disable URL masking"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict filtering (reject more examples)"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist")
        return 1
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize privacy filter
    privacy_filter = PrivacyFilter(
        mask_emails=not args.no_email_masking,
        mask_phones=not args.no_phone_masking,
        mask_urls=not args.no_url_masking,
        mask_numbers=True,
        preserve_style=not args.strict
    )
    
    try:
        stats = process_jsonl_file(args.input, args.output, privacy_filter)
        
        print("\n" + "="*50)
        print("PRIVACY FILTERING COMPLETE")
        print("="*50)
        print(f"Total examples: {stats['total_examples']}")
        print(f"Kept examples: {stats['filtered_examples']}")
        print(f"Rejected examples: {stats['rejected_examples']}")
        print(f"Rejection rate: {stats['rejected_examples']/stats['total_examples']*100:.1f}%")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
