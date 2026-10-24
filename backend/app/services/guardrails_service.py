import re
import logging

logger = logging.getLogger(__name__)

class GuardrailsService:
    @staticmethod
    def mask_pii(text: str) -> str:
        """
        Mask personally identifiable information (PII) such as emails, 
        phone numbers, and credit card numbers.
        Order matters: Mask credit cards first, then phones, then emails to avoid regex overlaps.
        """
        if not text:
            return text

        # 1. Mask credit card numbers first (16 digits)
        text = re.sub(r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b', '[REDACTED_CARD]', text)
        
        # 2. Mask phone numbers (supports international with +, area codes, and 7-digit local numbers)
        text = re.sub(r'(?:\+\d{1,3}[-.\s]?)?(?:\b\d{3}[-.\s]?\d{4}\b|\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b)', '[REDACTED_PHONE]', text)
        
        # 3. Mask email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', text)

        return text

    @staticmethod
    def validate_input_prompt(query: str) -> tuple[bool, str]:
        """
        Validate user input to prevent malicious prompts and prompt injection attacks.
        """
        if not query or len(query.strip()) < 3:
            return False, "Input is too short or empty."

        # Common prompt injection patterns
        jailbreak_patterns = [
            "ignore previous instructions",
            "system prompt",
            "bypass safety",
            "reveal internal instructions"
        ]

        query_lower = query.lower()
        for pattern in jailbreak_patterns:
            if pattern in query_lower:
                logger.warning(f"Detected potential prompt injection attempt: {pattern}")
                return False, "Query blocked due to security and safety policies."

        return True, "Query is safe."