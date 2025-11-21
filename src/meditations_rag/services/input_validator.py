import re
from typing import List

from meditations_rag.config import get_logger

logger = get_logger(__name__)


class InputValidator:
    """
    Validates user input using regex and keyword matching to detect
    potentially dangerous or unrelated prompts without using an LLM.
    """

    # Patterns that suggest an attempt to override system instructions
    INJECTION_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"ignore (all )?directions",
        r"forget (all )?previous instructions",
        r"system prompt",
        r"you are now",
        r"act as",
        r"roleplay as",
        r"simulate",
        r"jailbreak",
        r"override",
        r"developer mode",
    ]

    # Patterns for potentially harmful or dangerous content
    DANGEROUS_PATTERNS = [
        r"hack",
        r"exploit",
        r"vulnerability",
        r"attack",
        r"malware",
        r"virus",
        r"trojan",
        r"ransomware",
        r"phishing",
        r"social engineering",
        r"bomb",
        r"weapon",
        r"suicide",
        r"self-harm",
        r"kill",
        r"murder",
        r"terrorist",
        r"terrorism",
    ]

    # Basic profanity filter (can be expanded)
    PROFANITY_PATTERNS = [
        r"fuck",
        r"shit",
        r"bitch",
        r"asshole",
        r"cunt",
        r"dick",
        r"pussy",
        r"whore",
        r"slut",
    ]

    def __init__(self):
        self.injection_regex = self._compile_patterns(self.INJECTION_PATTERNS)
        self.dangerous_regex = self._compile_patterns(self.DANGEROUS_PATTERNS)
        self.profanity_regex = self._compile_patterns(self.PROFANITY_PATTERNS)

    def _compile_patterns(self, patterns: List[str]) -> re.Pattern:
        """Compiles a list of regex patterns into a single pattern object."""
        combined_pattern = "|".join(patterns)
        return re.compile(combined_pattern, re.IGNORECASE)

    def validate(self, query: str) -> bool:
        """
        Validates the query. Returns True if valid, False if invalid.
        """
        if not query or not query.strip():
            return False

        if self.injection_regex.search(query):
            logger.warning(f"Blocked potential prompt injection: {query}")
            return False

        if self.dangerous_regex.search(query):
            logger.warning(f"Blocked potential dangerous content: {query}")
            return False

        if self.profanity_regex.search(query):
            logger.warning(f"Blocked potential profanity: {query}")
            return False

        return True


# Global instance
input_validator = InputValidator()
