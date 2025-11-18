from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel
from meditations_rag.config import settings, get_logger
from meditations_rag.core.llm.base import LLMBase

logger = get_logger(__name__)

class GeminiLLM(LLMBase):
    def __init__(self):
        self.settings = settings.gemini

        if not self.settings.api_key:
            raise ValueError("Gemini API key is not configured.")

        self.client = genai.Client(api_key=self.settings.api_key.get_secret_value())
        logger.info(f"Initialized GeminiLLM with model: {self.settings.llm_model_name}")

    @retry(stop=stop_after_attempt(settings.gemini.max_retries), wait=wait_exponential())
    async def generate(self, prompt: str) -> str:
        # Build config with thinking support if enabled
        config_dict = {
            "max_output_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
        }
        
        # Add thinking config for Gemini 2.5 models when reasoning is enabled
        if self.settings.reasoning_enabled and "2.5" in self.settings.llm_model_name:
            config_dict["thinking_config"] = types.ThinkingConfig(
                thinking_budget=-1  # Dynamic thinking - model decides when and how much to think
            )
        
        config = types.GenerateContentConfig(**config_dict)
        
        response = await self.client.aio.models.generate_content(
            model=self.settings.llm_model_name,
            contents=prompt,
            config=config,
        )

        if response.text is None:
            return ""

        return response.text

    @retry(stop=stop_after_attempt(settings.gemini.max_retries), wait=wait_exponential())
    async def generate_structured(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        # Build config with thinking support if enabled
        config_dict = {
            "max_output_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "response_mime_type": "application/json",
            "response_schema": response_model.model_json_schema(),
        }
        
        # Add thinking config for Gemini 2.5 models when reasoning is enabled
        if self.settings.reasoning_enabled and "2.5" in self.settings.llm_model_name:
            config_dict["thinking_config"] = types.ThinkingConfig(
                thinking_budget=-1  # Dynamic thinking
            )
        
        config = types.GenerateContentConfig(**config_dict)
        
        response = await self.client.aio.models.generate_content(
            model=self.settings.llm_model_name,
            contents=prompt,
            config=config,
        )

        if response.text is None:
            raise ValueError("Failed to generate structured response")

        # Parse the JSON response into the Pydantic model
        return response_model.model_validate_json(response.text)