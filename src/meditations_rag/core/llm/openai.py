from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel
from meditations_rag.config import settings, get_logger
from meditations_rag.core.llm.base import LLMBase

logger = get_logger(__name__)

class OpenAILLM(LLMBase):
    def __init__(self):
        self.settings = settings.openai

        if not self.settings.api_key:
            raise ValueError("OpenAI API key is not configured.")

        self.client = AsyncOpenAI(api_key=self.settings.api_key.get_secret_value())
        logger.info(f"Initialized OpenAILLM with model: {self.settings.llm_model_name}")

    @retry(stop=stop_after_attempt(settings.openai.max_retries), wait=wait_exponential())
    async def generate(self, prompt: str) -> str:
        # Check if reasoning is enabled - reasoning models like o1/o3 use Responses API
        # For standard models, use Chat Completions API
        if self.settings.reasoning_enabled and self.settings.llm_model_name.startswith(("o1", "o3", "gpt-5")):
            # Use Responses API for reasoning models
            response = await self.client.responses.create(
                model=self.settings.llm_model_name,
                reasoning={"effort": "medium"},  # low, medium, or high
                input=[
                    {"role": "user", "content": prompt}
                ],
                max_output_tokens=self.settings.max_tokens,
            )
            
            if response.output_text is None:
                return ""
            
            return response.output_text
        else:
            # Use standard Chat Completions API
            response = await self.client.chat.completions.create(
                model=self.settings.llm_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
            )

            if response.choices[0].message.content is None:
                return ""

            return response.choices[0].message.content

    @retry(stop=stop_after_attempt(settings.openai.max_retries), wait=wait_exponential())
    async def generate_structured(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        # Reasoning models don't support structured output in the same way
        # Use standard Chat Completions API with response_format
        response = await self.client.chat.completions.parse(
            model=self.settings.llm_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_model,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
        )

        if response.choices[0].message.refusal:
            raise ValueError(f"Model refused to generate structured output: {response.choices[0].message.refusal}")

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Failed to parse structured response")

        return parsed

