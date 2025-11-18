from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel
from meditations_rag.config import settings, get_logger
from meditations_rag.core.llm.base import LLMBase

logger = get_logger(__name__)

class LocalLLM(LLMBase):
    def __init__(self):
        self.settings = settings.local_llm

        if not self.settings.api_base_url:
            raise ValueError("Local LLM API base URL is not configured.")

        self.client = AsyncOpenAI(
            api_key=self.settings.api_key.get_secret_value() if self.settings.api_key else None,
            base_url=self.settings.api_base_url,
        )
        logger.info(f"Initialized LocalLLM with model: {self.settings.llm_model_name or 'local-model'}")

    @retry(stop=stop_after_attempt(settings.local_llm.max_retries), wait=wait_exponential())
    async def generate(self, prompt: str) -> str:
        
        model_name = self.settings.llm_model_name or "local-model"
        system_message = "You are a helpful assistant."
        if self.settings.reasoning_enabled and any(keyword in model_name.lower() for keyword in ["think", "o1", "o3", "reason"]):
            system_message = "You are a helpful assistant. Think through problems step by step before providing your answer."
        
        response = await self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
        )

        if response.choices[0].message.content is None:
            return ""

        return response.choices[0].message.content

    @retry(stop=stop_after_attempt(settings.local_llm.max_retries), wait=wait_exponential())
    async def generate_structured(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        response = await self.client.chat.completions.parse(
            model=self.settings.llm_model_name or "local-model",
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