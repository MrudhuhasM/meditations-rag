from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from meditations_rag.config import get_logger, settings
from meditations_rag.core.llm.base_openai import BaseOpenAILLM

logger = get_logger(__name__)


class LocalLLM(BaseOpenAILLM):
    def __init__(self, model_name: str | None = None):
        local_settings = settings.local_llm

        if not local_settings.api_base_url:
            raise ValueError("Local LLM API base URL is not configured.")

        client = AsyncOpenAI(
            api_key=(
                local_settings.api_key.get_secret_value()
                if local_settings.api_key
                else None
            ),
            base_url=local_settings.api_base_url,
        )
        super().__init__(client, local_settings, "local_llm", model_name)
        logger.info(
            f"Initialized LocalLLM with model: {self.model_name or 'local-model'}"
        )

    @retry(
        stop=stop_after_attempt(settings.local_llm.max_retries), wait=wait_exponential()
    )
    async def generate(self, prompt: str) -> str:
        model_name = self.model_name or "local-model"
        system_message = "You are a helpful assistant."
        if self.settings.reasoning_enabled and any(
            keyword in model_name.lower() for keyword in ["think", "o1", "o3", "reason"]
        ):
            system_message = "You are a helpful assistant. Think through problems step by step before providing your answer."

        try:
            return await self._generate_chat_completion(
                prompt, system_message=system_message
            )
        except Exception as e:
            self._handle_error(e)

    @retry(
        stop=stop_after_attempt(settings.local_llm.max_retries), wait=wait_exponential()
    )
    async def generate_structured(
        self, prompt: str, response_model: type[BaseModel]
    ) -> BaseModel:
        return await self._generate_structured_common(prompt, response_model)
