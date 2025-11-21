from openai import APIConnectionError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from meditations_rag.config import get_logger, settings
from meditations_rag.core.exceptions import LLMConfigurationError, LLMResponseError
from meditations_rag.core.llm.base_openai import BaseOpenAILLM
from meditations_rag.core.rate_limiter import RateLimiterFactory, rate_limit

logger = get_logger(__name__)


class OpenAILLM(BaseOpenAILLM):
    def __init__(self, model_name: str | None = None):
        openai_settings = settings.openai

        if not openai_settings.api_key:
            raise LLMConfigurationError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY in your environment.",
                details={"provider": "openai"},
            )

        try:
            client = AsyncOpenAI(api_key=openai_settings.api_key.get_secret_value())
            super().__init__(client, openai_settings, "openai", model_name)
            logger.info(f"Initialized OpenAILLM with model: {self.model_name}")
        except Exception as e:
            raise LLMConfigurationError(
                f"Failed to initialize OpenAI client: {str(e)}",
                details={"provider": "openai", "error": str(e)},
            )

    @rate_limit(RateLimiterFactory.get_openai_limiter)
    @retry(
        stop=stop_after_attempt(settings.openai.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
    )
    async def generate(self, prompt: str) -> str:
        try:
            # Check if reasoning is enabled - reasoning models like o1/o3 use Responses API
            # For standard models, use Chat Completions API
            if self.settings.reasoning_enabled and self.model_name.startswith(
                ("o1", "o3", "gpt-5")
            ):
                logger.debug(
                    f"Using Responses API for reasoning model: {self.model_name}"
                )
                # Use Responses API for reasoning models
                response = await self.client.responses.create(
                    model=self.model_name,
                    reasoning={"effort": "medium"},  # low, medium, or high
                    input=[{"role": "user", "content": prompt}],
                    max_output_tokens=self.settings.max_tokens,
                )

                if response.output_text is None:
                    logger.warning("OpenAI Responses API returned None for output_text")
                    raise LLMResponseError(
                        "OpenAI Responses API returned empty output",
                        provider="openai",
                        response_data=response,
                        details={"model": self.model_name},
                    )

                logger.debug("Successfully generated response using Responses API")
                return response.output_text
            else:
                return await self._generate_chat_completion(prompt)

        except Exception as e:
            self._handle_error(e)

    @rate_limit(RateLimiterFactory.get_openai_limiter)
    @retry(
        stop=stop_after_attempt(settings.openai.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError)),
    )
    async def generate_structured(
        self, prompt: str, response_model: type[BaseModel]
    ) -> BaseModel:
        return await self._generate_structured_common(prompt, response_model)
