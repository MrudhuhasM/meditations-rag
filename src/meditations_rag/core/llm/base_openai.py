from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError, AuthenticationError, APIConnectionError
from pydantic import BaseModel
from meditations_rag.core.llm.base import LLMBase
from meditations_rag.core.exceptions import (
    LLMAPIError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMResponseError,
    LLMAuthenticationError,
)
from meditations_rag.config import get_logger

logger = get_logger(__name__)

class BaseOpenAILLM(LLMBase):
    """Base class for OpenAI-compatible LLM providers."""

    def __init__(self, client: AsyncOpenAI, settings, provider_name: str):
        self.client = client
        self.settings = settings
        self.provider_name = provider_name

    async def _generate_chat_completion(self, prompt: str, system_message: str = "You are a helpful assistant.") -> str:
        """Generate text using the standard Chat Completions API."""
        try:
            logger.debug(f"Using Chat Completions API for model: {self.settings.llm_model_name}")
            response = await self.client.chat.completions.create(
                model=self.settings.llm_model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
            )

            if not response.choices or len(response.choices) == 0:
                logger.error(f"{self.provider_name} API returned no choices")
                raise LLMResponseError(
                    f"{self.provider_name} API returned no choices in response",
                    provider=self.provider_name,
                    response_data=response,
                    details={"model": self.settings.llm_model_name}
                )

            if response.choices[0].message.content is None:
                logger.warning(f"{self.provider_name} API returned None for message content")
                raise LLMResponseError(
                    f"{self.provider_name} API returned empty message content",
                    provider=self.provider_name,
                    response_data=response.choices[0],
                    details={
                        "model": self.settings.llm_model_name,
                        "finish_reason": response.choices[0].finish_reason
                    }
                )

            logger.debug(f"Successfully generated response using Chat Completions API")
            return response.choices[0].message.content

        except Exception as e:
            self._handle_error(e)

    async def _generate_structured_common(self, prompt: str, response_model: type[BaseModel], system_message: str = "You are a helpful assistant.") -> BaseModel:
        """Generate structured output using the parse method."""
        try:
            logger.debug(f"Generating structured output with model: {self.settings.llm_model_name}")
            response = await self.client.chat.completions.parse(
                model=self.settings.llm_model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
            )

            if not response.choices or len(response.choices) == 0:
                logger.error(f"{self.provider_name} API returned no choices for structured output")
                raise LLMResponseError(
                    f"{self.provider_name} API returned no choices in structured response",
                    provider=self.provider_name,
                    response_data=response,
                    details={
                        "model": self.settings.llm_model_name,
                        "response_model": response_model.__name__
                    }
                )

            if response.choices[0].message.refusal:
                logger.warning(f"{self.provider_name} model refused structured output: {response.choices[0].message.refusal}")
                raise LLMResponseError(
                    f"Model refused to generate structured output: {response.choices[0].message.refusal}",
                    provider=self.provider_name,
                    response_data=response.choices[0].message,
                    details={
                        "model": self.settings.llm_model_name,
                        "response_model": response_model.__name__,
                        "refusal_reason": response.choices[0].message.refusal
                    }
                )

            parsed = response.choices[0].message.parsed
            if parsed is None:
                logger.error(f"Failed to parse {self.provider_name} structured response")
                raise LLMResponseError(
                    f"Failed to parse structured response from {self.provider_name}",
                    provider=self.provider_name,
                    response_data=response.choices[0].message,
                    details={
                        "model": self.settings.llm_model_name,
                        "response_model": response_model.__name__,
                        "finish_reason": response.choices[0].finish_reason
                    }
                )

            logger.debug(f"Successfully generated structured output")
            return parsed

        except Exception as e:
            self._handle_error(e, response_model=response_model)

    def _handle_error(self, e: Exception, response_model=None):
        """Common error handling for OpenAI-compatible APIs."""
        if isinstance(e, AuthenticationError):
            logger.error(f"{self.provider_name} authentication failed: {str(e)}")
            raise LLMAuthenticationError(
                f"{self.provider_name} authentication failed. Please verify your API key is correct and active.",
                provider=self.provider_name,
                details={"error": str(e), "model": self.settings.llm_model_name}
            ) from e
        
        if isinstance(e, RateLimitError):
            logger.warning(f"{self.provider_name} rate limit exceeded: {str(e)}")
            raise LLMRateLimitError(
                f"{self.provider_name} rate limit exceeded. Please wait before retrying.",
                provider=self.provider_name,
                details={
                    "error": str(e),
                    "model": self.settings.llm_model_name,
                    "max_retries": getattr(self.settings, 'max_retries', 3)
                }
            ) from e
        
        if isinstance(e, APITimeoutError):
            logger.error(f"{self.provider_name} API timeout: {str(e)}")
            raise LLMTimeoutError(
                f"{self.provider_name} API request timed out",
                provider=self.provider_name,
                details={
                    "error": str(e),
                    "model": self.settings.llm_model_name
                }
            ) from e
        
        if isinstance(e, APIConnectionError):
            logger.error(f"{self.provider_name} API connection error: {str(e)}")
            raise LLMAPIError(
                f"Failed to connect to {self.provider_name} API. Check your network connection.",
                provider=self.provider_name,
                model=self.settings.llm_model_name,
                error_type="connection_error",
                details={"error": str(e)}
            ) from e
        
        if isinstance(e, APIError):
            logger.error(f"{self.provider_name} API error: {str(e)}")
            raise LLMAPIError(
                f"{self.provider_name} API error: {str(e)}",
                provider=self.provider_name,
                model=self.settings.llm_model_name,
                status_code=getattr(e, 'status_code', None),
                error_type=getattr(e, 'type', None),
                details={"error": str(e)}
            ) from e
        
        if isinstance(e, LLMResponseError):
            raise e
        
        logger.error(f"Unexpected error during {self.provider_name} generation: {str(e)}", exc_info=True)
        details = {"error": str(e), "error_type": type(e).__name__}
        if response_model:
            details["response_model"] = response_model.__name__
            
        raise LLMAPIError(
            f"Unexpected error during {self.provider_name} generation: {str(e)}",
            provider=self.provider_name,
            model=self.settings.llm_model_name,
            error_type="unexpected_error",
            details=details
        ) from e
