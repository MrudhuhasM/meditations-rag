from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError, AuthenticationError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel
from meditations_rag.config import settings, get_logger
from meditations_rag.core.llm.base import LLMBase
from meditations_rag.core.rate_limiter import rate_limit, RateLimiterFactory
from meditations_rag.core.exceptions import (
    LLMConfigurationError,
    LLMAPIError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMResponseError,
    LLMAuthenticationError,
)

logger = get_logger(__name__)

class OpenAILLM(LLMBase):
    def __init__(self):
        self.settings = settings.openai

        if not self.settings.api_key:
            raise LLMConfigurationError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY in your environment.",
                details={"provider": "openai"}
            )

        try:
            self.client = AsyncOpenAI(api_key=self.settings.api_key.get_secret_value())
            logger.info(f"Initialized OpenAILLM with model: {self.settings.llm_model_name}")
        except Exception as e:
            raise LLMConfigurationError(
                f"Failed to initialize OpenAI client: {str(e)}",
                details={"provider": "openai", "error": str(e)}
            )

    @rate_limit(RateLimiterFactory.get_openai_limiter)
    @retry(
        stop=stop_after_attempt(settings.openai.max_retries), 
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError))
    )
    async def generate(self, prompt: str) -> str:
        try:
            # Check if reasoning is enabled - reasoning models like o1/o3 use Responses API
            # For standard models, use Chat Completions API
            if self.settings.reasoning_enabled and self.settings.llm_model_name.startswith(("o1", "o3", "gpt-5")):
                logger.debug(f"Using Responses API for reasoning model: {self.settings.llm_model_name}")
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
                    logger.warning("OpenAI Responses API returned None for output_text")
                    raise LLMResponseError(
                        "OpenAI Responses API returned empty output",
                        provider="openai",
                        response_data=response,
                        details={"model": self.settings.llm_model_name}
                    )
                
                logger.debug(f"Successfully generated response using Responses API")
                return response.output_text
            else:
                logger.debug(f"Using Chat Completions API for model: {self.settings.llm_model_name}")
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

                if not response.choices or len(response.choices) == 0:
                    logger.error("OpenAI API returned no choices")
                    raise LLMResponseError(
                        "OpenAI API returned no choices in response",
                        provider="openai",
                        response_data=response,
                        details={"model": self.settings.llm_model_name}
                    )

                if response.choices[0].message.content is None:
                    logger.warning("OpenAI API returned None for message content")
                    raise LLMResponseError(
                        "OpenAI API returned empty message content",
                        provider="openai",
                        response_data=response.choices[0],
                        details={
                            "model": self.settings.llm_model_name,
                            "finish_reason": response.choices[0].finish_reason
                        }
                    )

                logger.debug(f"Successfully generated response using Chat Completions API")
                return response.choices[0].message.content

        except AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {str(e)}")
            raise LLMAuthenticationError(
                f"OpenAI authentication failed. Please verify your API key is correct and active.",
                provider="openai",
                details={"error": str(e), "model": self.settings.llm_model_name}
            ) from e
        
        except RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {str(e)}")
            raise LLMRateLimitError(
                f"OpenAI rate limit exceeded. Please wait before retrying.",
                provider="openai",
                details={
                    "error": str(e),
                    "model": self.settings.llm_model_name,
                    "max_retries": settings.openai.max_retries
                }
            ) from e
        
        except APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {str(e)}")
            raise LLMTimeoutError(
                f"OpenAI API request timed out",
                provider="openai",
                details={
                    "error": str(e),
                    "model": self.settings.llm_model_name
                }
            ) from e
        
        except APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {str(e)}")
            raise LLMAPIError(
                f"Failed to connect to OpenAI API. Check your network connection.",
                provider="openai",
                model=self.settings.llm_model_name,
                error_type="connection_error",
                details={"error": str(e)}
            ) from e
        
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise LLMAPIError(
                f"OpenAI API error: {str(e)}",
                provider="openai",
                model=self.settings.llm_model_name,
                status_code=getattr(e, 'status_code', None),
                error_type=getattr(e, 'type', None),
                details={"error": str(e)}
            ) from e
        
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI generation: {str(e)}", exc_info=True)
            raise LLMAPIError(
                f"Unexpected error during OpenAI LLM generation: {str(e)}",
                provider="openai",
                model=self.settings.llm_model_name,
                error_type="unexpected_error",
                details={"error": str(e), "error_type": type(e).__name__}
            ) from e

    @rate_limit(RateLimiterFactory.get_openai_limiter)
    @retry(
        stop=stop_after_attempt(settings.openai.max_retries), 
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError))
    )
    async def generate_structured(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        try:
            logger.debug(f"Generating structured output with model: {self.settings.llm_model_name}")
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

            if not response.choices or len(response.choices) == 0:
                logger.error("OpenAI API returned no choices for structured output")
                raise LLMResponseError(
                    "OpenAI API returned no choices in structured response",
                    provider="openai",
                    response_data=response,
                    details={
                        "model": self.settings.llm_model_name,
                        "response_model": response_model.__name__
                    }
                )

            if response.choices[0].message.refusal:
                logger.warning(f"OpenAI model refused structured output: {response.choices[0].message.refusal}")
                raise LLMResponseError(
                    f"Model refused to generate structured output: {response.choices[0].message.refusal}",
                    provider="openai",
                    response_data=response.choices[0].message,
                    details={
                        "model": self.settings.llm_model_name,
                        "response_model": response_model.__name__,
                        "refusal_reason": response.choices[0].message.refusal
                    }
                )

            parsed = response.choices[0].message.parsed
            if parsed is None:
                logger.error("Failed to parse OpenAI structured response")
                raise LLMResponseError(
                    "Failed to parse structured response from OpenAI",
                    provider="openai",
                    response_data=response.choices[0].message,
                    details={
                        "model": self.settings.llm_model_name,
                        "response_model": response_model.__name__,
                        "finish_reason": response.choices[0].finish_reason
                    }
                )

            logger.debug(f"Successfully generated structured output")
            return parsed

        except AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {str(e)}")
            raise LLMAuthenticationError(
                f"OpenAI authentication failed. Please verify your API key is correct and active.",
                provider="openai",
                details={"error": str(e), "model": self.settings.llm_model_name}
            ) from e
        
        except RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {str(e)}")
            raise LLMRateLimitError(
                f"OpenAI rate limit exceeded. Please wait before retrying.",
                provider="openai",
                details={
                    "error": str(e),
                    "model": self.settings.llm_model_name,
                    "max_retries": settings.openai.max_retries
                }
            ) from e
        
        except APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {str(e)}")
            raise LLMTimeoutError(
                f"OpenAI API request timed out",
                provider="openai",
                details={
                    "error": str(e),
                    "model": self.settings.llm_model_name
                }
            ) from e
        
        except APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {str(e)}")
            raise LLMAPIError(
                f"Failed to connect to OpenAI API. Check your network connection.",
                provider="openai",
                model=self.settings.llm_model_name,
                error_type="connection_error",
                details={"error": str(e)}
            ) from e
        
        except APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise LLMAPIError(
                f"OpenAI API error: {str(e)}",
                provider="openai",
                model=self.settings.llm_model_name,
                status_code=getattr(e, 'status_code', None),
                error_type=getattr(e, 'type', None),
                details={"error": str(e)}
            ) from e
        
        except LLMResponseError:
            # Re-raise our custom exceptions
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI structured generation: {str(e)}", exc_info=True)
            raise LLMAPIError(
                f"Unexpected error during OpenAI structured generation: {str(e)}",
                provider="openai",
                model=self.settings.llm_model_name,
                error_type="unexpected_error",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "response_model": response_model.__name__
                }
            ) from e

