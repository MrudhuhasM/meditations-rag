from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from meditations_rag.config import get_logger, settings
from meditations_rag.core.exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from meditations_rag.core.llm.base import LLMBase
from meditations_rag.core.rate_limiter import RateLimiterFactory, rate_limit

logger = get_logger(__name__)


class GeminiLLM(LLMBase):
    def __init__(self, model_name: str | None = None):
        self.settings = settings.gemini
        self.model_name = model_name or self.settings.llm_model_name

        if not self.settings.api_key:
            raise LLMConfigurationError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your environment.",
                details={"provider": "gemini"},
            )

        try:
            self.client = genai.Client(api_key=self.settings.api_key.get_secret_value())
            logger.info(f"Initialized GeminiLLM with model: {self.model_name}")
        except Exception as e:
            raise LLMConfigurationError(
                f"Failed to initialize Gemini client: {str(e)}",
                details={"provider": "gemini", "error": str(e)},
            )

    @rate_limit(RateLimiterFactory.get_gemini_limiter)
    @retry(
        stop=stop_after_attempt(settings.gemini.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((errors.ServerError,)),
    )
    async def generate(self, prompt: str) -> str:
        try:
            # Build config with thinking support if enabled
            config_dict = {
                "max_output_tokens": self.settings.max_tokens,
                "temperature": self.settings.temperature,
            }

            # Add thinking config for Gemini 2.5 models when reasoning is enabled
            if self.settings.reasoning_enabled and "2.5" in self.model_name:
                logger.debug(
                    f"Enabling thinking mode for Gemini 2.5 model: {self.model_name}"
                )
                config_dict["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=-1  # Dynamic thinking - model decides when and how much to think
                )

            config = types.GenerateContentConfig(**config_dict)

            logger.debug(f"Calling Gemini API with model: {self.model_name}")
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            if not hasattr(response, "text") or response.text is None:
                logger.warning("Gemini API returned no text in response")
                raise LLMResponseError(
                    "Gemini API returned empty response",
                    provider="gemini",
                    response_data=str(response),
                    details={
                        "model": self.model_name,
                        "has_candidates": hasattr(response, "candidates"),
                        "candidates_count": (
                            len(response.candidates)
                            if hasattr(response, "candidates")
                            else 0
                        ),
                    },
                )

            logger.debug("Successfully generated response from Gemini")
            return response.text

        except errors.ClientError as e:
            # Check for authentication/permission issues
            error_message = str(e).lower()
            if (
                "auth" in error_message
                or "permission" in error_message
                or "api key" in error_message
            ):
                logger.error(f"Gemini authentication failed: {str(e)}")
                raise LLMAuthenticationError(
                    "Gemini authentication failed. Please verify your API key is correct and active.",
                    provider="gemini",
                    details={"error": str(e), "model": self.model_name},
                ) from e

            # Check for rate limiting
            if (
                "rate" in error_message
                or "quota" in error_message
                or "limit" in error_message
            ):
                logger.warning(f"Gemini rate limit exceeded: {str(e)}")
                raise LLMRateLimitError(
                    "Gemini rate limit exceeded. Please wait before retrying.",
                    provider="gemini",
                    details={
                        "error": str(e),
                        "model": self.model_name,
                        "max_retries": settings.gemini.max_retries,
                    },
                ) from e

            # Check for timeout
            if "timeout" in error_message or "deadline" in error_message:
                logger.error(f"Gemini API timeout: {str(e)}")
                raise LLMTimeoutError(
                    "Gemini API request timed out",
                    provider="gemini",
                    details={"error": str(e), "model": self.model_name},
                ) from e

            # Generic client error
            logger.error(f"Gemini client error: {str(e)}")
            raise LLMAPIError(
                f"Gemini client error: {str(e)}",
                provider="gemini",
                model=self.model_name,
                error_type="client_error",
                details={"error": str(e)},
            ) from e

        except errors.ServerError as e:
            logger.error(f"Gemini server error: {str(e)}")
            raise LLMAPIError(
                f"Gemini server error: {str(e)}",
                provider="gemini",
                model=self.model_name,
                error_type="server_error",
                details={"error": str(e)},
            ) from e

        except errors.APIError as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise LLMAPIError(
                f"Gemini API error: {str(e)}",
                provider="gemini",
                model=self.model_name,
                error_type=type(e).__name__,
                details={"error": str(e)},
            ) from e

        except Exception as e:
            logger.error(
                f"Unexpected error during Gemini generation: {str(e)}", exc_info=True
            )
            raise LLMAPIError(
                f"Unexpected error during Gemini LLM generation: {str(e)}",
                provider="gemini",
                model=self.model_name,
                error_type="unexpected_error",
                details={"error": str(e), "error_type": type(e).__name__},
            ) from e

    @rate_limit(RateLimiterFactory.get_gemini_limiter)
    @retry(
        stop=stop_after_attempt(settings.gemini.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((errors.ServerError,)),
    )
    async def generate_structured(
        self, prompt: str, response_model: type[BaseModel]
    ) -> BaseModel:
        try:
            # Build config with thinking support if enabled
            config_dict = {
                "max_output_tokens": self.settings.max_tokens,
                "temperature": self.settings.temperature,
                "response_mime_type": "application/json",
                "response_schema": response_model.model_json_schema(),
            }

            # Add thinking config for Gemini 2.5 models when reasoning is enabled
            if self.settings.reasoning_enabled and "2.5" in self.model_name:
                logger.debug("Enabling thinking mode for Gemini 2.5 structured output")
                config_dict["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=-1  # Dynamic thinking
                )

            config = types.GenerateContentConfig(**config_dict)

            logger.debug(
                f"Calling Gemini API for structured output with model: {self.model_name}"
            )
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            if not hasattr(response, "text") or response.text is None:
                logger.error("Gemini API returned no text for structured output")
                raise LLMResponseError(
                    "Failed to generate structured response from Gemini",
                    provider="gemini",
                    response_data=str(response),
                    details={
                        "model": self.model_name,
                        "response_model": response_model.__name__,
                        "has_candidates": hasattr(response, "candidates"),
                        "candidates_count": (
                            len(response.candidates)
                            if hasattr(response, "candidates")
                            else 0
                        ),
                    },
                )

            # Parse the JSON response into the Pydantic model
            try:
                logger.debug(f"Parsing Gemini response into {response_model.__name__}")
                return response_model.model_validate_json(response.text)
            except Exception as parse_error:
                logger.error(
                    f"Failed to parse Gemini response into Pydantic model: {str(parse_error)}"
                )
                raise LLMResponseError(
                    f"Failed to parse Gemini response into {response_model.__name__}: {str(parse_error)}",
                    provider="gemini",
                    response_data=response.text,
                    details={
                        "model": self.model_name,
                        "response_model": response_model.__name__,
                        "parse_error": str(parse_error),
                    },
                ) from parse_error

        except errors.ClientError as e:
            # Check for authentication/permission issues
            error_message = str(e).lower()
            if (
                "auth" in error_message
                or "permission" in error_message
                or "api key" in error_message
            ):
                logger.error(f"Gemini authentication failed: {str(e)}")
                raise LLMAuthenticationError(
                    "Gemini authentication failed. Please verify your API key is correct and active.",
                    provider="gemini",
                    details={"error": str(e), "model": self.model_name},
                ) from e

            # Check for rate limiting
            if (
                "rate" in error_message
                or "quota" in error_message
                or "limit" in error_message
            ):
                logger.warning(f"Gemini rate limit exceeded: {str(e)}")
                raise LLMRateLimitError(
                    "Gemini rate limit exceeded. Please wait before retrying.",
                    provider="gemini",
                    details={
                        "error": str(e),
                        "model": self.model_name,
                        "max_retries": settings.gemini.max_retries,
                    },
                ) from e

            # Check for timeout
            if "timeout" in error_message or "deadline" in error_message:
                logger.error(f"Gemini API timeout: {str(e)}")
                raise LLMTimeoutError(
                    "Gemini API request timed out",
                    provider="gemini",
                    details={"error": str(e), "model": self.model_name},
                ) from e

            # Generic client error
            logger.error(f"Gemini client error: {str(e)}")
            raise LLMAPIError(
                f"Gemini client error: {str(e)}",
                provider="gemini",
                model=self.model_name,
                error_type="client_error",
                details={"error": str(e)},
            ) from e

        except errors.ServerError as e:
            logger.error(f"Gemini server error: {str(e)}")
            raise LLMAPIError(
                f"Gemini server error: {str(e)}",
                provider="gemini",
                model=self.model_name,
                error_type="server_error",
                details={"error": str(e)},
            ) from e

        except errors.APIError as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise LLMAPIError(
                f"Gemini API error: {str(e)}",
                provider="gemini",
                model=self.model_name,
                error_type=type(e).__name__,
                details={"error": str(e)},
            ) from e

        except LLMResponseError:
            # Re-raise our custom exceptions
            raise

        except Exception as e:
            logger.error(
                f"Unexpected error during Gemini structured generation: {str(e)}",
                exc_info=True,
            )
            raise LLMAPIError(
                f"Unexpected error during Gemini structured generation: {str(e)}",
                provider="gemini",
                model=self.model_name,
                error_type="unexpected_error",
                details={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "response_model": response_model.__name__,
                },
            ) from e
