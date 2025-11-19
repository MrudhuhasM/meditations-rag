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


class OpenRouterLLM(LLMBase):
    '''OpenRouter LLM implementation using OpenAI SDK with custom base URL.'''
    
    def __init__(self):
        self.settings = settings.openrouter

        if not self.settings.api_key:
            raise LLMConfigurationError(
                'OpenRouter API key is not configured. Please set OPENROUTER_API_KEY in your environment.',
                details={'provider': 'openrouter'}
            )

        try:
            # Build default headers for OpenRouter attribution (optional)
            default_headers = {}
            if self.settings.http_referer:
                default_headers['HTTP-Referer'] = self.settings.http_referer
            if self.settings.x_title:
                default_headers['X-Title'] = self.settings.x_title

            # Initialize OpenAI client with OpenRouter base URL
            self.client = AsyncOpenAI(
                api_key=self.settings.api_key.get_secret_value(),
                base_url=self.settings.api_base,
                default_headers=default_headers if default_headers else None,
                timeout=self.settings.timeout
            )
            
            logger.info(f'Initialized OpenRouterLLM with model: {self.settings.llm_model_name}')
        except Exception as e:
            raise LLMConfigurationError(
                f'Failed to initialize OpenRouter client: {str(e)}',
                details={'provider': 'openrouter', 'error': str(e)}
            )

    @rate_limit(RateLimiterFactory.get_openrouter_limiter)
    @retry(
        stop=stop_after_attempt(settings.openrouter.max_retries), 
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError))
    )
    async def generate(self, prompt: str) -> str:
        '''
        Generate text response from OpenRouter.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Generated text response
        '''
        try:
            logger.debug(f'Calling OpenRouter API with model: {self.settings.llm_model_name}')
            response = await self.client.chat.completions.create(
                model=self.settings.llm_model_name,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt},
                ],
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
            )

            if not response.choices or len(response.choices) == 0:
                logger.error('OpenRouter API returned no choices')
                raise LLMResponseError(
                    'OpenRouter API returned no choices in response',
                    provider='openrouter',
                    response_data=response,
                    details={'model': self.settings.llm_model_name}
                )

            if response.choices[0].message.content is None:
                logger.warning('OpenRouter API returned None for message content')
                raise LLMResponseError(
                    'OpenRouter API returned empty message content',
                    provider='openrouter',
                    response_data=response.choices[0],
                    details={
                        'model': self.settings.llm_model_name,
                        'finish_reason': response.choices[0].finish_reason
                    }
                )

            logger.debug('Successfully generated response from OpenRouter')
            return response.choices[0].message.content

        except AuthenticationError as e:
            logger.error(f'OpenRouter authentication failed: {str(e)}')
            raise LLMAuthenticationError(
                f'OpenRouter authentication failed. Please verify your API key is correct and active.',
                provider='openrouter',
                details={'error': str(e), 'model': self.settings.llm_model_name}
            ) from e
        
        except RateLimitError as e:
            logger.warning(f'OpenRouter rate limit exceeded: {str(e)}')
            raise LLMRateLimitError(
                f'OpenRouter rate limit exceeded. Please wait before retrying.',
                provider='openrouter',
                details={
                    'error': str(e),
                    'model': self.settings.llm_model_name,
                    'max_retries': settings.openrouter.max_retries
                }
            ) from e
        
        except APITimeoutError as e:
            logger.error(f'OpenRouter API timeout: {str(e)}')
            raise LLMTimeoutError(
                f'OpenRouter API request timed out',
                provider='openrouter',
                timeout_seconds=self.settings.timeout,
                details={
                    'error': str(e),
                    'model': self.settings.llm_model_name
                }
            ) from e
        
        except APIConnectionError as e:
            logger.error(f'OpenRouter API connection error: {str(e)}')
            raise LLMAPIError(
                f'Failed to connect to OpenRouter API. Check your network connection.',
                provider='openrouter',
                model=self.settings.llm_model_name,
                error_type='connection_error',
                details={'error': str(e)}
            ) from e
        
        except APIError as e:
            logger.error(f'OpenRouter API error: {str(e)}')
            raise LLMAPIError(
                f'OpenRouter API error: {str(e)}',
                provider='openrouter',
                model=self.settings.llm_model_name,
                status_code=getattr(e, 'status_code', None),
                error_type=getattr(e, 'type', None),
                details={'error': str(e)}
            ) from e
        
        except Exception as e:
            logger.error(f'Unexpected error during OpenRouter generation: {str(e)}', exc_info=True)
            raise LLMAPIError(
                f'Unexpected error during OpenRouter LLM generation: {str(e)}',
                provider='openrouter',
                model=self.settings.llm_model_name,
                error_type='unexpected_error',
                details={'error': str(e), 'error_type': type(e).__name__}
            ) from e

    @rate_limit(RateLimiterFactory.get_openrouter_limiter)
    @retry(
        stop=stop_after_attempt(settings.openrouter.max_retries), 
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError))
    )
    async def generate_structured(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        '''
        Generate structured output from OpenRouter using response_format.
        
        Args:
            prompt: The input prompt
            response_model: Pydantic model class for structured output
            
        Returns:
            Parsed structured response
        '''
        try:
            logger.debug(f'Calling OpenRouter API for structured output with model: {self.settings.llm_model_name}')
            response = await self.client.chat.completions.parse(
                model=self.settings.llm_model_name,
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt},
                ],
                response_format=response_model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
            )

            if not response.choices or len(response.choices) == 0:
                logger.error('OpenRouter API returned no choices for structured output')
                raise LLMResponseError(
                    'OpenRouter API returned no choices in structured response',
                    provider='openrouter',
                    response_data=response,
                    details={
                        'model': self.settings.llm_model_name,
                        'response_model': response_model.__name__
                    }
                )

            if response.choices[0].message.refusal:
                logger.warning(f'OpenRouter model refused structured output: {response.choices[0].message.refusal}')
                raise LLMResponseError(
                    f'Model refused to generate structured output: {response.choices[0].message.refusal}',
                    provider='openrouter',
                    response_data=response.choices[0].message,
                    details={
                        'model': self.settings.llm_model_name,
                        'response_model': response_model.__name__,
                        'refusal_reason': response.choices[0].message.refusal
                    }
                )

            parsed = response.choices[0].message.parsed
            if parsed is None:
                logger.error('Failed to parse OpenRouter structured response')
                raise LLMResponseError(
                    'Failed to parse structured response from OpenRouter',
                    provider='openrouter',
                    response_data=response.choices[0].message,
                    details={
                        'model': self.settings.llm_model_name,
                        'response_model': response_model.__name__,
                        'finish_reason': response.choices[0].finish_reason
                    }
                )

            logger.debug('Successfully generated structured output from OpenRouter')
            return parsed

        except AuthenticationError as e:
            logger.error(f'OpenRouter authentication failed: {str(e)}')
            raise LLMAuthenticationError(
                f'OpenRouter authentication failed. Please verify your API key is correct and active.',
                provider='openrouter',
                details={'error': str(e), 'model': self.settings.llm_model_name}
            ) from e
        
        except RateLimitError as e:
            logger.warning(f'OpenRouter rate limit exceeded: {str(e)}')
            raise LLMRateLimitError(
                f'OpenRouter rate limit exceeded. Please wait before retrying.',
                provider='openrouter',
                details={
                    'error': str(e),
                    'model': self.settings.llm_model_name,
                    'max_retries': settings.openrouter.max_retries
                }
            ) from e
        
        except APITimeoutError as e:
            logger.error(f'OpenRouter API timeout: {str(e)}')
            raise LLMTimeoutError(
                f'OpenRouter API request timed out',
                provider='openrouter',
                timeout_seconds=self.settings.timeout,
                details={
                    'error': str(e),
                    'model': self.settings.llm_model_name
                }
            ) from e
        
        except APIConnectionError as e:
            logger.error(f'OpenRouter API connection error: {str(e)}')
            raise LLMAPIError(
                f'Failed to connect to OpenRouter API. Check your network connection.',
                provider='openrouter',
                model=self.settings.llm_model_name,
                error_type='connection_error',
                details={'error': str(e)}
            ) from e
        
        except APIError as e:
            logger.error(f'OpenRouter API error: {str(e)}')
            raise LLMAPIError(
                f'OpenRouter API error: {str(e)}',
                provider='openrouter',
                model=self.settings.llm_model_name,
                status_code=getattr(e, 'status_code', None),
                error_type=getattr(e, 'type', None),
                details={'error': str(e)}
            ) from e
        
        except LLMResponseError:
            # Re-raise our custom exceptions
            raise
        
        except Exception as e:
            logger.error(f'Unexpected error during OpenRouter structured generation: {str(e)}', exc_info=True)
            raise LLMAPIError(
                f'Unexpected error during OpenRouter structured generation: {str(e)}',
                provider='openrouter',
                model=self.settings.llm_model_name,
                error_type='unexpected_error',
                details={
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'response_model': response_model.__name__
                }
            ) from e
