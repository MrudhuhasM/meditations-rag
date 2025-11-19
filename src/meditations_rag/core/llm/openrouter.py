from openai import AsyncOpenAI, APIConnectionError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel
from meditations_rag.config import settings, get_logger
from meditations_rag.core.llm.base_openai import BaseOpenAILLM
from meditations_rag.core.rate_limiter import rate_limit, RateLimiterFactory
from meditations_rag.core.exceptions import LLMConfigurationError

logger = get_logger(__name__)


class OpenRouterLLM(BaseOpenAILLM):
    '''OpenRouter LLM implementation using OpenAI SDK with custom base URL.'''
    
    def __init__(self):
        openrouter_settings = settings.openrouter

        if not openrouter_settings.api_key:
            raise LLMConfigurationError(
                'OpenRouter API key is not configured. Please set OPENROUTER_API_KEY in your environment.',
                details={'provider': 'openrouter'}
            )

        try:
            # Build default headers for OpenRouter attribution (optional)
            default_headers = {}
            if openrouter_settings.http_referer:
                default_headers['HTTP-Referer'] = openrouter_settings.http_referer
            if openrouter_settings.x_title:
                default_headers['X-Title'] = openrouter_settings.x_title

            # Initialize OpenAI client with OpenRouter base URL
            client = AsyncOpenAI(
                api_key=openrouter_settings.api_key.get_secret_value(),
                base_url=openrouter_settings.api_base,
                default_headers=default_headers if default_headers else None,
                timeout=openrouter_settings.timeout
            )
            
            super().__init__(client, openrouter_settings, "openrouter")
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
            return await self._generate_chat_completion(prompt)
        except Exception as e:
            self._handle_error(e)

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
        return await self._generate_structured_common(prompt, response_model)

