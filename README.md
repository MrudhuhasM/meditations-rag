
This is a portfolio project to showcase my work to potential employers.

In this project i am building an Agentic RAG application 
My source for this project is Meditations by Marcus Aurelius, this is philosophical text that contains personal writings by Marcus Aurelius, Roman Emperor from 161 to 180 AD, in which he outlines his Stoic philosophy.
The application will be able to answer questions about the text, provide summaries, and generate insights based on the content of the book.

I will use the following technologies in this project:

LLM: OpenAI, OpenRouter, Local(Open AI), Gemini
Vector DB: Qdrant
Embedding: OpenAI, Local(Open AI), Gemini
Agent Framework: Langgraph

## LLM Providers

This project supports multiple LLM providers:

- **OpenAI**: Industry-standard models (GPT-4, GPT-4o, GPT-4o-mini)
- **OpenRouter**: Unified access to 100+ models from multiple providers (OpenAI, Anthropic, Google, Meta, etc.)
- **Gemini**: Google's latest models with competitive pricing
- **Local LLM**: Privacy-focused local models (Ollama, etc.)

All providers include:
- ✅ Automatic rate limiting
- ✅ Retry logic with exponential backoff
- ✅ Structured output support (Pydantic models)
- ✅ Comprehensive error handling
- ✅ Easy configuration via environment variables

See [docs/openrouter_provider.md](docs/openrouter_provider.md) for detailed OpenRouter documentation.
