"""
LLM Generator - Google Gemini Integration

Uses Google's Gemini API for answer generation.
FREE tier available at: https://aistudio.google.com/app/apikey
"""
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Retry configuration for rate limits
MAX_RETRIES = 5
INITIAL_BACKOFF = 3  # seconds

# Try new google.genai first, fall back to old google.generativeai
try:
    from google import genai
    from google.genai import types
    USE_NEW_API = True
    logger.info("Using new google.genai API")
except ImportError:
    import google.generativeai as genai_old
    USE_NEW_API = False
    logger.info("Using legacy google.generativeai API")


# Default prompt template for RAG
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant. Answer the question based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

Provide a helpful, accurate answer based on the context above. If the context is relevant, use it to answer. Be concise."""


class LLMGenerator:
    """
    LLM-based answer generator using Google Gemini.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_output_tokens: int = 1024
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        if USE_NEW_API:
            # New google.genai API
            self.client = genai.Client(api_key=api_key)
        else:
            # Legacy API
            genai_old.configure(api_key=api_key)
            self.model = genai_old.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
            )
        
        logger.info(f"Initialized Gemini model: {model_name}")
    
    def generate(
        self,
        prompt: str,
        context: str = "",
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response using the LLM with retry logic for rate limits."""
        # Build prompt
        if context:
            full_prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=prompt
            )
        else:
            full_prompt = prompt
        
        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                if USE_NEW_API:
                    # New API
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_output_tokens,
                        )
                    )
                    return response.text
                else:
                    # Legacy API
                    response = self.model.generate_content(full_prompt)
                    if response.parts:
                        return response.text
                    return "I couldn't generate a response."
                    
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    if attempt < MAX_RETRIES:
                        wait_time = INITIAL_BACKOFF * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {MAX_RETRIES} retries")
                        return "I'm currently experiencing high demand. Please wait a moment and try again."
                else:
                    # Non-rate-limit error, don't retry
                    logger.error(f"Error generating response: {e}")
                    return f"Error: {str(e)}"
        
        # If we get here, all retries failed
        logger.error(f"All retries failed: {last_error}")
        return "I'm currently experiencing high demand. Please wait a moment and try again."
    
    def generate_with_sources(
        self,
        question: str,
        context: str,
        sources: list
    ) -> dict:
        """Generate a response with source citations."""
        answer = self.generate(question, context=context)
        return {
            "answer": answer,
            "sources": sources,
            "model": self.model_name
        }
