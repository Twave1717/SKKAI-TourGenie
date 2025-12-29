"""Async LLM caller for alpha survey (speed optimization, not cost reduction).

This module provides async/parallel API calls for faster processing.
Note: Async does NOT reduce costs (same tokens), only improves speed.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import openai
from tqdm.asyncio import tqdm_asyncio

from .structured_output import AlphaSurveyOutput, get_alpha_survey_schema


async def call_llm_for_alpha_async(
    prompt: str,
    persona_id: str,
    model: str = "gpt-4.1",
    temperature: float = 0.1,
    max_retries: int = 3,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Optional[Dict[str, Any]]:
    """Async call to LLM API for alpha survey.

    Args:
        prompt: The alpha survey prompt
        persona_id: Reference ID of the persona
        model: LLM model to use
        temperature: Sampling temperature
        max_retries: Maximum retry attempts
        semaphore: Optional semaphore for rate limiting

    Returns:
        Parsed JSON response or None if failed
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.AsyncOpenAI(api_key=api_key)

    async def _make_request():
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a travel preference analyst. Analyze the persona's profile and provide detailed preference scores across all travel categories.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "alpha_survey",
                            "schema": get_alpha_survey_schema(),
                            "strict": True,
                        },
                    },
                )

                # Extract response text
                content = response.choices[0].message.content
                if not content:
                    continue

                # Parse JSON and validate with Pydantic
                result = json.loads(content)

                # Validate with Pydantic model
                validated = AlphaSurveyOutput(**result)

                # Ensure persona_id matches
                if validated.persona_id != persona_id:
                    result["persona_id"] = persona_id
                    validated = AlphaSurveyOutput(**result)

                return validated.model_dump()

            except json.JSONDecodeError as e:
                if attempt == max_retries - 1:
                    print(f"JSON parse error for {persona_id}: {e}")
                    return None
                await asyncio.sleep(1)
                continue

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"LLM API error for {persona_id}: {e}")
                    return None
                await asyncio.sleep(1)
                continue

        return None

    # Use semaphore for rate limiting if provided
    if semaphore:
        async with semaphore:
            return await _make_request()
    else:
        return await _make_request()


async def process_personas_async(
    prompt_data_list: List[Dict[str, Any]],
    model: str = "gpt-4.1",
    temperature: float = 0.1,
    max_retries: int = 3,
    max_concurrent: int = 10,
) -> List[Optional[Dict[str, Any]]]:
    """Process multiple personas concurrently with async API calls.

    Args:
        prompt_data_list: List of dicts with keys: persona_id, prompt
        model: LLM model to use
        temperature: Sampling temperature
        max_retries: Maximum retry attempts per call
        max_concurrent: Maximum concurrent API calls (for rate limiting)

    Returns:
        List of responses (same order as input)
    """
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks
    tasks = [
        call_llm_for_alpha_async(
            prompt=data["prompt"],
            persona_id=data["persona_id"],
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            semaphore=semaphore,
        )
        for data in prompt_data_list
    ]

    # Run with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc="Async alpha survey")

    return results


def run_async_pipeline(
    prompt_data_list: List[Dict[str, Any]],
    model: str = "gpt-4.1",
    temperature: float = 0.1,
    max_retries: int = 3,
    max_concurrent: int = 10,
) -> List[Optional[Dict[str, Any]]]:
    """Synchronous wrapper for async pipeline.

    Args:
        prompt_data_list: List of dicts with keys: persona_id, prompt
        model: LLM model to use
        temperature: Sampling temperature
        max_retries: Maximum retry attempts per call
        max_concurrent: Maximum concurrent API calls

    Returns:
        List of responses (same order as input)
    """
    return asyncio.run(
        process_personas_async(
            prompt_data_list,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            max_concurrent=max_concurrent,
        )
    )
