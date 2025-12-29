"""LLM caller for alpha survey."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import openai

from .structured_output import AlphaSurveyOutput, get_alpha_survey_schema


def call_llm_for_alpha(
    prompt: str,
    persona_id: str,
    model: str = "gpt-4.1",
    temperature: float = 0.1,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Call LLM API to get alpha survey response using OpenAI Structured Outputs.

    Args:
        prompt: The alpha survey prompt
        persona_id: Reference ID of the persona (e.g., stravl_1234)
        model: LLM model to use (default: gpt-4.1)
        temperature: Sampling temperature (default: 0.1 for balanced consistency)
        max_retries: Maximum number of retry attempts

    Returns:
        Parsed JSON response or None if failed
    """
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
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
            print(f"JSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            continue

        except Exception as e:
            print(f"LLM API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            continue

    return None


def call_llm_for_alpha_anthropic(
    prompt: str,
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.3,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Call Anthropic Claude API to get alpha survey response.

    Args:
        prompt: The alpha survey prompt
        model: Claude model to use
        temperature: Sampling temperature (default: 0.3 for consistency)
        max_retries: Maximum number of retry attempts

    Returns:
        Parsed JSON response or None if failed
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract response text
            content = response.content[0].text
            if not content:
                continue

            # Claude often wraps JSON in markdown code blocks, so clean it
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # Parse JSON
            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            print(f"JSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            continue

        except Exception as e:
            print(f"Anthropic API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            continue

    return None
