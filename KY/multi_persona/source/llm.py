"""
LLM 클라이언트 (OpenAI + Pydantic 구조화) + 스텁.
- chat.completions + response_format=json_object로 받아 schema.model_validate_json으로 파싱.
"""

import json
import os
import sys
import time
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, RateLimitError, OpenAI
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def generate_structured(self, prompt: str, schema: Type[T], **options) -> T:
        raise NotImplementedError("LLM backend를 구현하세요.")


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str = "gpt-4.1",
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, base_url=base_url)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다 (.env 확인).")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = os.getenv("LLM_DEBUG") == "1"

    def generate_structured(self, prompt: str, schema: Type[T], retries: int = 3, backoff: float = 1.5, **options) -> T:
        delay = 1.0
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                if self.debug:
                    sys.stderr.write(f"[LLM][req] model={self.model} attempt={attempt+1} prompt_preview={prompt[:200]!r}\n")
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=options.get("temperature", self.temperature),
                    max_tokens=options.get("max_tokens", self.max_tokens),
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or ""
                if self.debug:
                    usage = getattr(resp, "usage", None)
                    sys.stderr.write(f"[LLM][resp] usage={usage} content_preview={content[:200]!r}\n")
                try:
                    parsed = schema.model_validate_json(content)
                except Exception as parse_err:
                    if self.debug:
                        sys.stderr.write(f"[LLM][parse-error] err={parse_err}\n")
                    # best-effort 복구: 마지막 중괄호까지 자르거나 JSON substring 추출
                    recovered = self._recover_json(content)
                    parsed = schema.model_validate_json(recovered)
                return parsed
            except (RateLimitError, APIError, APIConnectionError) as e:
                last_err = e
                if self.debug:
                    sys.stderr.write(f"[LLM][warn] attempt={attempt+1} err={e}\n")
                if attempt == retries - 1:
                    break
                time.sleep(delay)
                delay *= backoff
            except Exception as e:
                last_err = e
                if self.debug:
                    sys.stderr.write(f"[LLM][error] attempt={attempt+1} err={e}\n")
                break
        raise RuntimeError(f"LLM 호출 실패: {last_err}")

    @staticmethod
    def _recover_json(content: str) -> str:
        """
        부분적으로 잘린 JSON을 복구 시도:
        - 마지막 닫는 중괄호 위치까지 자르기
        - 앞쪽에 첫 여는 중괄호 이후만 취하기
        """
        if not content:
            raise RuntimeError("LLM 응답이 비어 있어 JSON 복구 불가")
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("LLM 응답에 JSON 중괄호가 없어 복구 불가")
        return content[start : end + 1]


class StubClient(LLMClient):
    def __init__(self):
        super().__init__(model="stub")
        self.debug = os.getenv("LLM_DEBUG") == "1"

    def generate_structured(self, prompt: str, schema: Type[T], **options) -> T:
        if self.debug:
            sys.stderr.write(f"[LLM][stub] prompt_preview={prompt[:200]!r}\n")
        raise RuntimeError("Stub LLM: 실제 LLM을 사용하려면 --use-openai로 실행하세요.")


def stub_client() -> LLMClient:
    return StubClient()
