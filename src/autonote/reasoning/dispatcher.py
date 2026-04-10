"""LLM Dispatcher — routes reasoning tasks through autonote's unified LLM interface.

Replaces the meeting-copilot dispatcher that hardcoded Ollama + Claude API.
All LLM calls go through ``autonote.llm.query_llm()`` which supports any
provider via litellm (Ollama, OpenAI, Anthropic, DeepSeek, etc.) and
respects the user's preset configuration (fast, smart, cheap, local).

Since ``query_llm`` is synchronous and the reasoning engine is fully async,
every call is wrapped in ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from autonote.llm import query_llm
from autonote.reasoning.prompts import PROMPT_MAP

logger = logging.getLogger(__name__)


class LLMDispatcher:
    """Routes reasoning tasks to the user's configured LLM provider.

    Args:
        model: Model string or preset name (e.g. 'fast', 'smart', 'local').
               Defaults to the user's configured MODEL from .autonoterc.
    """

    def __init__(self, model: Optional[str] = None, source_file: Optional[str] = None) -> None:
        self.model = model
        self.source_file = source_file

    async def run(
        self,
        task_name: str,
        *,
        model_override: Optional[str] = None,
        **kwargs: str,
    ) -> str:
        """Run a reasoning task by formatting a prompt and calling the LLM.

        Args:
            task_name: Key in PROMPT_MAP (e.g. "summary", "action_items").
            model_override: If set, use this model instead of the dispatcher's default.
            **kwargs: Variables to format into the prompt template.

        Returns:
            The LLM-generated text response.

        Raises:
            KeyError: If task_name is not in PROMPT_MAP.
            RuntimeError: If the LLM call fails.
        """
        prompt_template = PROMPT_MAP[task_name]
        prompt = prompt_template.format(**kwargs)

        effective_model = model_override or self.model

        logger.debug(
            "Dispatching task '%s' (prompt length: %d chars)",
            task_name, len(prompt),
            extra={"structured": {
                "event": "llm_request",
                "task": task_name,
                "model": effective_model,
                "prompt": prompt,
            }},
        )

        # query_llm is synchronous — run in thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            query_llm,
            prompt=prompt,
            model=effective_model,
            stage=f"realtime_{task_name}",
            source_file=self.source_file,
        )

        logger.debug(
            "Task '%s' completed (%d chars)",
            task_name, len(result),
            extra={"structured": {
                "event": "llm_response",
                "task": task_name,
                "response": result,
                "response_len": len(result),
            }},
        )

        return result
