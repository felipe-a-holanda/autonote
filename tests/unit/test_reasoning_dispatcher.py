"""Tests for LLMDispatcher."""

import pytest
from unittest.mock import patch, MagicMock

from autonote.reasoning.dispatcher import LLMDispatcher


class TestLLMDispatcher:

    async def test_raises_on_unknown_task(self):
        dispatcher = LLMDispatcher(model="ollama/llama3")
        with pytest.raises(KeyError):
            await dispatcher.run("nonexistent_task")

    @patch("autonote.reasoning.dispatcher.asyncio.to_thread")
    async def test_run_calls_query_llm_with_formatted_prompt(self, mock_to_thread):
        async def fake_to_thread(fn, **kwargs):
            return fn(**kwargs)

        mock_to_thread.side_effect = fake_to_thread

        from autonote.reasoning.prompts import PROMPT_MAP
        # Use a real task name from the prompt map
        task_name = next(iter(PROMPT_MAP))
        template = PROMPT_MAP[task_name]
        # Build kwargs to satisfy the template
        import string
        formatter = string.Formatter()
        required_keys = {fname for _, fname, _, _ in formatter.parse(template) if fname}
        kwargs = {k: "test_value" for k in required_keys}

        with patch("autonote.reasoning.dispatcher.query_llm", return_value="result") as mock_llm:
            dispatcher = LLMDispatcher(model="ollama/test")
            result = await dispatcher.run(task_name, **kwargs)
            assert mock_llm.called
            call_kwargs = mock_llm.call_args[1]
            assert call_kwargs["model"] == "ollama/test"
            assert f"realtime_{task_name}" == call_kwargs["stage"]

    @patch("autonote.reasoning.dispatcher.asyncio.to_thread")
    async def test_run_uses_none_model_by_default(self, mock_to_thread):
        async def fake_to_thread(fn, **kwargs):
            return fn(**kwargs)
        mock_to_thread.side_effect = fake_to_thread

        from autonote.reasoning.prompts import PROMPT_MAP
        task_name = next(iter(PROMPT_MAP))
        template = PROMPT_MAP[task_name]
        import string
        required_keys = {fname for _, fname, _, _ in string.Formatter().parse(template) if fname}
        kwargs = {k: "x" for k in required_keys}

        with patch("autonote.reasoning.dispatcher.query_llm", return_value="ok") as mock_llm:
            dispatcher = LLMDispatcher()  # no model
            await dispatcher.run(task_name, **kwargs)
            assert mock_llm.call_args[1]["model"] is None
