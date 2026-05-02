from unittest.mock import MagicMock

import pytest

from words import _call_gpt


def _make_response(status="completed", text="번역 결과", response_id="resp_1"):
    return MagicMock(status=status, output_text=text, id=response_id)


def _make_client(*responses):
    client = MagicMock()
    client.responses.create.side_effect = list(responses)
    return client


def test_returns_text_and_id_on_first_success():
    client = _make_client(_make_response(text="hello", response_id="resp_abc"))
    text, response_id = _call_gpt(client, "translate", "english", "prompt", None)
    assert text == "hello"
    assert response_id == "resp_abc"
    assert client.responses.create.call_count == 1


def test_retries_until_success():
    client = _make_client(
        _make_response(status="incomplete"),
        _make_response(status="completed", text="ok", response_id="resp_2"),
    )
    text, response_id = _call_gpt(client, "translate", "english", "prompt", None)
    assert text == "ok"
    assert response_id == "resp_2"
    assert client.responses.create.call_count == 2


def test_raises_after_three_failed_attempts():
    client = _make_client(
        _make_response(status="incomplete"),
        _make_response(status="incomplete"),
        _make_response(status="incomplete"),
    )
    with pytest.raises(RuntimeError, match="GPT call failed"):
        _call_gpt(client, "translate", "english", "prompt", None)
    assert client.responses.create.call_count == 3


def test_does_not_pass_previous_response_id_when_none():
    client = _make_client(_make_response())
    _call_gpt(client, "translate", "english", "prompt", None)
    kwargs = client.responses.create.call_args.kwargs
    assert "previous_response_id" not in kwargs


def test_passes_previous_response_id_when_provided():
    client = _make_client(_make_response())
    _call_gpt(client, "translate", "english", "prompt", "resp_prev")
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs.get("previous_response_id") == "resp_prev"


def test_passes_prompt_and_system_instructions():
    client = _make_client(_make_response())
    _call_gpt(client, "translate", "english", "사용자_프롬프트", None)
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["input"] == "사용자_프롬프트"
    assert "영어" in kwargs["instructions"]
