from unittest.mock import MagicMock

import pytest

from words import GPT_TOKEN_LIMIT, _process_body


def _client_returning(*texts_and_ids):
    client = MagicMock()
    client.responses.create.side_effect = [
        MagicMock(status="completed", output_text=text, id=resp_id)
        for text, resp_id in texts_and_ids
    ]
    return client


def test_skips_empty_paragraphs():
    body = "문단1\n\n   \n\n문단2"
    client = _client_returning(("결과1", "resp_1"), ("결과2", "resp_2"))
    result = _process_body(client, "translate", "english", body)
    assert result == "결과1\n\n결과2"
    assert client.responses.create.call_count == 2


def test_threads_previous_response_id_between_calls():
    body = "문단1\n\n문단2\n\n문단3"
    client = _client_returning(
        ("r1", "resp_1"),
        ("r2", "resp_2"),
        ("r3", "resp_3"),
    )
    _process_body(client, "translate", "english", body)
    calls = client.responses.create.call_args_list
    assert "previous_response_id" not in calls[0].kwargs
    assert calls[1].kwargs["previous_response_id"] == "resp_1"
    assert calls[2].kwargs["previous_response_id"] == "resp_2"


def test_joins_results_with_double_newline():
    body = "p1\n\np2"
    client = _client_returning(("a", "r1"), ("b", "r2"))
    assert _process_body(client, "translate", "english", body) == "a\n\nb"


def test_token_limit_check_runs_before_gpt_call():
    huge = "단어 " * (GPT_TOKEN_LIMIT * 2)
    body = f"{huge}\n\n작은문단"
    client = MagicMock()
    with pytest.raises(ValueError, match="exceed"):
        _process_body(client, "translate", "english", body)
    client.responses.create.assert_not_called()


def test_proofread_mode_splits_on_dashes():
    body = "문단1\n-----\n문단2"
    client = _client_returning(("r1", "resp_1"), ("r2", "resp_2"))
    result = _process_body(client, "proofread", "english", body)
    assert result == "r1\n\nr2"
    assert client.responses.create.call_count == 2
