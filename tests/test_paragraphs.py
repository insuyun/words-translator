import pytest

from words import (
    GPT_TOKEN_LIMIT,
    _check_token_limit,
    _format_with_header,
    _parse_raw,
    _to_paragraphs,
)


def test_to_paragraphs_proofread_splits_on_five_or_more_dashes():
    text = "문단1\n-----\n문단2\n----------\n문단3"
    result = _to_paragraphs("proofread", text)
    assert len(result) == 3
    assert "문단1" in result[0]
    assert "문단2" in result[1]
    assert "문단3" in result[2]


def test_to_paragraphs_proofread_does_not_split_on_four_dashes():
    text = "문단1\n----\n문단2"
    result = _to_paragraphs("proofread", text)
    assert len(result) == 1


def test_to_paragraphs_translate_splits_on_blank_line():
    text = "문단1\n\n문단2\n\n문단3"
    result = _to_paragraphs("translate", text)
    assert result == ["문단1", "문단2", "문단3"]


def test_to_paragraphs_translate_single_newline_does_not_split():
    text = "문단1\n문단2"
    result = _to_paragraphs("translate", text)
    assert result == ["문단1\n문단2"]


def test_format_with_header_round_trip():
    text = "제목: 다윗\n본문: 대상 22:1-12\n---\n사람들이 모였다.\n"
    title, scripture, body = _parse_raw(text)
    formatted = _format_with_header(title, scripture, body)
    title2, scripture2, body2 = _parse_raw(formatted)
    assert (title, scripture, body) == (title2, scripture2, body2)


def test_format_with_header_shape():
    formatted = _format_with_header("t", "s", "본문")
    assert formatted == "제목: t\n본문: s\n---\n본문\n"


def test_check_token_limit_passes_below_limit():
    _check_token_limit(["짧은 문장입니다."])


def test_check_token_limit_raises_above_limit():
    huge = "단어 " * (GPT_TOKEN_LIMIT * 2)
    with pytest.raises(ValueError, match="exceed"):
        _check_token_limit([huge])


def test_check_token_limit_includes_paragraph_in_error():
    huge = "고유표식어 " + ("단어 " * (GPT_TOKEN_LIMIT * 2))
    with pytest.raises(ValueError) as exc_info:
        _check_token_limit([huge])
    assert "고유표식어" in str(exc_info.value.args[1])
