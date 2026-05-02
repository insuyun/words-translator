import pytest

from words import _parse_raw


def test_parses_title_scripture_and_body():
    text = "제목: 다윗이 성전을 준비하듯\n본문: 대상 22:1-12\n---\n첫 문단입니다.\n\n둘째 문단입니다.\n"
    title, scripture, body = _parse_raw(text)
    assert title == "다윗이 성전을 준비하듯"
    assert scripture == "대상 22:1-12"
    assert body == "첫 문단입니다.\n\n둘째 문단입니다."


def test_strips_whitespace_around_title_and_scripture():
    text = "제목:    제목입니다   \n본문:   요 3:16   \n---\n본문 시작\n"
    title, scripture, _ = _parse_raw(text)
    assert title == "제목입니다"
    assert scripture == "요 3:16"


def test_separator_with_many_dashes():
    text = "제목: t\n본문: s\n" + ("-" * 50) + "\n본문 내용\n"
    _, _, body = _parse_raw(text)
    assert body == "본문 내용"


def test_separator_with_four_dashes():
    text = "제목: t\n본문: s\n----\n본문 내용\n"
    _, _, body = _parse_raw(text)
    assert body == "본문 내용"


def test_missing_title_raises():
    text = "본문: 대상 22:1-12\n---\n본문\n"
    with pytest.raises(ValueError, match="제목"):
        _parse_raw(text)


def test_missing_scripture_raises():
    text = "제목: t\n---\n본문\n"
    with pytest.raises(ValueError, match="본문"):
        _parse_raw(text)


def test_missing_separator_raises():
    text = "제목: t\n본문: s\n본문 내용\n"
    with pytest.raises(ValueError, match="--- 구분선"):
        _parse_raw(text)


def test_empty_body_after_separator_raises():
    text = "제목: t\n본문: s\n---\n   \n"
    with pytest.raises(ValueError, match="설교 본문"):
        _parse_raw(text)


def test_empty_title_value_raises():
    text = "제목:   \n본문: s\n---\n내용\n"
    with pytest.raises(ValueError, match="제목"):
        _parse_raw(text)


def test_empty_scripture_value_raises():
    text = "제목: t\n본문:   \n---\n내용\n"
    with pytest.raises(ValueError, match="본문"):
        _parse_raw(text)


def test_multiple_missing_fields_reported_together():
    text = "본문 내용만 있음\n"
    with pytest.raises(ValueError) as exc_info:
        _parse_raw(text)
    msg = str(exc_info.value)
    assert "제목" in msg
    assert "본문" in msg
    assert "--- 구분선" in msg
