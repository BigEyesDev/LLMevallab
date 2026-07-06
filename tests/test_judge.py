import pytest

from src.evaluations.judge import _parse_judge_json, normalize_score


def test_normalize_score_maps_1_to_5_range():
    assert normalize_score(1) == 0.0
    assert normalize_score(3) == 0.5
    assert normalize_score(5) == 1.0


def test_parse_judge_json_valid():
    raw = '{"faithfulness": 4, "completeness": 3, "coherence": 5, "reasoning": "ok"}'
    result = _parse_judge_json(raw)
    assert result["faithfulness"] == 4
    assert result["completeness"] == 3
    assert result["coherence"] == 5
    assert result["reasoning"] == "ok"


def test_parse_judge_json_strips_markdown_fence():
    raw = '```json\n{"faithfulness": 2, "completeness": 2, "coherence": 2}\n```'
    result = _parse_judge_json(raw)
    assert result["faithfulness"] == 2


def test_parse_judge_json_clamps_out_of_range():
    raw = '{"faithfulness": 10, "completeness": 0, "coherence": 3}'
    result = _parse_judge_json(raw)
    assert result["faithfulness"] == 5
    assert result["completeness"] == 1


def test_parse_judge_json_invalid_returns_defaults():
    result = _parse_judge_json("not json at all")
    assert result["faithfulness"] == 1
    assert result["reasoning"] == "parse_error"
