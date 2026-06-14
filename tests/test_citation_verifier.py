from __future__ import annotations

from src.generation.citation_verifier import verify_citations


def test_verified_citation():
    response = "BTC IV rose sharply. [Source: Weekly Vol Report | 2024-11-04]"
    citations = ["Weekly Vol Report (2024-11-04)"]
    result = verify_citations(response, citations)

    assert result["path"] == "text"
    assert len(result["verified"]) == 1
    assert result["hallucinated"] == []


def test_hallucinated_citation():
    response = "Some claim. [Source: Unknown Doc | 2024-01-01]"
    citations = ["Weekly Vol Report (2024-11-04)"]
    result = verify_citations(response, citations)

    assert result["path"] == "text"
    assert result["verified"] == []
    assert len(result["hallucinated"]) == 1


def test_visual_path_short_circuits():
    result = verify_citations("any response text", included_citations=["visual"])

    assert result["path"] == "visual"
    assert result["verified"] is True
    assert result["hallucinated"] == []
