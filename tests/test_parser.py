from pathlib import Path

import pytest

from citelens.parser import build_paper, parse_pdf
from citelens.pdf import TextBlock

BODY = 10.0


def body(text: str, page: int = 1) -> TextBlock:
    return TextBlock(page=page, text=text, size=BODY, bold=False)


def heading(text: str, page: int = 1, size: float = 12.0) -> TextBlock:
    return TextBlock(page=page, text=text, size=size, bold=True)


TITLE = TextBlock(page=1, text="A Paper About Things", size=17.0, bold=True)


def test_title_is_largest_text_on_first_page() -> None:
    paper = build_paper([TITLE, body("Some text.")], fallback_title="x")
    assert paper.title == "A Paper About Things"
    assert [chunk.text for chunk in paper.chunks] == ["Some text."]


def test_numbered_headings_give_levels_and_section_paths() -> None:
    paper = build_paper(
        [
            TITLE,
            heading("3 Method"),
            body("Intro to method."),
            heading("3.1 Details", size=BODY),
            body("Detail."),
        ],
        fallback_title="x",
    )
    assert [(section.title, section.level) for section in paper.sections] == [
        ("3 Method", 1),
        ("3.1 Details", 2),
    ]
    assert [chunk.section_path for chunk in paper.chunks] == [
        ["3 Method"],
        ["3 Method", "3.1 Details"],
    ]


def test_small_bold_table_numbers_are_not_headings() -> None:
    paper = build_paper(
        [TITLE, TextBlock(page=1, text="4 Results", size=8.0, bold=True)],
        fallback_title="x",
    )
    assert paper.sections == []


def test_mid_sentence_split_is_merged_and_dehyphenated() -> None:
    paper = build_paper(
        [TITLE, body("We train a represen-"), body("tation model.")], fallback_title="x"
    )
    assert [chunk.text for chunk in paper.chunks] == [
        "We train a representation model."
    ]


def test_chunk_spans_point_into_full_text() -> None:
    paper = build_paper(
        [TITLE, body("First."), body("Second."), body("Third.")], fallback_title="x"
    )
    for chunk in paper.chunks:
        assert paper.full_text[chunk.char_start : chunk.char_end] == chunk.text


def test_numbered_references_split_even_mid_block() -> None:
    paper = build_paper(
        [
            TITLE,
            heading("References"),
            body("[1] A. Author. One. 2019. [2] B. Au-"),
            body("thor. Two. 2020."),
        ],
        fallback_title="x",
    )
    assert [reference.raw for reference in paper.references] == [
        "[1] A. Author. One. 2019.",
        "[2] B. Author. Two. 2020.",
    ]
    assert [reference.year for reference in paper.references] == [2019, 2020]


def test_author_year_references_regroup_split_entries() -> None:
    paper = build_paper(
        [
            TITLE,
            heading("References"),
            body("Alan Akbik and Roland Vollgraf."),
            body("2018. Contextual embeddings. arXiv:1808.04444."),
            body("Rie Ando. 2005. A framework."),
        ],
        fallback_title="x",
    )
    assert len(paper.references) == 2
    assert paper.references[0].arxiv_id == "1808.04444"


def test_references_end_at_the_next_heading() -> None:
    paper = build_paper(
        [
            TITLE,
            heading("References"),
            body("[1] A. One. 2019."),
            heading("A Appendix Stuff"),
            body("Extra."),
        ],
        fallback_title="x",
    )
    assert len(paper.references) == 1
    assert paper.chunks[-1].section_path == ["A Appendix Stuff"]


BERT = Path(__file__).parent.parent / "papers" / "1810.04805.pdf"


@pytest.mark.skipif(not BERT.exists(), reason="sample paper not downloaded")
def test_real_paper_bert() -> None:
    paper = parse_pdf(BERT)
    assert paper.title.startswith("BERT: Pre-training")
    assert paper.arxiv_id == "1810.04805"
    assert "3.1 Pre-training BERT" in [section.title for section in paper.sections]
    assert len(paper.references) > 50
