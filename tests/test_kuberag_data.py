import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "kuberag"


def load_corpus():
    # Split on newlines only: splitlines() also breaks on Unicode separators inside chunk text.
    return [json.loads(line) for line in (DATA / "corpus.jsonl").read_text().split("\n") if line.strip()]


def load_questions():
    return json.loads((DATA / "questions.json").read_text())["questions"]


def test_the_corpus_has_the_expected_chunks_from_kuberag():
    corpus = load_corpus()
    assert len(corpus) == 276
    assert all(c["text"].strip() for c in corpus)


def test_there_are_65_answerable_questions_with_gold_sources():
    questions = load_questions()
    assert len(questions) == 65
    assert all(q["gold_sources"] for q in questions)


def test_every_gold_source_exists_in_the_corpus():
    sources = {c["source"] for c in load_corpus()}
    for question in load_questions():
        assert set(question["gold_sources"]) <= sources, question["id"]
