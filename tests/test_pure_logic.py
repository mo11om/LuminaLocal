import csv

import pytest

from cv_matcher.agent import (
    decompose_requirements,
    filter_by_score,
    job_key,
    load_completed_jobs,
    verify_classification,
)
from cv_matcher.utils import chunk_id


class FakeDoc:
    def __init__(self, page_content):
        self.page_content = page_content

    def __eq__(self, other):
        return isinstance(other, FakeDoc) and self.page_content == other.page_content


def make_analysis(matched, missing, classification):
    return {
        "match_classification": classification,
        "matched_critical_skills": ["skill"] * matched,
        "missing_critical_skills": ["gap"] * missing,
    }


@pytest.mark.parametrize("matched,missing,expected", [
    (0, 6, "No Match"),      # 0%
    (2, 3, "Good Match"),    # 40% - lower boundary of Good
    (7, 3, "Strong Match"),  # 70% - lower boundary of Strong
    (6, 4, "Good Match"),    # 60%
    (10, 0, "Strong Match"), # 100%
])
def test_verify_classification_boundaries(matched, missing, expected):
    result = verify_classification(make_analysis(matched, missing, "Strong Match"))
    assert result["match_classification"] == expected


def test_verify_classification_just_below_good_threshold():
    # 39% must not round up into "Good Match"
    result = verify_classification(make_analysis(39, 61, "Good Match"))
    assert result["match_classification"] == "No Match"


def test_verify_classification_just_below_strong_threshold():
    result = verify_classification(make_analysis(69, 31, "Strong Match"))
    assert result["match_classification"] == "Good Match"


def test_verify_classification_leaves_agreeing_label_alone():
    result = verify_classification(make_analysis(8, 2, "Strong Match"))
    assert result["match_classification"] == "Strong Match"


def test_verify_classification_no_skills_does_not_divide_by_zero():
    data = {"match_classification": "Good Match",
            "matched_critical_skills": [], "missing_critical_skills": []}
    assert verify_classification(data)["match_classification"] == "Good Match"


def test_verify_classification_handles_none_lists():
    data = {"match_classification": "Good Match",
            "matched_critical_skills": None, "missing_critical_skills": None}
    assert verify_classification(data)["match_classification"] == "Good Match"


def test_chunk_id_is_stable_for_identical_content():
    assert chunk_id("Python, Docker, AWS") == chunk_id("Python, Docker, AWS")


def test_chunk_id_differs_for_different_content():
    assert chunk_id("Python") != chunk_id("Docker")


def test_chunk_id_handles_unicode():
    assert chunk_id("軟體工程師") == chunk_id("軟體工程師")


def test_decompose_requirements_splits_on_commas_newlines_and_conjunction():
    parts = decompose_requirements("Python, Docker\nKubernetes and AWS")
    assert parts == ["Python", "Docker", "Kubernetes", "AWS"]


def test_decompose_requirements_strips_whitespace_and_empties():
    assert decompose_requirements("Python,  , Docker,") == ["Python", "Docker"]


def test_decompose_requirements_empty_input():
    assert decompose_requirements("") == []
    assert decompose_requirements(None) == []


def test_decompose_requirements_single_skill():
    assert decompose_requirements("Python") == ["Python"]


def test_filter_by_score_keeps_only_at_or_above_threshold():
    good, weak = FakeDoc("relevant"), FakeDoc("noise")
    assert filter_by_score([(good, 0.8), (weak, 0.05)], 0.2) == [good]


def test_filter_by_score_is_inclusive_at_threshold():
    doc = FakeDoc("exactly at threshold")
    assert filter_by_score([(doc, 0.2)], 0.2) == [doc]


def test_filter_by_score_all_below_returns_empty():
    assert filter_by_score([(FakeDoc("a"), 0.01), (FakeDoc("b"), 0.1)], 0.5) == []


def test_job_key_uses_title_and_department():
    assert job_key({"job_title": "SWE", "department": "Platform"}) == ("SWE", "Platform")


def test_job_key_distinguishes_same_title_different_department():
    a = job_key({"job_title": "Engineer", "department": "Data"})
    b = job_key({"job_title": "Engineer", "department": "Infra"})
    assert a != b


def test_load_completed_jobs_missing_file_returns_empty(tmp_path):
    assert load_completed_jobs(str(tmp_path / "nope.csv")) == set()


def test_load_completed_jobs_reads_existing_rows(tmp_path):
    csv_path = tmp_path / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Job Title", "Department", "Classification"])
        writer.writeheader()
        writer.writerow({"Job Title": "SWE", "Department": "Platform", "Classification": "Strong Match"})
        writer.writerow({"Job Title": "SWE", "Department": "Data", "Classification": "No Match"})

    assert load_completed_jobs(str(csv_path)) == {("SWE", "Platform"), ("SWE", "Data")}


def test_load_completed_jobs_header_only_returns_empty(tmp_path):
    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["Job Title", "Department"]).writeheader()

    assert load_completed_jobs(str(csv_path)) == set()


def test_checkpoint_round_trip_skips_completed_jobs(tmp_path):
    jobs = [
        {"job_title": "SWE", "department": "Platform"},
        {"job_title": "SWE", "department": "Data"},
        {"job_title": "PM", "department": "Product"},
    ]
    csv_path = tmp_path / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Job Title", "Department"])
        writer.writeheader()
        writer.writerow({"Job Title": "SWE", "Department": "Platform"})

    done = load_completed_jobs(str(csv_path))
    pending = [j for j in jobs if job_key(j) not in done]

    assert [j["job_title"] for j in pending] == ["SWE", "PM"]
    assert pending[0]["department"] == "Data"
