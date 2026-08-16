from __future__ import annotations

import pytest

from hybridrag.eval.retrieval_metrics import precision_at_k, recall_at_k, reciprocal_rank


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"x", "y"}, k=3) == 0.0

    def test_partial_match(self):
        assert precision_at_k(["a", "b", "c"], {"a", "x"}, k=3) == pytest.approx(1 / 3)

    def test_divides_by_k_not_by_retrieved_count(self):
        # Only 2 retrieved but k=5 - a real IR failure (retrieved too few),
        # and precision should reflect that rather than dividing by 2.
        assert precision_at_k(["a", "b"], {"a", "b"}, k=5) == pytest.approx(2 / 5)

    def test_rejects_non_positive_k(self):
        with pytest.raises(ValueError):
            precision_at_k(["a"], {"a"}, k=0)


class TestRecallAtK:
    def test_finds_all_relevant_within_k(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_finds_none(self):
        assert recall_at_k(["a", "b", "c"], {"x"}, k=3) == 0.0

    def test_truncates_to_k(self):
        # relevant "b" is at rank 3, outside k=2
        assert recall_at_k(["a", "c", "b"], {"a", "b"}, k=2) == pytest.approx(0.5)

    def test_rejects_empty_relevant_set(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], set(), k=1)


class TestReciprocalRank:
    def test_first_result_is_relevant(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_relevant_result_at_rank_3(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant_result_found(self):
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_only_first_relevant_hit_counts(self):
        # "a" at rank 1 and "b" at rank 2 are both relevant - RR should use
        # rank 1 only, not average or otherwise combine both hits.
        assert reciprocal_rank(["a", "b"], {"a", "b"}) == 1.0
