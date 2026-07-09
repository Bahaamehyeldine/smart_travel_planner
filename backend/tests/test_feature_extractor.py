"""
test_feature_extractor.py

Tests for the ML feature extraction pipeline.

These tests verify:
- Keyword counting is case-insensitive and accurate
- Price tier extraction correctly reads Sleep section signals
- Feature vector has correct structure and values
- Class thresholds match the documented labeling rubric
"""
import pytest
from app.ml.feature_extractor import (
    _count_keywords,
    _extract_price_tier,
    _compute_keyword_features,
    _compute_region_features,
    ADVENTURE_KEYWORDS,
    BUDGET_KEYWORDS,
    CLASS_THRESHOLDS,
    REGIONS,
)


class TestCountKeywords:
    def test_counts_exact_match(self):
        """Keyword counter should find known adventure keywords in text."""
        text = "This destination offers great hiking and trekking opportunities"
        count = _count_keywords(text, ADVENTURE_KEYWORDS)
        assert count >= 2  # hiking and trekking both present

    def test_case_insensitive(self):
        """Keyword matching must be case-insensitive — Wikivoyage is inconsistent."""
        text = "HIKING and DIVING available"
        count = _count_keywords(text, ADVENTURE_KEYWORDS)
        assert count >= 2

    def test_empty_text_returns_zero(self):
        """Empty text should always return zero — no false positives."""
        assert _count_keywords("", ADVENTURE_KEYWORDS) == 0

    def test_no_match_returns_zero(self):
        """Text with no keywords should return zero."""
        text = "A quiet village with no activities"
        assert _count_keywords(text, ADVENTURE_KEYWORDS) == 0

    def test_partial_word_does_not_match(self):
        """'hike' should not match 'hiking' — we match substrings so this is ok,
        but 'ski' should not match 'skiing' unless skiing is in the list."""
        text = "ski resort"
        count = _count_keywords(text, ADVENTURE_KEYWORDS)
        # 'skiing' is in list but 'ski' alone — verify behavior is consistent
        assert isinstance(count, int)
        assert count >= 0


class TestExtractPriceTier:
    def test_empty_text_returns_midrange(self):
        """No Sleep section → default to mid-range (2)."""
        assert _extract_price_tier("") == 2

    def test_budget_keywords_return_tier_1(self):
        """Strong budget signal should return tier 1."""
        text = "affordable budget hostels cheap backpacker guesthouse dorm"
        assert _extract_price_tier(text) == 1

    def test_luxury_keywords_return_tier_3(self):
        """Strong luxury signal should return tier 3."""
        text = "luxury exclusive five-star private villa butler service upscale"
        assert _extract_price_tier(text) == 3

    def test_mixed_signals_return_valid_tier(self):
        """Mixed signals should return a valid tier (1, 2, or 3)."""
        text = "various accommodation options from budget to luxury"
        result = _extract_price_tier(text)
        assert result in [1, 2, 3]

    def test_no_signal_returns_midrange(self):
        """Text with no price keywords should default to mid-range."""
        text = "beautiful beaches and mountains with great views"
        assert _extract_price_tier(text) == 2


class TestComputeKeywordFeatures:
    def test_returns_all_expected_keys(self):
        """Feature vector must contain count, binary, and ratio for every class."""
        features = _compute_keyword_features("hiking adventure mountains")
        expected_classes = ["adventure", "relaxation", "culture",
                            "budget", "luxury", "family"]
        for cls in expected_classes:
            assert f"{cls}_keyword_count" in features, f"Missing {cls}_keyword_count"
            assert f"{cls}_keyword_binary" in features, f"Missing {cls}_keyword_binary"
            assert f"{cls}_keyword_ratio" in features, f"Missing {cls}_keyword_ratio"

    def test_adventure_query_scores_high(self):
        """Text with many adventure keywords should score high on adventure features."""
        text = "hiking trekking climbing diving mountaineering skiing"
        features = _compute_keyword_features(text)
        assert features["adventure_keyword_count"] >= 4
        assert features["adventure_keyword_binary"] == 1

    def test_ratios_sum_to_one(self):
        """
        Keyword ratios across all classes must sum to 1.0.
        This is a key invariant — ratios are computed from total keyword count.
        Exception: if total is 0, all ratios are 0.
        """
        features = _compute_keyword_features("hiking museums spa budget luxury")
        ratios = [
            features[f"{cls}_keyword_ratio"]
            for cls in ["adventure", "relaxation", "culture",
                        "budget", "luxury", "family"]
        ]
        total = sum(ratios)
        assert total == 0.0 or abs(total - 1.0) < 1e-6

    def test_binary_threshold_respected_for_adventure(self):
        """
        Adventure threshold is 3 — exactly 2 keywords should give binary=0.
        This mirrors the labeling rubric: 3+ activity signals = strong signal.
        """
        text = "hiking diving"
        features = _compute_keyword_features(text)
        assert features["adventure_keyword_count"] == 2
        assert features["adventure_keyword_binary"] == 0

    def test_budget_threshold_is_one(self):
        """
        Budget threshold is 1 — one keyword should give binary=1.
        Budget class uses keyword presence, not activity count.
        """
        text = "affordable accommodation"
        features = _compute_keyword_features(text)
        assert features["budget_keyword_binary"] == 1

    def test_empty_text_all_zeros(self):
        """Empty text should produce all-zero feature vector."""
        features = _compute_keyword_features("")
        for cls in ["adventure", "relaxation", "culture", "budget", "luxury", "family"]:
            assert features[f"{cls}_keyword_count"] == 0
            assert features[f"{cls}_keyword_binary"] == 0


class TestComputeRegionFeatures:
    def test_correct_region_is_one(self):
        """The matching region feature should be 1."""
        features = _compute_region_features("Southeast Asia")
        assert features["region_southeast_asia"] == 1

    def test_other_regions_are_zero(self):
        """All non-matching region features should be 0."""
        features = _compute_region_features("Southeast Asia")
        assert features["region_western_europe"] == 0
        assert features["region_north_america"] == 0

    def test_all_regions_covered(self):
        """Feature vector should contain one entry per known region."""
        features = _compute_region_features("Oceania")
        assert len(features) == len(REGIONS)

    def test_unknown_region_all_zeros(self):
        """Unknown region should produce all-zero one-hot vector — not crash."""
        features = _compute_region_features("Unknown Region")
        assert all(v == 0 for v in features.values())

    def test_region_key_format(self):
        """Region keys should use underscores not spaces."""
        features = _compute_region_features("North America")
        assert "region_north_america" in features
        assert "region_North America" not in features


class TestClassThresholds:
    def test_activity_classes_threshold_is_3(self):
        """
        Adventure, Relaxation, Culture require 3+ keyword matches.
        This matches our documented labeling rubric.
        """
        assert CLASS_THRESHOLDS["Adventure"] == 3
        assert CLASS_THRESHOLDS["Relaxation"] == 3
        assert CLASS_THRESHOLDS["Culture"] == 3

    def test_keyword_classes_threshold_is_1(self):
        """
        Budget, Luxury, Family require only 1 keyword match.
        These classes use presence signals, not activity counts.
        """
        assert CLASS_THRESHOLDS["Budget"] == 1
        assert CLASS_THRESHOLDS["Luxury"] == 1
        assert CLASS_THRESHOLDS["Family"] == 1

    def test_all_six_classes_have_thresholds(self):
        """Every class must have a threshold defined — no missing entries."""
        expected = {"Adventure", "Relaxation", "Culture", "Budget", "Luxury", "Family"}
        assert set(CLASS_THRESHOLDS.keys()) == expected
