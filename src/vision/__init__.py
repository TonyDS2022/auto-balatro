"""Computer vision module for Auto-Balatro."""

from .template_matcher import TemplateMatcher, Match, calculate_iou
from .feature_matcher import (
    FeatureMatcher,
    FeatureMatch,
    TemplateFeatures,
    HybridMatcher,
)
from .card_detector import CardDetector, DetectedCard
from .ocr import GameOCR, DigitRecognizer
from .state_detector import StateDetector

__all__ = [
    # Template matching (fast, no rotation)
    "TemplateMatcher",
    "Match",
    "calculate_iou",
    # Feature matching (rotation-invariant)
    "FeatureMatcher",
    "FeatureMatch",
    "TemplateFeatures",
    "HybridMatcher",
    # Card detection
    "CardDetector",
    "DetectedCard",
    # OCR
    "GameOCR",
    "DigitRecognizer",
    # State detection
    "StateDetector",
]
