"""Computer vision module for Auto-Balatro."""

from .template_matcher import TemplateMatcher, Match, calculate_iou
from .card_detector import CardDetector, DetectedCard
from .ocr import GameOCR, DigitRecognizer
from .state_detector import StateDetector

__all__ = [
    # Template matching
    "TemplateMatcher",
    "Match",
    "calculate_iou",
    # Card detection
    "CardDetector",
    "DetectedCard",
    # OCR
    "GameOCR",
    "DigitRecognizer",
    # State detection
    "StateDetector",
]
