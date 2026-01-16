"""OCR module for reading game text and numbers from Balatro.

This module provides optical character recognition functionality optimized
for reading game UI elements like scores, money, and counters. Supports
both template-based digit matching (faster) and Tesseract OCR (fallback).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Optional pytesseract import with graceful fallback
try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

logger = logging.getLogger(__name__)


# Default screen regions for 1920x1080 base resolution
# These are approximate and should be calibrated for accuracy
# Format: (left, top, width, height)
DEFAULT_REGIONS = {
    "score": (1650, 200, 250, 60),
    "money": (1700, 50, 150, 40),
    "chips_needed": (850, 580, 220, 50),
    "hands_remaining": (100, 650, 60, 40),
    "discards_remaining": (100, 720, 60, 40),
    "ante": (100, 100, 80, 40),
}


class GameOCR:
    """OCR engine for reading game UI elements.

    Provides methods to extract numeric values from various game UI regions
    using either template-based digit matching or Tesseract OCR.

    Attributes:
        ocr_backend: The OCR method to use ('digit_templates' or 'tesseract').
        digit_templates: Dictionary of digit templates for template matching.
        regions: Screen regions for various game elements.
    """

    # Class-level default regions (1920x1080 base resolution)
    SCORE_REGION: tuple[int, int, int, int] = (1650, 200, 250, 60)
    MONEY_REGION: tuple[int, int, int, int] = (1700, 50, 150, 40)
    CHIPS_NEEDED_REGION: tuple[int, int, int, int] = (850, 580, 220, 50)
    HANDS_REGION: tuple[int, int, int, int] = (100, 650, 60, 40)
    DISCARDS_REGION: tuple[int, int, int, int] = (100, 720, 60, 40)
    ANTE_REGION: tuple[int, int, int, int] = (100, 100, 80, 40)

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the OCR engine.

        Args:
            config: Optional configuration dictionary with keys:
                - ocr_backend (str): 'tesseract' or 'digit_templates' (default)
                - digit_template_dir (str): Path to digit template images
                - preprocessing (dict): Preprocessing options
                    - threshold_value (int): Binary threshold value (default: 127)
                    - invert (bool): Invert image colors (default: False)
                    - scale_factor (float): Scale factor for small images (default: 2.0)
                - regions (dict): Override default screen regions
        """
        config = config or {}

        self.ocr_backend: str = config.get("ocr_backend", "digit_templates")
        self._digit_template_dir: str = config.get("digit_template_dir", "data/templates/digits")

        # Preprocessing options
        preprocessing = config.get("preprocessing", {})
        self._threshold_value: int = preprocessing.get("threshold_value", 127)
        self._invert: bool = preprocessing.get("invert", False)
        self._scale_factor: float = preprocessing.get("scale_factor", 2.0)
        self._min_size: int = preprocessing.get("min_size", 20)

        # Screen regions (allow override from config)
        region_config = config.get("regions", {})
        self.regions: dict[str, tuple[int, int, int, int]] = {
            "score": region_config.get("score", self.SCORE_REGION),
            "money": region_config.get("money", self.MONEY_REGION),
            "chips_needed": region_config.get("chips_needed", self.CHIPS_NEEDED_REGION),
            "hands_remaining": region_config.get("hands_remaining", self.HANDS_REGION),
            "discards_remaining": region_config.get("discards_remaining", self.DISCARDS_REGION),
            "ante": region_config.get("ante", self.ANTE_REGION),
        }

        # Initialize digit templates if using template-based OCR
        self.digit_templates: dict[str, NDArray[np.uint8]] = {}
        if self.ocr_backend == "digit_templates":
            self._load_digit_templates()

        # Warn if tesseract requested but not available
        if self.ocr_backend == "tesseract" and not TESSERACT_AVAILABLE:
            logger.warning(
                "Tesseract OCR requested but pytesseract not installed. "
                "Install with: pip install pytesseract. Falling back to template matching."
            )
            self.ocr_backend = "digit_templates"
            self._load_digit_templates()

        logger.info(f"GameOCR initialized with backend: {self.ocr_backend}")

    def _load_digit_templates(self) -> None:
        """Load digit template images for template-based OCR.

        Templates should be named 0.png, 1.png, ..., 9.png in the template directory.
        If templates are not found, generates simple synthetic templates.
        """
        template_dir = Path(self._digit_template_dir)

        if template_dir.exists():
            for digit in range(10):
                template_path = template_dir / f"{digit}.png"
                if template_path.exists():
                    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self.digit_templates[str(digit)] = template
                        logger.debug(f"Loaded template for digit {digit}")

        # If no templates loaded, generate synthetic ones
        if not self.digit_templates:
            logger.info("No digit templates found, generating synthetic templates")
            self._generate_synthetic_templates()

    def _generate_synthetic_templates(self) -> None:
        """Generate simple synthetic digit templates using OpenCV.

        Creates basic digit images for template matching when real templates
        are not available.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2

        for digit in range(10):
            # Create a small image for the digit
            img = np.zeros((40, 30), dtype=np.uint8)
            text = str(digit)

            # Get text size for centering
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # Center the text
            x = (img.shape[1] - text_width) // 2
            y = (img.shape[0] + text_height) // 2

            cv2.putText(img, text, (x, y), font, font_scale, 255, thickness)

            self.digit_templates[str(digit)] = img

    def _preprocess_for_ocr(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Preprocess image for OCR recognition.

        Applies a series of image processing steps to improve OCR accuracy:
        - Convert to grayscale
        - Resize if too small
        - Increase contrast
        - Apply thresholding
        - Add border

        Args:
            image: Input BGR or grayscale image.

        Returns:
            Preprocessed grayscale image ready for OCR.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize if image is too small (scale up for better recognition)
        height, width = gray.shape[:2]
        if height < self._min_size or width < self._min_size:
            scale = self._scale_factor
            gray = cv2.resize(
                gray,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Optionally invert (useful if text is light on dark background)
        if self._invert:
            binary = cv2.bitwise_not(binary)

        # Add small border for better recognition at edges
        border_size = 5
        binary = cv2.copyMakeBorder(
            binary,
            border_size,
            border_size,
            border_size,
            border_size,
            cv2.BORDER_CONSTANT,
            value=255 if not self._invert else 0,
        )

        return binary

    def _template_digit_ocr(self, image: NDArray[np.uint8]) -> str:
        """Recognize digits using template matching.

        Slides digit templates across the image to find and identify each digit.
        Faster than Tesseract for game-specific fonts.

        Args:
            image: Preprocessed grayscale/binary image.

        Returns:
            String containing recognized digits, empty string if none found.
        """
        if not self.digit_templates:
            logger.warning("No digit templates available for OCR")
            return ""

        # Preprocess
        processed = self._preprocess_for_ocr(image)

        # Find all digit matches
        matches: list[tuple[int, str, float]] = []  # (x_position, digit, confidence)

        for digit, template in self.digit_templates.items():
            # Resize template to match processed image scale if needed
            template_h, template_w = template.shape[:2]
            processed_h = processed.shape[0]

            # Scale template to match image height approximately
            if processed_h > template_h * 1.5:
                scale = processed_h / (template_h * 1.2)
                template = cv2.resize(
                    template,
                    (int(template_w * scale), int(template_h * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )

            # Ensure template is not larger than image
            if template.shape[0] > processed.shape[0] or template.shape[1] > processed.shape[1]:
                continue

            # Template matching
            result = cv2.matchTemplate(processed, template, cv2.TM_CCOEFF_NORMED)

            # Find all matches above threshold
            threshold = 0.6
            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):  # Switch x and y
                confidence = result[pt[1], pt[0]]
                matches.append((pt[0], digit, confidence))

        if not matches:
            return ""

        # Sort by x position
        matches.sort(key=lambda x: x[0])

        # Remove overlapping matches (keep highest confidence)
        filtered_matches: list[tuple[int, str, float]] = []
        min_distance = 10  # Minimum pixels between digits

        for match in matches:
            x, digit, conf = match
            # Check if this match overlaps with any existing match
            overlaps = False
            for i, (fx, fd, fc) in enumerate(filtered_matches):
                if abs(x - fx) < min_distance:
                    overlaps = True
                    # Keep the one with higher confidence
                    if conf > fc:
                        filtered_matches[i] = match
                    break
            if not overlaps:
                filtered_matches.append(match)

        # Sort again by x position and extract digits
        filtered_matches.sort(key=lambda x: x[0])
        result_str = "".join(digit for _, digit, _ in filtered_matches)

        return result_str

    def _tesseract_ocr(self, image: NDArray[np.uint8]) -> str:
        """Recognize text using Tesseract OCR.

        Fallback OCR method using pytesseract. Configured for digit-only recognition.

        Args:
            image: Input BGR or grayscale image.

        Returns:
            String containing recognized characters, empty string on failure.
        """
        if not TESSERACT_AVAILABLE or pytesseract is None:
            logger.error("Tesseract OCR not available")
            return ""

        # Preprocess
        processed = self._preprocess_for_ocr(image)

        # Tesseract configuration for digits only
        # --psm 7: Treat the image as a single text line
        # --oem 3: Default OCR engine mode
        # tessedit_char_whitelist: Only recognize digits
        config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"

        try:
            text = pytesseract.image_to_string(processed, config=config)
            # Clean up result
            text = text.strip()
            # Remove any non-digit characters that might slip through
            text = re.sub(r"[^\d]", "", text)
            return text
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""

    def _extract_region(
        self,
        image: NDArray[np.uint8],
        region: tuple[int, int, int, int] | None,
    ) -> NDArray[np.uint8] | None:
        """Extract a region from the image.

        Args:
            image: Full screen image.
            region: (left, top, width, height) tuple defining the region.

        Returns:
            Extracted region as numpy array, or None if invalid.
        """
        if region is None:
            return image

        left, top, width, height = region

        # Validate region bounds
        img_height, img_width = image.shape[:2]
        if left < 0 or top < 0:
            logger.warning(f"Invalid region: negative coordinates {region}")
            return None
        if left + width > img_width or top + height > img_height:
            logger.warning(
                f"Region {region} exceeds image bounds ({img_width}x{img_height})"
            )
            # Clamp to image bounds
            width = min(width, img_width - left)
            height = min(height, img_height - top)

        if width <= 0 or height <= 0:
            return None

        return image[top : top + height, left : left + width]

    def _perform_ocr(self, image: NDArray[np.uint8]) -> str:
        """Perform OCR using the configured backend.

        Args:
            image: Image region to recognize.

        Returns:
            Recognized text string.
        """
        if self.ocr_backend == "tesseract" and TESSERACT_AVAILABLE:
            return self._tesseract_ocr(image)
        else:
            return self._template_digit_ocr(image)

    def _parse_number(self, text: str) -> int | None:
        """Parse a number string, handling commas and formatting.

        Args:
            text: String possibly containing a number with formatting.

        Returns:
            Parsed integer or None if parsing fails.
        """
        if not text:
            return None

        # Remove common formatting characters
        cleaned = text.replace(",", "").replace(".", "").replace(" ", "")
        # Remove any dollar signs
        cleaned = cleaned.replace("$", "")

        # Try to parse as integer
        try:
            return int(cleaned)
        except ValueError:
            logger.debug(f"Failed to parse number from: '{text}'")
            return None

    def read_number(
        self,
        image: NDArray[np.uint8],
        region: tuple[int, int, int, int] | None = None,
    ) -> int | None:
        """Extract a number from an image region.

        Args:
            image: Full screen image or pre-cropped region.
            region: Optional (left, top, width, height) tuple to extract.
                   If None, uses the entire image.

        Returns:
            Recognized integer value, or None if recognition fails.
        """
        # Extract region if specified
        roi = self._extract_region(image, region)
        if roi is None:
            return None

        # Perform OCR
        text = self._perform_ocr(roi)

        # Parse result
        return self._parse_number(text)

    def read_score(self, image: NDArray[np.uint8]) -> int | None:
        """Read the current score from the score display region.

        Handles large numbers that may include commas or other formatting.

        Args:
            image: Full screen capture image.

        Returns:
            Current score as integer, or None if unreadable.
        """
        return self.read_number(image, self.regions["score"])

    def read_money(self, image: NDArray[np.uint8]) -> int | None:
        """Read the dollar amount from the money display region.

        Handles the $ prefix in the game UI.

        Args:
            image: Full screen capture image.

        Returns:
            Current money as integer, or None if unreadable.
        """
        return self.read_number(image, self.regions["money"])

    def read_chips_needed(self, image: NDArray[np.uint8]) -> int | None:
        """Read the blind requirement (chips needed to beat the blind).

        Args:
            image: Full screen capture image.

        Returns:
            Chips needed as integer, or None if unreadable.
        """
        return self.read_number(image, self.regions["chips_needed"])

    def read_hands_remaining(self, image: NDArray[np.uint8]) -> int | None:
        """Read the hands remaining counter.

        Args:
            image: Full screen capture image.

        Returns:
            Hands remaining as integer, or None if unreadable.
        """
        return self.read_number(image, self.regions["hands_remaining"])

    def read_discards_remaining(self, image: NDArray[np.uint8]) -> int | None:
        """Read the discards remaining counter.

        Args:
            image: Full screen capture image.

        Returns:
            Discards remaining as integer, or None if unreadable.
        """
        return self.read_number(image, self.regions["discards_remaining"])

    def read_ante(self, image: NDArray[np.uint8]) -> int | None:
        """Read the current ante number (1-8).

        Args:
            image: Full screen capture image.

        Returns:
            Current ante as integer (1-8), or None if unreadable.
        """
        result = self.read_number(image, self.regions["ante"])

        # Validate ante is in expected range
        if result is not None and (result < 1 or result > 8):
            logger.warning(f"Read ante {result} outside expected range 1-8")

        return result

    def read_all_stats(
        self, image: NDArray[np.uint8]
    ) -> dict[str, int | None]:
        """Read all game statistics from a single screen capture.

        Convenience method to read all numeric UI elements at once.

        Args:
            image: Full screen capture image.

        Returns:
            Dictionary with keys: score, money, chips_needed, hands_remaining,
            discards_remaining, ante. Values are integers or None if unreadable.
        """
        return {
            "score": self.read_score(image),
            "money": self.read_money(image),
            "chips_needed": self.read_chips_needed(image),
            "hands_remaining": self.read_hands_remaining(image),
            "discards_remaining": self.read_discards_remaining(image),
            "ante": self.read_ante(image),
        }

    def calibrate_regions(
        self,
        image: NDArray[np.uint8],
        region_name: str,
        new_region: tuple[int, int, int, int],
    ) -> None:
        """Update a screen region for calibration.

        Allows runtime adjustment of screen regions for different resolutions
        or UI layouts.

        Args:
            image: Reference image for validation (unused currently).
            region_name: Name of the region to update (e.g., 'score', 'money').
            new_region: New (left, top, width, height) tuple for the region.
        """
        if region_name not in self.regions:
            logger.warning(f"Unknown region name: {region_name}")
            return

        self.regions[region_name] = new_region
        logger.info(f"Updated region '{region_name}' to {new_region}")

    def scale_regions_for_resolution(
        self,
        target_width: int,
        target_height: int,
        base_width: int = 1920,
        base_height: int = 1080,
    ) -> None:
        """Scale all regions for a different screen resolution.

        Adjusts all region coordinates proportionally based on the ratio
        between the target resolution and the base resolution.

        Args:
            target_width: Target screen width.
            target_height: Target screen height.
            base_width: Base resolution width (default: 1920).
            base_height: Base resolution height (default: 1080).
        """
        scale_x = target_width / base_width
        scale_y = target_height / base_height

        for name, region in self.regions.items():
            left, top, width, height = region
            self.regions[name] = (
                int(left * scale_x),
                int(top * scale_y),
                int(width * scale_x),
                int(height * scale_y),
            )

        logger.info(
            f"Scaled regions from {base_width}x{base_height} to {target_width}x{target_height}"
        )


# Alias for backwards compatibility
DigitRecognizer = GameOCR
