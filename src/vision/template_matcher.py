"""Template matching module for Auto-Balatro.

Provides multi-scale template matching with caching and non-maximum suppression
for detecting UI elements and cards in game screenshots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class Match:
    """Represents a template match result.

    Attributes:
        x: Top-left x coordinate of match.
        y: Top-left y coordinate of match.
        width: Template width at matched scale.
        height: Template height at matched scale.
        confidence: Match confidence score (0-1).
        scale: Scale factor that produced this match.
        template_name: Name of the matched template.
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float
    scale: float
    template_name: str

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of the match."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box as (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def calculate_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two bounding boxes.

    Args:
        box1: First box as (x1, y1, x2, y2).
        box2: Second box as (x1, y1, x2, y2).

    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Check for no intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


class TemplateMatcher:
    """Multi-scale template matching with caching and NMS.

    Loads template images from disk, caches them, and provides methods
    for matching templates against game screenshots at multiple scales.

    Attributes:
        template_dir: Directory containing template images.
        confidence_threshold: Minimum confidence for valid matches.
        grayscale: Whether to convert images to grayscale.
        scale_factors: List of scales to try during matching.
    """

    def __init__(
        self,
        template_dir: str = "data/templates",
        confidence_threshold: float = 0.8,
        grayscale: bool = True,
        scale_factors: list[float] | None = None,
    ) -> None:
        """Initialize the template matcher.

        Args:
            template_dir: Directory containing template PNG images.
            confidence_threshold: Minimum confidence score (0-1) for matches.
            grayscale: Convert images to grayscale before matching.
            scale_factors: List of scale factors to try. Defaults to
                [0.8, 0.9, 1.0, 1.1, 1.2] for handling resolution differences.
        """
        self.template_dir = Path(template_dir)
        self.confidence_threshold = confidence_threshold
        self.grayscale = grayscale
        self.scale_factors = scale_factors or [0.8, 0.9, 1.0, 1.1, 1.2]

        # Template cache: name -> numpy array
        self._cache: dict[str, NDArray[np.uint8]] = {}

        logger.info(
            f"TemplateMatcher initialized: dir={template_dir}, "
            f"threshold={confidence_threshold}, scales={self.scale_factors}"
        )

    def load_template(self, name: str) -> NDArray[np.uint8]:
        """Load a template image by name.

        Searches for the template in the template directory and its
        subdirectories. Caches loaded templates for reuse.

        Args:
            name: Template name without extension (e.g., 'button_play')
                  or with subdirectory (e.g., 'ui/button_play').

        Returns:
            Template image as numpy array (grayscale if self.grayscale).

        Raises:
            FileNotFoundError: If template file not found.
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Try to find the template
        template_path = self.template_dir / f"{name}.png"

        if not template_path.exists():
            # Try without subdirectory prefix
            base_name = Path(name).name
            for subdir in ["", "ui", "cards", "ranks", "suits", "digits"]:
                candidate = self.template_dir / subdir / f"{base_name}.png"
                if candidate.exists():
                    template_path = candidate
                    break

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {name} (searched in {self.template_dir})")

        # Load image
        if self.grayscale:
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        else:
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)

        if template is None:
            raise FileNotFoundError(f"Failed to load template: {template_path}")

        # Cache and return
        self._cache[name] = template
        logger.debug(f"Loaded template '{name}': {template.shape}")
        return template

    def match_template(
        self,
        image: NDArray[np.uint8],
        template_name: str,
        threshold: float | None = None,
    ) -> list[Match]:
        """Find all matches of a template in an image.

        Performs multi-scale template matching and applies non-maximum
        suppression to remove overlapping matches.

        Args:
            image: Image to search in (BGR or grayscale).
            template_name: Name of template to match.
            threshold: Minimum confidence threshold. Defaults to
                self.confidence_threshold.

        Returns:
            List of Match objects sorted by confidence (highest first).
        """
        threshold = threshold if threshold is not None else self.confidence_threshold

        try:
            template = self.load_template(template_name)
        except FileNotFoundError:
            logger.warning(f"Template not found: {template_name}")
            return []

        # Convert image to grayscale if needed
        if self.grayscale and len(image.shape) == 3:
            search_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            search_image = image

        matches: list[Match] = []
        img_h, img_w = search_image.shape[:2]
        template_h, template_w = template.shape[:2]

        for scale in self.scale_factors:
            # Scale the template
            scaled_w = int(template_w * scale)
            scaled_h = int(template_h * scale)

            # Skip if scaled template is larger than image
            if scaled_w > img_w or scaled_h > img_h:
                continue

            # Skip if scaled template is too small
            if scaled_w < 5 or scaled_h < 5:
                continue

            scaled_template = cv2.resize(
                template,
                (scaled_w, scaled_h),
                interpolation=cv2.INTER_LINEAR,
            )

            # Template matching
            result = cv2.matchTemplate(
                search_image, scaled_template, cv2.TM_CCOEFF_NORMED
            )

            # Find all locations above threshold
            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):  # Switch from (y,x) to (x,y)
                x, y = pt
                confidence = float(result[y, x])
                matches.append(
                    Match(
                        x=int(x),
                        y=int(y),
                        width=scaled_w,
                        height=scaled_h,
                        confidence=confidence,
                        scale=scale,
                        template_name=template_name,
                    )
                )

        # Apply non-maximum suppression
        matches = self._non_max_suppression(matches)

        # Sort by confidence (highest first)
        matches.sort(key=lambda m: m.confidence, reverse=True)

        return matches

    def match_multiple(
        self,
        image: NDArray[np.uint8],
        template_names: list[str],
        threshold: float | None = None,
    ) -> dict[str, list[Match]]:
        """Match multiple templates against the same image.

        Args:
            image: Image to search in.
            template_names: List of template names to match.
            threshold: Minimum confidence threshold.

        Returns:
            Dictionary mapping template name to list of matches.
        """
        results: dict[str, list[Match]] = {}

        for name in template_names:
            results[name] = self.match_template(image, name, threshold)

        return results

    def find_best_match(
        self,
        image: NDArray[np.uint8],
        template_name: str,
        threshold: float | None = None,
    ) -> Match | None:
        """Find the single best match of a template.

        Args:
            image: Image to search in.
            template_name: Name of template to match.
            threshold: Minimum confidence threshold.

        Returns:
            Best Match object, or None if no match above threshold.
        """
        matches = self.match_template(image, template_name, threshold)
        return matches[0] if matches else None

    def _non_max_suppression(
        self,
        matches: list[Match],
        overlap_thresh: float = 0.3,
    ) -> list[Match]:
        """Remove overlapping matches using non-maximum suppression.

        Keeps the highest confidence match when multiple matches overlap.

        Args:
            matches: List of matches to filter.
            overlap_thresh: IoU threshold for considering matches overlapping.

        Returns:
            Filtered list of matches with overlaps removed.
        """
        if len(matches) <= 1:
            return matches

        # Sort by confidence (highest first)
        sorted_matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
        kept: list[Match] = []

        for match in sorted_matches:
            # Check if this match overlaps with any kept match
            should_keep = True
            for kept_match in kept:
                iou = calculate_iou(match.bbox, kept_match.bbox)
                if iou > overlap_thresh:
                    should_keep = False
                    break

            if should_keep:
                kept.append(match)

        return kept

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()
        logger.debug("Template cache cleared")

    def preload_templates(self, names: list[str]) -> None:
        """Preload multiple templates into cache.

        Args:
            names: List of template names to load.
        """
        for name in names:
            try:
                self.load_template(name)
            except FileNotFoundError:
                logger.warning(f"Could not preload template: {name}")
