"""Feature-based matching for rotation-invariant detection.

Uses ORB (Oriented FAST and Rotated BRIEF) features for detecting
objects that may be rotated, such as joker cards in Balatro.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class FeatureMatch:
    """Result of feature-based matching.

    Attributes:
        x: Center x coordinate of detected object.
        y: Center y coordinate of detected object.
        width: Estimated width of detected object.
        height: Estimated height of detected object.
        angle: Rotation angle in degrees (0 = upright).
        confidence: Match confidence (0-1), based on inlier ratio.
        num_matches: Number of feature matches found.
        num_inliers: Number of inlier matches after RANSAC.
        template_name: Name of the matched template.
        homography: 3x3 homography matrix (if computed).
    """

    x: int
    y: int
    width: int
    height: int
    angle: float
    confidence: float
    num_matches: int
    num_inliers: int
    template_name: str
    homography: NDArray[np.float64] | None = None

    @property
    def center(self) -> tuple[int, int]:
        """Get center point."""
        return (self.x, self.y)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get axis-aligned bounding box as (x1, y1, x2, y2)."""
        half_w = self.width // 2
        half_h = self.height // 2
        return (self.x - half_w, self.y - half_h, self.x + half_w, self.y + half_h)

    @property
    def corners(self) -> list[tuple[int, int]]:
        """Get rotated corner points."""
        if self.homography is None:
            # Return axis-aligned corners
            x1, y1, x2, y2 = self.bbox
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        # Transform template corners through homography
        half_w = self.width // 2
        half_h = self.height // 2
        template_corners = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ], dtype=np.float32).reshape(-1, 1, 2)

        # Translate to center position
        transformed = cv2.perspectiveTransform(template_corners, self.homography)
        return [(int(p[0][0]), int(p[0][1])) for p in transformed]


@dataclass
class TemplateFeatures:
    """Cached features for a template image."""

    keypoints: tuple
    descriptors: NDArray[np.uint8]
    image: NDArray[np.uint8]
    width: int
    height: int


class FeatureMatcher:
    """Rotation-invariant feature matching using ORB.

    Detects objects regardless of rotation by matching ORB features
    and computing homography transformations.

    Suitable for:
    - Joker cards (can be rotated in UI)
    - Any UI element with rotation variation
    - Objects with distinctive texture/patterns

    Not suitable for:
    - Simple shapes or solid colors
    - Very small templates (< 32x32)

    Attributes:
        template_dir: Directory containing template images.
        min_matches: Minimum good matches required.
        confidence_threshold: Minimum inlier ratio for valid detection.
    """

    def __init__(
        self,
        template_dir: str = "data/templates",
        num_features: int = 500,
        min_matches: int = 10,
        confidence_threshold: float = 0.3,
        match_ratio: float = 0.75,
    ) -> None:
        """Initialize feature matcher.

        Args:
            template_dir: Directory containing template images.
            num_features: Maximum ORB features to detect.
            min_matches: Minimum matches required for valid detection.
            confidence_threshold: Minimum inlier ratio (inliers/matches).
            match_ratio: Lowe's ratio test threshold.
        """
        self.template_dir = Path(template_dir)
        self.min_matches = min_matches
        self.confidence_threshold = confidence_threshold
        self.match_ratio = match_ratio

        # Create ORB detector
        self._orb = cv2.ORB_create(nfeatures=num_features)

        # Create brute-force matcher with Hamming distance (for binary descriptors)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Template feature cache
        self._cache: dict[str, TemplateFeatures] = {}

        logger.info(
            f"FeatureMatcher initialized: features={num_features}, "
            f"min_matches={min_matches}, threshold={confidence_threshold}"
        )

    def load_template(self, name: str) -> TemplateFeatures:
        """Load and extract features from a template image.

        Args:
            name: Template name (without extension).

        Returns:
            TemplateFeatures with keypoints and descriptors.

        Raises:
            FileNotFoundError: If template not found.
            ValueError: If no features detected in template.
        """
        if name in self._cache:
            return self._cache[name]

        # Find template file
        template_path = self.template_dir / f"{name}.png"

        if not template_path.exists():
            # Search subdirectories
            for subdir in ["", "jokers", "ui", "cards"]:
                candidate = self.template_dir / subdir / f"{Path(name).name}.png"
                if candidate.exists():
                    template_path = candidate
                    break

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {name}")

        # Load as grayscale
        image = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to load: {template_path}")

        # Extract features
        keypoints, descriptors = self._orb.detectAndCompute(image, None)

        if descriptors is None or len(keypoints) < 4:
            raise ValueError(
                f"Template '{name}' has insufficient features ({len(keypoints) if keypoints else 0}). "
                "Need distinctive patterns for feature matching."
            )

        features = TemplateFeatures(
            keypoints=keypoints,
            descriptors=descriptors,
            image=image,
            width=image.shape[1],
            height=image.shape[0],
        )

        self._cache[name] = features
        logger.debug(f"Loaded template '{name}': {len(keypoints)} features")

        return features

    def match(
        self,
        image: NDArray[np.uint8],
        template_name: str,
        mask: NDArray[np.uint8] | None = None,
    ) -> list[FeatureMatch]:
        """Find all matches of a template in an image.

        Args:
            image: Image to search in (BGR or grayscale).
            template_name: Name of template to match.
            mask: Optional mask for search region.

        Returns:
            List of FeatureMatch objects, sorted by confidence.
        """
        try:
            template = self.load_template(template_name)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Cannot match template '{template_name}': {e}")
            return []

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect features in search image
        keypoints, descriptors = self._orb.detectAndCompute(gray, mask)

        if descriptors is None or len(keypoints) < self.min_matches:
            logger.debug(f"Insufficient features in search image: {len(keypoints) if keypoints else 0}")
            return []

        # Match descriptors using kNN
        try:
            raw_matches = self._matcher.knnMatch(template.descriptors, descriptors, k=2)
        except cv2.error as e:
            logger.warning(f"Matching failed: {e}")
            return []

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_matches:
            logger.debug(
                f"Insufficient good matches for '{template_name}': "
                f"{len(good_matches)} < {self.min_matches}"
            )
            return []

        # Extract matched point coordinates
        src_pts = np.float32([
            template.keypoints[m.queryIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            keypoints[m.trainIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)

        # Compute homography with RANSAC
        homography, mask_inliers = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0
        )

        if homography is None:
            logger.debug(f"Homography computation failed for '{template_name}'")
            return []

        # Count inliers
        num_inliers = int(mask_inliers.sum()) if mask_inliers is not None else 0
        confidence = num_inliers / len(good_matches)

        if confidence < self.confidence_threshold:
            logger.debug(
                f"Low confidence for '{template_name}': {confidence:.2f} < {self.confidence_threshold}"
            )
            return []

        # Compute transformed center and corners
        template_center = np.array([[[template.width / 2, template.height / 2]]], dtype=np.float32)
        transformed_center = cv2.perspectiveTransform(template_center, homography)
        cx, cy = transformed_center[0][0]

        # Compute rotation angle from homography
        angle = self._extract_rotation(homography)

        # Estimate size (may be scaled)
        corners = np.array([
            [[0, 0]],
            [[template.width, 0]],
            [[template.width, template.height]],
            [[0, template.height]],
        ], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners, homography)

        # Calculate bounding box size from transformed corners
        xs = transformed_corners[:, 0, 0]
        ys = transformed_corners[:, 0, 1]
        width = int(xs.max() - xs.min())
        height = int(ys.max() - ys.min())

        match_result = FeatureMatch(
            x=int(cx),
            y=int(cy),
            width=width,
            height=height,
            angle=angle,
            confidence=confidence,
            num_matches=len(good_matches),
            num_inliers=num_inliers,
            template_name=template_name,
            homography=homography,
        )

        return [match_result]

    def match_multiple(
        self,
        image: NDArray[np.uint8],
        template_names: list[str],
    ) -> dict[str, list[FeatureMatch]]:
        """Match multiple templates against the same image.

        Args:
            image: Image to search in.
            template_names: List of template names.

        Returns:
            Dictionary mapping template name to matches.
        """
        # Extract features once
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        results: dict[str, list[FeatureMatch]] = {}

        for name in template_names:
            results[name] = self.match(gray, name)

        return results

    def find_best_match(
        self,
        image: NDArray[np.uint8],
        template_names: list[str],
    ) -> FeatureMatch | None:
        """Find the best matching template from a list.

        Args:
            image: Image to search in.
            template_names: List of candidate templates.

        Returns:
            Best FeatureMatch, or None if no match found.
        """
        best_match: FeatureMatch | None = None

        for name in template_names:
            matches = self.match(image, name)
            if matches:
                if best_match is None or matches[0].confidence > best_match.confidence:
                    best_match = matches[0]

        return best_match

    def _extract_rotation(self, homography: NDArray[np.float64]) -> float:
        """Extract rotation angle from homography matrix.

        Args:
            homography: 3x3 homography matrix.

        Returns:
            Rotation angle in degrees.
        """
        # Extract rotation from the 2x2 upper-left submatrix
        # This is an approximation - assumes minimal perspective distortion
        h = homography
        angle_rad = math.atan2(h[1, 0], h[0, 0])
        return math.degrees(angle_rad)

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()
        logger.debug("Feature cache cleared")

    def preload_templates(self, names: list[str]) -> dict[str, bool]:
        """Preload templates and report which loaded successfully.

        Args:
            names: Template names to load.

        Returns:
            Dict mapping name to success status.
        """
        results = {}
        for name in names:
            try:
                self.load_template(name)
                results[name] = True
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Failed to preload '{name}': {e}")
                results[name] = False
        return results


class HybridMatcher:
    """Combines template matching and feature matching.

    Uses fast template matching for upright objects and
    feature matching for rotated objects.

    Recommended for Balatro:
    - Hand cards: template matching (always upright)
    - Joker cards: feature matching (may be rotated)
    - UI elements: template matching (fixed positions)
    """

    def __init__(
        self,
        template_dir: str = "data/templates",
        rotation_threshold: float = 5.0,
    ) -> None:
        """Initialize hybrid matcher.

        Args:
            template_dir: Directory containing templates.
            rotation_threshold: Angle threshold (degrees) to prefer feature matching.
        """
        from src.vision.template_matcher import TemplateMatcher

        self.template_matcher = TemplateMatcher(template_dir=template_dir)
        self.feature_matcher = FeatureMatcher(template_dir=template_dir)
        self.rotation_threshold = rotation_threshold

        logger.info("HybridMatcher initialized")

    def match(
        self,
        image: NDArray[np.uint8],
        template_name: str,
        expect_rotation: bool = False,
    ) -> list[FeatureMatch]:
        """Match template with automatic method selection.

        Args:
            image: Image to search.
            template_name: Template to find.
            expect_rotation: If True, use feature matching.

        Returns:
            List of matches (FeatureMatch format for consistency).
        """
        if expect_rotation:
            return self.feature_matcher.match(image, template_name)

        # Try template matching first (faster)
        template_matches = self.template_matcher.match_template(image, template_name)

        if template_matches:
            # Convert to FeatureMatch format
            return [
                FeatureMatch(
                    x=m.center[0],
                    y=m.center[1],
                    width=m.width,
                    height=m.height,
                    angle=0.0,
                    confidence=m.confidence,
                    num_matches=0,
                    num_inliers=0,
                    template_name=m.template_name,
                    homography=None,
                )
                for m in template_matches
            ]

        # Fall back to feature matching
        return self.feature_matcher.match(image, template_name)
