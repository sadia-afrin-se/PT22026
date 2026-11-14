"""
Inference pipeline: validate -> preprocess -> CNN + Grad-CAM -> report.

Kept separate from the UI so it can be reused by scripts and tests.
"""
import logging
import time
from dataclasses import dataclass, field

from PIL import Image

from src import config
from src.gradcam import overlay_heatmap, predict_with_gradcam
from src.model_loader import get_model
from src.preprocessing import prepare_display_image, preprocess, validate_image

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    label: str
    prob_pneumonia: float
    confidence: float
    threshold: float
    heatmap: Image.Image
    resized: Image.Image
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def probabilities(self) -> dict[str, float]:
        return {config.CLASS_NAMES[0]: 1 - self.prob_pneumonia,
                config.CLASS_NAMES[1]: self.prob_pneumonia}


def analyse(image: Image.Image, threshold: float = config.DEFAULT_THRESHOLD) -> Prediction:
    """Run the full pipeline on one PIL image."""
    warnings = validate_image(image)          # may raise InvalidImageError
    start = time.perf_counter()

    model = get_model()
    batch = preprocess(image)
    prob, cam = predict_with_gradcam(model, batch, threshold)

    label = config.CLASS_NAMES[1] if prob >= threshold else config.CLASS_NAMES[0]
    confidence = prob if label == config.CLASS_NAMES[1] else 1 - prob
    if 0.35 < prob < 0.65:
        warnings.append("The model is uncertain (score close to 0.5). "
                        "Treat this result with extra caution.")

    resized = prepare_display_image(image)
    heatmap = overlay_heatmap(resized, cam)
    elapsed = time.perf_counter() - start
    logger.info("Prediction %s p=%.4f (%.2fs)", label, prob, elapsed)

    return Prediction(label, prob, confidence, threshold, heatmap, resized,
                      warnings, elapsed)
