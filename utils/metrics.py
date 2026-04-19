# utils/metrics.py
try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError as exc:
    MeanAveragePrecision = None
    TORCHMETRICS_IMPORT_ERROR = exc
else:
    TORCHMETRICS_IMPORT_ERROR = None


class DetectionEvaluator:
    def __init__(self):
        if MeanAveragePrecision is None:
            raise ImportError(
                "DetectionEvaluator requires torchmetrics and pycocotools. "
                "Please install them before running validation."
            ) from TORCHMETRICS_IMPORT_ERROR
        self.metric = MeanAveragePrecision(iou_type="bbox")

    def update(self, preds, targets):
        self.metric.update(preds, targets)

    def compute(self):
        results = self.metric.compute()
        self.metric.reset()
        return results
