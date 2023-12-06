from typing import Any, Dict, List, Sequence, Tuple

import mmengine
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS


@METRICS.register_module()
class ClsScoreSaver(BaseMetric):
    def __init__(self, save_path: str, labels_file: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.save_path = save_path

        self.labels_name = []
        with open(labels_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue

                label = line.split(",")[0]
                self.labels_name.append(label)

    def process(
        self, data_batch: Sequence[Tuple[Any, Dict]], data_samples: Sequence[Dict]
    ) -> None:
        for sample in data_samples:
            assert sample["num_classes"] == len(self.labels_name)

            label = sample["pred_label"].item()
            pred_score = sample["pred_score"].cpu().numpy()
            item = {
                "frame_dir": sample["frame_dir"],
                "sql_id": sample["sql_id"],
                "pred_score": pred_score,
                "pred_label": label,
                "pred_label_score": pred_score[label],
                "pred_label_name": self.labels_name[label],
            }

            self.results.append(item)

    def compute_metrics(self, results: List) -> Dict:
        mmengine.dump(results, self.save_path)
        return {}
