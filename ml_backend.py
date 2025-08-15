import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys
from rtmlib import Wholebody, draw_skeleton, draw_bbox

DEVICE = 'cpu'
BACKEND = 'onnxruntime'
MODE = 'balanced'
KPT_THRESHOLD = 0.3

class RTMLibWholeBodyBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_name, self.output_name = get_single_tag_keys(
            self.parsed_label_config, 'Image', 'KeyPointLabels'
        )
        self.wholebody = Wholebody(mode=MODE, backend=BACKEND, device=DEVICE, to_openpose=False)
        print("Wholebody model initialized.")

    def predict(self, tasks, **kwargs):
        results = []
        for task in tasks:
            image_url = task['data'][self.input_name]
            response = requests.get(image_url)
            img_pil = Image.open(BytesIO(response.content)).convert('RGB')
            img_np = np.array(img_pil)[:, :, ::-1]

            bboxes = self.wholebody.det_model(img_np)
            if len(bboxes) > 0:
                keypoints, scores = self.wholebody.pose_model(img_np, bboxes=bboxes)
                height, width = img_np.shape[:2]
                points_list = [[float(kp[0])/width*100, float(kp[1])/height*100] for kp in keypoints[0]]
                prediction = {
                    "result": [
                        {
                            "from_name": self.output_name,
                            "to_name": self.input_name,
                            "type": "keypointlabels",
                            "value": {"points": points_list}
                        }
                    ]
                }
            else:
                prediction = {"result": []}
            results.append(prediction)
        return results
