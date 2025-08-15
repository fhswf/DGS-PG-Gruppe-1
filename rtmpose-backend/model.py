from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

import requests
from PIL import Image
import numpy as np
from io import BytesIO
from rtmlib import Wholebody

DEVICE = 'cpu'
BACKEND = 'onnxruntime'
MODE = 'balanced'
KPT_THRESHOLD = 0.3

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")
        self.wholebody = Wholebody(mode=MODE, backend=BACKEND, device=DEVICE, to_openpose=False)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        tasks: List of dicts from Label Studio, each containing 'data' with image URL
        returns: List of predictions
        """
        print("Starting prediction DGSSS...")
        results = []

        for task in tasks:
            print('Processing task:', task)
            #image_url = task['data'][self.input_name]
            image_url = task['data']['image']

            # Bild laden
            path = self.get_local_path(task['data']['image'], task_id=task['id'])
            #response = requests.get(path)
            #print(response)
            #img_pil = Image.open(BytesIO(response.content)).convert('RGB')
            img_pil = Image.open(path).convert('RGB')
            img_np = np.array(img_pil)[:, :, ::-1]  # PIL RGB → OpenCV BGR

            # 1. Person Detection
            bboxes = self.wholebody.det_model(img_np)  # (N,5) array

            # 2. Pose Estimation
            if len(bboxes) > 0:
                keypoints, scores = self.wholebody.pose_model(img_np, bboxes=bboxes)

                # Konvertiere Keypoints zu Prozentangaben für Label Studio
                height, width = img_np.shape[:2]
                points_list = []
                for person_kpts in keypoints:
                    points = [[float(kp[0])/width*100, float(kp[1])/height*100] for kp in person_kpts]
                    points_list.append(points)

                # Für jeden erkannten Menschen ein Label Studio keypoint result
                prediction = {
                    "result": [
                        {
                            "from_name": self.output_name,
                            "to_name": self.input_name,
                            "type": "keypointlabels",
                            "value": {
                                "points": points_list[0]  # nur erste Person z.B.
                            }
                        }
                    ]
                }
            else:
                # Keine Person erkannt → leeres Keypoint Ergebnis
                prediction = {"result": []}

            results.append(prediction)

        print("Prediction results GALIYAAAA:", results)

        return results

    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

