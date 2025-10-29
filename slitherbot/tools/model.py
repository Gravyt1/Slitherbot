from label_studio_ml.model import LabelStudioMLBase 
from ultralytics import YOLO 
from PIL import Image 
import requests
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()


class YOLOBackend(LabelStudioMLBase): 
    """
    Label Studio ML Backend for YOLO object detection.
    
    This backend integrates a trained YOLO model with Label Studio
    for automated pre-annotation of images.

    Don't forget to change the model path (:
    """
    
    def __init__(self, **kwargs): 
        super(YOLOBackend, self).__init__(**kwargs) 
        
        # Path to your trained YOLO model
        model_path = os.environ.get('MODEL_PATH', 'runs/detect/train/weights/best.pt')
        self.model = YOLO(model_path) 
        
        # Class names (update according to your dataset)
        self.class_names = ["border", "point", "snake"]
        
        # Label Studio configuration
        self.label_studio_url = os.environ.get('LABEL_STUDIO_URL', 'http://localhost:8080')
        self.api_token = os.environ.get('LABEL_STUDIO_API_TOKEN', '')
        
        print(f"ML Backend initialized")
        print(f"Model: {model_path}")
        print(f"Classes: {self.class_names}")
 
    def predict(self, tasks, **kwargs): 
        """
        Generate predictions for Label Studio tasks.
        
        Args:
            tasks (list): List of Label Studio tasks containing image data
            
        Returns:
            list: Predictions in Label Studio format
        """
        results = [] 
        
        for task in tasks: 
            image_url = task['data']['image']
            
            # Convert relative path to full URL
            if image_url.startswith('/data'):
                image_url = f"{self.label_studio_url}{image_url}"
            
            print(f"Processing: {image_url}")
            
            try:
                headers = {
                    'Authorization': f'Token {self.api_token}'
                }
                
                # Download image
                response = requests.get(image_url, headers=headers)
                response.raise_for_status()
                
                img = Image.open(BytesIO(response.content))
                width, height = img.size
                
                # Save temporarily for YOLO prediction
                temp_path = f"temp_{task['id']}.png"
                img.save(temp_path)
                
                # Run YOLO prediction
                preds = self.model.predict(source=temp_path)
                
                # Clean up temp file
                os.remove(temp_path)
                
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append({"result": []})
                continue
            
            ls_predictions = []
 
            # Convert YOLO predictions to Label Studio format
            if len(preds[0].boxes) > 0:
                boxes = preds[0].boxes.xyxy.tolist() 
                classes = preds[0].boxes.cls.tolist()
                confidences = preds[0].boxes.conf.tolist()
     
                for box, cls_idx, conf in zip(boxes, classes, confidences): 
                    x1, y1, x2, y2 = box 
                    ls_predictions.append({ 
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels", 
                        "value": { 
                            "x": x1 / width * 100, 
                            "y": y1 / height * 100, 
                            "width": (x2 - x1) / width * 100, 
                            "height": (y2 - y1) / height * 100, 
                            "rectanglelabels": [self.class_names[int(cls_idx)]] 
                        },
                        "score": float(conf)
                    })
                
                print(f"Generated {len(ls_predictions)} prediction(s)")
            else:
                print("No detections found")
 
            results.append({
                "result": ls_predictions
            })
            
        return results