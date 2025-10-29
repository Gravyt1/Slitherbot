from ultralytics import YOLO

def train_model(model_name="yolo11n.pt", data_config="slitherbot/dataset/config.yaml", test_image="slitherbot/dataset/test.png"):
    """
    Train a YOLO model on custom dataset.
    
    Args:
        model_name (str): Pre-trained YOLO model to use (default: yolo11n.pt)
        data_config (str): Path to dataset YAML configuration file
        test_image (str): Path to test image for prediction after training
    """
    # Initialize YOLO model
    model = YOLO("runs/detect/train/weights/best.pt")
    
    # Train the model
    print(f"Starting training with {model_name}...")
    model.train(
        data=data_config,
        imgsz=640,
        batch=4,
        epochs=75,
        workers=6,
        augment=True 
    )
    
    # Test prediction on a sample image
    print(f"\nRunning test prediction on {test_image}...")
    results = model.predict(source=test_image)
    results[0].show()
    
    print("\nTraining completed! Model saved in runs/detect/train/weights/")

if __name__ == "__main__":
    train_model()