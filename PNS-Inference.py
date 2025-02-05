import cv2
import torch
import numpy as np
import time
from torchvision import transforms

# Import your model and necessary modules
from lib.module.PNSPlusNetwork import PNSNet  # Adjust import based on your project structure
from scripts.config import config  # Import your configuration

class RealTimeInference:
    def __init__(self, model_path, video_source=0):
        # Initialize model
        self.model = PNSNet().cuda()
        state_dict = torch.load(model_path, map_location=torch.device('cuda'))
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Video capture
        self.cap = cv2.VideoCapture(video_source)
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 448)),  # Adjust based on your model's input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_frame(self, frame):
        # Convert frame to tensor and apply transforms
        frame_tensor = self.transform(frame).unsqueeze(0).cuda()
        return frame_tensor

    def postprocess_mask(self, mask):
        # Convert mask to numpy and resize to original frame size
        mask = mask.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                  self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return (mask > 0.5).astype(np.uint8) * 255  # Binary threshold

    def run_inference(self):
        prev_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Compute FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Inference
            with torch.no_grad():
                input_tensor = self.preprocess_frame(frame)
                pred_mask = self.model(input_tensor)
                mask = self.postprocess_mask(pred_mask)

            # Overlay mask
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

            # Add FPS text
            cv2.putText(overlay, f'FPS: {fps:.2f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display
            cv2.imshow('Real-time Polyp Segmentation', overlay)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    model_path = 'snapshot/PNSPlus/epoch_15/PNSPlus.pth'  # Adjust path to your model weights
    inference = RealTimeInference(model_path)
    inference.run_inference()

if __name__ == '__main__':
    main()