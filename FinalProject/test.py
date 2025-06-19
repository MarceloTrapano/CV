import lightning as L
import matplotlib.colors as mcolors

# load model and test on sample image
from utils import *
import matplotlib.pyplot as plt

def show_predictions(dataset_dir, batch_size=32, num_workers=4):
    model = LightingModel.load_from_checkpoint('checkpoints/best-checkpoint-v11.ckpt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.ToTensor(),
    ])
    
    data = LightingData(dataset_dir, batch_size, num_workers, transform=transform)
    predict_loader = data.predict_dataloader()
    
    
    # 1 sample from the dataset
    # show rgb image, predicted hsv image, and ground truth hsv image
    skip = 10
    for rgb, hsv in predict_loader:
        if skip > 0:
            skip -= 1
            continue
        rgb = rgb[0].to(device)
        hsv = hsv[0].to(device)
        
        with torch.no_grad():
            predicted_hsv = model(rgb.unsqueeze(0))
            
        predicted_hsv = predicted_hsv.squeeze(0).permute(1, 2, 0).cpu().numpy()
        rgb = rgb.permute(1, 2, 0).cpu().numpy()
        hsv = hsv.permute(1, 2, 0).cpu().numpy()
        predicted_rgb = mcolors.hsv_to_rgb(predicted_hsv)
        
        if TYPE == TransformType.HSV_TO_RGB:
            rgb, hsv, predicted_rgb = hsv, rgb, predicted_hsv
            predicted_hsv = mcolors.rgb_to_hsv(predicted_rgb)
        
        # show images
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.title('RGB Image')
        plt.imshow(rgb)
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.title('Predicted HSV Image')
        plt.imshow(predicted_hsv)
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.title('Ground Truth HSV Image')
        plt.imshow(hsv)
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.title('Predicted RGB from HSV')
        plt.imshow(predicted_rgb)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        break

if __name__ == "__main__":
    show_predictions('./Fruits Classification/all', batch_size=32, num_workers=4)