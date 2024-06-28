from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch.cuda

model = YOLO('yolov9e.pt')  

# Load and preprocess an image
image_path = 'lib\\05.webp'  #Path to your Image
image = cv2.imread(image_path)


image_resized = cv2.resize(image, (640, 640))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


results = model(source=image_path,device = device) #Use source = 0 to use your Webcam 


for result in results:
    detected_image = result.plot()


plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


output_path = 'output_image.jpg'
cv2.imwrite(output_path, detected_image)
