# CodeClauseInternship_ObjectDetectionSystem

# YOLOv9 Object Detection

This repository contains a Python script that performs object detection using the YOLOv9 model. The script processes an image, detects objects, and saves the output image with detected objects.

## Requirements

- Python 3.x
- OpenCV
- Matplotlib
- PyTorch
- ultralytics

## Installation

1. Clone this repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python matplotlib torch ultralytics
    ```

## Usage

1. Place the image you want to process in the `lib` directory and update the `image_path` variable in `index.py` to point to your image:
    ```python
    image_path = 'lib\\your_image.webp'
    ```

2. Run the script:
    ```bash
    python index.py
    ```

3. The script will detect objects in the image and display the result. The output image with detected objects will be saved as `output_image.jpg` in the current directory.

## Code Explanation

- Load the YOLOv9 model:
    ```python
    model = YOLO('yolov9e.pt')
    ```

- Load and preprocess the image:
    ```python
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (640, 640))
    ```

- Check for CUDA availability:
    ```python
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    ```

- Perform object detection:
    ```python
    results = model(source=image_path, device=device)
    ```

- Plot and save the detected image:
    ```python
    for result in results:
        detected_image = result.plot()

    plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, detected_image)
    ```

## Notes

- Ensure that the YOLOv9 model file (`yolov9e.pt`) is available in the same directory or provide the correct path.
- The script processes images with a default resolution of 640x640. Adjust the resolution as needed.

## License

This project is licensed under the MIT License.
