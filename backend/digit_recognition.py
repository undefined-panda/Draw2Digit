from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from backend.model import MNIST_Net

current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "example_images")

def model_prediction(image):
    """
    image is a numpy array
    """

    model = MNIST_Net()
    model_path = os.path.join(current_dir, "saved_models/digit_recognition_model_1.pth")
    model.load_state_dict(torch.load(model_path))

    tensor_image = torch.tensor(image, dtype=torch.float32)
    tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)

    output = model(tensor_image)
    pred = output.argmax(dim=1, keepdim=True).item()

    return pred

if __name__ == "__main__":
    digit = 5
    image = Image.open(f"{image_path}/{digit}.png")
    image = image.convert('L')

    pred = model_prediction(image)

    print(f"Prediction: {pred}")