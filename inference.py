import config
import torch
from PIL import Image
from torchvision import transforms
import sys

path_to_image = sys.argv[1]


def define_emotion(path_to_image):

    best_model = config.BEST_MODEL

    best_model_path = f"weights/{best_model}.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(best_model_path)
    model = model.to(device)

    img = Image.open(path_to_image)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    class_of_emotion = int(probabilities.argmax())
    list_of_emotions = [
        "anger",
        "contempt",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
        "uncertain",
    ]
    result = list_of_emotions[class_of_emotion]

    return result


if __name__ == "__main__":
    print("Emotion:", define_emotion(path_to_image))
