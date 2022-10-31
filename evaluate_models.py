import warnings

warnings.filterwarnings("ignore")

from glob import glob
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from sklearn import metrics
import config

batch_size = config.BATCH_SIZE
device = "cuda" if torch.cuda.is_available() else "cpu"


def pytorch_predict(model, test_loader, device):
    """
    Make prediction from a pytorch model
    """
    # set model to evaluate model
    model.eval()

    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)

    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        for data in test_loader:
            inputs = [i.to(device) for i in data[:-1]]
            labels = data[-1].to(device)

            outputs = model(*inputs)
            y_true = torch.cat((y_true, labels), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)

    y_true = y_true.cpu().numpy()
    _, y_pred = torch.max(all_outputs, 1)
    y_pred = y_pred.cpu().numpy()
    y_pred_prob = F.softmax(all_outputs, dim=1).cpu().numpy()

    return y_true, y_pred, y_pred_prob


def evaluave_models():
    """
    Evaluates all models in weights folder
    """

    models_list = [w.split("/")[1].split(".")[0] for w in glob("weights/*")]

    for model_name in models_list:
        input_size = 299 if model_name == "inception" else 224
        model = torch.load(f"weights/{model_name}.pt")
        model = model.to()
        data_transforms = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_dataset = datasets.ImageFolder("dataset/test", data_transforms)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        y_true, y_pred, y_pred_prob = pytorch_predict(model, test_dataloader, "cuda")

        print(f"[{model_name}] New model evaluation...")
        print(metrics.classification_report(y_true, y_pred))
        print("\n\n")


if __name__ == "__main__":
    evaluave_models()
