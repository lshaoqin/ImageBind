from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

def instantiate_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    print("Model instantiated.")

    return model, device

def generate_text_embeddings(inputs, model, device):
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(inputs, device)
    }

    with torch.no_grad():
        embeddings = model(inputs)

    return embeddings[ModalityType.TEXT]

def generate_vision_embeddings(inputs, model, device):
    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(inputs, device)
    }

    with torch.no_grad():
        embeddings = model(inputs)

    return embeddings[ModalityType.VISION]

def generate_audio_embeddings(inputs, model, device):
    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(inputs, device)
    }

    with torch.no_grad():
        embeddings = model(inputs)

    return embeddings[ModalityType.AUDIO]