from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from time import time

text_list=["A dog.", "A car", "A bird", "A phone in a bag", "A phone on a table", "Where is my phone?"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg", ".assets/phone_in_bag.jpg", ".assets/phone_on_table.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

start = time()

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
}

with torch.no_grad():
    embeddings = model(inputs)

end = time()
print("Time taken for text embeddings: ", end-start)

start = time()

# Load data
inputs = {
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

end = time()
print("Time taken for image embeddings: ", end-start)

'''
Time taken for text embeddings:  3.3920891284942627
Time taken for image embeddings:  16.159539461135864
'''

'''
start = time()

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)
end = time()
print("Time taken: ", end-start)
'''

'''
Vision x Text:  tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05, 1.7942e-11, 8.0421e-10, 2.8706e-08],
        [3.3824e-05, 9.9959e-01, 2.4110e-05, 8.4099e-06, 3.4318e-04, 8.7568e-07],
        [4.7996e-05, 1.3496e-02, 9.8645e-01, 4.1798e-08, 2.6521e-08, 9.1546e-07],
        [2.0748e-11, 2.3553e-11, 1.1435e-11, 9.9999e-01, 3.7244e-06, 6.9524e-06],
        [1.5426e-11, 5.4159e-12, 1.8023e-11, 1.7833e-07, 1.0000e+00, 2.1368e-07]])
Audio x Text:  tensor([[1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.]])
Vision x Audio:  tensor([[0.8070, 0.1088, 0.0842],
        [0.1036, 0.7884, 0.1079],
        [0.0018, 0.0022, 0.9960],
        [0.3730, 0.5165, 0.1105],
        [0.0503, 0.7143, 0.2354]])
Time taken (load + multiply):  34.935033321380615
Time taken (load + multiply on GPU): 6.752408266067505
Time taken (multiply only): 0.03707695007324219
'''