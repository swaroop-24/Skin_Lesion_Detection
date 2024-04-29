import torch
import torchvision.transforms as transforms
from PIL import Image
from Data_model import SkinLesionClassifier

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

model = SkinLesionClassifier()
model.load_state_dict(torch.load("skin_lesion_classifier2.pth"))
model.eval()

class_labels = ['Melanoma', 'Dysplastic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis', 'Dermatofibroma', 'Vascular lesion', 'Squamous cell carcinoma', 'Unknown']

def predict_image(image_path, model):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
    predicted_probs = torch.sigmoid(outputs).squeeze().numpy()
    max_prob_index = predicted_probs.argmax()
    if max_prob_index == 0:
        return "Melanoma"
    else:
        other_cancer = class_labels[max_prob_index]
        return f"Not Melanoma, most likely {other_cancer}"

image_path = "M4.jpg"
prediction = predict_image(image_path, model)
print("Prediction:", prediction)
