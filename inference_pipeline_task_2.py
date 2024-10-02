import torch
from utils.data_processing import preprocess_input
from dataloader import SimpleDataset
from utils.scoring import specificity
from sklearn.metrics import matthews_corrcoef, f1_score, cohen_kappa_score
import os
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import DataLoader 
from torchvision.models import ResNet18_Weights
from torchvision import models
import torch.nn as nn


import operator
import functools

from data_loader_task2 import load_data

image_size = 256

data_transforms = {
    'train': T.Compose([
        T.RandomRotation(15),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(image_size, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        T.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': T.Compose([
        T.Resize([image_size,image_size]),
        T.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': T.Compose([
        T.Resize([image_size,image_size]),
        #transforms.ColorJitter(brightness=.5, hue=.3)
        T.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ])
}


class InferenceTask2:
    def __init__(self, model_path, model_weights,*args, **kwargs):
        """
        Initializes the inference class with model paths and weights.

        Args:
            model_path (str): List of paths to the model files.
            model_weights (torch weights): List of weights for each model. Defaults to equal weights.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # DO NOT EDIT THIS PART        
        self.model = self.load_model(model_path, model_weights)
        print(f"Using device: {self.device}")

        self.i = 0 

    def load_model(self, model_path,model_weights,*args, **kwargs):
        """
        Loads a model from a given path.

        Args:
            model_path (str): Path to the model file.

        Returns:
            torch.nn.Module: Loaded model.
        """
        num_classes = 3 
        model = models.resnet18(weights=model_weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model
        
    
    def task2_inference(self, data_loader):
        """
        Performs inference on the data using the loaded model.

        Args:
            data_loader (DataLoader): DataLoader for the input data.

        Returns:
            list: True labels, predicted labels, and case IDs.
        """
        
        
        y_true = []
        y_pred = []
        cases = []
        with torch.no_grad():
            for images, labels, case in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                cases.extend(case)

        y_true = [item if isinstance(item, list) else [item] for item in y_true]
        y_pred = [item if isinstance(item, list) else [item] for item in y_pred]
        cases = [item if isinstance(item, list) else [item] for item in cases]
        
        return y_true, y_pred, cases
    

    def simple_inference(self, data_loader):
        """
        Performs inference on the data using the loaded model. 

        Args:
            data_loader (DataLoader): DataLoader for the input data.

        Returns:
            list: True labels, predicted labels, and case IDs.
        """
        
        y_true = []
        y_pred = []
        cases = []

        with torch.no_grad():
            for batch in tqdm(data_loader):
                
                oct_slice_xt = batch["oct_slice_xt"].to(self.device)
                localizers_xt = batch["localizers_xt"].to(self.device)
                clinical_data = batch["clinical_data"]
                label = batch["label"]
                case_id = batch["case_id"]
                output = self.model(oct_slice_xt)
                prediction = list(output.argmax(dim=1).cpu().detach())
                y_pred.append(prediction)
                y_true.append(label)
                cases.append(case_id)

        return y_true, y_pred, cases

    def scoring(self, y_true, y_pred):
        """
        DO NOT EDIT THIS CODE

        Calculates F1 score, Matthews Correlation Coefficient, and Specificity for a classification task.

        Args:
            y_true (list): True labels.
            y_pred (list): Predicted labels.

        Returns:
            dict: Dictionary containing F1 score, Matthews Correlation Coefficient, Specificity, and Quadratic-weighted Kappa metrics.
        """
        return {
            "F1_score": f1_score(y_true, y_pred, average="micro"),
            "Rk-correlation": matthews_corrcoef(y_true, y_pred),
            "Specificity": specificity(y_true, y_pred),
            "Quadratic-weighted_Kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic")
        }



    def run(self, data_loader,use_ensemble = True, use_tta=False,n_augmentations=5):
        """
        Runs the inference and saves results.

        Args:
            data_loader (DataLoader): DataLoader for the input data.
            use_tta (bool): Whether to use test time augmentation.
            n_augmentations (int): Number of augmentations to apply for TTA.

        Returns:
            dict: Dictionary containing various scores.
        """
               
        
        y_true, y_pred, cases = self.task2_inference(data_loader)
    
            
        y_true = functools.reduce(operator.iconcat, y_true, [])
        y_pred = functools.reduce(operator.iconcat, y_pred, [])
        cases = functools.reduce(operator.iconcat, cases, [])          

        output_file = f"output/results_task2_team_{os.environ['Team_name']}_method_{self.i}.csv"
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'cases': cases})
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        self.i +=1
        return self.scoring(y_true, y_pred), y_true, y_pred

# Main execution
print(f"Starting the inference for the team: {os.environ['Team_name']}")

# Load data
X, y, cases = load_data('csv/df_task2_val_challenge_with_labels.csv', 'data/')

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
cases_tensor = torch.tensor(cases, dtype=torch.long)

val_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor,cases_tensor)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

model_path = 'models/trained_resnet18_baseline_alltrainandval_50ep.pth'

# model_weights_contribution = [0.6, 0.4]  # Example weights for the models
model_weights = ResNet18_Weights.IMAGENET1K_V1
inference_task2 = InferenceTask2(model_path, model_weights)

scores, y_true, y_pred = inference_task2.run(val_loader)
print(f" Obtained scores for inference method 1: F1_score: {scores['F1_score']}, Rk-correlation: {scores['Rk-correlation']}, Specificity: {scores['Specificity']}, Quadratic-weighted_Kappa: {scores['Quadratic-weighted_Kappa']}")

