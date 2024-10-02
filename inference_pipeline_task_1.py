import torch
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef
from utils.data_processing import preprocess_input
from dataloader import SimpleDataset
from utils.scoring import specificity
from tqdm import tqdm
# from models.example_model_task1 import SimpleModel1
# from models.example_model_task1v2 import SimpleModel1v2

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageOps

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim


from torch.utils.data import DataLoader 
import torchvision.transforms as T

from torchvision.models import ResNet18_Weights
from torchvision import models

import operator
import functools


##################################################### ADDED ###########################################
class EvolutionModel_basis(nn.Module):
    def __init__(self, num_classes):
        super(EvolutionModel_basis, self).__init__()

        # Initialize resnet1 with pretrained weights from torchvision
        self.resnet1 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet1_in_features = self.resnet1.fc.in_features
        self.resnet1.fc = nn.Identity()  # Remove the fully connected layer

        # Initialize resnet2 with pretrained weights from torchvision
        self.resnet2 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet2_in_features = self.resnet2.fc.in_features
        self.resnet2.fc = nn.Identity()  # Remove the fully connected layer

        # New fully connected layer
        # Assuming both resnets have the same number of in_features
        combined_in_features = resnet1_in_features + resnet2_in_features
        self.fc1 = nn.Linear(combined_in_features, num_classes)

    def forward(self, x1, x2):
        out1 = self.resnet1(x1)
        out2 = self.resnet2(x2)
        combined = torch.cat((out1, out2), dim=1)  # Concatenate along the feature dimension
        out = self.fc1(combined)
        return out


class ImagePairDataset(Dataset):
    def __init__(self, X1, X2, y, cases, transform=None):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.cases = cases
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image1 = self.X1[idx]
        image2 = self.X2[idx]
        label = self.y[idx]
        case = self.cases[idx]

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label, case

def load_data(csv_file_path, image_folder_path, num_samples=None, target_size=(224, 224)):
    # Load the CSV file
    csv = pd.read_csv(csv_file_path)
    print(f"CSV columns: {csv.columns}")
    
    # Initialize lists to hold images and labels
    images_at_t0 = []
    images_at_t1 = []
    labels = []
    cases = []
    
    # Iterate over the rows in the CSV
    for index, row in csv.iterrows():
        if num_samples is not None and index >= num_samples:
            break
        
        # Print progress
        print(f'Loading image {index + 1}/{len(csv)}')
        
        # Load and preprocess images
        image_t0 = Image.open(os.path.join(image_folder_path, row['image_at_ti']))
        image_t0 = ImageOps.fit(image_t0, target_size, Image.Resampling.LANCZOS)
        image_array_t0 = np.array(image_t0)
        
        image_t1 = Image.open(os.path.join(image_folder_path, row['image_at_ti+1']))
        image_t1 = ImageOps.fit(image_t1, target_size, Image.Resampling.LANCZOS)
        image_array_t1 = np.array(image_t1)
        
        # Append images and labels
        images_at_t0.append(image_array_t0)
        images_at_t1.append(image_array_t1)
        labels.append(row['label'])

        # Append cases
        cases.append(row['case'])
            
    return np.array(images_at_t0), np.array(images_at_t1), np.array(labels), np.array(cases)

#########################################################################################################################

class InferenceTask1:
    def __init__(self, model_paths, model_names,model_weights=None,*args, **kwargs):
        """
        Initializes the inference class with model paths and weights.

        Args:
            model_paths (list): List of paths to the model files.
            model_weights (list, optional): List of weights for each model. Defaults to equal weights.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [self.load_model(model_name,model_path) for model_name,model_path in zip(model_names,model_paths)]
        self.model = self.models[0] 
        print(f"Using device: {self.device}")
        self.i = 0
        if model_weights is None:
            self.model_weights = [1.0 / len(model_paths)] * len(model_paths)
        else:
            self.model_weights = model_weights

    def load_model(self, model_name, model_path,*args, **kwargs):
        """
        Loads a model from a given path and it's class name.

        Args:
            model_name (str): name of the model class.
            model_path (str): Path to the model file.
            

        Returns:
            torch.nn.Module: Loaded model.
        """
        num_classes = 4
        model = EvolutionModel_basis(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def simple_inference(self, data_loader):
        """
        Performs inference on the data using the loaded model.

        Args:
            data_loader (DataLoader): DataLoader for the input data.

        Returns:
            list: True labels, predicted labels, and case IDs.
        """
        
        ## The proposed example only use the pair of OCT slice, but you are free to update if your pipeline involve
        ## localizer and the clinical, udapte accordingly 
        
        y_true = []
        y_pred = []
        cases = []
        with torch.no_grad():
            for images1, images2, labels, case in data_loader:
                images1, images2, labels = images1.to(self.device), images2.to(self.device), labels.to(self.device)
                outputs = self.model(images1, images2)
                # loss = criterion(outputs, labels)
                # val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                cases.extend(case)

        y_true = [item if isinstance(item, list) else [item] for item in y_true]
        y_pred = [item if isinstance(item, list) else [item] for item in y_pred]
        cases = [item if isinstance(item, list) else [item] for item in cases]
        return y_true, y_pred, cases
    
    def scoring(self, y_true, y_pred):
        """
        DO NOT EDIT THIS CODE
        Calculates various scoring metrics.

        Args:
            y_true (list): True labels.
            y_pred (list): Predicted labels.

        Returns:
            dict: Dictionary containing various scores.
        """
        return {
            "F1_score": f1_score(y_true, y_pred, average="micro"),
            "Rk-correlation": matthews_corrcoef(y_true, y_pred),
            "Specificity": specificity(y_true, y_pred),
        }


    def test_time_augmentation(self, input_tensor, n_augmentations=5):
        """
        Applies test time augmentation to the input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor.
            n_augmentations (int): Number of augmentations to apply.

        Returns:
            torch.Tensor: Averaged output from the model after applying augmentations.
        """
        augmentations = T.Compose([
            T.RandomHorizontalFlip(p=1.0),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
        
        outputs = []
        for _ in range(n_augmentations):
            augmented_input = augmentations(input_tensor).to(self.device)
            with torch.no_grad():
                outputs.append(self.models[0](augmented_input))
        return torch.mean(torch.stack(outputs,dim=0), dim=0)
                    

    def simple_ensemble_inference(self, data_loader):
        """
        Performs inference using model ensembling and test time augmentation.

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
                
                oct_slice_xt0 = batch["oct_slice_xt0"].to(self.device)
                oct_slice_xt1 = batch["oct_slice_xt1"].to(self.device)
                localizers_xt0 = batch["localizers_xt0"].to(self.device)
                localizers_xt1 = batch["localizers_xt1"].to(self.device)
                clinical_data = batch["clinical_data"]
                label = batch["label"]
                case_id = batch["case_id"]
                                


                inputs = torch.concat([oct_slice_xt0, oct_slice_xt1], dim=1)
                outputs = []
                for model, weight in zip(self.models, self.model_weights):
                    output = model(inputs)
                    outputs.append(output * weight)
                averaged_output = torch.mean(torch.stack(outputs), dim=0)
                prediction = list(averaged_output.argmax(dim=1).cpu().detach())
                
                y_pred.append(prediction)
                y_true.append(label)
                cases.append(case_id)

        return y_true, y_pred, cases

    def simple_test_time_inference(self, data_loader, n_augmentations=5):
        """
        Performs inference with test time augmentation and model ensembling.

        Args:
            data_loader (DataLoader): DataLoader for the input data.
            n_augmentations (int): Number of augmentations to apply for TTA.

        Returns:
            list: True labels, predicted labels, and case IDs.
        """
        y_true = []
        y_pred = []
        cases = []

        with torch.no_grad():
            
            for batch in tqdm(data_loader):
                
                oct_slice_xt0 = batch["oct_slice_xt0"].to(self.device)
                oct_slice_xt1 = batch["oct_slice_xt1"].to(self.device)
                localizers_xt0 = batch["localizers_xt0"].to(self.device)
                localizers_xt1 = batch["localizers_xt1"].to(self.device)
                clinical_data = batch["clinical_data"]
                label = batch["label"]
                case_id = batch["case_id"]
                tta_outputs = []
                for _ in range(n_augmentations):
  
                    inputs = torch.concat([oct_slice_xt0, oct_slice_xt1], dim=1)
                    augmented_inputs = self.test_time_augmentation(inputs, n_augmentations)

                    prediction = list(augmented_inputs.argmax(dim=1).cpu().detach())
                    

                    
                    y_pred.append(prediction)
                    y_true.append(label)
                    cases.append(case_id)
        return y_true, y_pred, cases


    def run(self, data_loader, use_ensemble = True, use_tta=False, n_augmentations=5):
        """
        Runs the inference and saves results.

        Args:
            data_loader (DataLoader): DataLoader for the input data.
            use_tta (bool): Whether to use test time augmentation.
            n_augmentations (int): Number of augmentations to apply for TTA.

        Returns:
            dict: Dictionary containing various scores.
        """
        
        
        ## You can test as much inference pipeline you which
        # in your local machine. You will have to select
        # two shot to for the final submission. 
        # The inference should always return a list of batch containing label,prediction,cases 
        # The method run should always return the scores  #TODO twee shots? welke nog meer?
        
        if use_tta:
            y_true, y_pred, cases = self.simple_test_time_inference(data_loader, n_augmentations)
        elif use_ensemble:
            y_true, y_pred, cases = self.simple_ensemble_inference(data_loader)
        
        #### elif:
            # Any custom inference that you want to apply
        
        else:
            y_true, y_pred, cases = self.simple_inference(data_loader)
            
            


        # DO NOT EDIT THIS PART


        y_true = functools.reduce(operator.iconcat, y_true, [])
        y_pred = functools.reduce(operator.iconcat, y_pred, [])
        cases = functools.reduce(operator.iconcat, cases, [])
        
        output_file = f"output/results_task1_team_{os.environ['Team_name']}_method_{self.i}.csv"
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'cases': cases})
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        self.i +=1
        return self.scoring(y_true, y_pred)

# Main execution
print(f"Starting the inference for the team: {os.environ['Team_name']}")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


X1, X2, y, cases = load_data('csv/df_task1_val_challenge_with_labels.csv', 'data/', num_samples=None)
dataset = ImagePairDataset(X1, X2, y, cases, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

model_paths = ['models/evolution_model_basis_final_phase.pth']
model_names = ["Evolution_model_basis"]
model_weights_contribution = [1]  # Example weights for the models
inference_task1 = InferenceTask1(model_paths, model_names, model_weights_contribution)


scores_1 = inference_task1.run(data_loader, use_tta=False, use_ensemble=False, n_augmentations=5)
print(f"Obtained scores for inference method 1: F1_score: {scores_1['F1_score']}, Rk-correlation: {scores_1['Rk-correlation']}, Specificity: {scores_1['Specificity']}")

# scores_2 = inference_task1.run(data_loader, use_tta=False, use_ensemble=False) 
# print(f"Obtained scores for inference method 2: F1_score: {scores_2['F1_score']}, Rk-correlation: {scores_2['Rk-correlation']}, Specificity: {scores_2['Specificity']}")
