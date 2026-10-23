import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision.models import resnet18, ResNet18_Weights

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

class HybridQXAIModel(nn.Module):
    def __init__(self):
        super(HybridQXAIModel, self).__init__()
        base_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # --- TRANSFER LEARNING: FREEZE RESNET LAYERS ---
        # Ithu konjam images (12 photos) vechu train pannum pothu 
        # pre-trained weights-ah protect pannum, elathayum 'Deforested' nu predict panrathu thavirkkum.
        for param in base_resnet.parameters():
            param.requires_grad = False
            
        self.feature_extractor = nn.Sequential(*list(base_resnet.children())[:-2])
        self.fc_reduce = nn.Linear(512 * 7 * 7, 4)

        qc = QuantumCircuit(4)
        feature_map = ZZFeatureMap(feature_dimension=4, reps=1)
        ansatz = RealAmplitudes(num_qubits=4, reps=1)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = SamplerQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )

        self.quantum_layer = TorchConnector(qnn)
        self.classifier = nn.Linear(16, 2)

    def forward(self, x, noise_enabled=False, zne_enabled=True):
        features = self.feature_extractor(x)
        flat_features = torch.flatten(features, 1)
        q_input = self.fc_reduce(flat_features)
        q_input = torch.sigmoid(q_input) * np.pi

        q_out = self.quantum_layer(q_input)

        if noise_enabled:
            if zne_enabled:
                q_out = q_out * 0.94 + torch.randn_like(q_out) * 0.02
            else:
                q_out = q_out * 0.70 + torch.randn_like(q_out) * 0.15

        out = self.classifier(q_out)
        return out, features

def load_trained_model(weights_path="qxai_model_weights.pth"):
    model = HybridQXAIModel()
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        print("✅ Loaded trained model weights successfully!")
    else:
        print("⚠️ Model weights not found! Running initialized base model with frozen feature extractor.")
    model.eval()
    return model

def generate_qxai_heatmap(original_img_np, feature_maps):
    heatmap = torch.mean(feature_maps, dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    heatmap_resized = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
    high_intensity_pixels = np.sum(heatmap_resized > 0.60)
    deforested_ratio = high_intensity_pixels / heatmap_resized.size

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img_np, 0.55, heatmap_colored, 0.45, 0)
    return overlay, deforested_ratio

def calculate_carbon_loss(deforested_ratio, total_area_sq_km=100.0):
    hectares = (total_area_sq_km * deforested_ratio) * 100
    estimated_co2_loss_tons = hectares * 150.0
    return hectares, estimated_co2_loss_tons

def process_sar_fusion(optical_img_np):
    gray = cv2.cvtColor(optical_img_np, cv2.COLOR_RGB2GRAY)
    sar_simulated = cv2.Laplacian(gray, cv2.CV_64F)
    sar_simulated = cv2.convertScaleAbs(sar_simulated)
    return cv2.applyColorMap(sar_simulated, cv2.COLORMAP_VIRIDIS)