from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

class PrescriptionDataset(Dataset):
    def __init__(self, prescriptions, num_herbs, num_symptoms):
        self.prescriptions = prescriptions
        self.num_herbs = num_herbs
        self.num_symptoms = num_symptoms

    def __len__(self):
        return len(self.prescriptions)

    def __getitem__(self, idx):
        p = self.prescriptions[idx]
        # 症状输入 (多标签one-hot)
        symptoms = torch.zeros(self.num_symptoms, dtype=torch.int)
        symptoms[p['symptoms']] = 1
        # 药材标签 (多标签one-hot)
        herbs = torch.zeros(self.num_herbs, dtype=torch.float)
        herbs[p['herbs']] = 1.0
        return {'symptoms': symptoms, 'herbs': herbs}