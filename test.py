"""
Testing the patch classifier.
"""

from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder

from src.utils import batched
from src.dataset import PartImageNetDataset
from src.process import patch_embedding, patch_labelling, generate_couples

# Constants definition
PATH = "./PartImageNet"
PROCESSOR = "facebook/dino-vitb16"
MODEL = "facebook/dino-vitb16"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

BATCH_SIZE = 128
N_COUPLES = 256
N_LABELS = 4
COUPLE_SIZE = 1536

# Loading data and models
data = PartImageNetDataset(root=PATH, split="test", shuffle=True, random_state=42)
processor = AutoImageProcessor.from_pretrained(PROCESSOR)
model = AutoModel.from_pretrained(MODEL).to(DEVICE)
print(DEVICE)

# Data processing
X_test = []
y_test = []
for batch in tqdm(batched(data, BATCH_SIZE)):
    patch_embeddings = patch_embedding(
        data=batch, processor=processor, model=model, device=DEVICE
    )
    patch_labels = patch_labelling(
        data=batch, labels=data.categories, n_patches=patch_embeddings.shape[1]
    )
    couples_embeddings, couples_labels = generate_couples(
        patch_embeddings=patch_embeddings,
        patch_labels=patch_labels,
        n_couples=N_COUPLES,
        random_state=42,
    )

    X_test.extend(couples_embeddings)
    y_test.extend(couples_labels)
    break

X_test = torch.stack(X_test, dim=0)
y_test = torch.Tensor(y_test).reshape(-1, 1)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
    torch.Tensor(y_test)
)
y_test = torch.Tensor(ohe.transform(y_test)).to(DEVICE)

# Model testing
class PatchClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(COUPLE_SIZE, 512)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(256, N_LABELS)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.output(x)
        return x

clf = PatchClassifier().to(DEVICE)
clf.load_state_dict(torch.load('patch-classifier-parameters.pt'))
clf.eval()
y_pred = clf(X_test)
acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
print("Accuracy:", acc)

with open('./result.txt', 'w') as f:
    f.write(str(acc))
