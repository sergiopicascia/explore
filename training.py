"""
Training a patch classifier.
"""

from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
import torch
from torch import nn, optim
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
data = PartImageNetDataset(root=PATH, split="train", shuffle=True, random_state=42)
processor = AutoImageProcessor.from_pretrained(PROCESSOR)
model = AutoModel.from_pretrained(MODEL).to(DEVICE)

# Data processing
X_train = []
y_train = []
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

    X_train.extend(couples_embeddings)
    y_train.extend(couples_labels)

X_train = torch.stack(X_train, dim=0)
y_train = torch.Tensor(y_train).reshape(-1, 1)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
    torch.Tensor(y_train)
)
y_train = torch.Tensor(ohe.transform(y_train)).to(DEVICE)


# Model training
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
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf.parameters(), lr=0.001)

N_EPOCHS = 100
BATCH_SIZE = 64
for epoch in range(N_EPOCHS):
    for i in range(0, len(X_train), BATCH_SIZE):
        X_batch = X_train[i : i + BATCH_SIZE]
        y_pred = clf(X_batch)
        y_batch = y_train[i : i + BATCH_SIZE]
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    y_pred = clf(X_train)
    ce = loss_fn(y_pred, y_train)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_train, 1)).float().mean()
    print(f"Finished epoch {epoch}, loss {ce}, accuracy {acc}")

torch.save(clf.state_dict(), "./model/patch-classifier-parameters.pt")
