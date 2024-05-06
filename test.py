"""
Testing the patch classifier.
"""

from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
import torch
from torch import nn
import torchvision
from sklearn.preprocessing import OneHotEncoder

from src.utils import batched
from src.dataset import PartImageNetDataset
from src.process import patch_embedding, patch_labelling, generate_couples, draw_boxes

# Constants definition
PATH = "/Users/sergiopicascia/Documents/University/Research/Datasets/Object Part Detection/PartImageNet"
PROCESSOR = "facebook/dino-vitb16"
MODEL = "facebook/dino-vitb16"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

BATCH_SIZE = 128
N_COUPLES = 19110
N_LABELS = 4
COUPLE_SIZE = 1536

# Loading data and models
data = PartImageNetDataset(root=PATH, split="test", shuffle=True, random_state=42)
processor = AutoImageProcessor.from_pretrained(PROCESSOR)
model = AutoModel.from_pretrained(MODEL).to(DEVICE)


# Model testing
class PatchClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(COUPLE_SIZE, 512)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(512, 256)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(256, N_LABELS)
        # self.hidden3 = nn.Linear(256, N_LABELS)
        # self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.output(x)
        # x = self.output(self.hidden3(x))
        return x


clf = PatchClassifier().to(DEVICE)
clf.load_state_dict(
    torch.load("./model/patch-classifier-parameters.pt", map_location=DEVICE)
)
clf.eval()

THRESHOLD = 0.5
tp = 0
fn = 0
fp = 0

# Data processing
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
        balance_labels=False,
        random_state=42,
    )

    X_test = couples_embeddings
    y_test = couples_labels

    X_test = torch.stack(X_test, dim=0)
    y_test = torch.Tensor(y_test).reshape(-1, 1)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        torch.Tensor(y_test)
    )
    y_test = torch.Tensor(ohe.transform(y_test)).to(DEVICE)

    y_pred = clf(X_test)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
    print("Accuracy:", acc)

    for i, img in enumerate(batch):
        y_pred_boxes = draw_boxes(img, y_pred[i])
        y_true_boxes = [a["bbox"] for a in img["annotations"]]
        iou = torchvision.ops.box_iou(y_true_boxes, y_pred_boxes)
        for row in iou:
            if torch.any(row >= THRESHOLD):
                tp += 1
            else:
                fn += 1

        for col in iou.T:
            if not torch.any(col >= THRESHOLD):
                fp += 1

    print(f"Precision: {tp/(tp+fp)}, Recall: {tp/(tp+fn)}")
