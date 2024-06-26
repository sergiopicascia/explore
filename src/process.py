"""
Methods for patch processing.
"""

from itertools import combinations
import math
import torch
import numpy as np
from shapely import Polygon
import networkx as nx


def patch_embedding(data, processor, model, device):
    """Computes the embedding of the patches.

    Args:
        data: iterable containing data on images and annotations.
        processor: image processor.
        model: image embedding model.
        device: device where to allocate data.

    Returns:
        torch.Tensor: embeddings of the patches.
    """
    images_pixels = [image[0]["image"] for image in data]
    inputs = processor(images=images_pixels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    patch_embeddings = outputs.last_hidden_state[:, 1:, :]

    return patch_embeddings


def patch_labelling(data, labels, n_patches):
    """Assigns a label to each patch according to overlap with annotations.

    Args:
        data: iterable containing data on images and annotations.
        labels: categories present in the annotation.
        n_patches (_type_): number of patches in each image.

    Returns:
        list: labels assigned to patches for each image.
    """
    patches_labels = []
    for image in data:
        image_pixels = image[0]["image"]

        # Divide original image in patches
        row_steps = np.linspace(
            start=0,
            stop=image_pixels.shape[0] + 1,
            num=int(math.sqrt(n_patches)) + 1,
            dtype=int,
        )
        col_steps = np.linspace(
            start=0,
            stop=image_pixels.shape[1] + 1,
            num=int(math.sqrt(n_patches)) + 1,
            dtype=int,
        )

        image_patches_polygons = []
        for i in range(int(math.sqrt(n_patches))):
            for j in range(int(math.sqrt(n_patches))):
                # Generate a polygon of the patch
                image_patches_polygons.append(
                    Polygon.from_bounds(
                        col_steps[j],
                        row_steps[i],
                        col_steps[j + 1] - 1,
                        row_steps[i + 1] - 1,
                    )
                )

        # Generate polygons of the annotations
        annotations_polygons = [
            Polygon(
                np.array(annotation["segmentation"][0]).reshape(
                    (int(len(annotation["segmentation"][0]) / 2), 2)
                )
            )
            for annotation in image[0]["annotations"]
            if len(annotation["segmentation"][0]) > 4
        ]

        # Compute labels for patches
        image_patches_labels = []
        i = 0
        for _ in range(14):
            for _ in range(14):
                candidates = np.zeros((len(annotations_polygons),))
                for polygon_idx, polygon in enumerate(annotations_polygons):
                    candidates[polygon_idx] = (
                        polygon.buffer(0).intersection(image_patches_polygons[i]).area
                        / polygon.area
                    )
                if np.any(candidates):
                    candidate_idx = np.argmax(candidates)
                    annotation_idx = image[0]["annotations"][candidate_idx]["id"]
                    category_idx = image[0]["annotations"][candidate_idx]["category_id"]
                    image_patches_labels.append(
                        (
                            annotation_idx,
                            labels[category_idx]["name"],
                            labels[category_idx]["supercategory"],
                        )
                    )
                else:
                    image_patches_labels.append(("-", "Background", "Background"))
                i += 1

        patches_labels.append(image_patches_labels)

    return patches_labels


def generate_couples(
    patch_embeddings, patch_labels, n_couples, balance_labels, random_state
):
    """Generates couples of patches for each image.

    Args:
        patch_embeddings: embeddings of the patches.
        patch_labels: labels assigned to the patches.
        n_couples: maximum number of couples generated per image.
        balance_labels: if True, sample an equal number of couples for each label category.
        random_state: random seed.

    Returns:
        list: embeddings of the couples
        list: labels of the couples
    """
    rng = np.random.default_rng(random_state)

    couples_embeddings = []
    couples_labels = []
    for embeddings, labels in zip(patch_embeddings, patch_labels):
        candidate_couples = list(
            combinations(range(len(labels)), 2)
        )  # generate all possible couples
        candidate_couples_labels = np.zeros(
            len(candidate_couples)
        )  # assign label to each couple
        for j, (p1, p2) in enumerate(candidate_couples):
            if labels[p1] == labels[p2]:
                candidate_couples_labels[j] = 0
            elif labels[p1][1] == labels[p2][1]:
                candidate_couples_labels[j] = 1
            elif labels[p1][2] == labels[p2][2]:
                candidate_couples_labels[j] = 2
            else:
                candidate_couples_labels[j] = 3

        labels_unique, labels_counts = np.unique(
            candidate_couples_labels, return_counts=True
        )  # return label count
        if balance_labels:
            m_couples = (
                min(np.append(labels_counts, n_couples)) // labels_unique.size
            )  # minimum between label count and n_couples, divided by the number of unique labels
            for label in labels_unique:
                couples_indexes = rng.choice(
                    np.nonzero(candidate_couples_labels == label)[0],
                    size=m_couples,
                    replace=False,
                    shuffle=False,
                )
                for couple_index in couples_indexes:
                    couple = candidate_couples[couple_index]
                    couples_embeddings.append(
                        torch.cat([embeddings[couple[0]], embeddings[couple[1]]])
                    )
                    couples_labels.append(label)
        else:
            couples_indexes = rng.choice(
                len(candidate_couples_labels),
                size=n_couples,
                replace=False,
                shuffle=False,
            )
            for couple_index in couples_indexes:
                couple = candidate_couples[couple_index]
                couples_embeddings.append(
                    torch.cat([embeddings[couple[0]], embeddings[couple[1]]])
                )
                couples_labels.append(candidate_couples_labels[couple_index])

    return couples_embeddings, couples_labels


def draw_boxes(image_data, predictions, n_patches=196):
    side_patches = int(math.sqrt(n_patches))
    labels = torch.argmax(predictions, axis=1)
    patch_couples = list(combinations(range(n_patches), 2))
    graphs = []
    for i in torch.unique(labels):
        g = nx.Graph()
        g.add_nodes_from(range(n_patches))
        for j, couple in enumerate(patch_couples):
            if i >= labels[j]:
                g.add_edge(couple[0], couple[1])
        graphs.append(g)

    width = image_data[0]["image"].shape[0]
    height = image_data[0]["image"].shape[1]

    row_steps = np.linspace(
        start=0,
        stop=width + 1,
        num=side_patches + 1,
        dtype=int,
    )
    col_steps = np.linspace(
        start=0,
        stop=height + 1,
        num=side_patches + 1,
        dtype=int,
    )

    labels_square = np.array(range(n_patches)).reshape(side_patches, side_patches)

    boxes = []
    for community in nx.community.louvain_communities(graphs[0]):
        community_pixels = np.zeros((width, height))
        for i in range(side_patches):
            for j in range(side_patches):
                if labels_square[i][j] in community:
                    community_pixels[
                        row_steps[i] : row_steps[i + 1], col_steps[j] : col_steps[j + 1]
                    ] = 1

        x_min = np.unique(np.where(community_pixels == 1)[1])[0]
        x_max = np.unique(np.where(community_pixels == 1)[1])[-1]
        y_min = np.unique(np.where(community_pixels == 1)[0])[0]
        y_max = np.unique(np.where(community_pixels == 1)[0])[-1]
        boxes.append([x_min, y_min, x_max, y_max])

    return boxes
