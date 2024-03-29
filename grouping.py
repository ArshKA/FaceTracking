import numpy as np

def group_bbox(frame_dict, labels):
    result_dict = {label: {} for label in np.unique(labels) if label >= 0}

    for frame_id, bbox_list in frame_dict.items():
        for label_idx, bbox in bbox_list:
            if label_idx >= 0:
                result_dict[labels[label_idx]][frame_id] = bbox

    return result_dict


def group_probabilities(probs, labels):
    if len(labels) != len(probs):
        raise ValueError(f'Dimentions of labels {labels.shape} does not match probabilities {probs.shape}')

    labels_count = max(labels) + 1
    class_probabilities = np.zeros((labels_count, probs.shape[1]))
    counts = np.zeros(labels_count)

    for i in range(len(labels)):
        class_probabilities[labels[i]] += probs[i]
        counts[labels[i]] += 1

    averages = class_probabilities / np.expand_dims(counts, axis=1)
    return averages


def process_predictions(predictions, attributes, threshold=0.5):

    processed_predictions = {}

    for label_id, label_predictions in enumerate(predictions):
        processed_predictions[label_id] = []

        for prediction_id, prob in enumerate(label_predictions):
            if prob > threshold:
                processed_predictions[label_id].append(attributes[prediction_id])

    return processed_predictions