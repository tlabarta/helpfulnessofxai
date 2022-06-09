import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def explain(model, img_pre_torch, labels, k, i=0):
    """
    :param preds: class predictions
    :param labels: class labels
    :param k: top k confidence scores to be returned
    :param i: image number, a convenience for naming purposes
    saves the output graph into
    :return: a list of tuples with axis 0 being confidence scores, axis 1 - corresponding class labels
    """
    # predict confidence_scores (i.e. probabilities)
    output = model.predict(img_pre_torch)
    predictions_tensor = torch.nn.functional.softmax(output[0], dim=0)

    # transform predictions to numpy array
    predictions = predictions_tensor.detach().cpu().numpy()
    predictions = np.squeeze(predictions)
    np.set_printoptions(formatter={'float_kind': '{:f}'.format})

    # sort array in descending order
    sorted_confidence_scores = np.sort(predictions)[::-1]
    sorted_confidence_scores = sorted_confidence_scores[0:k]

    # extract corresponding class labels
    ind = np.argsort(predictions)[::-1][0:k]
    sorted_predicted_labels = np.array([])
    for j in range(0, k):
        str_label = str(ind[j])
        sorted_predicted_labels = np.append(sorted_predicted_labels, labels[str_label][1])

    # list of scores and class labels
    confidence_scores = np.vstack((sorted_confidence_scores, sorted_predicted_labels)).T

    # labels for the questionaire
    sorted_predicted_labels = ["1st most probable class\n (the predicted class)", "2nd most probable class", "3rd most probable class"]
    

    plt.figure(figsize=(8, 4))
    splot=sns.barplot(x=sorted_confidence_scores[::-1]*100,y=sorted_predicted_labels[::-1], color="blue")
    plt.xlim(0, 100)
    plt.xlabel("AI's confidence for the given classes \n in %", size=20)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.bar_label(splot.containers[0], labels=[f"{round(score, 1)}%" if round(score, 1) >= 0.1 else "smaller than 0.1%" for score in sorted_confidence_scores[::-1]*100], size=13)
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()

    return fig