from experiment_setting import skel
from data_curation import simpletext, toxicity, fact_checking
import shutil
import pandas as pd
from datasets import Dataset, DatasetDict
import torch

import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from GPT2_with_memo import MeMoGPT2


def distribution_from_hidden(rep):
    acts = torch.abs(rep)
    probs = acts / acts.sum(dim=-1, keepdim=True)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    mean_entropy = entropy.mean()
    return mean_entropy

def PCA_plot(X, labels, title_graph="GPT2 Representation PCA", save_path = "gpt2_pca_plot.png"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print("PCA SHAPE:", X_pca.shape)

    unique_labels = sorted(list(set(labels)))

    label_to_id = {
        label: i for i, label in enumerate(unique_labels)
    }

    numeric_labels = [
        label_to_id[x]
        for x in labels
    ]
    
    
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=numeric_labels
    )

    texts = []

    for i, txt in enumerate(labels):
        text = plt.annotate(
            txt,
            (X_pca[i, 0], X_pca[i, 1]),
            fontsize=9
        )
        texts.append(text)

# Automatically move labels to avoid overlap
    adjust_text(
        texts,
        arrowprops=dict(
            arrowstyle='-',
            color='lightgray',
            lw=0.5
        )
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title(title_graph)

    handles, _ = scatter.legend_elements()

    plt.legend(
        handles,
        unique_labels,
        title="Semantic Class"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved PCA plot to: {save_path}")


def top_k_bros(rep, k = 50):
    topk = torch.topk(torch.abs(rep), k=k, dim=-1)
    mask = torch.zeros_like(rep)
    mask.scatter_(1, topk.indices, 1)
    sparsity = mask.mean()
    return topk.indices


important_neurons = 0
def ablate_neurons(module, inp, output):
    output = output.clone()
    output[:, :, important_neurons] = 0.0

    return output


def get_representations(dataloader):
    reps = []
    labels = []
    for batch in dataloader:
        outputs = model(
            batch["input_ids"],
            output_hidden_states=True
        )
        rep = outputs.hidden_states[-1][:, -1, :]
        reps.append(rep.detach().cpu())
        labels.extend(batch["label"])








from transformers import AutoModelForSequenceClassification

def VANILLA_run(toy_dataset, eval_toy, gpt2name = "CausalNLP/gpt2-hf_multilingual-90", save=False):

    batch_size = 8
    number_of_epochs = 4
    
    eval_texts = eval_toy["text"].tolist()
    eval_labels = eval_toy["labels"].tolist()

    try:
        model = AutoModelForSequenceClassification.from_pretrained(save)
        print(f"##### CHECKPOINT {save} ALREADY FOUND SO GPT2 WON'T BE FINETUNNED (again) #####")
    except:
        pristine = skel(MNAME = gpt2name, raw_dataset=[toy_dataset, 5, "multi_label_classification"])
        tokenizer, model = pristine.run(bsize = batch_size, epochs = number_of_epochs, save=save)

    return eval_texts, eval_labels