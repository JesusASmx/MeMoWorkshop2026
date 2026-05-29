from experiment_setting import skel
from data_curation import simpletext
import shutil
import pandas as pd
from datasets import Dataset, DatasetDict
import torch

import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from GPT2_with_memo import MeMoGPT2
#from MeMo_experiments import MEMO_run
from Vanilla_experiments import VANILLA_run
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

#from MeMoPyTorch.modelling_memo import MeMo
#from MeMoPyTorch.modelling_memo_tokenizer_original import MeMoTokenizer


def distribution_from_hidden(rep):
    acts = torch.abs(rep)
    probs = acts / acts.sum(dim=-1, keepdim=True)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    mean_entropy = entropy.mean()
    max_entropy = torch.log(
        torch.tensor(rep.shape[-1], dtype=torch.float32)
    )
    return {
        "mean_entropy": mean_entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": mean_entropy / max_entropy
    }

def PCA_plot(X, labels, title_graph="GPT2 Representation PCA", save_path = "gpt2_pca_plot.png"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print("PCA SHAPE:", X_pca.shape)

    unique_labels = sorted(list(set(labels)))

    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_to_id[x] for x in labels]
        
    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=numeric_labels
    )

    texts = []
    for i, txt in enumerate(labels):
        text = plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]), fontsize=9)
        texts.append(text)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray',lw=0.5))

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title_graph)
    handles, _ = scatter.legend_elements()
    plt.legend(handles, unique_labels, title="Semantic Class")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

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

def mech_interp(rep, eval_labels, plot_path = "./FIGURES/gpt2_pca_plot.png", model=None, inputs=None, original_logits=None, topk_k=30, ablate_k=10):
    
    ### ACTIVATION ENTROPY. 
    entropy = distribution_from_hidden(rep) ### THIS IS THE VALUE.
    print(f"ACTIVATION ENTROPY OF THE LAST HIDDEN STATE:\n {entropy}")

    ### PCA & TOP-K BROS.  
    #PCA:
    X = rep.cpu().numpy()
    PCA_plot(X, eval_labels, title_graph="GPT2 Representation PCA", save_path = plot_path)
    sil_score = silhouette_score(X, eval_labels)
    print(f"SILLOUHETE SCORE: {sil_score}")
    #TOP-K Bros:
    topk = top_k_bros(rep, k = topk_k)
    important_neurons = topk.tolist()
    print("\nTOP NEURONS:")
    for x in important_neurons:
        print(x)
    
    ### NEURON ABLATION
    if model is not None and inputs is not None and original_logits is not None:
        print(f"\nABLATING TOP-{ablate_k} NEURONS...")

        neurons_to_ablate = important_neurons[:ablate_k]
        rep_ablated = rep.clone()
        rep_ablated[:, neurons_to_ablate] = 0.0 ## ABLATION ITSELF
        all_ablated_logits = []
        with torch.no_grad():
            for batch_idx, inputs in enumerate(all_inputs):
                inputs = {
                    k: v.to(device)
                    for k, v in inputs.items()
                }
                current_batch_size = inputs["input_ids"].shape[0]

                start = batch_idx * batch_size
                end = start + current_batch_size

                rep_batch = rep_ablated[start:end].to(device)

                def ablation_hook(module, module_input, module_output):
                # GPT2 blocks return tuple(hidden_states, ...)
                    if isinstance(module_output, tuple):
                        hidden_states = module_output[0]
                        hidden_states[:, -1, :] = rep_batch
                        return (hidden_states,) + module_output[1:]
                    else:
                        module_output[:, -1, :] = rep_batch
                        return module_output

                hook = model.transformer.h[-1].register_forward_hook(
                    ablation_hook
                )
                outputs_ablated = model(
                    **inputs,
                    return_dict=True
                )
                hook.remove()
                all_ablated_logits.append(
                    outputs_ablated.logits.cpu()
                )

        ablated_logits = torch.cat(
            all_ablated_logits,
            dim=0
        ).to(device)
        logit_shift = torch.mean(
            torch.abs(original_logits - ablated_logits)
        ).item()
        print(f"\nMEAN LOGIT SHIFT AFTER ABLATION: {logit_shift}")
        orig_probs = torch.softmax(
            original_logits,
            dim=-1
        )
        ablated_probs = torch.softmax(
            ablated_logits,
            dim=-1
        )
        kl = torch.nn.functional.kl_div(
            ablated_probs.log(),
            orig_probs,
            reduction="batchmean"
        ).item()
        print(f"KL DIVERGENCE AFTER ABLATION: {kl}")

        return {
            "entropy": entropy,
            "silhouette_score": sil_score,
            "important_neurons": important_neurons,
            "logit_shift": logit_shift,
            "kl_divergence": kl
        }
    
    else:
        return {
            "entropy": entropy,
            "silhouette_score": sil_score,
            "important_neurons": important_neurons
        }




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':

    batch_size = 8
    number_of_epochs = 4

    toy_dataset = simpletext(multilabel=False, limit=1000) #1000
    eval_toy = simpletext(multilabel=False, ablated=25) ##EVAL DATA FOR MECH INTERP:

    MNAME = "GPT2_simplestories_4epochs_lite"
    GPT2_NAME = "CausalNLP/gpt2-hf_multilingual-90" #"erwanf/gpt2-mini"

    eval_texts, eval_labels = VANILLA_run(toy_dataset, eval_toy, gpt2name=GPT2_NAME, save=MNAME)

    model = AutoModelForSequenceClassification.from_pretrained(MNAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MNAME)


    #### MECH INTERP VANILLA:
    print("########################### MECH INTERP VANILLA ###########################\n\n")

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 8

    eval_texts = eval_toy["text"].tolist()
    eval_labels = eval_toy["labels"].tolist()
    all_hidden_states = []
    all_logits = []
    all_inputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_texts), batch_size), desc="Evaluation Batch"):
            batch_texts = eval_texts[i:i + batch_size]
            
            the_inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True
            ).to(device)

            the_outputs = model(
                **the_inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            stored_inputs = {
                    k: v.to(device)
                    for k, v in the_inputs.items()
                }
            all_inputs.append(stored_inputs)

            all_logits.append(the_outputs.logits.cpu())
            last_hidden = the_outputs.hidden_states[-1].cpu()
            all_hidden_states.append(last_hidden)
            
            del the_inputs, the_outputs
            torch.cuda.empty_cache()

    all_logits = torch.cat(all_logits, dim=0).to(device)
    all_hidden_states = torch.cat(all_hidden_states, dim=0).to(device)

    rep = all_hidden_states[:, -1, :]

    results = mech_interp(
        rep,
        eval_labels,
        model=model,
        inputs=all_inputs,
        original_logits=all_logits,
        topk_k=30,
        ablate_k=10
    )


    ########################## MEMO:

    GPT2_NAME = "CausalNLP/gpt2-hf_multilingual-90" #"erwanf/gpt2-mini"
    
    memo = skel(MNAME = GPT2_NAME, MeMo=True, raw_dataset=[toy_dataset, 5, "multi_label_classification"])
    metokenizer, memodel = memo.run(bsize = batch_size, epochs = 4)# number_of_epochs)

    #### MECH INTERP:
    memodel.eval()
    metokenizer.pad_token = metokenizer.eos_token

    batch_size = 8

    eval_texts = eval_toy["text"].tolist()
    eval_labels = eval_toy["labels"].tolist()
    all_hidden_states = []
    all_logits = []
    all_inputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_texts), batch_size), desc="Evaluation Batch"):
            batch_texts = eval_texts[i:i + batch_size]
            
            the_inputs = metokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True
            ).to(device)

            the_inputs.pop('token_type_ids')

            the_outputs = memodel(
                **the_inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            stored_inputs = {
                    k: v.to(device)
                    for k, v in the_inputs.items()
                }
            all_inputs.append(stored_inputs)

            all_logits.append(the_outputs["logits"].cpu())
            last_hidden = the_outputs["hidden_states"][-1].cpu()
            all_hidden_states.append(last_hidden)
            
            del the_inputs, the_outputs
            torch.cuda.empty_cache()

    all_logits = torch.cat(all_logits, dim=0).to(device)
    all_hidden_states = torch.cat(all_hidden_states, dim=0).to(device)

    rep = all_hidden_states[:, -1, :]

    results = mech_interp(
        rep,
        eval_labels,
        plot_path = "./FIGURES/gpt2MeMo_pca_plot.png",
        model=model,
        inputs=all_inputs,
        original_logits=all_logits,
        topk_k=30,
        ablate_k=10
    )