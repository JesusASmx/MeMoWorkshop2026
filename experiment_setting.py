from datasets import Dataset, DatasetDict, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

from GPT2_with_memo import MeMoGPT2

from MeMoPyTorch.modelling_memo_tokenizer import MeMoTokenizer

import numpy as np

class skel():
    def __init__(self, raw_dataset:list, MNAME = "CausalNLP/gpt2-hf_multilingual-90", MeMo=False):
        
        self.raw_dataset = raw_dataset[0]
        self.num_labels = raw_dataset[1]
        self.problem_type = raw_dataset[2]

        self.tokenizer = MeMoTokenizer.from_pretrained(MNAME, 
                                                    padding_side='left', truncation_side='left', 
                                                    max_length=4096, head_number=4) #4096
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id

        if not MeMo:
            self.model = AutoModelForSequenceClassification.from_pretrained(MNAME, problem_type=self.problem_type, num_labels=self.num_labels, ignore_mismatched_sizes=True)
        
        else:
            self.model = MeMoGPT2(MNAME = MNAME, problem_type=self.problem_type, num_labels=self.num_labels, other_tokenizer=self.tokenizer)
            
            #for param in self.model.MeMo_projection.parameters():
            #    param.requires_grad = False  ### WE DON'T WANT TO CHANGE THE MEMORIES WITH THE TRAINING XD.


    def tokenize_function(self, example):
        return self.tokenizer(
                example["text"], 
                padding="max_length",
                max_length=512,
                truncation=True
            )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred

        if self.problem_type == "single_label_classification":
            preds = logits.argmax(axis=1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro")
            return {"accuracy": acc, "macro_f1": f1}
        
        else:
            probs = 1 / (1 + np.exp(-logits))
            preds = (probs >= 0.5).astype(int)
            macro_f1 = f1_score(labels, preds, average="macro")
            micro_f1 = f1_score(labels, preds, average="micro")
            subset_acc = accuracy_score(labels, preds)
            return {"macro_f1": macro_f1, "micro_f1": micro_f1, "subset_accuracy": subset_acc }



    def run(self, bsize = 4, epochs = 1, save=False):
        tokenized_datasets = self.raw_dataset.map(self.tokenize_function, batched=True)#.to('cpu')
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)


        training_args = TrainingArguments(
            #use_cpu=True,
            output_dir="./output",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            learning_rate=2e-5,
            per_device_train_batch_size=bsize,
            per_device_eval_batch_size=bsize,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics
            )

        trainer.train()

        metrics = trainer.evaluate()
        print(f"\n        #######################################################################################\n        ###                                  FINAL METRICS                                  ###\n        #######################################################################################\n")
        for x in metrics:
            print("*) "+x+":", metrics[x])
        print("\n\n")

        if save:
            trainer.save_model(save)
            self.tokenizer.save_pretrained(save)

        return self.tokenizer, self.model