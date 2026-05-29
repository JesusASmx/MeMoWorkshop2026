import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import GPT2ForSequenceClassification

from MeMoPyTorch.modelling_memo import MeMo
from MeMoPyTorch.modelling_memo_tokenizer import MeMoTokenizer


class MeMoGPT2(nn.Module):

    def __init__(self, MNAME, problem_type, num_labels=2, other_tokenizer=0):
        super().__init__()

        self.gpt2 = GPT2ForSequenceClassification.from_pretrained(MNAME, problem_type=problem_type, num_labels=num_labels, ignore_mismatched_sizes=True)
        self.gpt2.config.pad_token_id = (
            self.gpt2.config.eos_token_id
        )

        hidden = self.gpt2.config.hidden_size


        self.tok = other_tokenizer

        self.MEMO_tokenizer = other_tokenizer
        
        #MeMoTokenizer.from_pretrained(MNAME, #"EleutherAI/gpt-neox-20b", 
                                                    #padding_side='left', truncation_side='left', 
                                                    #max_length=4096, head_number=4)
        
        #self.MEMO_tokenizer.pad_token = self.MEMO_tokenizer.eos_token
        #self.MEMO_tokenizer.pad_token_id = self.MEMO_tokenizer.pad_token_id

        self.MeModel = MeMo(inner_dim=2048, 
            num_of_heads=4,
            num_of_layers=3,
            chunk_length=4096, #4096,
            num_embeddings=self.MEMO_tokenizer.vocab_size, 
            padding_idx=self.MEMO_tokenizer.pad_token_id, 
            device="cuda")
        
        #train_texts = data_to_memorize["train"]["text"]
        #for text in tqdm(train_texts, desc="MeMorizing train texts"):
        #    encoded = MEMO_tokenizer.get_text_batch_encoding([text])
        #    MeModel.memorize_text(encoded)

        #memory_vector = MeModel.get_last_layer()
        #self.memory_tensor = (memory_vector.CMM.weight.detach().mean(dim=0))
        
        self.MeMo_projection = nn.Linear(
            2048, #self.memory_tensor.shape[-1], inner_dm == 2048
            hidden
        )



    def reconstruct(self, input_ids, attention_mask):
        reconstructed = []

        for ids, mask in zip(input_ids, attention_mask):
            valid_ids = ids[mask == 1]
            text = self.tok.decode(valid_ids, skip_special_tokens=True)
            reconstructed.append(text)

        return reconstructed
    


    def forward(self, input_ids, attention_mask, labels=None, output_hidden_states=False, return_dict=True):

        batch_size = input_ids.size(0)
        token_embeds = self.gpt2.transformer.wte(input_ids)

        reconstr_text = self.reconstruct(input_ids=input_ids, attention_mask=attention_mask) ## Not the cleanest way but my blood is 150% coffe and its 5 am. ChatGPT Is clueless abt "how2 deez sheet" and I just want to paint another miniature while waiting for another run.

        for text in reconstr_text: #tqdm(reconstr_text, desc="Text per batch"):
            encoded = self.MEMO_tokenizer.get_text_batch_encoding([text])
            self.MeModel.memorize_text(encoded)
        
        memory_vector = self.MeModel.get_last_layer()
        self.memory_tensor = (memory_vector.CMM.weight.detach().mean(dim=0))
        
        prefix = self.MeMo_projection(self.memory_tensor)

        prefix = prefix.unsqueeze(0).expand(batch_size,-1)
        prefix = prefix.unsqueeze(1)

        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        prefix_mask = torch.ones((batch_size, 1), device=attention_mask.device)

        atn_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.gpt2(
                inputs_embeds=inputs_embeds,
                attention_mask=atn_mask,
                labels=labels,
                use_cache=False,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        
        if output_hidden_states:
            return {
                "loss": outputs.loss,
                "logits": outputs.logits,
                "hidden_states": outputs.hidden_states
            }

        else:
            return {
                "loss": outputs.loss,
                "logits": outputs.logits,
            }