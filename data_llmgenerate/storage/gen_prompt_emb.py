import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model, Qwen2Model, Qwen2Tokenizer, AutoTokenizer, AutoModel

class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path = 'FRED',
        model_name = "qwen",
        device = 'cuda:0',
        input_len = 96,
        d_model = 768,
        layer = 12,
        divide = 'train'
    ):  
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len =  input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.len = self.input_len-1
        
        local_model_path = "Qwen3-1.7B"
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(local_model_path, local_files_only=True).to(self.device)

    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):
        values = in_data[i, :, j].flatten().tolist()
        values_str = ", ".join([str(int(value)) for value in values])

        trends = torch.sum(torch.diff(in_data[i, :, j].flatten()))
        trends_str = f"{trends.item():0f}"
        
        if self.data_path in ['G56_high']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:00"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:50"
        else: 
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:00"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:55"

        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("[t1]", start_date).replace("[t2]", end_date).replace("[id]", str(j+1))

        tokenized_prompt = self.tokenizer.encode(in_prompt, return_tensors="pt").to(self.device)
        return tokenized_prompt

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.model(tokenized_prompt).last_hidden_state
        return prompt_embeddings

    def generate_embeddings(self, in_data, in_data_mark):
            input_templates = {
                'G56_high': "[SCENARIO] Please generate the overload trafic flow series based on the normal traffc flow data at the current period. [TIME INTERVALS] Time intervals: [t1] to [t2], at 10-minute intervals. [NORMAL TRAFFIC FLOW] Traffc flow: Node-[id]: value1, ..., valuen, valuen every 10 minutes. ",
                'G60_high': "[SCENARIO] Please generate the overload trafic flow series based on the normal traffc flow data at the current period. [TIME INTERVALS] Time intervals: [t1] to [t2], at 5-minute intervals. [NORMAL TRAFFIC FLOW] Traffc flow: Node-[id]: value1, ..., valuen, valuen every 5 minutes. ",
            }
            
            input_template = input_templates.get(self.data_path, input_templates['G56_high'])
            
            tokenized_prompts = []
            max_token_count = 0
            for i in range(len(in_data)):
                for j in range(in_data.shape[2]):
                    tokenized_prompt = self._prepare_prompt(input_template, in_data, in_data_mark, i, j).to(self.device)
                    max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                    tokenized_prompts.append((i, tokenized_prompt.to(self.device), j))

            in_prompt_emb = torch.zeros((len(in_data), max_token_count, self.d_model, in_data.shape[2]), dtype=torch.float32, device=self.device)

            for i, tokenized_prompt, j in tokenized_prompts:
                prompt_embeddings = self.forward(tokenized_prompt)
                padding_length = max_token_count - tokenized_prompt.shape[1]
                if padding_length > 0:
                    last_token_embedding = prompt_embeddings[:, -1, :].unsqueeze(1)
                    padding = last_token_embedding.repeat(1, padding_length, 1)
                    prompt_embeddings_padded = torch.cat([prompt_embeddings, padding], dim=1)
                else:
                    prompt_embeddings_padded = prompt_embeddings
                        
                in_prompt_emb[i, :max_token_count, :, j] = prompt_embeddings_padded
                last_token_emb = in_prompt_emb[:, max_token_count-1:max_token_count, :, :]
                last_token_emb = last_token_emb.squeeze()

            return last_token_emb