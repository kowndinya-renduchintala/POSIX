import copy
from typing import Tuple

import torch
from torch.nn.utils import rnn
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import LanguageModel

IGNORE_INDEX=-100

class HFModel(LanguageModel):
    def __init__(self, model_name_or_path: str, **kwargs):
        self.model_name_or_path=model_name_or_path
        self.tokenizer=AutoTokenizer.from_pretrained(model_name_or_path)
        self.model=AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)

        # define pad_token for tokenizer if it is not set
        if self.tokenizer.pad_token:
            print(f"Padding token already set to {self.tokenizer.pad_token}")
        elif self.tokenizer.unk_token:
            print(f"Setting pad token to {self.tokenizer.unk_token}")
            self.tokenizer.pad_token=self.tokenizer.unk_token
        elif self.tokenizer.eos_token:
            print(f"Setting pad token to {self.tokenizer.eos_token}")
            self.tokenizer.pad_token=self.tokenizer.eos_token
        else:
            print(f"Adding special token <|pad|> as pad token")
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    def get_responses(self, prompts: list[str], batched=False, **kwargs) -> Tuple[list[list[int]], list[str]]:
        response_tokens=[]
        responses=[]
        if batched:
            response_tokens, responses=self._get_responses_batched(prompts, **kwargs)
        else:
            N=len(prompts)
            for prompt in prompts:
                resp_tokens, resp=self._get_response(prompt, **kwargs)
                response_tokens.append(resp_tokens)
                responses.append(resp)
        return response_tokens, responses
    
    def _get_response(self, prompt: str, **kwargs) -> Tuple[list[int], str]:
        tokenized_prompt=self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized_prompt.input_ids.to(self.model.device)
        attention_mask=tokenized_prompt.attention_mask.to(self.model.device)
        N_prompt=input_ids.shape[1]
        output=self.model.generate(input_ids=input_ids, attention_mask=attention_mask, pad_token_id=self.tokenizer.pad_token_id, **kwargs)
        response_tokens=output[0][N_prompt:]
        response=self.tokenizer.decode(response_tokens, skip_special_tokens=False)
        return response_tokens.tolist(), response
    
    def _get_responses_batched(self, prompts: list[str], **kwargs) -> Tuple[list[list[int]], list[str]]:
        tokenized_prompts=self.tokenizer(prompts)
        input_ids, attention_mask=tuple([torch.tensor(tokenized_prompt_feature[::-1]) for tokenized_prompt_feature in tokenized_prompts[key]] for key in ["input_ids", "attention_mask"])
        input_ids=rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(dims=[1])
        attention_mask=rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0).flip(dims=[1])
        N_padded=input_ids.shape[1]
        generated_tokens=self.model.generate(
            input_ids=input_ids.to(self.model.device), 
            attention_mask=attention_mask.to(self.model.device), 
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
        )
        generated_tokens=generated_tokens[:, N_padded:]
        responses=self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        return generated_tokens.tolist(), responses

    def compute_log_probabilties(self, prompts: list[str], responses: list[list[int]], batched=False) -> list[float]:
        log_probs=[]
        if batched:
            log_probs=self._compute_log_probabilities_batched(prompts, responses)
        else:
            N=len(prompts)
            for i in range(N):
                log_prob=self._compute_log_probability(prompts[i], responses[i])
                log_probs.append(log_prob)
        return log_probs
    
    def _compute_log_probability(self, prompt: str, response_tokens: list[int]) -> float:
        prompt_tokens=self.tokenizer(prompt).input_ids
        prompt_response_tokens=prompt_tokens+response_tokens
        N_prompt=len(prompt_tokens)
        N_prompt_response=len(prompt_response_tokens)

        input_ids = torch.tensor([prompt_response_tokens]).to(self.model.device)

        with torch.no_grad():
            logits=self.model(input_ids).logits
        
        response_logits=logits[:, N_prompt-1:N_prompt_response-1,:]
        response_probs=torch.softmax(response_logits, dim=-1)

        log_response_probs=torch.log(response_probs)
        final_logprob=0.0
        for k in range(len(response_tokens)):
            final_logprob+=log_response_probs[0,k,response_tokens[k]].item()

        return final_logprob
    
    def _compute_log_probabilities_batched(self, prompts: list[str], responses: list[list[int]]) -> list[float]:
        tokenized_prompts=self.tokenizer(prompts)
        tokenized_responses={"input_ids": responses, "attention_mask": [[1]*len(response) for response in responses]}
        tokenized_prompt_responses={key: [p+r for p, r in zip(tokenized_prompts[key], tokenized_responses[key])] for key in ["input_ids", "attention_mask"]}
        all_labels=copy.deepcopy(tokenized_prompt_responses["input_ids"])
        prompt_lengths=[len(tokenized_prompt) for tokenized_prompt in tokenized_prompts["input_ids"]]
        for labels, prompt_length in zip(all_labels, prompt_lengths):
            labels[:prompt_length]=[IGNORE_INDEX]*prompt_length
        tokenized_prompt_responses["labels"]=all_labels
        input_ids, attention_mask, labels=tuple([torch.tensor(tokenized_prompt_feature[::-1]) for tokenized_prompt_feature in tokenized_prompt_responses[key]] for key in ["input_ids", "attention_mask", "labels"])
        input_ids=rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).flip(dims=[1])
        attention_mask=rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0).flip(dims=[1])
        labels=rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX).flip(dims=[1])

        final_logprobs=[]
        with torch.no_grad():
            batch=dict(
                input_ids=input_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device),
                labels=labels.to(self.model.device)
            )
            outputs=self.model(**batch)
            logprobs=torch.log(torch.softmax(outputs.logits, dim=-1))
            for i in range(logprobs.shape[0]):
                final_logprob=0.0
                i1=(labels[i]!=IGNORE_INDEX).nonzero()[0]
                response_tokens=input_ids[i][i1:]
                response_logprobs=logprobs[i][i1-1:-1]
                for j, token in enumerate(response_tokens):
                    final_logprob+=response_logprobs[j, token].item()
                final_logprobs.append(final_logprob)
        return final_logprobs