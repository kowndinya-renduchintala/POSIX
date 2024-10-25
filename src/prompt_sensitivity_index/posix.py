import json
from typing import Tuple
from tqdm.auto import tqdm
from dataclasses import dataclass

from .models import LanguageModel

@dataclass
class PosixConfig:
    max_new_tokens: int = 5
    batched: bool = False
    batch_size: int = 4

@dataclass
class PosixTrace:
    prompts: list[list[str]]
    responses: list[list[str]]
    logprob_matrices: list[list[list[float]]]
    prompt_sensitivities: list[float]
    posix: float

def get_prompt_sensitivity(language_model: LanguageModel, intent_aligned_prompt_sets: list[list[str]], config: PosixConfig, verbose=False) -> Tuple[float, PosixTrace]:
    """
    Get the Prompt Sensitivity Index (POSIX) for a given set of intent-aligned prompts.
    Args:
        language_model: A LanguageModel object that will be used to score the prompts.
        intent_aligned_prompts: A list of lists of prompts that are aligned by intent.
        batched: Whether to score the prompts in a batched manner. Defaults to False.
    Returns:
        A tuple containing the Prompt Sensitivity Index (POSIX) and a PosixTrace object.
    """
    batched=config.batched
    batch_size=config.batch_size
    generation_config={
        "max_new_tokens": config.max_new_tokens,
        "do_sample": False,
    }

    N_prompt_sets=len(intent_aligned_prompt_sets)
    if verbose:
        print(f"Total Number of Intent-Aligned Prompt Sets: {N_prompt_sets}")
    
    response_tokens=[]
    responses=[]
    logprob_matrices=[]
    prompt_sensitivities=[]

    pbar=tqdm(range(len(intent_aligned_prompt_sets)))
    for i in range(N_prompt_sets):
        intent_aligned_prompts=intent_aligned_prompt_sets[i]
        N_prompts=len(intent_aligned_prompts)

        response_tokens.append([])
        responses.append([])

        if batched:
            for j in range(0, len(intent_aligned_prompts), batch_size):
                batched_prompts=intent_aligned_prompts[j:j+config.batch_size]
                batched_response_tokens, batched_responses=language_model.get_responses(batched_prompts, batched=True, **generation_config)
                response_tokens[-1].extend(batched_response_tokens)
                responses[-1].extend(batched_responses)
        else:
            batched_response_tokens, batched_responses=language_model.get_responses(intent_aligned_prompts, batched=False, **generation_config)
            response_tokens[-1].extend(batched_response_tokens)
            responses[-1].extend(batched_responses)

        logprob_matrix=[[0 for _ in range(N_prompts)] for _ in range(N_prompts)]
        for i in range(N_prompts):
            if batched:
                for j in range(0, len(intent_aligned_prompts), batch_size):
                    batched_logprobs=language_model.compute_log_probabilties([intent_aligned_prompts[i]]*N_prompts, response_tokens[-1][j:j+config.batch_size], batched=True)
                    for k in range(len(batched_logprobs)):
                        logprob_matrix[i][j+k]=batched_logprobs[k]
            else:
                logprob_matrix[i]=language_model.compute_log_probabilties([intent_aligned_prompts[i]]*N_prompts, response_tokens[-1], batched=False)
        
        logprob_matrices.append(logprob_matrix)
        psi=0.0
        for i in range(N_prompts):
            for j in range(N_prompts):
                psi+=(abs(logprob_matrix[j][i]-logprob_matrix[j][j]))/config.max_new_tokens
        
        prompt_sensitivity=psi/(N_prompts*(N_prompts-1))
        prompt_sensitivities.append(prompt_sensitivity)

        if verbose:
            print(f"Prompt Sensitivity for Prompt Set {i}: {prompt_sensitivity}")
        pbar.update(1)

    posix=sum(prompt_sensitivities)/N_prompt_sets
    if verbose:
        print(f"Prompt Sensitivity Index (POSIX): {posix}")

    return posix, PosixTrace(intent_aligned_prompt_sets, responses, logprob_matrices, prompt_sensitivities, posix)

def write_trace_to_json(trace: PosixTrace, output_path: str):
    N=len(trace.prompts)
    to_write=[]
    for i in range(N):
        to_write.append({
            "prompts": trace.prompts[i],
            "responses": trace.responses[i],
            "log_probability_matrix": trace.logprob_matrices[i],
            "prompt_sensitivity": trace.prompt_sensitivities[i]
        })
    with open(output_path, "w") as f:
        json.dump(to_write, f, indent=4)