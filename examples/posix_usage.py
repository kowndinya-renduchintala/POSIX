from prompt_sensitivity_index.models import HFModel
from prompt_sensitivity_index.posix import (
    PosixConfig, 
    PosixTrace, 
    get_prompt_sensitivity, 
    write_trace_to_json
)

def main():
    intent_aligned_prompts=[
        [
            "Q: What is the capital of France?\nA: ",
            "Q: WHat is te capital city of France?\nA: ",
            "Q: what is teh cpital of france??\nA: "
        ],
        [
            "Q: What is the national animal of India?\nA: ",
            "Q: What's the national animl of India?\nA: ",
            "Q: WHat is teh national animal of India?\nA: "
        ],
        [
            "Q: Tell me the meaning of rendezvous?\nA: ",
            "Q: WHat is the meaning of rendezvous?\nA: ",
            "Q: What does rendezvous mean?\nA: "
        ]
    ]
    model=HFModel("openai-community/gpt2")
    config=PosixConfig(max_new_tokens=5, batched=False)
    posix, trace=get_prompt_sensitivity(model, intent_aligned_prompts, config, verbose=True)
    print(f"Prompt Sensitivity Index: {posix}")
    write_trace_to_json(trace, "posix_trace.json")

if __name__=="__main__":
    main()