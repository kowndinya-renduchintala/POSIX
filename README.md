<p align="center">
    <br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        <img src="https://github.com/kowndinya-renduchintala/POSIX/blob/main/posix_logo.png" width="500" />
    </br>
    <br>
        <strong> PrOmpt Sensitivity IndeX</strong>
    </br>
    <br>
        (Companion Software for our EMNLP 2024 Paper)
    </br>
</p>

<p align="center">
    <a href="https://github.com/kowndinya-renduchintala/POSIX/blob/main/LICENSE.txt">
        <img alt="GitHub" src="https://img.shields.io/github/license/kowndinya-renduchintala/POSIX?color=blue">
    </a>
    <a href="#">
        <img alt="GitHub Stars" src="https://img.shields.io/github/stars/kowndinya-renduchintala/POSIX">
    </a>
    <a href="#">
        <img alt="GitHub Forks" src="https://img.shields.io/github/forks/kowndinya-renduchintala/POSIX">
    </a>
</p>

# About POSIX

*POSIX* is a simple and easy-to-use Python Library to evaluate prompt sensitivity of Language Models. 

# Installation

The simplest way to install *POSIX* is via `pip`:
```bash
pip install prompt-sensitivity-index
```

You can also install *POSIX* from source:
```bash
git clone https://github.com/kowndinya-renduchintala/POSIX.git
cd POSIX
pip install .
```

# Example Usage

```python
from prompt_sensitivity_index.models import HFModel
from prompt_sensitivity_index.posix import (
    PosixConfig, 
    PosixTrace, 
    get_prompt_sensitivity, 
    write_trace_to_json
)

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
```

# Citation

If you use *POSIX* in your research, please cite our EMNLP 2024 paper :blush: -


[POSIX: A Prompt Sensitivity Index For Large Language Models](https://aclanthology.org/2024.findings-emnlp.852) (Chatterjee et al., Findings 2024)

```
@inproceedings{chatterjee-etal-2024-posix,
    title = "{POSIX}: A Prompt Sensitivity Index For Large Language Models",
    author = "Chatterjee, Anwoy  and
      Renduchintala, H S V N S Kowndinya  and
      Bhatia, Sumit  and
      Chakraborty, Tanmoy",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.852",
    pages = "14550--14565",
    abstract = "Despite their remarkable capabilities, Large Language Models (LLMs) are found to be surprisingly sensitive to minor variations in prompts, often generating significantly divergent outputs in response to minor variations in the prompts, such as spelling errors, alteration of wording or the prompt template. However, while assessing the quality of an LLM, the focus often tends to be solely on its performance on downstream tasks, while very little to no attention is paid to prompt sensitivity. To fill this gap, we propose POSIX {--} a novel PrOmpt Sensitivity IndeX as a reliable measure of prompt sensitivity, thereby offering a more comprehensive evaluation of LLM performance. The key idea behind POSIX is to capture the relative change in loglikelihood of a given response upon replacing the corresponding prompt with a different intent-preserving prompt. We provide thorough empirical evidence demonstrating the efficacy of POSIX in capturing prompt sensitivity and subsequently use it to measure and thereby compare prompt sensitivity of various open source LLMs. We find that merely increasing the parameter count or instruction tuning does not necessarily reduce prompt sensitivity whereas adding some few-shot exemplars, even just one, almost always leads to significant decrease in prompt sensitivity. We also find that alterations to prompt template lead to the highest sensitivity in the case of MCQ type tasks, whereas paraphrasing results in the highest sensitivity in open-ended generation tasks. The code for reproducing our results is open-sourced at https://github.com/kowndinya-renduchintala/POSIX.",
}
```

# License

*POSIX* is licensed under the MIT License. See [LICENSE](LICENSE) for more information.