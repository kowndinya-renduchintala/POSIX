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

If you use *POSIX* in your research, please cite of our EMNLP 2024 paper (aclanthology version coming soon...)

[_POSIX: A Prompt Sensitivity Index For Large Language Models_](https://arxiv.org/abs/2410.02185):

```
@article{chatterjee2024posix,
  title={POSIX: A Prompt Sensitivity Index For Large Language Models},
  author={Chatterjee, Anwoy and Renduchintala, HSVNS Kowndinya and Bhatia, Sumit and Chakraborty, Tanmoy},
  journal={arXiv preprint arXiv:2410.02185},
  year={2024}
}
```

# License

*POSIX* is licensed under the MIT License. See [LICENSE](LICENSE) for more information.