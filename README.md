### LoRA Fine-Tune

Using LoRA to fine-tune LLaMA-3-8B for [Mask] guessing task.

### Dataset

Mask Wiki Documents from TriviaQA_Test (165052 Samples after filting)

### Dataset Generation Prompt

```python
{
  "instruction": "Your Task is to GUESS the complete sentence to fill in the [Mask] section of the Wiki Document.",
  "input": wiki_title + '\n' + masked_wiki_document,
  "output": mask_sentence,
}
```
### References

[LLaMA3-8B-Instruct Lora 微调](https://github.com/datawhalechina/self-llm/diffs/2?base_sha=75f1a738e5a8d57f66e5a4186903354068746a26&head_user=0-yy-0&name=master&pull_number=90&qualified_name=refs%2Fheads%2Fmaster&sha1=75f1a738e5a8d57f66e5a4186903354068746a26&sha2=a7c237d90a207ba89529303d3c34c505246466a6&short_path=da2fe8c&unchanged=expanded&w=false)

[深入浅出 LoRA](https://zhuanlan.zhihu.com/p/650197598)
