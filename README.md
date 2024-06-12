Using LoRA to fine-tune LLaMA-3-8B for [Mask] guessing task.

Dataset: Mask Wiki Documents from TriviaQA_Test (165052 Samples after filting)

Prompt:
'''json
{
  "instruction": "Your Task is to GUESS the complete sentence to fill in the [Mask] section of the Wiki Document.",
  "input": title + '\n' + masked_text,
  "output": mask_sentence,
}
'''
