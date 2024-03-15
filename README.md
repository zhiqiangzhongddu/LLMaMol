# LLM4Mol

## Instruction to execution

1. Message (Prompt) template design.
   1. Message templates (``code/prompt_template``). 
   2. Prompt templates (``code/prompt_template``). 
2. Caption generation. (Optional)
   1. We apply pre-trained generative model to generate captions for molecules ([smiles2caption](https://github.com/blender-nlp/MolT5)).  
   2. Generated captions are saved in ``input/caption``. 
3. Description generation. (Optional)
   1. We generate descriptions of molecule graph structure accompanied by atom features ([rdkit.Chem](https://rdkit.org/docs/source/rdkit.Chem.html)). 
   2. Generated descriptions are saved in ``input/description``.
4. Predictor Message generation. 
   1. Generated Predictor messages are saved in ``input/message``. 
   2. Generated prompts are saved in ``input/prompt``.
   3. ``llm.template = IF(D)``: ask LLM to provide useful features for the task, (with given description).
   4. ``llm.template = IP(D)``: ask LLM to make predictions for the task, (with given description).
   5. ``llm.template = IE(D)``: ask LLM to make predictions for the task and provide explanations, (with given description).
   6. ``llm.template = FS(D)-3``: given 3 example knowledge, ask LLM to make predictions for the task and provide explanations, (with given description).
5. Query LLMs for response.
   1. Considering the consistency and popularity, we use ChatGPT for now. 
6. Make predictions based on LMs.
   1. Generated embeddings are saved as ``output/prt_lms/ogbg-molbace/IF/microsoft/deberta-base-seed42.emb``
7. Make predictions based on GNNs.
   1. Generated predictions are saved as ``output/gnns/ogbg-molbace/gin-v-raw-seed42/predictions.pt``

Notebook files in ``code/notebook`` are for demo test. 

## Example execution code

### Generate Caption

Example: 
```
python -m code.generate_caption dataset ogbg-molbace demo_test True device 0
```

### Generate Description and Compress it using ChatGPT

Example: 
```
python -m code.generate_description dataset ogbg-molbace demo_test True
python -m code.generate_compressor_message dataset ogbg-molbace demo_test True
python -m code.query_chatgpt dataset ogbg-molbace llm.template compress_des demo_test True
```

### Generate Prompt 
We no longer use it since generated prompts differ from SOTA messages for ChatGPT. 
We use Message (below) instead. 

Example: 
```
python -m code.generate_prompt dataset ogbg-molbace demo_test True
```

### Generate Predictor Message

Example: 
```
python -m code.generate_predictor_message dataset ogbg-molbace demo_test True
```

### Query ChatGPT

Example: 
```
python -m code.query_chatgpt dataset ogbg-molbace llm.template IF demo_test True
```

### Run LM model to evaluate Text quality

```
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1 python -m code.mainLM dataset ogbg-molbace data.text raw seed 42
```

### Run GNN model to evaluate

```
python -m code.mainGNN dataset ogbg-molbace data.feature IF seed 42
```
