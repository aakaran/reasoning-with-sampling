# Reasoning with Sampling


### [Paper]() | [Project Page](https://aakaran.github.io/training_free_reasoning/)

[![rws](teaser.png)](teaser.png)


This repo contains the official PyTorch implementation of Reasoning with Sampling.
> [**Reasoning with Sampling: Your Base Model is Smarter Than You Think**]()<br>
> [Aayush Karan](https://aakaran.github.io/), [Yilun Du](https://yilundu.github.io/)
> <br>Harvard<br>



## Setup

Run the following script to setup environment.

```bash
git clone https://github.com/aakaran/reasoning-with-sampling.git
cd reasoning-with-sampling
conda create -n psamp python=3.10
conda activate psamp
pip install -r requirements.txt
```


## Sampling
The llm_experiments folder contains slurm scripts to run power sampling for MATH500 (```power_samp_math.py```), whose .json is included in llm_experiments/data, as well as HumanEval (```power_samp_he.py```), GPQA Diamond (```power_samp_gpqa.py```), and AlpacaEval 2.0 (```power_samp_alpaca.py```), whose corresponding eval sets can be downloaded from their official repos. 

To run power sampling on MATH500 with 8 seeds and the eval set split across 5 shards:
```bash
sbatch llm_experiments/scripts/power_samp_math.sh
```
The output is several .csv files (based on the shard and seed number) that store the response outputs, correct answers, original prompts, etc. 

## Evaluation
**Single-shot Reasoning**

To grade the responses for single-shot reasoning, collect the .csv files for a given seed run in a folder (e.g. ```results/qwen_math/MATH```) and pass it into ```eval_math.py```:

```bash
python llm_experiments/eval_math.py --folder=results/qwen_math/MATH
```

```eval_gpqa.py``` is similar, and for ```eval_he.py```, an additional ```--output_fname``` argument is required, as HumanEval collects all responses in a jsonl file (e.g. ```--output_fname=qwen_math_he```).

For AlpacaEval 2.0, ```eval_alpaca.py``` collects a ```--folder``` into one json file ```--output_fname```. For evaluating the json file, follow the instructions in the official repo: https://github.com/tatsu-lab/alpaca_eval


**Pass@k Performance**

For pass@k performance, collect the .csv files across seeds in a folder again (e.g. ```results/qwen_math/MATH```) and pass into ```passk_math.py```:
```bash
python llm_experiments/passk_math.py --folder=results/qwen_math/MATH
```
The output is a plot of the pass@k performance. As with single-shot reasoning, ```eval_gpqa.py``` and ```eval_he.py``` are similar, but for the latter an additional ```--output_fname``` argument is required.


