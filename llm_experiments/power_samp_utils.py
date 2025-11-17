import os

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import Dataset, load_dataset, concatenate_datasets


import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers

from grader_utils.parse_utils import parse_answer
from constants import *

### DESCRIPTION ###
# power sampling to sample from p^{alpha}, where p is the base model
# takes in 1/alpha (temperature) as an argument (default 0.25), and mcmc_power_samp implements sampling from p^{alpha} 


class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

    # returns log probs
    @torch.no_grad()
    def next_token(self, prefix):
        device = self.device
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=device)
        prefix_cond = torch_prefix if torch_prefix.size(1) <= self.block_size else torch_prefix[:, -self.block_size:]
        output = self.model(prefix_cond)
        logits = output.logits
        logits = logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)



# returns probabilities (normed)
def normalize(dist):
    probs = F.softmax(dist, dim=-1)
    return probs

# returns sum of logits (product of distributions p*q)
def dist_product(logit_p, logit_q):
    return logit_p+logit_q

# returns logit scaled by temp (temperature scaling p^(1/tau))
def dist_temp_scale(logit_p, temp):
    return logit_p * torch.tensor(1 / temp, dtype=logit_p.dtype, device=logit_p.device)

# low-temperature sampling proposal distribution
def naive_temp(p : AutoregressiveSampler, context, temp, seq_len):
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    output = p.model.generate(
        input_ids=input_ids,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]


    idx = tokens.view(unscaled_logits.shape[0], 1, 1)

    log_probs_unnorm = (1/temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)

    return prop, log_probs_norm, log_probs_unnorm


# alpha = infty power sampling; temp is for proposal distribution
def max_swap(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    c = len(context)
    print(f'Temp: {temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []


    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0


    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # llm query takes the burden of time
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            log_r = sum(target_log_prob_prop) - sum(target_log_prob_cur)

            if log_r > 0:
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

# power sampling with autoregressive mcmc
def mcmc_power_samp(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    c = len(context)
    print(f'alpha: {1/temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []


    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0


    for _ in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # llm query takes the burden of time
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)

            if np.random.rand() < np.exp(log_r):
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


### SOFT THINKING IMPLEMENTATION ###
# Soft Thinking: reasoning in continuous concept space using soft token embeddings
# instead of discrete tokens

class SoftThinkingSampler:
    """
    Soft Thinking sampler that generates in continuous concept space.

    Instead of sampling discrete tokens, creates soft tokens as probability-weighted
    mixtures of token embeddings: h_t = Σ_v p(v|context) * embedding(v)

    Enhanced with:
    - Noise injection (Dirichlet, Gumbel-Softmax) for exploration
    - Dual temperature control (thinking vs output)
    - Multi-criteria early stopping (entropy + length)
    """
    def __init__(self, model, tokenizer, device,
                 max_topk=10, min_p=0.001,
                 early_stopping_entropy_threshold=0.05,
                 early_stopping_length_threshold=None,
                 thinking_temperature=1.0,
                 output_temperature=1.0,
                 noise_type='none',  # 'none', 'dirichlet', 'gumbel'
                 noise_alpha=0.1,  # For Dirichlet noise
                 gumbel_tau=1.0):  # For Gumbel-Softmax temperature
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = model.config.max_position_embeddings

        # Soft thinking hyperparameters
        self.max_topk = max_topk  # Top-k tokens to consider for soft token
        self.min_p = min_p  # Minimum probability threshold
        self.early_stopping_entropy_threshold = early_stopping_entropy_threshold
        self.early_stopping_length_threshold = early_stopping_length_threshold  # Max thinking steps
        self.thinking_temperature = thinking_temperature  # Temperature during thinking
        self.output_temperature = output_temperature  # Temperature for final output

        # Noise injection parameters
        self.noise_type = noise_type
        self.noise_alpha = noise_alpha  # Dirichlet concentration parameter
        self.gumbel_tau = gumbel_tau  # Gumbel-Softmax temperature

        # Get embedding layer
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            # For LLaMA-style models
            self.embed_layer = model.model.embed_tokens
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            # For GPT-style models
            self.embed_layer = model.transformer.wte
        elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
            # For some decoder models
            self.embed_layer = model.model.decoder.embed_tokens
        else:
            raise ValueError("Could not find embedding layer in model")

    def apply_noise_to_probs(self, probs):
        """
        Apply noise to probabilities for exploration.

        Args:
            probs: Probability distribution tensor

        Returns:
            Perturbed probability distribution
        """
        if self.noise_type == 'none':
            return probs

        elif self.noise_type == 'dirichlet':
            # Dirichlet noise: sample from Dirichlet(alpha * probs)
            # This maintains the general shape but adds stochasticity
            alpha = torch.ones_like(probs) * self.noise_alpha
            concentration = alpha + probs * (1 - self.noise_alpha)

            # Sample from Dirichlet distribution
            dirichlet = torch.distributions.Dirichlet(concentration)
            perturbed_probs = dirichlet.sample()

            # Renormalize
            perturbed_probs = perturbed_probs / perturbed_probs.sum()
            return perturbed_probs

        elif self.noise_type == 'gumbel':
            # Gumbel-Softmax: add Gumbel noise to logits before softmax
            # This is a reparameterization trick for categorical sampling
            logits = torch.log(probs + 1e-10)

            # Sample Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)

            # Apply Gumbel-Softmax
            perturbed_logits = (logits + gumbel_noise) / self.gumbel_tau
            perturbed_probs = F.softmax(perturbed_logits, dim=-1)
            return perturbed_probs

        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    @torch.no_grad()
    def create_soft_token(self, prefix, use_thinking_temp=True):
        """
        Create a soft token as probability-weighted mixture of token embeddings.

        Args:
            prefix: List of token IDs representing the context
            use_thinking_temp: Whether to use thinking temperature (True) or output temperature (False)

        Returns:
            soft_embedding: Tensor of shape (1, 1, hidden_dim) representing soft token
            log_probs: Log probabilities of top-k tokens
            top_tokens: Token IDs of top-k candidates
            entropy: Entropy of the distribution
        """
        device = self.device
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=device)

        # Truncate if needed
        prefix_cond = torch_prefix if torch_prefix.size(1) <= self.block_size else torch_prefix[:, -self.block_size:]

        # Get logits from model
        output = self.model(prefix_cond)
        logits = output.logits[0, -1, :]  # Shape: (vocab_size,)

        # Apply temperature scaling (different for thinking vs output)
        temperature = self.thinking_temperature if use_thinking_temp else self.output_temperature
        logits = logits / temperature

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Apply noise injection for exploration (only during thinking phase)
        if use_thinking_temp and self.noise_type != 'none':
            probs = self.apply_noise_to_probs(probs)

        # Calculate entropy for early stopping
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        # Filter by min_p threshold
        mask = probs >= self.min_p

        # Get top-k tokens
        topk_probs, topk_indices = torch.topk(probs, min(self.max_topk, (probs >= self.min_p).sum().item()))

        # Renormalize top-k probabilities
        topk_probs = topk_probs / topk_probs.sum()

        # Get embeddings for top-k tokens
        topk_embeddings = self.embed_layer(topk_indices)  # Shape: (k, hidden_dim)

        # Create soft token as weighted mixture
        soft_embedding = (topk_probs.unsqueeze(-1) * topk_embeddings).sum(dim=0, keepdim=True).unsqueeze(0)  # Shape: (1, 1, hidden_dim)

        log_probs = torch.log(topk_probs + 1e-10)

        return soft_embedding, log_probs, topk_indices, entropy.item()

    @torch.no_grad()
    def generate_with_soft_thinking(self, prefix, max_new_tokens, num_thinking_steps=3):
        """
        Generate tokens using soft thinking with enhanced early stopping.

        Args:
            prefix: Initial context (list of token IDs)
            max_new_tokens: Maximum number of tokens to generate
            num_thinking_steps: Number of soft thinking steps before committing to hard token

        Returns:
            tokens: Generated token IDs
            log_probs: Log probabilities
            soft_token_info: Information about soft tokens used
        """
        device = self.device
        generated_tokens = []
        all_log_probs = []
        soft_token_info = []

        current_prefix = prefix.copy()

        for step in range(max_new_tokens):
            # Perform soft thinking steps
            soft_embeddings_list = []
            entropies = []

            for thinking_step in range(num_thinking_steps):
                if thinking_step == 0:
                    # First step: use discrete prefix with thinking temperature
                    soft_emb, log_probs, top_tokens, entropy = self.create_soft_token(
                        current_prefix, use_thinking_temp=True
                    )
                else:
                    # Subsequent steps: use soft embeddings
                    soft_emb, log_probs, top_tokens, entropy = self.create_soft_token_from_soft(
                        current_prefix, soft_embeddings_list
                    )

                soft_embeddings_list.append(soft_emb)
                entropies.append(entropy)

                # Multi-criteria early stopping
                # 1. Entropy-based: stop if model is highly confident
                if entropy < self.early_stopping_entropy_threshold:
                    break

                # 2. Length-based: stop if reached max thinking steps
                if self.early_stopping_length_threshold is not None:
                    if len(soft_embeddings_list) >= self.early_stopping_length_threshold:
                        break

            # Commit to hard token: use final soft embedding with output temperature
            final_soft_emb = soft_embeddings_list[-1]
            final_token, final_log_prob = self.commit_to_hard_token(
                current_prefix, final_soft_emb, use_thinking_temp=False
            )

            generated_tokens.append(final_token)
            all_log_probs.append(final_log_prob)
            current_prefix.append(final_token)

            soft_token_info.append({
                'thinking_steps': len(soft_embeddings_list),
                'entropies': entropies,
                'final_entropy': entropies[-1],
                'stopped_by': 'entropy' if entropies[-1] < self.early_stopping_entropy_threshold else
                              'length' if (self.early_stopping_length_threshold and len(soft_embeddings_list) >= self.early_stopping_length_threshold) else
                              'max_steps'
            })

            # Check for EOS
            if final_token == self.tokenizer.eos_token_id:
                break

        return generated_tokens, all_log_probs, soft_token_info

    @torch.no_grad()
    def create_soft_token_from_soft(self, discrete_prefix, soft_embeddings):
        """
        Create next soft token given discrete prefix and soft thinking embeddings.

        Args:
            discrete_prefix: List of discrete token IDs
            soft_embeddings: List of soft embedding tensors from previous thinking steps

        Returns:
            soft_embedding, log_probs, top_tokens, entropy (same as create_soft_token)
        """
        device = self.device

        # Convert discrete prefix to embeddings
        discrete_tokens = torch.tensor([discrete_prefix], dtype=torch.long, device=device)
        discrete_embs = self.embed_layer(discrete_tokens)  # Shape: (1, len(prefix), hidden_dim)

        # Concatenate with soft embeddings
        all_embeddings = torch.cat([discrete_embs] + soft_embeddings, dim=1)

        # Truncate if needed
        if all_embeddings.size(1) > self.block_size:
            all_embeddings = all_embeddings[:, -self.block_size:, :]

        # Forward pass with embeddings
        output = self.model(inputs_embeds=all_embeddings)
        logits = output.logits[0, -1, :]

        # Apply thinking temperature
        logits = logits / self.thinking_temperature
        probs = F.softmax(logits, dim=-1)

        # Apply noise injection during thinking
        if self.noise_type != 'none':
            probs = self.apply_noise_to_probs(probs)

        # Calculate entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        # Get top-k
        topk_probs, topk_indices = torch.topk(probs, min(self.max_topk, (probs >= self.min_p).sum().item()))
        topk_probs = topk_probs / topk_probs.sum()

        # Create soft embedding
        topk_embeddings = self.embed_layer(topk_indices)
        soft_embedding = (topk_probs.unsqueeze(-1) * topk_embeddings).sum(dim=0, keepdim=True).unsqueeze(0)

        log_probs = torch.log(topk_probs + 1e-10)

        return soft_embedding, log_probs, topk_indices, entropy.item()

    @torch.no_grad()
    def commit_to_hard_token(self, discrete_prefix, final_soft_embedding, use_thinking_temp=False):
        """
        Commit soft thinking to a hard token.

        Args:
            discrete_prefix: List of discrete token IDs
            final_soft_embedding: Final soft embedding from thinking process
            use_thinking_temp: Whether to use thinking temperature (typically False for final output)

        Returns:
            token_id: Selected hard token
            log_prob: Log probability of selected token
        """
        device = self.device

        # Convert discrete prefix to embeddings
        discrete_tokens = torch.tensor([discrete_prefix], dtype=torch.long, device=device)
        discrete_embs = self.embed_layer(discrete_tokens)

        # Concatenate with final soft embedding
        all_embeddings = torch.cat([discrete_embs, final_soft_embedding], dim=1)

        # Truncate if needed
        if all_embeddings.size(1) > self.block_size:
            all_embeddings = all_embeddings[:, -self.block_size:, :]

        # Forward pass
        output = self.model(inputs_embeds=all_embeddings)
        logits = output.logits[0, -1, :]

        # Use output temperature for final token selection (sharper/more confident)
        temperature = self.thinking_temperature if use_thinking_temp else self.output_temperature

        # Sample token
        probs = F.softmax(logits / temperature, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()
        log_prob = torch.log(probs[token_id] + 1e-10).item()

        return token_id, log_prob


def soft_thinking_generate(p: SoftThinkingSampler, context, max_new_tokens,
                           num_thinking_steps=3):
    """
    Generate using Soft Thinking methodology.

    Args:
        p: SoftThinkingSampler instance
        context: Initial context (list of token IDs)
        max_new_tokens: Maximum tokens to generate
        num_thinking_steps: Number of soft thinking steps per token

    Returns:
        generated_ids: Full sequence (context + generated)
        log_probs: Log probabilities of generated tokens
        soft_info: Information about soft thinking process
        avg_thinking_steps: Average number of thinking steps used
    """
    c = len(context)
    generated_tokens, log_probs, soft_info = p.generate_with_soft_thinking(
        context, max_new_tokens, num_thinking_steps
    )

    # Compute average thinking steps
    avg_thinking_steps = sum(info['thinking_steps'] for info in soft_info) / len(soft_info) if soft_info else 0

    # Return full sequence
    full_sequence = context + generated_tokens

    return full_sequence, log_probs, soft_info, avg_thinking_steps


def format_prompt(question, model, tokenizer, cot=True):
    if model == "qwen":
        format_str = PROMPT + question
        if cot:
            format_str+=COT
        else:
            format_str+=BASE

    elif model == "qwen_math":
        format_str = PROMPT + question
        if cot:
            format_str+=COT
        else:
            format_str+=BASE

    elif model == "qwen_math_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "tulu":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    return format_str
