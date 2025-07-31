# 24B2406_llm_from_scratch
Building a GPT-Style Large Language Model From Scratch
This repository, contains the complete source code for building, pre-training, and finetuning a decoder-style Generative Pre-trained Transformer (GPT) model from the ground up using PyTorch. The project is structured chapter-by-chapter, progressively building from the core components of a transformer to a fully functional, instruction-finetuned language model.
Project Overview
The primary goal of this repository is to provide a clear, educational, and hands-on implementation of a Large Language Model. The code follows a logical progression:
1.	Core Architecture: Implementing the fundamental building blocks, including the multi-head causal attention mechanism and the transformer block.
2.	Pre-training: Training the model on a text corpus to learn language patterns and generate coherent text. This includes loading official pre-trained weights from OpenAI's GPT-2.
3.	Finetuning: Adapting the pre-trained model for specialized downstream tasks, including:
o	Text Classification: Finetuning the model to classify text (e.g., as spam or not spam).
o	Instruction Following: Finetuning the model to understand and respond to user prompts and instructions, turning it into a helpful assistant.
4.	Evaluation: Assessing the performance of the finetuned model, including using another LLM (via Ollama) for automated evaluation of instruction-following capabilities.
Project Structure
The repository is organized into directories corresponding to the chapters of development. Each script is self-contained but builds upon the concepts of the previous ones.
•	multihead_attention.py
o	Implements the core Multi-Head Attention module with causal masking. This is the heart of the transformer architecture, allowing the model to weigh the importance of different tokens in a sequence.
•	implementation_of_a_decoder-style_GPT_model.py
o	Constructs the full GPT model architecture.
o	Integrates the MultiHeadAttention module into TransformerBlocks.
o	Includes implementations for LayerNorm, GELU activation, and the FeedForward network.
o	Stacks the transformer blocks to create the final GPTModel.
•	train_and_generate.py
o	Contains the complete pipeline for pre-training the GPT model.
o	Includes functions for text generation with advanced decoding strategies like temperature scaling and top-k sampling.
o	Provides the training loop (train_model_simple), loss calculation utilities, and plotting functions.
•	gpt_download.py
o	A utility script to download the official pretrained GPT-2 weights and configuration files from OpenAI's repository. It uses TensorFlow to load the original checkpoint files.
•	finetune_classifier.py
o	Demonstrates how to adapt the pretrained GPT model for a text classification task.
o	Includes code for preparing a dataset (spam vs. not spam), creating a custom SpamDataset and DataLoader, and modifying the model with a classification head.
o	Contains the finetuning loop and evaluation logic based on accuracy.
•	instruction_finetuning.py
o	Contains the pipeline for instruction-finetuning the pretrained GPT model.
o	Prepares an instruction-based dataset and uses a custom collate function for dynamic padding.
o	Finetunes the model to act as a helpful assistant that can follow user prompts.
•	ollama_tets.py (ollama_evaluate.py)
o	A standalone script for evaluating the instruction-finetuned model.
o	It uses a locally running LLM (e.g., Llama 3) via Ollama to score the model's responses against a reference, providing an automated way to assess performance.
Features
•	From-Scratch GPT-2 Architecture: A clean and well-commented implementation of a GPT-2 style model in PyTorch.
•	Multi-Head Causal Attention: The core component of the transformer, implemented with efficiency and clarity.
•	Pre-training Pipeline: A complete training script to train the model on a custom text corpus.
•	Advanced Text Generation: Functions for generating text with decoding strategies like greedy search, temperature scaling, and top-k sampling.
•	Pretrained Weight Loading: Utility to download and load official GPT-2 weights (124M, 355M, etc.) into the custom model architecture.
•	Classification Finetuning: A full example of adapting the model for a binary text classification task, including data preparation, training, and evaluation.
•	Instruction Finetuning: A complete pipeline to finetune the model on an instruction dataset, enabling it to follow commands and answer questions.
•	LLM-based Evaluation: An automated evaluation script that leverages Ollama to score the finetuned model's performance on instruction-following tasks.
How to Use
1. Setup
First, clone the repository and install the required dependencies.
git clone [https://github.com/your-username/uthun.git](https://github.com/your-username/uthun.git)
cd uthun
pip install -r requirements.txt

You will also need to install Ollama for the evaluation script. Follow the instructions at ollama.com. After installation, pull the Llama 3 model:
ollama pull llama3

2. Running the Scripts
The scripts are designed to be run in a sequence that follows the model's development lifecycle.
A. Model Implementation and Pre-training (Optional)
These scripts allow you to train a model from scratch. This is computationally intensive and best performed on a machine with a GPU.
# This script can be run to see the model generate text (with random weights)
python implementation_of_a_decoder-style_GPT_model.py

# This script contains the training loop for pre-training
# (Best to run on a GPU with a large dataset)
python train_and_generate.py

B. Finetuning for Text Classification
This script loads a pretrained GPT-2 model, adapts it for spam classification, and finetunes it.
# This will download the dataset, load GPT-2, finetune, and evaluate
python finetune_classifier.py

C. Finetuning for Instruction Following
This is the final stage, where the model learns to be a helpful assistant.
# This script will download a larger GPT-2 model and finetune it on an instruction dataset
python instruction_finetuning.py

D. Evaluating the Instruction-Finetuned Model
After running the instruction finetuning script, it will generate a JSON file with the model's responses. You can then use the evaluation script to score these responses.
First, ensure the Ollama server is running in a separate terminal:
ollama serve

Then, run the evaluation script:
# The finetuning script creates 'instruction-data-with-response.json'
python ollama_tets


