# LLM Transformer Project
=======================
### MATH598C: LLMs

We have implemented a transformer from scratch using the PyTorch python library. 

### In this Repository:
#### transformer.ipynb 
Run jupyter Notebook 

### Broken functionality into .py files for testing purposes:

#### config.py
Configurations class 

#### dataset.py 
Class to help load book dataset with helper functions

#### main.py
Main to pull everything together

#### model.py
Code for transformer, MLP, attention Head, and the transformer Block 

#### tokenizer.py
Tokenizer implementation

#### train.py
Code for training loop

#### train.py
Code for training loop

#### .gitgnore
Specifies what to exclude when committing to repository

#### pyproject.toml
Defined dependencies and other necessary infromation to run succesfully

#### README.md
Documentation 

#### contributions.txt
Defines everyones contributions 

#### writeup.txt
Text file with design choices and challenges 

#### tests:
pytests defined in two files:
##### conftest.py and test_transformer.py

### Running the Code:
1. Clone repository 
Using https
```bash 
git clone https://github.com/melody-gold/math598-llm-project.git 
```
or you can use SSH 
```bash 
git clone git@github.com:melody-gold/math598-llm-project.git
```
2. Set up Python Environment:
```bash 
python -m venv .venv
source .venv/bin/activate
uv sync
```
3. Run Code by running the Jupyter Notebook called transformer.ipynb

4. Run Tests (Optional)
```bash 
pytest
```

### Contributions
Everyone's contributions are indicated in contributions.txt. please refer to it for contributions. 

### Results
TO DO

