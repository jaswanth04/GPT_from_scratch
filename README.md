# GPT development from scratch

## Introduction

GPT stands for Generative Pre Trained Transformer.

This is based on the Attention is All you need paper.  
We tried to create a Decoder only model, that is trained on the tiny shakesphere dataset. This does not compare to original GPT-4 model, where the vocab is tokens considered. Here the vocabulary that we are considering is the letters present in the dataset. 

This repo has been implemented in pytorch

## Training details

This has been trained for 20k epochs, on cuda (RTX 3080Ti) which took around 25 mins to train. The loss used is cross entropy. 

## How to run

```
pip install -r requirements.txt
python v2.py
```

Using the above command the script trains the model, and also generates 10000 words of shakeshpere text. The text is understandable. But it does not make sense yet. It would need better engineering where we might need to use more data, and also better tokens, and train for more epochs.