# QaDialMoE
Code for paper "QaDialMoE: Question-answering Dialogue based Fact Verification with Mixture of Experts".

# Dependency
+ python 3.5+
+ pytorch 1.0+

# Usage
Download RoBERTa-Large model from https://huggingface.co/roberta-large. Place all the files into a new folder "roberta_large" under the root.
## Dataset
Please see the details in `data` folder.
## Training
To train the QaDialMoE:
```shell script
python run_healthmoe.py --do_train --do_eval
python run_colloquial_t5.py --do_train --do_eval
python run_faviqmoe_dpr.py --do_train --do_eval
```

# Acknowledgement
Our implementation is built on the source code from [SaMoE](https://github.com/THUMLP/SaMoE). Thanks for their contributions.
