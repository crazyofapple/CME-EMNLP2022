# CME-EMNLP2022
Calibration Meets Explanation: A Simple and Effective Approach for Model Confidence Estimates, EMNLP 2022

Code and datasets for our paper. 


## Instructions

Use the following instructions to set up the dependencies:

```bash
$ conda activate [your_anaconda3_environment]
$ pip install -r requirements.txt
```

### Datasets

Use `tar -zxf calibration_data.tar.gz` to unpack the archive, and place it in the root directory.

### Training
```bash
bash example.sh
```


### Evaluating Calibration


```bash
bash eval.sh
```
