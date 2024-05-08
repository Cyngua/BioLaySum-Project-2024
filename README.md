# BioLaySum Project
CPSC477 Final Project/Shared Task: Lay Summarization of Biomedical Research Articles @ BioNLP Workshop, ACL 2024 <br>
Group Members: Xincheng Cai, Mengmeng Du<br>
## Abstract
Biomedical research articles contain vital information for a wide audience, yet their complex language and specialized terminology often hinder comprehension for non-experts. Inspired by the BIONLP 2024 workshop, we propose a NLP solution to generate lay summaries, which are more readable to diverse audiences. We implemented two transformer-based models, specifically BART and BART-PubMed. Our study investigates the performance of these models across different biomedical topics and explores methods to improve summarization quality through definition retrieval from Webster Medical Dictionary. By enhancing the readability of biomedical publications, our work aims to promote knowledge accessibility to scientific information.

## Project Workflow
Our project mainly consist of three parts as shown in the workflow diagram:
* **Experiment I**: BART vs BART-PubMed on the entire dataset: Compared the finetuning performance on the test dataset using metrics from three aspects: relevance, readability and factuality. 
* **Experiemnt II**: BART-PubMed model across various topics: Finetune BART-PubMed model separately on 6 subsets of the dataset stratified by keywords. 
* **Experiment III**: Terminology replacement: To further enhance performance, we implemented terminology replacement with definitions retrieved from a medical dictionary, [Webster Medical Library](https://www.merriam-webster.com/medical), and evaluated its impact on model performance.
![Project workflow diagram](workflow.jpg)

## Environment Setup and Computing Infrastructure
* Experiment I: We have a file called `env.yml` for virtual environment setup, which is provided by Yale CPSC452/552: Deep learning theory and application. The training process was run on the cpsc462 cluster [Yale McCleary](https://docs.ycrc.yale.edu/clusters/mccleary/), which supports `gpu_devel` with 6-core cpu. Here is the code to create conda environment:

``` sh
# create conda environment
$ conda env create -f env.yml

# update conda environment
$ conda env update -n cpsc552 --file env.yml
```

* Experiment II and III: We based our last two experiments on Google Colab Pro environment plus external library installed. All the training was run on **T-4 GPU** by Google Colab. Run the following codes in colab notebooks to install the most essential libraries for this project.

```
!pip install accelerate -U
!pip install transformers datasets evaluate
!pip install textstat
!pip install rouge_score
!pip install bert_score
!pip install summac
```


## Methodologies
Please refer to our report methods section for further details about models and evaluation metrics.
### Models
* BART(Bidirectional and Auto-Regressive Transformers): a denoising autoencoder built upon a sequence-to-sequence architecture, pretrained on XSum news dataset
* BART-PubMed: BART based model, pretrained on the PubMed dataset

### Evaluation Metrics
We evaluated our model performance from three perspectives: Relevance, Readability, and Factuality. The metrics are listed as follows:
|                 | Evaluation Metrics         |
|-----------------|----------------------------|
| **Relevance**   | ROUGE (1, 2, and L) and BERTScore |
| **Readability** | FKGL, CLI, DCRS            |
| **Factuality**  | SummaC                     |


## Training Details and 


## Dataset Description
[Link to the datasets](https://drive.google.com/drive/folders/1sfmYlHL9FcAjKpLzjW4CO_izJmVbcZ-g?usp=sharing), which is a google drive folder accessible to Yale community. We utilized eLife biomedical journals in this project, an open-access peer-reviewed journal with a specific focus on biomedical and life sciences. The original datasets are provided by the shared task in .jsonl format, where each line represents a JSON object with the fields outlined 

| Column       | Description            |
|--------------|------------------------|
| Lay summary  | Article lay summary   |
| Article      | Article main text      |
| Headings     | Article headings       |
| Keywords     | Topic of the article   |
| ID           | Article ID             |

The datasets have been pre-split for model training and validation, consisting of 4,346 instances earmarked for training and 241 for validation. The folder also contains the data modified with a definition replacement preprocessing step, and the data rearranged according to the keywords. All these modified datasets are in a json format.
