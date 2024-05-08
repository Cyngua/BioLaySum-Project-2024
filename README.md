# BioLaySum Project
CPSC477 Final Project/Shared Task: Lay Summarization of Biomedical Research Articles @ BioNLP Workshop, ACL 2024 <br>
Group Members: Xincheng Cai, Mengmeng Du<br>
## Project Workflow
Our project mainly consist of three parts as shown in the workflow diagram:
* Experiment I: BART vs BART-PubMed on the entire dataset: Compared the finetuning performance on the test dataset using metrics from three aspects: relevance, readability and factuality. 
* Experiemnt II: BART-PubMed model across various topics: Finetune BART-PubMed model separately on 6 subsets of the dataset stratified by keywords. 
* Experiment III: Terminology replacement: To further enhance performance, we implemented terminology replacement with definitions retrieved from a medical dictionary, [Webster Medical Library](https://www.merriam-webster.com/medical), and evaluated its impact on model performance.
![Project workflow diagram](workflow.jpg)
## Methodologies
### Models

### Evaluation Metrics


## Computing Infrastructure


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
