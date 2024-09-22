# Paper: Automatically Selecting Follow-up Questions for Deficient Bug Reports
In the following, we briefly describe the different components that are included in this project and the softwares required to run the experiments.

## Project Structure
The project includes the following files and folders:

  - __/data__: A folder that contains inputs that are used for the experiments
  - dataset.csv: CSV file that contains 25k bug reports with follow-up questions and answers
  - github_issue_titles.csv: Titles of those 25k bug reports
  - github_issue_labels.csv: Labels of those 25k bug reports
  - github_repo_labels.csv: Repository labels of those 25k bug reports
  - post_data.tsv: 25k bug reports processed by Lucene
  - qa_data.tsv: 25k bug reports with 10 candidate follow-up questions selected by Lucene
  - test_ids.txt: Test dataset ids
  - train_ids.txt: Train dataset ids
  - __/embeddings__: A folder that contains the embeddings we have used
  - __/script__: Contains the scripts for running the experiments
      - __/models__: Contains the scripts of the model
        - run_main.sh: The entry point of the experiment
  - __/survey__: A folder that contains surevey related files

## Software Requirements
We used the following software to run our experiments
  * python3.6
  * torch
  * torchvision
  * spacy>=2.2.4
  * numpy pandas gensim jsonschema
  * conda

## Setup
Conda
```
conda install numpy pandas gensim jsonschema
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
python -m spacy download en_core_web_sm
```

Conda currently does not provide required version of spacy, so we use pip to install spacy:

```
conda install pip
pip install spacy==2.2.4
```

## Running Experiments
Step 1: Install software requirements mentioned above.

Step 2: Update the filepaths and parameters in *script/models/run_main.sh*

Step 3: `./script/models/run_main.sh`