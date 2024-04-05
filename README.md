# DAGPap24

This repo includes all content relevant to [DAGPap24 (Detection of Artificially Generated Scientific Papers 2024)](https://www.codabench.org/competitions/2431/#/pages-tab) competition, organised for SDP 2024

## Environment setup

This project is setup with [Poetry](https://python-poetry.org/docs/) environment mangement, 
To install Poetry (Linux, macOS, Windows WSL), run:

```
curl -sSL https://install.python-poetry.org | python3 -
```

Then, to set up the env and dependecies, run:

```
poetry install
```

For more info on Poetry, please visit [the poetry docs](https://python-poetry.org/docs/).

## Data

- [Download training dataset](https://drive.google.com/file/d/1hJ-JtC0i8LBpD1hF3xWfRjkax42uE2NP/)
- [Download dev dataset](https://drive.google.com/file/d/1rurhsY7cbS1JoYtE4h2-vTVFUdMP8fFo/)

### Train data

The training set has the following columns:

- text – an excerpt from the article's full text;
- tokens – same as text, split by whitespaces;
- annotations – a list of triples [[start_id, end_id, label_id]] where ids indicate start and end token ids of a span, and label_id indicates its provenance ('human', 'NLTK_synonym_replacement', 'chatgpt', or 'summarized');
- token_label_ids – a list mapping each token from tokens with a corresponding label id (from 0 to 3), according to annotations.
```
>>> train_df = pd.read_parquet('train_data.parquet', engine='fastparquet')
>>> train_df[['text', 'tokens']].head(2)
	                                            text	                        annotations
index		
15096	Across the world, Emergency Departments are fa...	[[0, 3779, human], [3780, 7601, NLTK_synonym_r...
14428	lung Crab is the in the lead make of cancer-re...	[[0, 4166, NLTK_synonym_replacement], [4167, 2...


>>> train_df[["tokens", "token_label_ids"]].head(2)
	                                            tokens	                    token_label_ids
index		
15096	[Across, the, world,, Emergency, Departments, ...	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
14428	[lung, Crab, is, the, in, the, lead, make, of,...	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...

```

### Dev / Test data

The development and test sets have the following columns:

- text
- tokens
```
>>> dev_df = pd.read_parquet('dev_data.parquet', engine='fastparquet')
>>> dev_df.head()
                                                    text                                             tokens
index                                                                                                      
12313  Phylogenetic networks are a generalization of ...  [Phylogenetic, networks, are, a, generalizatio...
3172   Prediction modelling is more closely aligned w...  [Prediction, modelling, is, more, closely, ali...
6451   The heat transfer exhibits the flow of heat (t...  [The, heat, transfer, exhibits, the, flow, of,...
4351   a common experience during superficial ultraso...  [a, common, experience, during, superficial, u...
22694  Code metadata Current code version v1.5.9 Perm...  [Code, metadata, Current, code, version, v1.5....
```

## Evaluation

We're using Macro F1 score on `token_label_ids`. For each Full text, we're tokenizing/splitting the text on whitespace, and labeling each token. The final score is the average f1 across all full texts in the test set.

You can test your solution's performance offline by using the provided evaluation script [eval_f1.py](src/eval_f1.py)

Usage:
```
poetry run python -m src.eval_f1 --true_labels_file <file-in-data-dir-with-true-labels>.parquet --pred_file predictions.parquet
```

> Both files (true labels file and predictions file) must be located in [data](data)

### To submit a solution:

1. Prepare a Parquet file **predictions.parquet**(filename matters) with index "index" and a column "preds" structured as follows:

index      preds
14104    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
21468    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, ...
The length of a list "preds" should match the corresponding length of a "tokens" list in the dev/test set (i.e., one prediction per token).

Here preds is a list of integers from 0 to 3 corresponding to token-level predictions:

- 0 for 'human'
- 1 for 'NLTK_synonym_replacement'
- 2 for 'chatgpt'
- 3 for 'summarized'

See "[sample_dev_submission.zip](https://drive.google.com/file/d/1Xm59UlLJ-aemDG-PZ3SCBJa_5EkzGLXq)" (from the [Data page](https://www.codabench.org/competitions/2431/#/pages-tab)) for an example. It is an all-zero submission that corresponds to predicting that all text is human-written.

2. Compress (zip) the prediction file, for example, with a bash command:

```
zip my_new_submission.zip predictions.parquet
```

Please note that you should not compress the folder with the prediction files. Instead, you need to compress the files themselves: i.e., if you are submitting from Macbook (and you do not want to compress via terminal), you need to select all the prediction files that you want to send and press "Compress".

3. Submit your solution via the My submissions tab.

## Baseline

We're also providing a baseline where we use [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) to approach the task as a token classification task. To run the baseline, execute:

```
poetry run python -m src.ml_gen_detection.dagpap24_baseline
```

|         **Pred model** 	| **Average Macro Macro F1 Score** 	|
|-----------------------:	|-----------------------------------|
|        All zeros score 	|                            0.36 	|
| DistilBERT (20 epochs) 	|                            0.84 	|