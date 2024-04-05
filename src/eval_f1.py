import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import f1_score

from src.utils import get_project_root

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    """
    This is the evaluation script. For each text in the test set, we calculate the
    weighted average F1 score across all tokens, and for the final score
    we average all the F1 scores.
    """
    parser = argparse.ArgumentParser(
        description="Competition evaluation script"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config_baseline.yml",
    )

    parser.add_argument(
        "--true_labels_file", type=str, default="train_data.parquet"
    )

    parser.add_argument(
        "--pred_file", type=str, default="preds_random.parquet"
    )

    args = parser.parse_args()

    # loading config params
    project_root: Path = get_project_root()
    with open(str(project_root / "config" / args.config_file)) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # import true labels
    path_to_test_data = str(
        project_root / params["data"]["path_to_data"] / args.true_labels_file
    )
    logger.info(f"Loading true labels from {path_to_test_data}")
    true_labels = pd.read_parquet(
        path_to_test_data,
        engine="fastparquet",
        columns=["index", "token_label_ids"],
    )
    if true_labels.index.name != "index":
        true_labels.set_index("index", inplace=True)

    try:
        assert true_labels.index.name == "index" and true_labels.columns.values.tolist() == ["token_label_ids"]
        logger.info("True labels file format OK")
    except AssertionError:
        logger.error("True label file format error")

    # import pred labels
    path_to_pred_labels = str(
        project_root / params["data"]["path_to_data"] / args.pred_file
    )
    logger.info(f"Loading pred labels from {path_to_pred_labels}")
    pred_labels = pd.read_parquet(path_to_pred_labels)

    if pred_labels.index.name != "index":
        if "index" in pred_labels.columns:
            pred_labels.set_index("index", inplace=True)    
        else:
            logger.error("Pred file does not contain 'index' column")
            raise KeyError("Pred file does not contain 'index' column")
    elif pred_labels.columns.values.tolist() == ["preds"]:
        logger.info("Pred file format OK") 
    
    try:
        assert set(pred_labels.index).intersection(set(true_labels.index)) == set(pred_labels.index)
        logger.info("Pred indexes OK")
    except AssertionError:
        logger.error("Pred file does not contain all indexes of true file")
        raise KeyError("Pred file does not contain all indexes of true file")

    df = pred_labels.join(true_labels, how="left")

    try:
        assert len(df) == len(pred_labels)
    except AssertionError:
        logger.error("data files len mismatch")

    logger.info(f"Getting macro average F1-Score per text")
    
    df["f1_score"] = df.apply(
        lambda x: f1_score(
            x["token_label_ids"], x["preds"], average="macro"
        ),
        axis=1,
    )

    logger.info(
        f"Average F1 score across all labels is: {df['f1_score'].mean()*100:0.4f}%"
    )
