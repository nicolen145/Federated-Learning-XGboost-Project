"""my-app: A Flower / XGBoost app."""

import warnings

import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict

from my_app_xgboost.task import load_data, replace_keys


from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss, 
)

warnings.filterwarnings("ignore", category=UserWarning)

# Flower ClientApp
app = ClientApp()

def _local_boost(bst_input, num_local_round, train_dmatrix):
    # Update trees based on local training data.
    for i in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

    # Bagging: extract the last N=num_local_round trees for sever aggregation
    bst = bst_input[
        bst_input.num_boosted_rounds()
        - num_local_round : bst_input.num_boosted_rounds()
    ]
    return bst


@app.train()
def train(msg: Message, context: Context) -> Message:
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_dmatrix, _, _, num_train, _, _ = load_data(partition_id, num_partitions)

    # Read from run config
    num_local_round = context.run_config["local-epochs"]
    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    global_round = msg.content["config"]["server-round"]
    if global_round == 1:
        # First round local training
        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=num_local_round,
        )
    else:
        bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())

        # Load global model into booster
        bst.load_model(global_model)

        # Local training
        bst = _local_boost(bst, num_local_round, train_dmatrix)

    # Save model
    local_model = bst.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)

    # Construct reply message
    # Note: we store the model as the first item in a list into ArrayRecord,
    # which can be accessed using index ["0"].
    model_record = ArrayRecord([model_np])
    metrics = {
        "num-examples": num_train,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global XGBoost model on the local validation set (binary)."""

    # Load local data 
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valid_dmatrix, test_dmatrix, _, num_val, num_test = load_data(partition_id, num_partitions)

    # Load XGBoost params from run_config 
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    # Load global model from the server message 
    bst = xgb.Booster(params=params)

    global_model_bytes = msg.content["arrays"]["0"].numpy().tobytes()
    global_model = bytearray(global_model_bytes)

    bst.load_model(global_model)

    # Predict on validation set
    y_true = valid_dmatrix.get_label().astype(int)

    # For binary:logistic, predict returns probabilities for class 1
    y_pred_proba = bst.predict(valid_dmatrix)  # shape (n,) or (n, 1)

    # Handle both (n,) and (n,1)
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba[:, 0]

    # Threshold at 0.5 to get class labels
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Compute metrics (binary classification) 
    auc = float(roc_auc_score(y_true, y_pred_proba))
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    loss = float(log_loss(y_true, y_pred_proba))

    metrics = {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": loss,
        "num-examples": int(num_val),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)


