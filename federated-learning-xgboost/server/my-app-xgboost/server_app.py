"""my-app: A Flower / XGBoost app."""

import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging
from my_app_xgboost.task import replace_keys
import os
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from clearml import Task, Logger


# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    task = Task.init(
    project_name="fl_project",
    task_name=f"FL XGBoost | rounds={num_rounds} | clients=3",
    )
    task.connect({
        "num_rounds": num_rounds,
        "fraction_train": fraction_train,
        "fraction_evaluate": fraction_evaluate,
        "xgb_params": params,
    })
    logger = Logger.current_logger()


    # Init global model
    # Init with an empty object; the XGBooster will be created
    # and trained on the client side.
    global_model = b""
    # Note: we store the model as the first item in a list into ArrayRecord,
    # which can be accessed using index ["0"].
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    # Initialize FedXgbBagging strategy
    strategy = FedXgbBagging(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # Start strategy, run FedXgbBagging for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk (only if it exists)
    bst = xgb.Booster(params=params)

    arr_rec = result.arrays  # This should be an ArrayRecord or None

    if arr_rec is None:
        print("Warning: no global arrays returned by strategy, skipping model save.")
    else:
        # Try string key first (as in the official tutorial)
        key = "0"
        if key not in arr_rec:
            # Fallback: try integer key
            if 0 in arr_rec:
                key = 0
            else:
                print("Warning: no entry '0' (string) or 0 (int) in result.arrays, skipping model save.")
                return

        global_model = bytearray(arr_rec[key].numpy().tobytes())
        bst.load_model(global_model)

        print("\nSaving final model to disk...")
        bst.save_model("final_model.json")


    
    # ==========================
    # create metrics plots
    # ==========================

    # result.evaluate_metrics_clientapp: dict[round -> MetricRecord]
    eval_metrics = result.evaluate_metrics_clientapp

    if not eval_metrics:
        print("No evaluation metrics found, skipping plots.")
        return

    for r in sorted(eval_metrics.keys()):
        mr = eval_metrics[r]
        if "auc" in mr:
            logger.report_scalar("AUC", "global", float(mr["auc"]), r)
        if "accuracy" in mr:
            logger.report_scalar("Accuracy", "global", float(mr["accuracy"]), r)
        if "f1" in mr:
            logger.report_scalar("F1", "global", float(mr["f1"]), r)
        if "loss" in mr:
            logger.report_scalar("Loss", "global", float(mr["loss"]), r)


    # Make sure output dir exists
    os.makedirs("outputs", exist_ok=True)

    # Sort rounds
    rounds = sorted(eval_metrics.keys())

    # Helper: extract metric by name (if it exists)
    def extract_metric(name):
        values = []
        for r in rounds:
            mr = eval_metrics[r]  # MetricRecord behaves like dict
            if name in mr:
                values.append(float(mr[name]))
            else:
                values.append(None)
        return values

    auc = extract_metric("auc")
    acc = extract_metric("accuracy")
    f1  = extract_metric("f1")
    loss = extract_metric("loss")

    # Plot AUC
    if any(v is not None for v in auc):
        plt.figure()
        plt.plot(rounds, auc, marker="o")
        plt.xlabel("Round")
        plt.ylabel("AUC")
        plt.title("AUC per FL Round")
        plt.grid(True)
        plt.savefig(os.path.join("outputs", "auc_per_round.png"))
        plt.close()
        print("Saved outputs/auc_per_round.png")

    # Plot Accuracy (if exists)
    if any(v is not None for v in acc):
        plt.figure()
        plt.plot(rounds, acc, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title("Accuracy per FL Round")
        plt.grid(True)
        plt.savefig(os.path.join("outputs", "accuracy_per_round.png"))
        plt.close()
        print("Saved outputs/accuracy_per_round.png")

    # Plot F1 (if exists)
    if any(v is not None for v in f1):
        plt.figure()
        plt.plot(rounds, f1, marker="o")
        plt.xlabel("Round")
        plt.ylabel("F1")
        plt.title("F1 per FL Round")
        plt.grid(True)
        plt.savefig(os.path.join("outputs", "f1_per_round.png"))
        plt.close()
        print("Saved outputs/f1_per_round.png")

    # Plot Loss (if exists)
    if any(v is not None for v in loss):
        plt.figure()
        plt.plot(rounds, loss, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.title("Loss per FL Round")
        plt.grid(True)
        plt.savefig(os.path.join("outputs", "loss_per_round.png"))
        plt.close()
        print("Saved outputs/loss_per_round.png")


    # combined plot (AUC + Acc + F1)
    if any(v is not None for v in auc + acc + f1):
        plt.figure()
        if any(v is not None for v in auc):
            plt.plot(rounds, auc, marker="o", label="AUC")
        if any(v is not None for v in acc):
            plt.plot(rounds, acc, marker="o", label="Accuracy")
        if any(v is not None for v in f1):
            plt.plot(rounds, f1, marker="o", label="F1")

        plt.xlabel("Round")
        plt.ylabel("Score")
        plt.title("Metrics per FL Round")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join("outputs", "metrics_per_round.png"))
        plt.close()
        print("Saved outputs/metrics_per_round.png")


    
    # Save final model locally
    MODEL_PATH = os.path.join("outputs", "final_model.json")
    bst.save_model(MODEL_PATH)

    # Upload model to ClearML
    task.upload_artifact(name="final_model", artifact_object=MODEL_PATH)

    for fn in os.listdir("outputs"):
        if fn.lower().endswith(".png"):
            task.upload_artifact(fn, artifact_object=os.path.join("outputs", fn))

