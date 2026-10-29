"""Shared TensorFlow model building, fold fitting, and experiment helpers."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..features import feature_table
from ..preprocessing import Preprocessor


def tensorflow():
    """Import the tested backend only when training/inference needs it."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Install turbopdz[train] or turbopdz[inference] for TensorFlow models") from exc
    return tf


def feature_config(config: dict, track: str) -> dict:
    """Use common features with an independent override for each model track."""
    result = {key: deepcopy(value) for key, value in config["features"].items() if key not in {"zml", "sigmaml"}}
    result.update(deepcopy(config["features"].get(track, {})))
    return result


def training_config(config: dict, track: str) -> dict:
    """Resolve independent sigmaML training overrides without inheriting zML HPO."""
    result = deepcopy(config["training"])
    if track == "sigmaml":
        result.pop("params_path", None)
        sigma = config["sigmaml"]
        for key in ("epochs", "loss", "verbose", "patience", "reduce_lr_patience",
                    "reduce_lr_factor", "prefetch", "sample_weight_column", "params_path"):
            if key in sigma:
                result[key] = deepcopy(sigma[key])
        result["params"].update(deepcopy(sigma.get("params", {})))
        result["hpo"] = deepcopy(sigma.get("hpo", {"mode": "fixed"}))
        result["loss"] = sigma.get("loss", "mae")
    elif track != "zml":
        raise ValueError("Model track must be zml or sigmaml")
    return result


def selected_parameters(training: dict) -> tuple[dict, int]:
    """Reuse explicitly exported parameters/epochs, never source model weights."""
    source = training.get("params_path") or training.get("hpo", {}).get("params_path")
    if source:
        values = json.loads(Path(source).read_text())
        return values.get("best_params", values), int(values.get("final_epochs", training["epochs"]))
    if training.get("hpo", {}).get("mode") == "reuse_parameters":
        raise ValueError("reuse_parameters requires an explicit params_path")
    return deepcopy(training["params"]), int(training["epochs"])


def build_model(input_dim: int, params: dict, *, seed: int, track: str,
                z_max: float, loss: str = "mae"):
    """Build the paper's dense/dropout MLP, with sigmoid zML or linear log-scale."""
    tf = tensorflow()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,))], name=f"turboPDZ_{track}")
    for layer in range(int(params["n_layers"])):
        model.add(tf.keras.layers.Dense(int(params.get(f"n_units_l{layer}", 64)),
                                       activation=params.get("activation", "relu"),
                                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+layer)))
        model.add(tf.keras.layers.Dropout(float(params.get("dropout_rate", 0)), seed=seed+layer))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid" if track == "zml" else "linear"))
    if track == "zml":
        if z_max <= 0:
            raise ValueError("Sigmoid zML requires a positive maximum redshift")
        model.add(tf.keras.layers.Rescaling(float(z_max)))
    model.compile(optimizer=tf.keras.optimizers.AdamW(
        learning_rate=float(params.get("learning_rate", 0.001)),
        weight_decay=float(params.get("weight_decay", 0))), loss=loss)
    return model


def _batches(x, y, weights, batch_size: int, *, seed: int, shuffle: bool, prefetch: int):
    tf = tensorflow()
    values = (x, y) if weights is None else (x, y, weights)
    dataset = tf.data.Dataset.from_tensor_slices(values)
    if shuffle:
        dataset = dataset.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.batch(batch_size).with_options(options).prefetch(prefetch)


def predict_model(model, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Batch eager inference, avoiding machine-dependent tf.data thread pools."""
    predictions = [np.asarray(model(x[start:start+batch_size], training=False)).ravel()
                   for start in range(0, len(x), batch_size)]
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)


def fit_rows(dataset, features, fit_idx, params, *, config, track="zml", targets=None,
             valid_idx=None, epochs=None, seed=None):
    """Fit preprocessing/model using only explicit fit indices and optional validation."""
    tf = tensorflow()
    tf.keras.backend.clear_session()
    seed = config["seed"] if seed is None else seed
    fcfg = feature_config(config, track)
    pp = Preprocessor.fit(dataset.pdz[fit_idx], features.iloc[fit_idx],
                          variance=params.get("explained_variance_pdf", 0.99),
                          use_pca=fcfg.get("pdz_pca", True), missing=fcfg.get("missing", "raise"), seed=seed)
    x = pp.transform(dataset.pdz[fit_idx], features.iloc[fit_idx])
    labels = dataset.z_spec if targets is None else targets
    training = training_config(config, track)
    model = build_model(x.shape[1], params, seed=seed, track=track,
                        z_max=float(config.get("dataset", {}).get("z_max", dataset.z_grid.max())),
                        loss=training.get("loss", "mae"))
    weights = None
    weight_column = training.get("sample_weight_column")
    if weight_column:
        weights = dataset.observables[weight_column].to_numpy(dtype=np.float32)[fit_idx]
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Sample weights must be finite, non-negative, and not all zero")
        weights = weights / weights.mean()
    batch_size = int(params.get("batch_size", 128))
    train = _batches(x, np.asarray(labels[fit_idx], dtype=np.float32), weights, batch_size,
                     seed=seed, shuffle=True, prefetch=int(training.get("prefetch", 1)))
    validation = None
    callbacks = []
    if valid_idx is not None:
        valid_x = pp.transform(dataset.pdz[valid_idx], features.iloc[valid_idx])
        validation = _batches(valid_x, np.asarray(labels[valid_idx], dtype=np.float32), None,
                              batch_size, seed=seed, shuffle=False, prefetch=1)
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=training.get("patience", 18),
                                                      restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(patience=training.get("reduce_lr_patience", 6),
                                                          factor=training.get("reduce_lr_factor", 0.5), min_lr=1e-5)]
    # tf.data performs the shuffling above; Keras' array-only shuffle flag is
    # ignored for datasets and otherwise emits a warning for every fold.
    history = model.fit(train, validation_data=validation, epochs=int(epochs or training["epochs"]),
                        shuffle=False, callbacks=callbacks, verbose=int(training.get("verbose", 1)))
    model._turbopdz_history = pd.DataFrame(history.history).rename_axis("epoch").reset_index()
    model._turbopdz_history["epoch"] += 1
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1) if validation is not None else len(history.history["loss"])
    return model, pp, best_epoch
