from pathlib import Path
import joblib
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from typing import Iterable, Mapping, Tuple, Optional, Literal, Any, Sequence, Dict
from types import ModuleType
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve, classification_report, log_loss
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM, SVC
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Optional TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Reshape, Input, Embedding
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from detector_framework.data_processing.processing import (
    ModelSettings, RegressionData, get_windows_and_futures, preproc_transform, form_one_hot_encoder
)

# Configure Matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
else:
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        matplotlib.use('Agg')
plt.ion()


def get_keyword_filenames(keyword, file_dir: Path):
    """
    Retrieve a list of filenames in a provided directory matching keyword.
    """
    keywords = {keyword} if isinstance(keyword, str) else set(keyword)
    return [p.name for p in file_dir.iterdir() if p.is_file() and any(kw in p.name for kw in keywords)]


def validate_settings(model_settings: ModelSettings, valid_models: set, valid_modes: set):
    if model_settings.model_type not in valid_models:
        raise ValueError(f"model_type must be one of {sorted(valid_models)!r}, got {model_settings.model_type!r}")

    if model_settings.preproc_approach not in valid_modes:
        raise ValueError(f"preproc_approach must be one of {sorted(valid_modes)!r}, got {model_settings.preproc_approach!r}")

    if model_settings.model_type not in str(model_settings.model_path):
        raise AssertionError(f"model_type {model_settings.model_type!r} not in model_path {model_settings.model_path!r}")


def calculate_sample_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    return class_weights[y.astype(int)]


def roc_auc_plot(y_test: np.ndarray, y_scores: np.ndarray, sample_weight=None) -> None:
    auc = roc_auc_score(y_test, y_scores, sample_weight=sample_weight)
    fpr, tpr, _ = roc_curve(y_test, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show(block=True)


def score_plot(y_pred: np.ndarray, malware_test: np.ndarray):
    len_benign = len(y_pred) - len(malware_test)
    benign_x = np.arange(len_benign)
    benign_y = y_pred[benign_x]

    len_malware = len(malware_test)
    malware_x = np.arange(len_benign, len_benign + len_malware)
    malware_y = y_pred[malware_x]

    _, ax = plt.subplots(1, 1, figsize=(10, 4), sharey=True)
    ax.plot(benign_x, benign_y, color="blue")
    ax.plot(malware_x, malware_y, color="red")
    plt.tight_layout()
    plt.show(block=True)


if TF_AVAILABLE:
    class LSTMWrapper:
        def __init__(self, X_train: np.ndarray):
            embedding_vector_length = 32
            self.model = Sequential()
            self.model.add(Embedding(int(np.max(X_train)) + 100, embedding_vector_length))
            self.model.add(LSTM(100))
            self.model.add(Dense(1, activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)

        def predict(self, X_test: np.ndarray) -> np.ndarray:
            probas = self.model.predict(X_test)
            return (probas > 0.5).astype(int).reshape(-1)

        def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
            return self.model.predict(X_test)
else:
    class LSTMWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow is not available for LSTMWrapper")


def regression_error(model_settings: ModelSettings, regression_data: RegressionData):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for regression_error (LSTM)")

    benign_wdws = regression_data.benign_windows
    benign_futures = regression_data.benign_futures
    malware_wdws = regression_data.malware_windows
    malware_futures = regression_data.malware_futures

    unique_vals = np.unique(benign_wdws.reshape(-1))
    lookup = np.full(unique_vals.max() + 1, -1, dtype=int)
    lookup[unique_vals] = np.arange(len(unique_vals))

    mapped_benign_wdws = lookup[benign_wdws]
    mapped_benign_futures = lookup[benign_futures]
    mapped_malware_wdws = lookup[malware_wdws]
    mapped_malware_futures = lookup[malware_futures]

    X_train, X_test, y_train, y_test = train_test_split(
        mapped_benign_wdws, mapped_benign_futures, test_size=0.3, random_state=42
    )

    X_test = np.concatenate((X_test, mapped_malware_wdws))
    y_test = np.concatenate((y_test, mapped_malware_futures))

    enc = form_one_hot_encoder(X_train)

    def encode_and_reshape(data):
        a, b = data.shape
        encoded = enc.transform(data.reshape(-1, 1)).toarray()
        return encoded.reshape(a, b, -1)

    X_train = encode_and_reshape(X_train)
    X_test = encode_and_reshape(X_test)
    y_train = encode_and_reshape(y_train)
    y_test = encode_and_reshape(y_test)

    scaler = StandardScaler().fit(X_train.reshape(-1, 1))

    def scale_and_reshape(data):
        original_shape = data.shape
        scaled = scaler.transform(data.reshape(-1, 1))
        return scaled.reshape(original_shape)

    X_train = scale_and_reshape(X_train)
    X_test = scale_and_reshape(X_test)
    y_train = scale_and_reshape(y_train)
    y_test = scale_and_reshape(y_test)

    if not model_settings.new_model and os.path.exists(model_settings.model_path):
        print(f"Loading model from {model_settings.model_path}")
        model = load_model(model_settings.model_path)
    else:
        tf.random.set_seed(42)

        split = int(0.8 * len(X_train))
        X_train_final, X_val = X_train[:split], X_train[split:]
        y_train_final, y_val = y_train[:split], y_train[split:]

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=1e-4,
            restore_best_weights=True
        )

        model = Sequential([
            Input(shape=(model_settings.window_length, len(enc.categories_[0]))),
            Dense(64, activation='relu'),
            LSTM(64),
            Dense(len(enc.categories_[0]), activation='linear'),
            Reshape((1, len(enc.categories_[0])))
        ])

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=300, batch_size=64, verbose=2,
            callbacks=[early_stop]
        )
        model.save(model_settings.model_path)

    y_pred = model.predict(X_test)
    deltas = np.sum(np.abs(y_pred - y_test), axis=2)

    if model_settings.plot:
        score_plot(deltas, mapped_malware_wdws)

    y_discrete = np.zeros(len(y_test))
    y_discrete[-len(malware_futures):] = 1
    sample_weights = calculate_sample_weights(y_discrete)

    auc = roc_auc_score(y_discrete, deltas, sample_weight=sample_weights)
    print(f"ROC AUC Score: {auc:.3f}")

    if model_settings.plot:
        roc_auc_plot(y_discrete, deltas, sample_weights)


def binary_supervised_error(model_settings: ModelSettings, benign: np.ndarray, malware: np.ndarray) -> None:
    model_type = model_settings.model_type

    X = np.concatenate((benign, malware))
    y = np.zeros(len(benign) + len(malware))
    y[len(benign):] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    if model_type == "xgb":
        pipeline = XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42
        )
    elif model_type == "svc":
        pipeline = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', probability=True, class_weight="balanced")
        )
    elif model_type == "lstm":
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        tf.random.set_seed(42)
        pipeline = LSTMWrapper(X_train)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    y_scores = pipeline.predict_proba(X_test)
    if y_scores.ndim > 1 and y_scores.shape[1] > 1:
        y_scores = y_scores[:, 1]

    sample_weights = calculate_sample_weights(y_test)
    auc = roc_auc_score(y_test, y_scores, sample_weight=sample_weights)
    print(f"ROC AUC Score: {auc:.3f}")

    if model_settings.plot:
        roc_auc_plot(y_test, y_scores, sample_weights)


def unsupervised_error(model_settings: ModelSettings, benign: np.ndarray, malware: np.ndarray):
    model_type = model_settings.model_type

    X_train, X_test_benign = train_test_split(benign, test_size=0.3, random_state=42)

    X_test = np.concatenate((X_test_benign, malware))
    y_test = np.zeros(len(X_test))
    y_test[:len(X_test_benign)] = 1

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "isolation_forest": IsolationForest(contamination=0.01),
        "minimum_covariance_determinant": EllipticEnvelope(contamination=0.01),
        "local_outlier_factor": LocalOutlierFactor(contamination=0.01, novelty=True),
        "svc": OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    }

    if model_type not in models:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = models[model_type]
    model.fit(X_train)
    y_pred = model.score_samples(X_test)

    sample_weights = calculate_sample_weights(y_test)
    auc = roc_auc_score(y_test, y_pred, sample_weight=sample_weights)
    print(f"ROC AUC Score: {auc:.3f}")

    if model_settings.plot:
        score_plot(y_pred, malware)
        roc_auc_plot(y_test, y_pred, sample_weights)


def multiclass_error(model_settings: ModelSettings, X: np.ndarray, y: np.ndarray):
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    sample_weights_train = calculate_sample_weights(y_train)

    lb = LabelBinarizer()
    lb.fit(y_train)
    y_test_ohe = lb.transform(y_test)

    dtree_model = DecisionTreeClassifier(max_depth=7).fit(X_train, y_train, sample_weight=sample_weights_train)
    y_pred_probas = dtree_model.predict_proba(X_test)

    sample_weights_test = calculate_sample_weights(y_test)

    loss_ohe = log_loss(y_test_ohe, y_pred_probas, sample_weight=sample_weights_test)
    print(f"Log Loss Score: {loss_ohe:.5f}")

    y_pred = lb.inverse_transform(y_pred_probas)
    print(classification_report(y_test, y_pred, sample_weight=sample_weights_test))

    if model_settings.plot:
        cm = confusion_matrix(y_test, y_pred, sample_weight=sample_weights_test, normalize='true')
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.DataFrame(cm), annot=True, fmt='.2f', cmap='Blues',
                    cbar_kws={'label': 'Normalized Count'})
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show(block=True)

    model_path = Path(model_settings.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path = Path(model_settings.settings_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(dtree_model, model_settings.model_path, compress=("zlib", 3))
    joblib.dump(model_settings, model_settings.settings_path, compress=("zlib", 3))

    return loss_ohe


def regression_analysis(model_settings, benign_path, benign_list, malware_path, malware_list):
    validate_settings(model_settings, {"lstm"}, {"windowed"})
    benign_wdws, benign_futures = get_windows_and_futures(model_settings, benign_path, benign_list)
    malware_wdws, malware_futures = get_windows_and_futures(model_settings, malware_path, malware_list)

    malware_wdws = malware_wdws[:5000]
    malware_futures = malware_futures[:5000]

    regression_data = RegressionData(benign_wdws, benign_futures, malware_wdws, malware_futures)
    regression_error(model_settings, regression_data)


def binary_supervised_analysis(model_settings, benign_path, benign_list, malware_path, malware_list):
    validate_settings(model_settings, {"svc", "xgb", "lstm"},
                      {"zero-padded_trace", "syscall_frequency", "windowed_features", "windowed"})
    benign_transformed = preproc_transform(model_settings, benign_path, benign_list)
    malware_transformed = preproc_transform(model_settings, malware_path, malware_list)

    if model_settings.preproc_approach == "windowed":
        benign_transformed = benign_transformed[0]
        malware_transformed = malware_transformed[0]

    malware_transformed = malware_transformed[:5000]

    binary_supervised_error(model_settings, benign_transformed, malware_transformed)


def unsupervised_analysis(model_settings, benign_path, benign_list, malware_path, malware_list):
    validate_settings(model_settings, {"isolation_forest", "minimum_covariance_determinant", "local_outlier_factor", "svc"},
                      {"zero-padded_trace", "syscall_frequency", "windowed_features", "windowed"})
    benign_transformed = preproc_transform(model_settings, benign_path, benign_list)
    malware_transformed = preproc_transform(model_settings, malware_path, malware_list)

    if model_settings.preproc_approach == "windowed":
        benign_transformed = benign_transformed[0]
        malware_transformed = malware_transformed[0]

    malware_transformed = malware_transformed[:5000]

    unsupervised_error(model_settings, benign_transformed, malware_transformed)


def multiclass_analysis(model_settings, benign_path, benign_dict: dict, malware_path, malware_dict: dict):
    transformed_data = []
    labels = []

    for key, val in {**benign_dict, **malware_dict}.items():
        path = benign_path if key in benign_dict else malware_path
        tmp_list = get_keyword_filenames(val, path)
        transformed = preproc_transform(model_settings, path, tmp_list)
        transformed_data.append(transformed)
        labels.append(np.full(len(transformed), key))

    X = np.concatenate(transformed_data)
    y = np.concatenate(labels)

    return multiclass_error(model_settings, X, y)


def get_prescored_predictions(stage_keys: list, stage_windows: list, prescored_dir: Path) -> (np.ndarray, np.ndarray):
    trace_classes = []
    trace_values = []

    for ttp, window_seq_len in zip(stage_keys, stage_windows):
        prescored_path = prescored_dir / f"{ttp}_prescored.joblib"
        label_class, label_val = joblib.load(prescored_path)

        idx = np.random.randint(0, len(label_class) - window_seq_len)

        trace_classes.append(label_class[idx:idx + window_seq_len])
        trace_values.append(label_val[idx:idx + window_seq_len])

    return np.concatenate(trace_classes), np.concatenate(trace_values)


def get_live_predictions(stage_keys: list, stage_windows: list, classifier, model_settings: ModelSettings,
                         malware_path: Path) -> (np.ndarray, np.ndarray):
    trace_classes = []
    trace_values = []

    for ttp, window_seq_len in zip(stage_keys, stage_windows):
        malware_list = get_keyword_filenames(ttp, malware_path)
        transformed = preproc_transform(model_settings, malware_path, malware_list)

        idx = np.random.randint(0, len(transformed) - window_seq_len)
        transformed = transformed[idx:idx + window_seq_len]

        y_pred_ohe = classifier.predict_proba(transformed)
        label_class = np.argmax(y_pred_ohe, axis=1)
        label_val = y_pred_ohe[np.arange(len(label_class)), label_class]

        trace_classes.append(label_class)
        trace_values.append(label_val)

    return np.concatenate(trace_classes), np.concatenate(trace_values)


def files_and_labels_to_X_y(
    paths: Iterable[Path],
    signal_module: ModuleType,
    malware_map: Mapping[int, list],
    window_size_time: float,
    window_stride_time: float,
    train_test_split: float,
    *,
    strict: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y) from a collection of files.

    - Each file is turned into a feature matrix X_i via `signal_module.file_df_feature_extraction`.
    - Its label is taken from `malware_map[file_path.name]` and broadcast to the number of rows in X_i.
    - All X_i are concatenated along axis 0; same for y_i.

    Parameters
    ----------
    paths : iterable of Path
        Files to process.
    signal_module : module or object
        Must provide `get_file_df(Path)` and `file_df_feature_extraction(df, window_size_time, window_stride_time)`.
    malware_map : dict-like
        Maps filename (str) → integer label.
    window_size_time, window_stride_time : floats
        Feature extraction parameters.
    strict : bool
        If True, raise on missing label/file issues; if False, skip problematic files.
    **kwargs : Any
        Additional keyword arguments passed to the signal module's feature extraction functions.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for p in paths:
        label = None
        for key, malware_list in malware_map.items():
            if any(malware in p.name for malware in malware_list):
                label = key
                break

        if label is None:
            if strict:
                raise KeyError(f"No label found in malware_map for file: {p.name}")
            continue

        df = signal_module.get_file_df(p, **kwargs)

        extract = getattr(signal_module, "file_df_feature_extraction_parallel", None)
        if extract is None:
            extract = getattr(signal_module, "file_df_feature_extraction")

        X_i = extract(df, window_size_time, window_stride_time, **kwargs)

        # Skip files that produced zero windows (optional)
        if X_i is None or X_i.size == 0:
            if strict:
                # If strict, consider zero-window an error
                raise ValueError(f"Feature extraction produced no rows for: {p}")
            else:
                continue

        y_i = np.full(X_i.shape[0], label, dtype=np.int32)

        X_list.append(X_i)
        y_list.append(y_i)

    if not X_list:
        # No data; return empty shapes
        return np.empty((0, 0), dtype=float), np.empty((0,), dtype=np.int32), np.empty((0, 0), dtype=float), np.empty((0,), dtype=np.int32)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Ensure y is integer-typed
    if not np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.int32, copy=False)

    labels = np.unique(y)

    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    for label in labels:
        tmp_X = X[y == label]
        tmp_y = y[y == label]

        n = tmp_X.shape[0]
        idx = int(np.floor(train_test_split * n))
        X_train, X_test = np.split(tmp_X, [idx], axis=0)
        y_train, y_test = np.split(tmp_y, [idx], axis=0)

        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    return X_train, y_train, X_test, y_test


def compute_train_test_sample_weights(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    """
    Compute class-balanced sample weights for train/test.

    We derive class weights from the *training* distribution, then apply
    those weights to both y_train and y_test to get per-sample weights.

    Returns
    -------
    train_sample_weights : (n_train,) float array
    test_sample_weights  : (n_test,) float array
    class_weight_map     : dict[label -> weight]
    """
    y_train = np.asarray(y_train).ravel()
    y_test  = np.asarray(y_test).ravel()

    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )  # shape: (n_classes,)

    class_weight_map: Dict[int, float] = {
        int(label): float(w) for label, w in zip(classes, class_weights)
    }

    # Optional: sanity-check that test set doesn't contain unseen labels
    unseen = np.setdiff1d(np.unique(y_test), classes)
    if unseen.size > 0:
        raise ValueError(
            f"y_test contains labels not seen in y_train: {unseen.tolist()}"
        )

    train_sample_weights = compute_sample_weight(class_weight=class_weight_map, y=y_train)
    test_sample_weights  = compute_sample_weight(class_weight=class_weight_map, y=y_test)

    return train_sample_weights, test_sample_weights, class_weight_map


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    *,
    max_depth: int = None,
    random_state: Optional[int] = 42,
) -> tuple[DecisionTreeClassifier, object, float]:
    """
    Train a DecisionTreeClassifier with required per-sample weights.

    Returns
    -------
    (DecisionTreeClassifier, LabelBinarizer, score)
    """
    y_arr = np.asarray(y_train).ravel()
    n_samples = np.shape(X_train)[0]

    w = np.asarray(sample_weight, dtype=float).ravel()
    if w.shape[0] != n_samples or y_arr.shape[0] != n_samples:
        raise ValueError(
            f"Inconsistent lengths: X={n_samples}, y={y_arr.shape[0]}, sample_weight={w.shape[0]}"
        )
    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative.")
    if not np.isfinite(w).all():
        raise ValueError("sample_weight must be finite.")
    if w.sum() == 0:
        raise ValueError("sample_weight must not sum to zero.")

    lb = LabelBinarizer().fit(y_arr)

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_arr, sample_weight=w)
    score = clf.score(X_train, y_arr, sample_weight=w)

    return clf, lb, score


def prediction_analysis(
    y_true: np.ndarray,                  # (n,) labels OR (n,k) one-hot
    y_proba: np.ndarray,                 # (n,k) predicted probabilities
    *,
    lb: LabelBinarizer,                  # fitted on the same class set
    sample_weight: Optional[np.ndarray] = None,
    plot: bool = False,
    normalize: Literal['true','pred','all', None] = 'true',
    ax: Optional[plt.Axes] = None,
    title: str = "Confusion Matrix",
) -> Dict[str, Any]:
    """
    Compute log-loss, classification report, and (optionally) plot a confusion matrix.
    Returns a dict with 'log_loss', 'classification_report', and 'confusion_matrix'.
    """
    if y_true.ndim == 2:
        y_true_labels = lb.inverse_transform(y_true)
    else:
        y_true_labels = y_true

    y_pred_labels = lb.inverse_transform(y_proba)

    loss = log_loss(y_true, y_proba, sample_weight=sample_weight)
    report = classification_report(y_true_labels, y_pred_labels, sample_weight=sample_weight)
    cm = confusion_matrix(y_true_labels, y_pred_labels, sample_weight=sample_weight, normalize=normalize)

    print(f"Log Loss: {loss:.4f}")
    print("Classification Report:\n", report)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=lb.classes_,
            yticklabels=lb.classes_,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        if ax is None:
            plt.show()

    return {
        "log_loss": loss,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def train_and_test_report(X: np.ndarray, y: np.ndarray) -> None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    sw_train, sw_test, _ = compute_train_test_sample_weights(y_train, y_test)
    clf, lb, _ = train_model(X_train, y_train, sw_train)
    y_proba = clf.predict_proba(X_test)
    prediction_analysis(y_test, y_proba, lb=lb, sample_weight=sw_test, plot=True)


def train_and_save_model(X: np.ndarray, y: np.ndarray, save_path: Path) -> float:
    sample_weights, _, _ = compute_train_test_sample_weights(y, y)
    clf, lb, score = train_model(X, y, sample_weights)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, save_path)
    return score


def build_features(signal_df_dict, signal_modules, window_size_time, window_stride_time, feature_dict=None, *,
                   preserve_time=True):
    if feature_dict is None:
        feature_dict = {}

    for signal, actions in tqdm(signal_df_dict.items()):
        mod = signal_modules[signal]
        # Prefer a parallel extractor if the module provides one; else fall back
        extract = getattr(mod, "file_df_feature_extraction_parallel", None)
        if extract is None:
            extract = getattr(mod, "file_df_feature_extraction")
        for action, df_list in actions.items():
            rows = [extract(df, window_size_time, window_stride_time, preserve_time=preserve_time)
                    for df in df_list]

            feature_dict.setdefault(signal, {})
            feature_dict[signal].setdefault(action, [])
            out = feature_dict[signal][action]
            out.extend(rows)

    return feature_dict


def outer_train_loop(parameter_dict: dict, window_size_time: float, window_stride_time: float, tts: float, train: bool) -> list:
    train_scores = []

    for params in parameter_dict.values():
        if not params["signal_selection"]:
            continue

        data_dir = params["data_dir"]
        malware_dict = params["malware_dictionary"]
        signal_fe = params["feature_extraction_module"]
        save_path = params["save_path"]

        data_paths = [p for p in data_dir.iterdir() if p.is_file()]
        data_paths.sort()

        malware_keys = [item for sublist in malware_dict.values() for item in sublist]
        malware_keys = set(malware_keys)

        filtered = [
            path for path in data_paths
            if any(key in path.name for key in malware_keys)
        ]
        data_paths = filtered

        X, y, xt, yt = files_and_labels_to_X_y(
            data_paths,
            signal_fe,
            malware_dict,
            window_size_time,
            window_stride_time,
            train_test_split=tts
        )

        if train:
            train_score = train_and_save_model(X, y, save_path)
            train_scores.append(train_score)
            print(f"Train Score: {train_score}")
        else:
            train_and_test_report(X, y)

    return train_scores
