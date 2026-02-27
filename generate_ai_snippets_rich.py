import random
import json
from typing import Dict, Any, List

OUTPUT_FILE = "ai_snippets_generated.jsonl"


# ============================
#  UZUN / ORTA AI PATTERNLERİ
# ============================

def pattern_simple_ai(i: int) -> str:
    return f'''
def compute_sum_{i}(values: list[Any]) -> int:
    """
    Computes the numeric sum with defensive checks.
    Includes validation, filtering and deterministic output.
    """
    if values is None:
        raise ValueError("Input cannot be None")

    clean = []
    for item in values:
        try:
            if isinstance(item, (int, float)):
                clean.append(float(item))
        except:
            continue
    return sum(clean)
    '''


def pattern_class_ai(i: int) -> str:
    return f'''
class DataNormalizer_{i}:
    """
    Provides normalization utilities for numeric sequences.
    Auto-detects invalid values and applies min-max scaling.
    """

    def __init__(self):
        self._cache = {{}}

    def _ensure_numeric(self, seq):
        clean = []
        for x in seq:
            if isinstance(x, (int, float)):
                clean.append(x)
        return clean

    def normalize(self, seq):
        nums = self._ensure_numeric(seq)
        if not nums:
            return []
        mn, mx = min(nums), max(nums)
        span = mx - mn if mx != mn else 1
        return [(x - mn) / span for x in nums]
    '''


def pattern_pipeline(i: int) -> str:
    return f'''
def pipeline_process_{i}(data):
    """
    Runs a multi-stage pipeline:
    - validation
    - flatten
    - filter numeric
    - compute stats
    """
    if not isinstance(data, (list, tuple)):
        raise TypeError("Pipeline expects a list or tuple")

    def flatten(x):
        for item in x:
            if isinstance(item, (list, tuple)):
                yield from flatten(item)
            else:
                yield item

    flat = list(flatten(data))
    nums = [x for x in flat if isinstance(x, (int, float))]

    if not nums:
        return {{}}

    return {{
        "mean": sum(nums)/len(nums),
        "min": min(nums),
        "max": max(nums)
    }}
    '''


def pattern_config(i: int) -> str:
    return f'''
class Processor_{i}:
    """
    Processor with artificial configuration structure.
    """

    def __init__(self, config: dict):
        self.config = config or {{}}

    def run(self, payload):
        mode = self.config.get("mode", "safe")
        if mode == "verbose":
            print("Verbose mode active")

        if not isinstance(payload, list):
            raise ValueError("Payload must be a list")

        nums = [x for x in payload if isinstance(x, (int, float))]
        return {{
            "count": len(nums),
            "sum": sum(nums),
            "avg": sum(nums)/len(nums) if nums else 0
        }}
    '''


def pattern_errors(i: int) -> str:
    return f'''
def safe_reduce_{i}(items):
    """
    Redundant multi-layer safe reducer.
    """
    if items is None:
        raise ValueError("Items cannot be None")

    total = 0
    for x in items:
        try:
            try:
                total += float(x)
            except TypeError:
                continue
        except:
            pass
    return total
    '''


def pattern_nested(i: int) -> str:
    return f'''
def nested_analysis_{i}(tree):
    """
    Performs unnecessarily nested traversal (AI-style).
    """

    def visit(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from visit(v)
        elif isinstance(node, list):
            for v in node:
                yield from visit(v)
        else:
            yield node

    flat = list(visit(tree))
    nums = [x for x in flat if isinstance(x, (int, float))]
    return {{
        "total": sum(nums),
        "len": len(nums)
    }}
    '''


def pattern_module(i: int) -> str:
    return f'''
def validate_{i}(x):
    if x is None:
        raise ValueError("Missing input")
    return x

def convert_{i}(seq):
    return [float(x) for x in seq if isinstance(x, (int, float))]

def summarize_{i}(seq):
    clean = convert_{i}(seq)
    return {{
        "min": min(clean),
        "max": max(clean),
        "mean": sum(clean)/len(clean)
    }}
    '''


def pattern_comment(i: int) -> str:
    return f'''
def analyze_series_{i}(data):
    """
    Over-commented numeric analysis function.
    """
    if not isinstance(data, list):
        raise TypeError("Expected list")

    numeric = []
    for x in data:
        if isinstance(x, (int, float)):
            numeric.append(x)

    return {{
        "min": min(numeric),
        "max": max(numeric),
        "range": max(numeric)-min(numeric)
    }}
    '''


# ============================
#  KISA AI PATTERNLERİ
# ============================

def pattern_short_safe_div(i: int) -> str:
    return f'''
def safe_divide_{i}(a, b):
    """Performs division with structured error handling."""
    try:
        return a / b
    except ZeroDivisionError:
        return None
    except TypeError:
        return None
    '''


def pattern_short_extract_numeric(i: int) -> str:
    return f'''
def extract_numeric_{i}(payload):
    """Filters numeric values from mixed payload."""
    clean = []
    for x in payload:
        if isinstance(x, (int, float)):
            clean.append(x)
    return clean
    '''


def pattern_short_compute_stats(i: int) -> str:
    return f'''
def compute_stats_{i}(items):
    """Returns aggregated statistics with min, max, mean."""
    nums = [x for x in items if isinstance(x, (int, float))]
    if not nums:
        return {{}}
    return {{
        "min": min(nums),
        "max": max(nums),
        "mean": sum(nums)/len(nums)
    }}
    '''


def pattern_short_validate_payload(i: int) -> str:
    return f'''
def validate_payload_{i}(obj):
    """Ensures payload matches expected structure."""
    if not isinstance(obj, dict):
        raise TypeError("Expected dict")
    return {{"keys": list(obj.keys())}}
    '''


def pattern_short_flatten_list(i: int) -> str:
    return f'''
def flatten_list_{i}(seq):
    """Flattens nested lists one level deep."""
    out = []
    for item in seq:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out
    '''


def pattern_short_structured_sum(i: int) -> str:
    return f'''
def structured_sum_{i}(values):
    """Computes sum with basic input validation."""
    if values is None:
        return 0
    return sum(v for v in values if isinstance(v, (int, float)))
    '''


def pattern_short_count_tokens(i: int) -> str:
    return f'''
def count_tokens_{i}(text):
    """Counts tokens by splitting on whitespace."""
    if not isinstance(text, str):
        return 0
    return len(text.split())
    '''


# ====================================
#  CRITICAL: LLM META PATTERN
# ====================================

def pattern_llm_meta(i: int) -> str:
    return f'''
def meta_process_{i}(payload):
    """
    Automatically generated meta-processing function.
    Simulates AI-style abstraction using rule maps and internal lambdas.
    """

    RULES = {{
        "validate": lambda x: isinstance(x, (list, dict)),
        "extract_numbers": lambda x: [v for v in x if isinstance(v, (int, float))] 
                                     if isinstance(x, list) else [],
        "summary": lambda seq: {{
            "count": len(seq),
            "min": min(seq) if seq else None,
            "max": max(seq) if seq else None,
            "avg": sum(seq)/len(seq) if seq else None
        }},
    }}

    if not RULES["validate"](payload):
        return {{"error": "invalid input"}}

    nums = RULES["extract_numbers"](payload)
    return RULES["summary"](nums)
    '''


# ====================================
#  CRITICAL: EXECUTION GRAPH PATTERN
# ====================================

def pattern_exec_graph(i: int) -> str:
    return f'''
class ExecutionNode_{i}:
    \"\"\"Autogenerated AI-style execution graph node.\"\"\"

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn
        self.result = None
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def run(self, data):
        self.result = self.fn(data)
        for child in self.children:
            child.run(self.result)
        return self.result


def build_graph_{i}():
    import math

    root = ExecutionNode_{i}("root", lambda x: [v for v in x if isinstance(v, (int, float))])
    norm = ExecutionNode_{i}("normalize", lambda seq: [v / max(seq) for v in seq] if seq else [])
    stats = ExecutionNode_{i}("stats", lambda seq: {{
        "min": min(seq) if seq else None,
        "max": max(seq) if seq else None,
        "mean": sum(seq)/len(seq) if seq else None,
    }})

    root.add_child(norm)
    norm.add_child(stats)
    return root
    '''


# ============================
#  ML / NLP PATTERNLERİ
# ============================

def pattern_sklearn_kmeans(i: int) -> str:
    return f'''
from sklearn.cluster import KMeans
import numpy as np

def cluster_points_{i}():
    \"\"\"Clusters sample points using KMeans (AI-style).\"\"\" 
    X = np.array([[1,2], [1.5,1.8], [5,8], [8,8], [1,0.6], [9,11]])
    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    model.fit(X)
    return {{
        "labels": model.labels_.tolist(),
        "centers": model.cluster_centers_.tolist()
    }}
    '''


def pattern_vectorizer_bow(i: int) -> str:
    return f'''
from sklearn.feature_extraction.text import CountVectorizer

def build_bow_matrix_{i}(texts):
    \"\"\"Builds a Bag-of-Words matrix from raw texts.\"\"\" 
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return {{
        "matrix": X.toarray().tolist(),
        "vocab": vectorizer.get_feature_names_out().tolist()
    }}
    '''


def pattern_keras_mlp(i: int) -> str:
    return f'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_mlp_model_{i}(input_dim: int):
    \"\"\"Constructs a simple Keras MLP model.\"\"\" 
    model = Sequential([
        Dense(32, activation="relu", input_shape=(input_dim,)),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
    '''


def pattern_lru_fib(i: int) -> str:
    return f'''
from functools import lru_cache

@lru_cache
def fib_{i}(n: int) -> int:
    \"\"\"Memoized fibonacci calculation.\"\"\" 
    if n < 2:
        return n
    return fib_{i}(n - 1) + fib_{i}(n - 2)
    '''


def pattern_config_class(i: int) -> str:
    return f'''
class Config_{i}:
    \"\"\"Simple configuration holder for ML pipelines.\"\"\" 
    def __init__(self, lr: float, epochs: int, batch: int):
        self.lr = lr
        self.epochs = epochs
        self.batch = batch

    def as_dict(self):
        return {{"lr": self.lr, "epochs": self.epochs, "batch": self.batch}}
    '''
    
def pattern_ai_fib(i):
    return f'''
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_{i}(n: int) -> int:
    """Memoized fibonacci implementation."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fib_{i}(n-1) + fib_{i}(n-2)
    '''

def pattern_ai_word_freq(i):
    return f'''
from collections import Counter

def word_freq_{i}(text: str):
    """Compute lowercase word frequencies from raw string input."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    words = text.lower().split()
    return dict(Counter(words))
    '''
def pattern_ai_task_manager(i):
    return f'''
from dataclasses import dataclass, field
from typing import List

@dataclass
class Task_{i}:
    title: str
    done: bool = False

@dataclass
class TaskManager_{i}:
    tasks: List[Task_{i}] = field(default_factory=list)

    def add(self, title: str):
        self.tasks.append(Task_{i}(title))

    def mark(self, idx: int):
        if 0 <= idx < len(self.tasks):
            self.tasks[idx].done = True
        else:
            raise IndexError("Invalid index")
    '''
def pattern_ai_stats(i):
    return f'''
import statistics

def stats_{i}(values):
    """Return count, mean, median, variance."""
    if not values:
        raise ValueError("empty list")
    return {{
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "variance": statistics.pvariance(values)
    }}
    '''
def pattern_ai_json_filter(i):
    return f'''
import json

def load_and_filter_{i}(path: str):
    """Load a JSON list and return only active items."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [x for x in data if x.get("is_active")]
    '''
def pattern_ai_clean_texts(i):
    return f'''
import re

def clean_texts_{i}(texts):
    """Lowercase + remove special characters + normalize spaces."""
    out = []
    for t in texts:
        t = re.sub(r"[^a-zA-Z0-9\\s]", "", t)
        t = re.sub(r"\\s+", " ", t).strip().lower()
        out.append(t)
    return out
    '''
def pattern_ai_predict_wrapper(i: int) -> str:
    return f'''
import numpy as np

def validate_and_predict_{i}(model, sample):
    """
    AI-style prediction wrapper with:
    - capability check
    - type validation
    - reshape
    - structured dictionary output
    """

    if not hasattr(model, "predict"):
        raise ValueError("Model does not implement predict().")

    if not isinstance(sample, (list, tuple)):
        raise TypeError("Sample must be list-like.")

    arr = np.array(sample).reshape(1, -1)
    pred = model.predict(arr)[0]

    return {{
        "input": arr.tolist(),
        "prediction": float(pred),
    }}
    '''
def pattern_ai_basic_utils(i):
    return f'''
def compute_avg_{i}(values):
    """
    Compute arithmetic mean of a numeric list with
    strict type checking and defensive validation.
    """
    if not isinstance(values, list):
        raise TypeError("values must be a list")
    if not values:
        return 0.0

    total = 0.0
    for v in values:
        total += v
    return total / len(values)
    '''
def pattern_ai_safe_division(i):
    return f'''
def safe_division_{i}(a, b):
    """
    AI-style safe division with structured error reporting.
    """
    try:
        return a / b
    except ZeroDivisionError:
        return 0.0
    except TypeError:
        return 0.0
    '''
def pattern_ai_filter_items(i):
    return f'''
from typing import List, Callable, TypeVar

T = TypeVar("T")

def filter_items_{i}(items: List[T], cond: Callable[[T], bool]) -> List[T]:
    """
    Generic filter implementation with explicit functional typing.
    """
    out = []
    for it in items:
        if cond(it):
            out.append(it)
    return out
    '''
def pattern_ai_report(i):
    return f'''
def generate_report_{i}(data):
    """
    Structured statistical report generator following
    deterministic aggregation semantics.
    """
    if not data:
        return {{"count": 0, "max": None, "min": None, "average": 0}}

    return {{
        "count": len(data),
        "max": max(data),
        "min": min(data),
        "average": sum(data) / len(data)
    }}
    '''
def pattern_ai_text_processor(i):
    return f'''
class TextProcessor_{i}:
    """
    AI-style text processing class with utility methods.
    """

    @staticmethod
    def count_words(text: str) -> int:
        return len(text.split())

    @staticmethod
    def to_title_case(text: str) -> str:
        return text.title()
    '''
def pattern_ai_mini_linear_reg(i):
    return f'''
import numpy as np
from sklearn.linear_model import LinearRegression

def mini_linear_reg_{i}(X, y, new_sample):
    """Train a small LinearRegression model and return prediction."""
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict([new_sample])[0]
    return float(pred)
    '''
def pattern_ai_minmax_norm(i):
    return f'''
import numpy as np

def minmax_norm_{i}(values):
    """AI-style numeric min-max normalization utility."""
    arr = np.asarray(values, dtype=float)
    mn = arr.min()
    mx = arr.max()
    return ((arr - mn) / (mx - mn + 1e-9)).tolist()
    '''
def pattern_ai_light_validator(i):
    return f'''
from typing import Dict, Any

def validate_payload_{i}(payload: Dict[str, Any]) -> bool:
    """Lightweight structured payload validator."""
    required = {{"id": int, "name": str, "score": float}}
    for key, t in required.items():
        val = payload.get(key)
        if not isinstance(val, t):
            return False
    return True
    '''
def pattern_ai_prime(i):
    return f'''
def is_prime_{i}(number: int) -> bool:
    """
    Determines primality using structured validation and 
    early-termination optimization for downstream consistency.
    """
    if not isinstance(number, int):
        raise TypeError("number must be int")

    if number < 2:
        return False

    limit = int(number ** 0.5)
    for divisor in range(2, limit + 1):
        if number % divisor == 0:
            return False

    return True
    '''
def pattern_ai_safe_division(i):
    return f'''
def safe_division_{i}(a, b):
    """
    Performs division with structured exception handling 
    and deterministic fallback behavior.
    """
    try:
        return float(a) / float(b)
    except ZeroDivisionError:
        return 0.0
    except Exception:
        return 0.0
    '''
def pattern_ai_average(i):
    return f'''
def compute_average_{i}(values):
    """
    Computes arithmetic mean with input normalization and 
    predictable output structure.
    """
    if not isinstance(values, (list, tuple)):
        raise TypeError("values must be list-like")

    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return {{"count": 0, "avg": 0.0}}

    return {{"count": len(nums), "avg": sum(nums) / len(nums)}}
    '''
def pattern_ai_char_counter(i):
    return f'''
def count_characters_{i}(text: str):
    """
    Counts character frequencies using deterministic structure and 
    defensive type validation.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    counts = {{}}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    return counts
    '''
def pattern_ai_validate_payload(i):
    return f'''
def validate_payload_{i}(payload):
    """
    Validates structured payload with explicit type enforcement 
    for downstream reliability.
    """
    required = {{"id": int, "name": str, "score": float}}

    if not isinstance(payload, dict):
        return False

    for key, typ in required.items():
        if key not in payload or not isinstance(payload[key], typ):
            return False

    return True
    '''
def pattern_ai_normalization(i):
    return f'''
import numpy as np

def normalize_features_{i}(values):
    """
    Applies stable min-max normalization with epsilon smoothing
    for consistent downstream preprocessing.
    """
    arr = np.array(values, dtype=float)
    mn, mx = float(arr.min()), float(arr.max())
    eps = 1e-9
    return ((arr - mn) / (mx - mn + eps)).tolist()
    '''
def pattern_ai_preprocess_scaler(i):
    return f'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_features_{i}(values):
    """Normalize numeric features into [0,1] range for ML pipelines."""
    arr = np.array(values, dtype=float).reshape(-1,1)
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(arr)
    return transformed.flatten().tolist()
'''

# ============================
#  PATTERN LİSTESİ
# ============================

AI_PATTERNS = [
    pattern_simple_ai,
    pattern_class_ai,
    pattern_pipeline,
    pattern_config,
    pattern_errors,
    pattern_nested,
    pattern_module,
    pattern_comment,

    pattern_short_safe_div,
    pattern_short_extract_numeric,
    pattern_short_compute_stats,
    pattern_short_validate_payload,
    pattern_short_flatten_list,
    pattern_short_structured_sum,
    pattern_short_count_tokens,

    pattern_llm_meta,
    pattern_exec_graph,

    pattern_sklearn_kmeans,
    pattern_vectorizer_bow,
    pattern_keras_mlp,
    pattern_lru_fib,
    pattern_config_class,
    pattern_ai_predict_wrapper,
     pattern_ai_word_freq,
    pattern_ai_fib,
    pattern_ai_task_manager,
    pattern_ai_stats,
    pattern_ai_json_filter,
    pattern_ai_clean_texts,
pattern_ai_basic_utils,
    pattern_ai_safe_division,
    pattern_ai_filter_items,
    pattern_ai_report,
    pattern_ai_text_processor,
    pattern_ai_mini_linear_reg,
    pattern_ai_minmax_norm,
    pattern_ai_light_validator,
    pattern_ai_prime,
    pattern_ai_safe_division,
    pattern_ai_average,
    pattern_ai_char_counter,
    pattern_ai_validate_payload,
    pattern_ai_normalization,
    pattern_ai_preprocess_scaler,

]


# ============================
#  DATASET ÜRETİMİ
# ============================

def generate_dataset(count: int = 5000) -> List[Dict[str, Any]]:
    dataset = []
    for i in range(count):
        pattern_func = random.choice(AI_PATTERNS)
        code = pattern_func(i)
        dataset.append({
            "id": f"AI_{i}",
            "text": code.strip(),
            "label": "1",
            "lang": "python",
        })
    return dataset


def main():
    dataset = generate_dataset(5000)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=== Yeni AI dataset üretimi tamamlandı (tüm patternler + meta + exec graph) ===")


if __name__ == "__main__":
    main()