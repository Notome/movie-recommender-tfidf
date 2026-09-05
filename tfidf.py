import re
from collections import Counter
from math import log

import numpy as np


class TfIdfVectorizer:
    def __init__(self, stop_words_path: str):
        with open(stop_words_path, "r") as file:
            self.stop_words = set(file.read().split())

        self.documents: dict[str, list[str]] = {}
        self.tf: dict[str, dict[str, float]] = {}
        self.idf: dict[str, float] = {}
        self.vocab: list[str] = []
        self.word_index: dict[str, int] = {}
        self.doc_ids: list[str] = []
        self.matrix: np.ndarray | None = None
        self._norms: np.ndarray | None = None

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"[a-z]+", text.lower())
        return [w for w in words if w not in self.stop_words]

    def _compute_tf(self, tokens: list[str]) -> dict[str, float]:
        counter = Counter(tokens)
        total = len(tokens)
        return {word: count / total for word, count in counter.items()}

    def _compute_idf(self) -> dict[str, float]:
        doc_freq = Counter()
        for tokens in self.documents.values():
            doc_freq.update(set(tokens))

        n_docs = len(self.documents)
        return {word: log(n_docs / df) for word, df in doc_freq.items()}

    def fit(self, corpus: dict[str, str]) -> None:
        self.documents = {title: self._tokenize(text) for title, text in corpus.items()}

        self.tf = {title: self._compute_tf(tokens) for title, tokens in self.documents.items()}
        self.idf = self._compute_idf()
        self.vocab = sorted(self.idf.keys())
        self.word_index = {word: i for i, word in enumerate(self.vocab)}

    def transform(self) -> np.ndarray:
        self.doc_ids = list(self.documents.keys())
        matrix = np.zeros((len(self.doc_ids), len(self.vocab)), dtype=np.float32)

        for row, title in enumerate(self.doc_ids):
            words = list(self.tf[title].keys())
            cols = [self.word_index[w] for w in words]
            values = [self.tf[title][w] * self.idf[w] for w in words]
            matrix[row, cols] = values

        self.matrix = matrix
        self._norms = np.linalg.norm(matrix, axis=1)
        return matrix

    def fit_transform(self, corpus: dict[str, str]) -> np.ndarray:
        self.fit(corpus)
        return self.transform()

    def cosine_similarity(self, A: np.ndarray, B: np.ndarray) -> float:
        denom = np.linalg.norm(A) * np.linalg.norm(B)
        return float(np.dot(A, B) / denom) if denom else 0.0

    def similarity_matrix(self) -> np.ndarray:
        if self.matrix is None:
            raise ValueError("Сначала вызовите fit_transform / transform")

        norms = np.where(self._norms == 0, 1.0, self._norms)  # защита от деления на 0
        normalized = self.matrix / norms[:, None]
        return normalized @ normalized.T

    def top_n_similar(self, n: int = 10) -> dict[str, list[tuple[str, float]]]:
        sim = self.similarity_matrix()
        np.fill_diagonal(sim, -np.inf)

        result = {}
        for row, doc_id in enumerate(self.doc_ids):
            top_idx = np.argpartition(-sim[row], min(n, len(sim[row]) - 1))[:n]
            top_idx = top_idx[np.argsort(-sim[row][top_idx])]
            result[doc_id] = [(self.doc_ids[j], float(sim[row, j])) for j in top_idx]

        return result