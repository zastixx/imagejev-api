"""ImageJev: The Image-Native Decision Engine.

    Jev:       TEXT   + DECISION SCHEMA -> structured decision
    ImageJev:  IMAGE  + DECISION + SCHEMA -> structured decision

Combines frozen CLIP vision-language embeddings with calibrated closed-schema decision math.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Sequence, Union

import numpy as np

from .backends import Backend, OpenCLIPBackend
from .types import ChoiceResult, NoulResult, ScoreResult

# ---------------------------------------------------------------------------------------- Math Helpers


def _softmax(z: np.ndarray) -> np.ndarray:
    """Compute softmax over a 1D logit vector."""
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def _softmax_batch(z: np.ndarray) -> np.ndarray:
    """Compute softmax over a 2D logit matrix of shape (B, N)."""
    max_z = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - max_z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _confidence(p: np.ndarray) -> float:
    """Normalized confidence margin.

    Returns 1.0 when all probability mass is on a single option,
    and 0.0 for a uniform random distribution across N options.
    Formula: (N * max(p) - 1) / (N - 1)
    """
    n = len(p)
    return 1.0 if n < 2 else float(np.clip((n * np.max(p) - 1.0) / (n - 1.0), 0.0, 1.0))


def _dispersion(probs: Dict[int, float]) -> float:
    """Fraction of a Score schema's probability mass outside the peak level and its immediate neighbors."""
    peak = max(probs, key=probs.get)
    adjacent = {peak, peak - 1, peak + 1} & set(probs)
    return float(np.clip(1.0 - sum(probs[k] for k in adjacent), 0.0, 1.0))


# ---------------------------------------------------------------------------------------- Templates & Warnings

_OBJECT_TEMPLATE = "{name}: a photo of a {name}"

_DEFAULT_NULL_ANCHOR = "unrelated photo, other random object, or different background"
_DEFAULT_RELEVANCE_THRESHOLD = 0.20

_JUDGMENT_WORDS = {
    "yes", "no", "true", "false", "pass", "fail", "good", "bad",
    "positive", "negative", "acceptable", "unacceptable",
    "suitable", "unsuitable", "approved", "rejected", "valid", "invalid",
}


# ---------------------------------------------------------------------------------------- Core Engine


class ImageJev:
    """Main ImageJev Decision Engine.

    Parameters:
        backend: Optional custom backend implementing the Backend protocol.
                 If None, initializes an OpenCLIPBackend with the specified model and device.
        model: Vision backbone architecture name (default: "ViT-B-32").
        pretrained: Pretrained weights tag (default: "laion2b_s34b_b79k").
        device: Device string ('cuda', 'mps', 'cpu', or None for auto-detect).
    """

    def __init__(
        self,
        backend: Backend | None = None,
        model: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ):
        if backend is not None:
            self.backend = backend
        else:
            self.backend = OpenCLIPBackend(name=model, pretrained=pretrained, device=device)
        self._cache: Dict[str, np.ndarray] = {}

    def _embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self.backend.encode_text([text])[0]
        return self._cache[text]

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            encoded = self.backend.encode_text(uncached)
            for t, v in zip(uncached, encoded):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    def decide(
        self,
        image: Any,
        decision: str,
        schema: dict,
    ) -> Union[ChoiceResult, ScoreResult, NoulResult]:
        """Make a structured decision on a single visual input.

        Args:
            image: Path (str or Path), URL (http/https), PIL.Image, or raw bytes.
            decision: Plain-language question or framing statement.
            schema: Decision schema dictionary:
                    - {"type": "choice", "options": {...}}
                    - {"type": "score", "levels": [...]}
                    - {"type": "noul", "true": "...", "false": "..."}

        Returns:
            ChoiceResult, ScoreResult, or NoulResult based on schema type.
        """
        schema_type = schema.get("type")
        if schema_type == "choice":
            return self._choice(image, schema)
        if schema_type == "score":
            return self._score(image, schema)
        if schema_type == "noul":
            return self._noul(image, decision, schema)
        raise ValueError(
            f'schema["type"] must be "choice", "score", or "noul", got {schema_type!r}'
        )

    def decide_batch(
        self,
        images: Sequence[Any],
        decision: str,
        schema: dict,
    ) -> List[Union[ChoiceResult, ScoreResult, NoulResult]]:
        """Make structured decisions on a batch of visual inputs with high GPU/CPU throughput.

        Args:
            images: Sequence of images (Paths, URLs, PIL.Images, or bytes).
            decision: Plain-language question or framing statement.
            schema: Decision schema dictionary.

        Returns:
            List of ChoiceResult, ScoreResult, or NoulResult instances.
        """
        if not images:
            return []

        schema_type = schema.get("type")
        if schema_type == "choice":
            return self._choice_batch(images, schema)
        if schema_type == "score":
            return self._score_batch(images, schema)
        if schema_type == "noul":
            return self._noul_batch(images, decision, schema)
        raise ValueError(
            f'schema["type"] must be "choice", "score", or "noul", got {schema_type!r}'
        )

    def decide_multi(
        self,
        image: Any,
        questions: Dict[str, dict],
    ) -> Dict[str, Union[ChoiceResult, ScoreResult, NoulResult]]:
        """Evaluate multiple questions simultaneously on a SINGLE image pass.

        Encodes the visual input once, then evaluates all question schemas in memory.

        Args:
            image: Path, URL, PIL.Image, or raw bytes.
            questions: Mapping of question_id -> {"decision": "...", "schema": {...}} or
                       {"instructions": "...", "type": "...", ...}

        Returns:
            Dictionary of question_id -> Result object.
        """
        if not questions:
            return {}

        img_vec = self.backend.encode_image(image)
        results = {}

        for qid, qdata in questions.items():
            # Handle both {"decision": "...", "schema": {...}} and flat format
            decision = qdata.get("decision") or qdata.get("instructions") or qdata.get("prompt") or ""
            schema = qdata.get("schema") or qdata
            schema_type = schema.get("type")

            if schema_type == "choice":
                names, prompts = self._prompts_for_choice(schema.get("options") or schema.get("criteria", {}))
                allow_none = bool(schema.get("allow_none", False))
                thresh = float(schema.get("relevance_threshold", _DEFAULT_RELEVANCE_THRESHOLD))
                null_desc = str(schema.get("null_anchor", _DEFAULT_NULL_ANCHOR))

                all_prompts = prompts + [null_desc] if allow_none else prompts
                text_vecs = self._embed_batch(all_prompts)
                cos_sims = text_vecs @ img_vec
                in_sims = cos_sims[:len(names)]
                max_rel = float(np.max(in_sims))
                is_rel = bool(max_rel >= thresh)

                logits = self.backend.logit_scale * cos_sims
                probs = _softmax(logits)
                top_idx = int(np.argmax(probs))

                if allow_none and (top_idx == len(names) or not is_rel):
                    results[qid] = ChoiceResult(
                        type="choice",
                        choice=None,
                        probability=float(probs[len(names)]) if allow_none else float(probs[top_idx]),
                        confidence=max_rel,
                        is_relevant=False,
                        relevance=round(max_rel, 4),
                        probabilities={name: float(p) for name, p in zip(names, probs[:len(names)])},
                    )
                else:
                    opt_probs = probs[:len(names)] / np.sum(probs[:len(names)]) if allow_none else probs
                    best_idx = int(np.argmax(opt_probs))
                    results[qid] = ChoiceResult(
                        type="choice",
                        choice=names[best_idx],
                        probability=float(opt_probs[best_idx]),
                        confidence=_confidence(opt_probs),
                        is_relevant=is_rel,
                        relevance=round(max_rel, 4),
                        probabilities={name: float(p) for name, p in zip(names, opt_probs)},
                    )
            elif schema_type == "score":
                levels = list(schema.get("levels") or schema.get("criteria", []))
                if not 2 <= len(levels) <= 10:
                    raise ValueError(f"score schema for {qid!r} requires 2-10 levels, got {len(levels)}")
                text_vecs = self._embed_batch(levels)
                logits = self.backend.logit_scale * (text_vecs @ img_vec)
                probs = _softmax(logits)
                peak_idx = int(np.argmax(probs))
                probs_dict = {k: float(x) for k, x in enumerate(probs)}
                results[qid] = ScoreResult(
                    type="score",
                    score=float(probs @ np.arange(len(probs))),
                    probability=float(probs[peak_idx]),
                    confidence=_confidence(probs),
                    dispersion=_dispersion(probs_dict),
                    legend=dict(enumerate(levels)),
                    probabilities=probs_dict,
                )
            elif schema_type == "noul":
                true_text, false_text = self._resolve_noul_prompts(decision, schema)
                vecs = self._embed_batch([true_text, false_text])
                logits = self.backend.logit_scale * (vecs @ img_vec)
                prob_true = float(1.0 / (1.0 + np.exp(-(logits[0] - logits[1]))))
                results[qid] = NoulResult(type="noul", probability=prob_true, noul=prob_true)
            else:
                raise ValueError(f'schema["type"] for {qid!r} must be "choice", "score", or "noul", got {schema_type!r}')

        return results

    # -------------------------------------------------------------------------------- Choice Primitive

    def _prompts_for_choice(self, options: Union[dict, list]) -> tuple[list[str], list[str]]:
        if isinstance(options, dict):
            names, descs = list(options.keys()), list(options.values())
        elif isinstance(options, (list, tuple)):
            names, descs = list(options), [None] * len(options)
        else:
            raise TypeError(f"choice schema 'options' must be a dict or list, got {type(options)}")

        prompts, risky = [], []
        for name, desc in zip(names, descs):
            if desc:
                prompts.append(f"{name}: {desc}")
            else:
                prompts.append(_OBJECT_TEMPLATE.format(name=name))
                if str(name).strip().lower() in _JUDGMENT_WORDS:
                    risky.append(name)

        if risky:
            warnings.warn(
                f"choice option(s) {risky} look like judgment words, not object names, but have no "
                f"description, so they received the generic object-class template "
                f"('{_OBJECT_TEMPLATE.format(name=risky[0])}'). For subjective or quality judgments, "
                f"provide explicit option descriptions in the schema, or use schema={{'type': 'noul'}}.",
                UserWarning,
                stacklevel=3,
            )
        return names, prompts

    def _choice(self, image: Any, schema: dict) -> ChoiceResult:
        names, prompts = self._prompts_for_choice(schema["options"])
        allow_none = bool(schema.get("allow_none", False))
        thresh = float(schema.get("relevance_threshold", _DEFAULT_RELEVANCE_THRESHOLD))
        null_desc = str(schema.get("null_anchor", _DEFAULT_NULL_ANCHOR))

        all_prompts = prompts + [null_desc] if allow_none else prompts
        img_vec = self.backend.encode_image(image)
        text_vecs = self._embed_batch(all_prompts)
        cos_sims = text_vecs @ img_vec
        in_sims = cos_sims[:len(names)]
        max_rel = float(np.max(in_sims))
        is_rel = bool(max_rel >= thresh)

        logits = self.backend.logit_scale * cos_sims
        probs = _softmax(logits)
        top_idx = int(np.argmax(probs))

        if allow_none and (top_idx == len(names) or not is_rel):
            return ChoiceResult(
                choice=None,
                probability=float(probs[len(names)]),
                confidence=max_rel,
                is_relevant=False,
                relevance=round(max_rel, 4),
                probabilities={name: float(p) for name, p in zip(names, probs[:len(names)])},
            )

        opt_probs = probs[:len(names)] / np.sum(probs[:len(names)]) if allow_none else probs
        best_idx = int(np.argmax(opt_probs))

        return ChoiceResult(
            choice=names[best_idx],
            probability=float(opt_probs[best_idx]),
            confidence=_confidence(opt_probs),
            is_relevant=is_rel,
            relevance=round(max_rel, 4),
            probabilities={name: float(p) for name, p in zip(names, opt_probs)},
        )

    def _choice_batch(self, images: Sequence[Any], schema: dict) -> List[ChoiceResult]:
        names, prompts = self._prompts_for_choice(schema["options"])
        allow_none = bool(schema.get("allow_none", False))
        thresh = float(schema.get("relevance_threshold", _DEFAULT_RELEVANCE_THRESHOLD))
        null_desc = str(schema.get("null_anchor", _DEFAULT_NULL_ANCHOR))

        all_prompts = prompts + [null_desc] if allow_none else prompts
        img_mat = self.backend.encode_images(images)  # (B, D)
        text_mat = self._embed_batch(all_prompts)  # (N_all, D)
        cos_mat = img_mat @ text_mat.T  # (B, N_all)
        logits = self.backend.logit_scale * cos_mat  # (B, N_all)
        probs = _softmax_batch(logits)  # (B, N_all)

        results = []
        for i, p in enumerate(probs):
            in_sims = cos_mat[i, :len(names)]
            max_rel = float(np.max(in_sims))
            is_rel = bool(max_rel >= thresh)
            top_idx = int(np.argmax(p))

            if allow_none and (top_idx == len(names) or not is_rel):
                results.append(
                    ChoiceResult(
                        choice=None,
                        probability=float(p[len(names)]),
                        confidence=max_rel,
                        is_relevant=False,
                        relevance=round(max_rel, 4),
                        probabilities={name: float(val) for name, val in zip(names, p[:len(names)])},
                    )
                )
            else:
                opt_probs = p[:len(names)] / np.sum(p[:len(names)]) if allow_none else p
                best_idx = int(np.argmax(opt_probs))
                results.append(
                    ChoiceResult(
                        choice=names[best_idx],
                        probability=float(opt_probs[best_idx]),
                        confidence=_confidence(opt_probs),
                        is_relevant=is_rel,
                        relevance=round(max_rel, 4),
                        probabilities={name: float(val) for name, val in zip(names, opt_probs)},
                    )
                )
        return results

    # -------------------------------------------------------------------------------- Score Primitive

    def _score(self, image: Any, schema: dict) -> ScoreResult:
        levels = list(schema.get("levels", []))
        if not 2 <= len(levels) <= 10:
            raise ValueError(f"score schema requires 2-10 levels, got {len(levels)}")

        img_vec = self.backend.encode_image(image)
        text_vecs = self._embed_batch(levels)
        logits = self.backend.logit_scale * (text_vecs @ img_vec)
        probs = _softmax(logits)
        peak_idx = int(np.argmax(probs))
        probs_dict = {k: float(x) for k, x in enumerate(probs)}

        return ScoreResult(
            score=float(probs @ np.arange(len(probs))),
            probability=float(probs[peak_idx]),
            confidence=_confidence(probs),
            dispersion=_dispersion(probs_dict),
            legend=dict(enumerate(levels)),
            probabilities=probs_dict,
        )

    def _score_batch(self, images: Sequence[Any], schema: dict) -> List[ScoreResult]:
        levels = list(schema.get("levels", []))
        if not 2 <= len(levels) <= 10:
            raise ValueError(f"score schema requires 2-10 levels, got {len(levels)}")

        img_mat = self.backend.encode_images(images)  # (B, D)
        text_mat = self._embed_batch(levels)  # (N, D)
        logits = self.backend.logit_scale * (img_mat @ text_mat.T)  # (B, N)
        probs = _softmax_batch(logits)  # (B, N)
        legend_map = dict(enumerate(levels))

        results = []
        for p in probs:
            peak_idx = int(np.argmax(p))
            probs_dict = {k: float(x) for k, x in enumerate(p)}
            results.append(
                ScoreResult(
                    score=float(p @ np.arange(len(p))),
                    probability=float(p[peak_idx]),
                    confidence=_confidence(p),
                    dispersion=_dispersion(probs_dict),
                    legend=legend_map,
                    probabilities=probs_dict,
                )
            )
        return results

    # -------------------------------------------------------------------------------- Noul Primitive

    def _resolve_noul_prompts(self, decision: str, schema: dict) -> tuple[str, str]:
        true_text = schema.get("true")
        false_text = schema.get("false")
        if not true_text or not false_text:
            true_text = true_text or f"Yes. {decision}"
            false_text = false_text or f"No. {decision}"
            stripped = decision.strip()
            if stripped.endswith("?") or (
                stripped.split()[:1]
                and stripped.split()[0].lower() in (
                    "does", "is", "are", "can", "will", "has", "have", "do", "did", "was", "were"
                )
            ):
                warnings.warn(
                    f'decision ("{decision}") looks like a question without explicit true/false criteria. '
                    f'Using generic Yes./No. wrappers. For optimal contrastive sharpness, '
                    f'provide explicit true/false criteria, e.g. {{"true": "a cat is present", "false": "no cats"}}.',
                    UserWarning,
                    stacklevel=4,
                )
        return true_text, false_text

    def _noul(self, image: Any, decision: str, schema: dict) -> NoulResult:
        true_text, false_text = self._resolve_noul_prompts(decision, schema)
        img_vec = self.backend.encode_image(image)
        vecs = self._embed_batch([true_text, false_text])
        logits = self.backend.logit_scale * (vecs @ img_vec)
        prob_true = float(1.0 / (1.0 + np.exp(-(logits[0] - logits[1]))))
        return NoulResult(probability=prob_true)

    def _noul_batch(self, images: Sequence[Any], decision: str, schema: dict) -> List[NoulResult]:
        true_text, false_text = self._resolve_noul_prompts(decision, schema)
        img_mat = self.backend.encode_images(images)  # (B, D)
        vecs = self._embed_batch([true_text, false_text])  # (2, D)
        logits = self.backend.logit_scale * (img_mat @ vecs.T)  # (B, 2)
        diff = logits[:, 0] - logits[:, 1]
        probs = 1.0 / (1.0 + np.exp(-diff))
        return [NoulResult(probability=float(p)) for p in probs]
