import re
import numpy as np
from typing import Dict, Any, List, Tuple, Callable, Optional

# Simple in-memory caches to reduce duplicated embedding calls within process lifetime
_EMBED_CACHE: Dict[str, np.ndarray] = {}
_DOC_FIELD_EMB_CACHE: Dict[str, Dict[str, np.ndarray]] = {}

# small cosine helper
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size==0 or b.size==0:
        return 0.0
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b) / denom)

# Helper: batch embed texts using provided batch embedding function
def batch_embed_texts(get_embeddings_batch: Callable[[List[str]], List[np.ndarray]], texts: List[str]) -> List[np.ndarray]:
    results: List[np.ndarray] = [None] * len(texts)
    to_request = []
    idx_map = {}
    for i, t in enumerate(texts):
        key = (t or "").strip()
        if not key:
            results[i] = np.zeros((1536,), dtype=np.float32)
            continue
        if key in _EMBED_CACHE:
            results[i] = _EMBED_CACHE[key]
        else:
            idx_map[len(to_request)] = i
            to_request.append(key)
    if to_request:
        vecs = get_embeddings_batch(to_request)
        for j, v in enumerate(vecs):
            i = idx_map[j]
            _EMBED_CACHE[to_request[j]] = v
            results[i] = v
    return results

# Document -> 4 field texts
def doc_to_4texts(doc: Dict[str, Any]) -> Dict[str, str]:
    ef = doc.get("extractedFeature") or {}
    # color: colorCd or main_color/sub_color
    color = " ".join(filter(None, [doc.get("colorCd"), ef.get("main_color"), ef.get("sub_color")]))
    # activity: signs of activity/behavior from specialMark/noticeable_features/health_impression/age/weight
    activity_parts = []
    if doc.get("kindFullNm"):
        activity_parts.append(doc.get("kindFullNm"))
    if doc.get("specialMark"):
        activity_parts.append(doc.get("specialMark"))
    if ef.get("noticeable_features"):
        activity_parts.append(ef.get("noticeable_features"))
    if ef.get("health_impression"):
        activity_parts.append(ef.get("health_impression"))
    if doc.get("age"):
        activity_parts.append(f"age: {doc.get('age')}")
    if doc.get("weight"):
        activity_parts.append(f"weight: {doc.get('weight')}")
    activity = " | ".join(activity_parts)
    # appearance: fur_pattern, fur_length, fur_texture, noticeable physical description
    appearance_parts = []
    if ef.get("fur_pattern"): appearance_parts.append(ef.get("fur_pattern"))
    if ef.get("fur_length"): appearance_parts.append(ef.get("fur_length"))
    if ef.get("fur_texture"): appearance_parts.append(ef.get("fur_texture"))
    if ef.get("eye_shape"): appearance_parts.append(ef.get("eye_shape"))
    if ef.get("ear_shape"): appearance_parts.append(ef.get("ear_shape"))
    if ef.get("tail_shape"): appearance_parts.append(ef.get("tail_shape"))
    if ef.get("noticeable_features"): appearance_parts.append(ef.get("noticeable_features"))
    appearance = " | ".join(appearance_parts)
    # personality: specialMark often contains personality keywords; include explicit fields
    personality = doc.get("specialMark") or ""
    # Ensure non-empty strings
    return {
        "color": color.strip(),
        "activity": activity.strip(),
        "appearance": appearance.strip(),
        "personality": personality.strip()
    }

# User (profile + query) -> 4 field texts
def user_to_4texts(user_query: Optional[str], profile: Optional[Dict[str, Any]]) -> Dict[str, str]:
    profile = profile or {}
    # color: extract color keywords from user_query or profile (favoriteAnimals maybe color?)
    color_parts = []
    if user_query:
        # simple regex for known color words (extend as needed)
        colors = ["검정", "검은", "흰","하얀", "갈색", "치즈", "회색", "베이지", "크림", "검은색", "흰색"]
        for c in colors:
            if c in user_query:
                color_parts.append(c)
    # activity: from profile.activityLevel and user_query mentions like '활발'
    activity_parts = []
    if profile.get("activityLevel"):
        activity_parts.append(f"활동수준: {profile.get('activityLevel')}")
    if user_query:
        if any(k in user_query for k in ["활발", "에너지", "활동적", "활동"]):
            activity_parts.append("활발")
        if any(k in user_query for k in ["차분", "조용", "온순"]):
            activity_parts.append("차분")
    # appearance: from userQuery (e.g. '털이 짧은', '점박이')
    appearance_parts = []
    if user_query:
        # naive: include the raw query as appearance candidate too
        appearance_parts.append(user_query)
    # personality: preferredPersonality, expectations
    personality_parts = []
    if profile.get("preferredPersonality"):
        personality_parts.append(", ".join(profile.get("preferredPersonality") or []))
    if profile.get("expectations"):
        personality_parts.append(", ".join(profile.get("expectations") or []))
    return {
        "color": " ".join(color_parts).strip(),
        "activity": " | ".join(activity_parts).strip(),
        "appearance": " | ".join(appearance_parts).strip(),
        "personality": " | ".join(personality_parts).strip()
    }

# Ensure doc field embeddings exist (will use get_embeddings_batch to compute if absent)
def ensure_doc_field_embeddings(doc: Dict[str, Any], get_embeddings_batch: Callable[[List[str]], List[np.ndarray]],
                                persist_fn: Optional[Callable[[str, Dict[str, List[float]]], None]] = None) -> Dict[str, np.ndarray]:
    doc_key = str(doc.get("desertionNo") or doc.get("_id"))
    if doc_key in _DOC_FIELD_EMB_CACHE:
        return _DOC_FIELD_EMB_CACHE[doc_key]
    fields = doc_to_4texts(doc)
    texts = []
    keys = []
    for k, v in fields.items():
        if v:
            texts.append(v)
            keys.append(k)
    if not texts:
        emb_map = {k: np.zeros((1536,), dtype=np.float32) for k in ["color","activity","appearance","personality"]}
        _DOC_FIELD_EMB_CACHE[doc_key] = emb_map
        return emb_map
    vecs = batch_embed_texts(get_embeddings_batch, texts)
    emb_map = {k: v for k, v in zip(keys, vecs)}
    # ensure all keys exist
    for k in ["color","activity","appearance","personality"]:
        if k not in emb_map:
            emb_map[k] = np.zeros((1536,), dtype=np.float32)
    _DOC_FIELD_EMB_CACHE[doc_key] = emb_map
    if persist_fn:
        # persist as list of floats for storage
        try:
            serial = {k: v.tolist() for k, v in emb_map.items()}
            persist_fn(doc_key, serial)
        except Exception:
            pass
    return emb_map

# compute per-field similarity and weighted aggregation
def compute_4field_match(query_embs: Dict[str, Optional[np.ndarray]], profile_embs: Dict[str, Optional[np.ndarray]],
                         doc_field_embs: Dict[str, np.ndarray], weights: Dict[str, float] = None) -> Tuple[float, Dict[str, float]]:
    # query_embs/profile_embs are dicts with keys color/activity/appearance/personality -> vector or None
    field_scores = {}
    for k in ["color","activity","appearance","personality"]:
        qv = query_embs.get(k)
        pv = profile_embs.get(k)
        fvec = doc_field_embs.get(k)
        best = 0.0
        if qv is not None and qv.size:
            sim = cosine(qv, fvec)
            best = max(best, float(sim))
        if pv is not None and pv.size:
            sim = cosine(pv, fvec)
            best = max(best, float(sim))
        field_scores[k] = best
    # default weight distribution (tunable)
    default = {"color":0.15,"activity":0.35,"appearance":0.2,"personality":0.30}
    wmap = weights or default
    total_w = sum(wmap.get(k,0.0) for k in field_scores.keys())
    if total_w <= 0:
        total_w = 1.0
        wmap = {k: 1.0/4 for k in field_scores.keys()}
    score = sum(field_scores[k]* (wmap.get(k,0.0)/total_w) for k in field_scores.keys())
    return float(score), field_scores