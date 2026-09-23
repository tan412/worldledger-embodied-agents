"""Deterministic atomic-action mining from task language.

The miner is intentionally conservative: it recognizes only an explicit
vocabulary, preserves the original text, and returns unmatched fragments
instead of silently dropping them.  It does not infer a robot command.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .skills import SkillStep


@dataclass(frozen=True)
class ActionPattern:
    skill: str
    zh: tuple[str, ...]
    en: tuple[str, ...]
    confidence: float = 1.0


ACTION_PATTERNS = (
    ActionPattern("approach", ("接近", "靠近", "移向", "伸向"), ("approach", "reach")),
    ActionPattern("align", ("对齐", "校准", "定位", "调整", "排齐", "摆齐"),
                  ("align", "aligned", "aligning", "calibrate",
                   "reposition", "repositioning")),
    ActionPattern("grasp", ("抓取", "抓住", "抓起", "夹取", "拿起", "拿住",
                            "拿取", "拿", "夹", "捏"),
                  ("grasp", "grasped", "grab", "grabbing", "pick up",
                   "picked up", "take")),
    ActionPattern("lift", ("提起", "抬起", "举起"), ("lift", "raise")),
    ActionPattern("transport", ("搬运", "搬到", "运送", "移动到", "拿到"),
                  ("transport", "transporting", "move", "moving", "carry",
                   "carrying")),
    ActionPattern("place", ("放置", "放到", "放入", "放在", "置于", "放回", "置放"),
                  ("place", "placed", "placing", "put", "set down", "put back")),
    ActionPattern("release", ("松开", "释放", "松爪"), ("release", "let go")),
    ActionPattern("handover", ("交接", "传递", "递给", "转交"), ("handover", "pass")),
    ActionPattern("open", ("打开", "开启"), ("open", "opening")),
    ActionPattern("close", ("关闭", "关上"), ("close", "closing", "shut")),
    ActionPattern("wipe", ("擦拭", "擦", "清洁"), ("wipe", "wiping", "clean", "cleaning")),
    ActionPattern("flatten", ("展开", "摊平", "铺平", "摊开"), ("flatten", "spread", "lay flat")),
    ActionPattern("fold", ("折叠", "折一下", "折好"), ("fold", "folding", "folded")),
    ActionPattern("smooth", ("整理褶皱", "抚平", "抹平", "平整"), ("smooth", "smoothing")),
    ActionPattern("hang", ("悬挂", "挂起", "挂上", "挂"), ("hang", "hanging")),
    ActionPattern("insert", ("插入", "塞入", "装入"), ("insert", "inserting", "put into")),
    ActionPattern("rotate", ("旋转", "转动", "拧"), ("rotate", "rotating", "turn")),
    ActionPattern("push", ("推动", "推"), ("push", "pushing")),
    ActionPattern("pull", ("拉动", "拉"), ("pull", "pulling")),
    ActionPattern("sort", ("分类", "分拣", "归类", "整理"), ("sort", "sorting", "categorize")),
    ActionPattern("pour", ("倒入", "倾倒", "倒出"), ("pour", "pouring", "tip")),
    ActionPattern("press", ("按压", "按下", "压住"), ("press", "pressing", "push down")),
)


_GLUE = re.compile(
    r"^(右手|左手|双手|一只手|另一只手|把|将|对|并|然后|再|"
    r"一次|一遍|the|a|an|with|using|and|then|to|into|onto|from|of|on|in|"
    r"once)$",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[\s,，。；;、:：.!！？?（）()\[\]【】<>《》→➡]+")
_DIRECT_CONNECTOR = re.compile(
    r"^(?:并|并且|and|then|随后|接着|之后|再)$", re.IGNORECASE)


@dataclass
class MiningResult:
    source_text: str
    source_text_en: str = ""
    steps: list[SkillStep] = field(default_factory=list)
    unrecognized: list[str] = field(default_factory=list)
    recognized_spans: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def skills(self) -> list[str]:
        return [step.skill for step in self.steps]

    def to_dict(self) -> dict:
        return {
            "source_text": self.source_text,
            "source_text_en": self.source_text_en,
            "steps": [step.to_dict() for step in self.steps],
            "unrecognized": list(self.unrecognized),
            "recognized_spans": list(self.recognized_spans),
            "confidence": self.confidence,
            "notes": list(self.notes),
        }


def _candidate_patterns(text: str, language: str) -> list[tuple[int, int, ActionPattern, str]]:
    candidates = []
    for pattern in ACTION_PATTERNS:
        terms = pattern.zh if language == "zh" else pattern.en
        for term in terms:
            flags = re.IGNORECASE if language == "en" else 0
            for match in re.finditer(re.escape(term), text, flags):
                candidates.append((match.start(), match.end(), pattern, term))
    # Longest match first at the same location; then restore textual order.
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected = []
    occupied_until = -1
    for item in candidates:
        if item[0] < occupied_until:
            continue
        selected.append(item)
        occupied_until = item[1]
    return selected


def _residual_tokens(text: str, spans: list[tuple[int, int, ActionPattern, str]],
                     language: str) -> list[str]:
    if not text:
        return []
    marks = []
    cursor = 0
    for start, end, _pattern, _term in spans:
        if start > cursor:
            marks.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        marks.append(text[cursor:])

    residual = []
    for chunk in marks:
        pieces = [p for p in _PUNCT.split(chunk) if p]
        if language == "zh":
            # Keep contiguous Chinese noun phrases together; only remove glue.
            tokens = pieces
        else:
            tokens = re.findall(r"[A-Za-z0-9_'-]+", chunk)
        for token in tokens:
            token = token.strip()
            if token and not _GLUE.match(token):
                residual.append(token)
    return residual


def _mine_one_language(text: str, language: str):
    spans = _candidate_patterns(text, language)
    collapsed = []
    for span in spans:
        if collapsed and collapsed[-1][2].skill == span[2].skill:
            bridge = text[collapsed[-1][1]:span[0]].strip()
            if _DIRECT_CONNECTOR.match(bridge):
                # "展开并摊平" is one flatten atom.  A noun or a comma
                # between two matches keeps them as separate actions.
                collapsed[-1] = (
                    collapsed[-1][0], span[1], collapsed[-1][2],
                    collapsed[-1][3],
                )
                continue
        collapsed.append(span)
    spans = collapsed
    unknown = _residual_tokens(text, spans, language)
    return spans, unknown


def mine_text(text: str, text_en: str = "", *,
              segment_id: int | None = None,
              object_id: str | None = None,
              object_kind: str | None = None,
              arm: str = "any") -> MiningResult:
    """Extract ordered atoms from Chinese/English task text."""
    text = (text or "").strip()
    text_en = (text_en or "").strip()
    zh_spans, zh_unknown = _mine_one_language(text, "zh")
    en_spans, en_unknown = _mine_one_language(text_en, "en")

    # Prefer the language with more recognized atoms.  If both are present,
    # keep the higher-confidence ordered sequence and retain both residuals.
    if len(en_spans) > len(zh_spans):
        spans, language, unknown = en_spans, "en", en_unknown
        selected_text = text_en
    else:
        spans, language, unknown = zh_spans, "zh", zh_unknown
        selected_text = text

    steps = []
    recognized = []
    for index, (start, end, pattern, term) in enumerate(spans):
        steps.append(SkillStep(
            skill=pattern.skill,
            object_id=object_id,
            object_kind=object_kind,
            arm=arm,
            source_segment_id=segment_id,
            source_text=text,
            source_text_en=text_en,
            confidence=pattern.confidence,
            metadata={"source_term": term, "source_language": language,
                      "source_index": index},
        ))
        recognized.append({"skill": pattern.skill, "term": term,
                           "start": start, "end": end,
                           "language": language})

    # De-duplicate residual fragments while preserving order.
    dedup_unknown = list(dict.fromkeys(unknown))
    total = len(steps) + len(dedup_unknown)
    confidence = round(len(steps) / total, 4) if total else 0.0
    notes = []
    if text and not spans and text_en:
        notes.append("中文未命中动作词，使用英文文本")
    if dedup_unknown:
        notes.append("未识别片段已保留，不能据此生成隐含技能")
    return MiningResult(
        source_text=text,
        source_text_en=text_en,
        steps=steps,
        unrecognized=dedup_unknown,
        recognized_spans=recognized,
        confidence=confidence,
        notes=notes,
    )


def mine_segments(segments: Iterable[dict]) -> list[MiningResult]:
    results = []
    for index, segment in enumerate(segments):
        result = mine_text(
            segment.get("text") or segment.get("action_description") or "",
            segment.get("text_en") or segment.get("action_description_en") or "",
            segment_id=segment.get("id", index + 1),
            object_id=segment.get("object_id"),
            object_kind=segment.get("object_kind"),
            arm=segment.get("arm", "any"),
        )
        results.append(result)
    return results


def mine_evidence_package(pkg) -> list[MiningResult]:
    """Mine annotation segments without changing the evidence package."""
    stream = pkg.get("annotation.language_segments")
    if stream is None or stream.data is None:
        return []
    return mine_segments(stream.data)


def flatten_steps(results: Iterable[MiningResult]) -> list[SkillStep]:
    steps = []
    for result in results:
        steps.extend(result.steps)
    return steps
