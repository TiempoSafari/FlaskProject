from typing import Dict, Any, List, Tuple


def expand_wildcards(model_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    支持 transitions 中 from="*"：
    - from="*": 展开为所有 states（生成多条迁移）
    说明：
    - 这是为了兼容模型喜欢用 '*' 表示“任意状态触发”的写法
    - 展开后再进入 normalize/validate
    """
    if not isinstance(model_json, dict):
        return model_json

    states = model_json.get("states") or []
    transitions = model_json.get("transitions") or []

    if not isinstance(states, list) or not isinstance(transitions, list):
        return model_json

    new_transitions: List[Dict[str, Any]] = []
    warnings: List[str] = model_json.get("_normalize_warnings", [])

    for i, t in enumerate(transitions):
        if not isinstance(t, dict):
            warnings.append(f"transitions[{i}] 不是对象，已忽略")
            continue

        ev = t.get("event")
        fr = t.get("from")
        to = t.get("to")
        acts = t.get("actions", [])

        if fr == "*":
            # 展开
            for s in states:
                new_transitions.append({
                    "event": ev,
                    "from": s,
                    "to": to,
                    "actions": list(acts) if isinstance(acts, list) else []
                })
            warnings.append(f"已展开通配符 from='*'：event={ev} to={to} -> {len(states)} 条")
        else:
            new_transitions.append(t)

    model_json["transitions"] = new_transitions
    if warnings:
        model_json["_normalize_warnings"] = warnings
    return model_json


def normalize_transitions(model_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    归一化 transitions：
    - 去掉完全重复的迁移（同 event/from/to/actions）
    - 处理冲突重复（同 event/from 但 to/actions 不一致）：
        默认保留第一条，其余丢弃，同时写入 model_json["_normalize_warnings"]
    """
    transitions = model_json.get("transitions", [])
    if not isinstance(transitions, list):
        model_json["transitions"] = []
        model_json["_normalize_warnings"] = ["transitions 不是 list，已重置为空数组"]
        return model_json

    warnings: List[str] = model_json.get("_normalize_warnings", [])
    kept: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for i, t in enumerate(transitions):
        if not isinstance(t, dict):
            warnings.append(f"transitions[{i}] 不是对象，已忽略")
            continue

        ev = t.get("event")
        fr = t.get("from")
        to = t.get("to")
        acts = t.get("actions", [])

        # 基础字段缺失就忽略
        if not isinstance(ev, str) or not isinstance(fr, str) or not isinstance(to, str):
            warnings.append(f"transitions[{i}] event/from/to 非字符串，已忽略: {t}")
            continue

        if acts is None:
            acts = []
        if not isinstance(acts, list):
            warnings.append(f"transitions[{i}].actions 非数组，已强制置空: {t}")
            acts = []

        acts2 = [str(a) for a in acts]
        t_norm = {"event": ev, "from": fr, "to": to, "actions": acts2}

        key = (ev, fr)
        if key not in kept:
            kept[key] = t_norm
            continue

        prev = kept[key]

        # 完全一致：忽略即可
        if prev["to"] == t_norm["to"] and prev.get("actions", []) == t_norm.get("actions", []):
            warnings.append(f"发现完全重复迁移，已去重: (event,from)=({ev},{fr})")
            continue

        # 冲突：默认保留第一条
        warnings.append(
            "发现冲突重复迁移，已保留第一条并丢弃后续："
            f"(event,from)=({ev},{fr}) | keep={prev} | drop={t_norm}"
        )

    model_json["transitions"] = list(kept.values())
    if warnings:
        model_json["_normalize_warnings"] = warnings
    else:
        model_json.pop("_normalize_warnings", None)
    return model_json


def validate_model_json(model_json: dict):
    """
    强校验：确保前端渲染 + ZIPC 导出不会炸。
    注意：应在 expand_wildcards + normalize_transitions 后调用。
    """
    if not isinstance(model_json, dict):
        raise ValueError("顶层必须是 JSON 对象")

    states = model_json.get("states")
    events = model_json.get("events")
    actions = model_json.get("actions", {})
    transitions = model_json.get("transitions", [])
    initial_state = model_json.get("initial_state")

    if not isinstance(states, list) or not states or not all(isinstance(s, str) and s.strip() for s in states):
        raise ValueError("states 必须是非空字符串数组")

    if not isinstance(events, list) or not events or not all(isinstance(e, str) and e.strip() for e in events):
        raise ValueError("events 必须是非空字符串数组")

    if not isinstance(actions, dict):
        raise ValueError("actions 必须是对象/字典：{动作名: 描述字符串}")

    for k, v in actions.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError("actions 的 key（动作名）必须是非空字符串")
        if not isinstance(v, str):
            raise ValueError("actions 的 value（描述）必须是字符串（可为空）")

    if initial_state is None or not isinstance(initial_state, str) or not initial_state.strip():
        initial_state = states[0]
        model_json["initial_state"] = initial_state

    if initial_state not in states:
        raise ValueError("initial_state 必须出现在 states 中")

    if not isinstance(transitions, list):
        raise ValueError("transitions 必须是数组")

    # (event, from) 唯一性兜底检查
    seen = set()

    for i, t in enumerate(transitions):
        if not isinstance(t, dict):
            raise ValueError(f"transitions[{i}] 必须是对象")

        ev = t.get("event")
        fr = t.get("from")
        to = t.get("to")
        acts = t.get("actions", [])

        if not (isinstance(ev, str) and ev in events):
            raise ValueError(f"transitions[{i}].event 必须在 events 中定义")
        if not (isinstance(fr, str) and fr in states):
            raise ValueError(f"transitions[{i}].from 必须在 states 中定义")
        if not (isinstance(to, str) and to in states):
            raise ValueError(f"transitions[{i}].to 必须在 states 中定义")

        key = (ev, fr)
        if key in seen:
            raise ValueError(f"normalize 后仍存在重复迁移：(event, from)=({ev}, {fr})")
        seen.add(key)

        if acts is None:
            acts = []
            t["actions"] = []

        if not isinstance(acts, list) or not all(isinstance(a, str) for a in acts):
            raise ValueError(f"transitions[{i}].actions 必须是字符串数组")

        for a in acts:
            if a not in actions:
                raise ValueError(f"transitions[{i}] 引用了未定义动作: {a}")


def generate_zipc_txt(model_json: dict) -> str:
    """
    生成 ZIPC STM TEXT FORMAT Ver1.01 文本。
    """
    validate_model_json(model_json)

    states = model_json["states"]
    events = model_json["events"]
    actions = model_json.get("actions", {})
    initial_state = model_json.get("initial_state", states[0])

    state_index = {s: i for i, s in enumerate(states)}

    lines: List[str] = []

    # ===== Header =====
    lines.append("{%ZIPC STM TEXT FORMAT Ver1.01%}")
    lines.append("%Header%")
    lines.append("0 0 0 0 0 0 0 1 {} {}".format(len(events), len(states)))
    lines.append("E type normal")
    lines.append("void")
    lines.append("state_machine")
    lines.append(str(state_index[initial_state]))
    lines.append("")

    # ===== Event =====
    lines.append("%Event%")
    lines.append(str(len(events)))
    for e in events:
        lines.append("%StmCellEv%")
        lines.append(" ".join(["-1"] * 15))
        lines.append(e)
        lines.append("")

    lines.append("%Event Group%")
    lines.append("0")

    # ===== State =====
    lines.append("%State%")
    lines.append(str(len(states)))
    for s in states:
        lines.append("%StmCellSt%")
        lines.append(" ".join(["-1"] * 15))
        lines.append(s)
        lines.append("")

    lines.append("%State Group%")
    lines.append("0")

    # ===== Action =====
    lines.append("%Action%")
    lines.append(str(len(actions)))
    for i, (name, desc) in enumerate(actions.items()):
        lines.append(str(i))
        lines.append("%StmCellAc%")
        lines.append("-1 0 2")
        lines.append(name)

    # ===== Footer =====
    lines.append("%Event Start Activity%")
    lines.append("0")
    lines.append("%Event End Activity%")
    lines.append("0")
    lines.append("%State Start Activity%")
    lines.append("0")
    lines.append("%State End Activity%")
    lines.append("0")
    lines.append("%State Mode Activity%")
    lines.append("0")

    return "\n".join(lines)
