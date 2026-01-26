import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import requests
import re

from config import Config
from rag import get_rag_index, build_rag_context
from zipc_exporter import validate_model_json, expand_wildcards, normalize_transitions


def call_llm(prompt: str, model_name: str, temperature: float) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "input": {
            "messages": [
                {"role": "system", "content": "你是状态机建模工程师，仅输出 JSON"},
                {"role": "user", "content": prompt},
            ]
        },
        "parameters": {"temperature": temperature, "max_tokens": 3000},
    }
    headers = {
        "Authorization": f"Bearer {Config.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
    }
    response = requests.post(Config.DASHSCOPE_API_ENDPOINT, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    text = (response.json().get("output", {}).get("text") or "").strip()
    extracted = extract_json_object(text)
    return json.loads(extracted)


def extract_json_object(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    match = re.search(r"\{[\s\S]*\}\s*$", t)
    if match:
        return match.group(0).strip()
    return t


def prepare_prompt(requirement_doc: str, rag_context: str) -> str:
    return f"""
你是工业控制系统的状态机建模工程师。

你的任务是：
根据需求文档，生成【状态机的结构化 JSON 描述】，该 JSON 将作为唯一事实源，
用于前端自动渲染状态转移矩阵，以及后续导出 ZIPC 状态机。

【必须严格遵守】
1. 只输出 JSON
2. 不要 Markdown
3. 不要解释
4. JSON 必须可被 json.loads() 解析

【JSON 结构】
{{
  "states": [],
  "events": [],
  "actions": {{}},
  "initial_state": "",
  "transitions": [
    {{
      "event": "",
      "from": "",
      "to": "",
      "actions": []
    }}
  ]
}}

【actions 字段要求（非常重要）】
- actions 必须是对象/字典：{{"动作名": "描述字符串(可为空)"}}
- 例如：{{"StartMotor": "启动电机", "StopMotor": ""}}

【语义约束】
- transitions 中引用的状态/事件/动作必须已定义
- 同一 (event, from) 最多一条迁移（必须唯一）
- 如需表示“任意状态触发”，允许使用 from="*"（后端会展开为所有 states）
- 无迁移则不要生成 transition
- 不要生成未定义的状态/事件/动作
- states/events 不能为空

【参考资料（优先遵循规范，其次参考样例）】
{rag_context}

【需求文档】
{requirement_doc}
""".strip()


def score_against_expected(actual: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, float]:
    def jaccard(a: List[str], b: List[str]) -> float:
        set_a = set(a or [])
        set_b = set(b or [])
        if not set_a and not set_b:
            return 1.0
        return len(set_a & set_b) / max(1, len(set_a | set_b))

    return {
        "states_jaccard": jaccard(actual.get("states", []), expected.get("states", [])),
        "events_jaccard": jaccard(actual.get("events", []), expected.get("events", [])),
    }


def normalize_model_json(data: Dict[str, Any]) -> Dict[str, Any]:
    data = expand_wildcards(data)
    data = normalize_transitions(data)
    validate_model_json(data)
    return data


def evaluate_case(requirement_doc: str, expected: Dict[str, Any] | None, rag_context: str) -> Tuple[bool, Dict[str, float]]:
    prompt = prepare_prompt(requirement_doc, rag_context)
    model_json = call_llm(prompt, Config.DEFAULT_MODEL, Config.DEFAULT_TEMPERATURE)
    normalized = normalize_model_json(model_json)
    if expected:
        return True, score_against_expected(normalized, expected)
    return True, {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="data/eval_cases.jsonl")
    parser.add_argument("--use-rag", action="store_true")
    args = parser.parse_args()

    if not Config.DASHSCOPE_API_KEY:
        raise SystemExit("Missing DASHSCOPE_API_KEY")

    rag_index = get_rag_index(Config.RAG_INDEX_DIR) if args.use_rag else None
    results = []

    with open(args.cases, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            requirement_doc = row["requirementDoc"]
            expected = row.get("expected")
            rag_context = ""
            if rag_index:
                matches = rag_index.retrieve(requirement_doc, Config.RAG_TOP_K, Config.RAG_MIN_SCORE)
                rag_context = build_rag_context(matches)
            ok, scores = evaluate_case(requirement_doc, expected, rag_context)
            results.append({"ok": ok, "scores": scores})

    total = len(results)
    valid_rate = sum(1 for r in results if r["ok"]) / max(1, total)
    avg_states = sum(r["scores"].get("states_jaccard", 0) for r in results) / max(1, total)
    avg_events = sum(r["scores"].get("events_jaccard", 0) for r in results) / max(1, total)

    print(f"cases={total} valid_rate={valid_rate:.2%} states_jaccard={avg_states:.3f} events_jaccard={avg_events:.3f}")


if __name__ == "__main__":
    main()
