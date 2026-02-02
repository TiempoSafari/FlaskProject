import json
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from openpyxl import Workbook
import requests


SYMBOL_ACTION_NONE = "/"
SYMBOL_ACTION_ILLEGAL = "×"
SYMBOL_TRANSITION_STAY = "-"


@dataclass
class Rule:
    rule_id: str
    rule_type: str
    source: str
    description: str
    condition: Dict[str, Any]
    correction: Optional[Dict[str, Any]] = None


@dataclass
class RuleIssue:
    rule_id: str
    rule_type: str
    source: str
    description: str
    table: str
    state: str
    event: str
    detail: str
    suggestion: Optional[Dict[str, Any]] = None


def _clean_text(text: str) -> str:
    """压缩空白字符，便于从需求文本中做关键字判断。"""
    return re.sub(r"\s+", " ", text or "").strip()


def _table_entries(jsonc_data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """提取所有符合 SEAT 结构的表（包含 S/E/SEAT 的对象）。"""
    return [
        (name, table)
        for name, table in (jsonc_data or {}).items()
        if isinstance(table, dict) and "S" in table and "E" in table and "SEAT" in table
    ]


def _extract_json_object(text: str) -> str:
    """从大模型输出中提取最外层 JSON 对象或数组。"""
    if not text:
        return text
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    match_obj = re.search(r"\{[\s\S]*\}\s*$", t)
    if match_obj:
        return match_obj.group(0).strip()
    match_list = re.search(r"\[[\s\S]*\]\s*$", t)
    if match_list:
        return match_list.group(0).strip()
    return t


def _strip_jsonc_comments(text: str) -> str:
    """移除 JSONC 的 // 与 /* */ 注释（不处理字符串内部注释）。"""
    if not text:
        return text
    no_block = re.sub(r"/\*[\s\S]*?\*/", "", text)
    return re.sub(r"//.*$", "", no_block, flags=re.MULTILINE)


def build_knowledge_graph(jsonc_data: Dict[str, Any]) -> nx.DiGraph:
    """
    构建知识图谱（NetworkX 有向图）。
    节点：仅使用 State（状态）节点。
    边：State -> State 的 TRANSITION 边，动作/事件等作为边属性。
    """
    graph = nx.DiGraph()
    tables = _table_entries(jsonc_data)

    for table_name, table in tables:
        states = table.get("S", [])
        seat = table.get("SEAT", {})

        # 1) 添加状态节点
        for state in states:
            if isinstance(state, dict):
                state_name = state.get("name")
                state_type = state.get("type", "atomic")
                parent = state.get("parent") or (table.get("meta", {}).get("parent"))
            else:
                state_name = str(state)
                state_type = "atomic"
                parent = table.get("meta", {}).get("parent")

            node_id = f"State::{state_name}"
            graph.add_node(
                node_id,
                type="State",
                name=state_name,
                state_type=state_type,
                parent=parent or "",
                table=table_name,
            )

        # 2) 添加转移边（事件/动作作为属性）
        for event_name, row in seat.items():
            for state_name, cell in (row or {}).items():
                if not isinstance(cell, dict):
                    continue
                target = cell.get("T") or SYMBOL_TRANSITION_STAY
                actions = cell.get("A") or []
                if target in (SYMBOL_TRANSITION_STAY, "null"):
                    continue

                graph.add_edge(
                    f"State::{state_name}",
                    f"State::{target}",
                    type="TRANSITION",
                    table=table_name,
                    event=event_name,
                    actions=[a for a in actions if a not in (SYMBOL_ACTION_NONE, SYMBOL_ACTION_ILLEGAL)],
                )

        # 3) CONTAIN：复合状态包含子状态
        for state in states:
            if not isinstance(state, dict):
                continue
            if state.get("type") != "compound" or not state.get("children"):
                continue
            child_table = jsonc_data.get(state["children"], {})
            for child_state in child_table.get("S", []):
                child_name = child_state.get("name") if isinstance(child_state, dict) else str(child_state)
                graph.add_edge(
                    f"State::{state['name']}",
                    f"State::{child_name}",
                    type="CONTAIN",
                )

    return graph


def summarize_knowledge_graph(graph: nx.DiGraph, limit: int = 200) -> Dict[str, Any]:
    """将知识图谱转为结构化摘要（按类型聚合 + 转移边属性）。"""
    states = []
    transitions = []
    contains = []

    for node, data in graph.nodes(data=True):
        if data.get("type") == "State":
            states.append({
                "id": node,
                "name": data.get("name"),
                "state_type": data.get("state_type"),
                "table": data.get("table"),
            })
        if len(states) >= limit:
            break

    for source, target, data in graph.edges(data=True):
        edge_type = data.get("type")
        if edge_type == "TRANSITION":
            transitions.append({
                "from": source,
                "to": target,
                "event": data.get("event"),
                "actions": data.get("actions"),
                "table": data.get("table"),
            })
        elif edge_type == "CONTAIN":
            contains.append({
                "parent": source,
                "child": target,
            })
        if len(transitions) + len(contains) >= limit:
            break

    return {
        "states": states,
        "transitions": transitions,
        "contains": contains,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
    }


def _configure_matplotlib_fonts() -> Optional[str]:
    """为知识图谱图片选择可用的中文字体，避免出现方框乱码。"""
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")
    custom_font = os.getenv("SEAT_KG_FONT")
    if custom_font and os.path.isfile(custom_font):
        matplotlib.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=custom_font).get_name()]
        matplotlib.rcParams["axes.unicode_minus"] = False
        return custom_font

    font_candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Songti.ttc",
        "C:\\\\Windows\\\\Fonts\\\\msyh.ttc",
        "C:\\\\Windows\\\\Fonts\\\\simhei.ttf",
        "C:\\\\Windows\\\\Fonts\\\\simfang.ttf",
    ]
    for path in font_candidates:
        if os.path.isfile(path):
            matplotlib.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=path).get_name()]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return path

    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred_fonts:
        if name in available_fonts:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None


def visualize_knowledge_graph(graph: nx.DiGraph, output_path: str) -> None:
    """绘制知识图谱 PNG（只做可视化，不改变图数据）。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = _configure_matplotlib_fonts()

    if not graph.nodes:
        return
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, k=0.8, seed=42)
    node_labels = {node: data.get("name", node) for node, data in graph.nodes(data=True)}
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="#c7ddff")
    nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.3)
    font_family = None
    if font_path and os.path.isfile(font_path):
        font_family = font_manager.FontProperties(fname=font_path).get_name()
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, font_family=font_family)
    edge_labels = {}
    for source, target, data in graph.edges(data=True):
        edge_type = data.get("type")
        if edge_type == "TRANSITION":
            event = data.get("event") or ""
            actions = data.get("actions") or []
            action_text = ",".join(actions) if actions else "/"
            edge_labels[(source, target)] = f"{event}\nA:{action_text}"
        elif edge_type == "CONTAIN":
            edge_labels[(source, target)] = "CONTAIN"
        else:
            edge_labels[(source, target)] = edge_type or ""
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, font_family=font_family)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _normalize_rule_dict(rule: Dict[str, Any]) -> Rule:
    """将字典规则转换为 Rule 对象，并做最小字段校验。"""
    if not isinstance(rule, dict):
        raise ValueError("规则必须是对象")
    required = ["rule_id", "rule_type", "source", "description", "condition"]
    for field in required:
        if field not in rule:
            raise ValueError(f"规则缺少字段: {field}")
    return Rule(
        rule_id=str(rule["rule_id"]),
        rule_type=str(rule["rule_type"]),
        source=str(rule["source"]),
        description=str(rule["description"]),
        condition=rule["condition"] if isinstance(rule["condition"], dict) else {},
        correction=rule.get("correction"),
    )


def generate_rules_with_llm(
    requirement_doc: str,
    knowledge_graph: nx.DiGraph,
    llm_config: Dict[str, Any],
) -> List[Rule]:
    """调用百炼大模型，根据需求文档 + 知识图谱生成规则集。"""
    api_key = llm_config.get("api_key")
    api_endpoint = llm_config.get("api_endpoint")
    model_name = llm_config.get("model_name")
    temperature = llm_config.get("temperature", 0.2)

    if not api_key or not api_endpoint or not model_name:
        raise ValueError("规则生成需要 api_key/api_endpoint/model_name")

    graph_summary = summarize_knowledge_graph(knowledge_graph)
    prompt = f"""
你是SEAT模型规则抽取专家。根据以下需求文档与知识图谱摘要，生成冲突规则与遗漏规则。

【知识图谱含义说明】
1) 节点类型：
  - State：状态节点，来源于 JSONC 的 S（状态集合）。
2) 边类型：
  - TRANSITION：State -> State，表示状态转移。事件/动作作为边属性：
      - event：触发事件名称
      - actions：动作列表（已过滤“/”“×”）
  - CONTAIN：State -> State，表示复合状态包含子状态。
3) 规则应利用图谱结构与需求文本进行一致性检查：
  - 冲突：与需求或图谱逻辑矛盾（例如同一状态+事件产生多个不同转移）。
  - 遗漏：需求或图谱应有但未明确说明的转移动作/目标。

【需求文档】
{requirement_doc}

【知识图谱摘要】
{json.dumps(graph_summary, ensure_ascii=False)}

【输出要求】
1) 仅输出 JSON 数组，不要 Markdown，不要解释。
2) 数组元素为规则对象，字段结构如下：
  {{
    "rule_id": "C001",
    "rule_type": "冲突/遗漏",
    "source": "需求显式约束/需求隐含约束/SEAT公理/领域常识",
    "description": "规则描述",
    "condition": {{
      "state": "状态名或*",
      "event": "事件名或[事件列表]或*",
      "action": "动作名或[动作列表]或条件",
      "transfer": "目标状态名或条件"
    }},
    "correction": {{
      "action": "动作名或[动作列表]",
      "transfer": "目标状态名"
    }}
  }}
3) condition 中字段可按需省略，但必须与需求和图谱语义一致。
4) correction 字段必须使用 action/transfer，不要使用 next_state 等其他字段名。
5) 规则必须基于需求文档与图谱推导，不能写死固定规则。
6) 如果无法生成规则，也必须输出空数组 []。
""".strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8",
    }

    payload = {
        "model": model_name,
        "input": {
            "messages": [
                {"role": "system", "content": "你是规则抽取专家，仅输出JSON数组"},
                {"role": "user", "content": prompt},
            ]
        },
        "parameters": {
            "temperature": float(temperature),
            "max_tokens": 2000,
        },
    }

    response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    raw_text = (response.json().get("output", {}).get("text") or "").strip()
    extracted = _extract_json_object(raw_text)
    data = json.loads(_strip_jsonc_comments(extracted))

    if not isinstance(data, list):
        raise ValueError("规则生成结果必须是数组")
    return [_normalize_rule_dict(item) for item in data]


def extract_rules(requirement_doc: str, jsonc_data: Dict[str, Any]) -> List[Rule]:
    """已弃用：规则改由大模型基于知识图谱与需求生成。"""
    raise RuntimeError("extract_rules 已弃用，请使用 generate_rules_with_llm")


def write_rule_configs(rules: Iterable[Rule], output_dir: str) -> Dict[str, str]:
    """把规则写成 4 个层级的 .conf 文件，方便人工查看/维护。"""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "base_rules.conf": os.path.join(output_dir, "base_rules.conf"),
        "demand_explicit_rules.conf": os.path.join(output_dir, "demand_explicit_rules.conf"),
        "demand_implicit_rules.conf": os.path.join(output_dir, "demand_implicit_rules.conf"),
        "domain_rules.conf": os.path.join(output_dir, "domain_rules.conf"),
    }

    groups = {"base": [], "explicit": [], "implicit": [], "domain": []}
    for rule in rules:
        if rule.source == "SEAT公理":
            groups["base"].append(rule)
        elif rule.source == "需求显式约束":
            groups["explicit"].append(rule)
        elif rule.source == "需求隐含约束":
            groups["implicit"].append(rule)
        else:
            groups["domain"].append(rule)

    def _format_rule(rule: Rule) -> str:
        lines = [
            f"[{rule.rule_id}] 类型:{rule.rule_type} 来源:{rule.source} 描述:{rule.description}",
            "条件:",
        ]
        for k, v in (rule.condition or {}).items():
            lines.append(f"  {k} = {json.dumps(v, ensure_ascii=False)}")
        if rule.correction:
            lines.append(
                "判定:若满足条件，则标记为{}，修正方案:{}".format(
                    rule.rule_type,
                    json.dumps(rule.correction, ensure_ascii=False),
                )
            )
        return "\n".join(lines)

    mapping = {
        "base": paths["base_rules.conf"],
        "explicit": paths["demand_explicit_rules.conf"],
        "implicit": paths["demand_implicit_rules.conf"],
        "domain": paths["domain_rules.conf"],
    }

    for key, path in mapping.items():
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n\n".join(_format_rule(rule) for rule in groups[key]))

    return paths


def _match_rule(rule: Rule, state: str, event: str, actions: List[str], target: Optional[str]) -> bool:
    """判断某个状态-事件单元格是否命中某条规则。"""
    cond = rule.condition or {}
    if cond.get("state") not in (None, "*", state):
        return False
    if "event" in cond:
        event_cond = cond["event"]
        if event_cond != "*":
            if isinstance(event_cond, list) and event not in event_cond:
                return False
            if isinstance(event_cond, str) and event_cond != event:
                return False
    if "action_not" in cond and any(a != cond["action_not"] for a in actions):
        return True
    if "transfer_not" in cond and target != cond["transfer_not"]:
        return True
    if cond.get("action_missing") and not actions:
        return True
    if cond.get("transfer_missing") and not target:
        return True
    if "action_conflict" in cond:
        conflict_terms = cond["action_conflict"]
        if all(any(term in a for a in actions) for term in conflict_terms):
            return True
    if cond.get("state") in ("*", state) and cond.get("event") in ("*", event):
        if rule.rule_id == "C001":
            return False
    if cond.get("state") == state and cond.get("event") == event:
        return True
    return False


def validate_jsonc_rules(jsonc_data: Dict[str, Any], rules: List[Rule]) -> Dict[str, Any]:
    """遍历所有“事件 × 状态”的组合，生成冲突/遗漏报告。"""
    issues: List[RuleIssue] = []
    for table_name, table in _table_entries(jsonc_data):
        seat = table.get("SEAT", {})
        events = table.get("E", [])
        states = [s.get("name") if isinstance(s, dict) else s for s in table.get("S", [])]

        for event in events:
            for state in states:
                cell = (seat.get(event) or {}).get(state)
                actions: List[str] = []
                target: Optional[str] = None
                if isinstance(cell, dict):
                    actions = cell.get("A") or []
                    target = cell.get("T")

                for rule in rules:
                    if _match_rule(rule, state, event, actions, target):
                        issues.append(
                            RuleIssue(
                                rule_id=rule.rule_id,
                                rule_type=rule.rule_type,
                                source=rule.source,
                                description=rule.description,
                                table=table_name,
                                state=state,
                                event=event,
                                detail=f"{state}+{event}",
                                suggestion=rule.correction,
                            )
                        )

    return {
        "conflict_count": len([i for i in issues if i.rule_type == "冲突"]),
        "missing_count": len([i for i in issues if i.rule_type == "遗漏"]),
        "conflicts": [i.__dict__ for i in issues if i.rule_type == "冲突"],
        "missings": [i.__dict__ for i in issues if i.rule_type == "遗漏"],
    }


def apply_validation_fixes(jsonc_data: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    """根据校验报告自动修正缺失/冲突的单元格。"""
    if not report:
        return jsonc_data
    data = json.loads(json.dumps(jsonc_data, ensure_ascii=False))
    issues = report.get("conflicts", []) + report.get("missings", [])

    for issue in issues:
        table = data.get(issue["table"], {})
        seat = table.get("SEAT", {})
        event = issue["event"]
        state = issue["state"]
        seat.setdefault(event, {})
        if state not in seat[event]:
            seat[event][state] = {"A": [], "T": SYMBOL_TRANSITION_STAY}
        cell = seat[event][state]
        suggestion = issue.get("suggestion") or {}
        if "action" in suggestion:
            action_val = suggestion["action"]
            cell["A"] = action_val if isinstance(action_val, list) else [action_val]
        if "transfer" in suggestion:
            cell["T"] = suggestion["transfer"]
        seat[event][state] = cell
        table["SEAT"] = seat
        data[issue["table"]] = table

    return data


def jsonc_to_seat_matrix(jsonc_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """把 JSONC 转换为 Markdown + Excel 矩阵文件。"""
    os.makedirs(output_dir, exist_ok=True)
    matrices = {}
    for table_name, table in _table_entries(jsonc_data):
        states = [s.get("name") if isinstance(s, dict) else s for s in table.get("S", [])]
        events = table.get("E", [])
        seat = table.get("SEAT", {})

        lines = ["| 事件\\状态 | " + " | ".join(states) + " |", "|" + "---|" * (len(states) + 1)]
        for event in events:
            row = [f"{event}"]
            for state in states:
                cell = (seat.get(event) or {}).get(state) or {}
                actions = cell.get("A") or []
                target = cell.get("T") or SYMBOL_TRANSITION_STAY
                action_text = ",".join(actions) if actions else SYMBOL_ACTION_NONE
                row.append(f"动作：{action_text}；转移：{target}")
            lines.append("| " + " | ".join(row) + " |")

        note = (
            "\n\n### 状态变迁矩阵说明\n"
            "1. 符号约定：/=无操作，×=非法操作，-=保持当前状态；\n"
            f"2. 初始状态：{table.get('meta', {}).get('initial', '')}；\n"
            "3. 表格来源：JSONC中间结构自动转换。\n"
        )
        markdown = "\n".join(lines) + note

        markdown_path = os.path.join(output_dir, f"{table_name}_seat_matrix.md")
        with open(markdown_path, "w", encoding="utf-8") as handle:
            handle.write(markdown)

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = table_name
        sheet.append(["事件\\状态"] + states)
        for event in events:
            row = [event]
            for state in states:
                cell = (seat.get(event) or {}).get(state) or {}
                actions = cell.get("A") or []
                target = cell.get("T") or SYMBOL_TRANSITION_STAY
                action_text = ",".join(actions) if actions else SYMBOL_ACTION_NONE
                row.append(f"动作：{action_text}；转移：{target}")
            sheet.append(row)

        excel_path = os.path.join(output_dir, f"{table_name}_seat_matrix.xlsx")
        workbook.save(excel_path)

        matrices[table_name] = {
            "markdown": markdown,
            "markdown_path": markdown_path,
            "excel_path": excel_path,
        }

    return matrices


def run_full_pipeline(
    requirement_doc: str,
    jsonc_data: Dict[str, Any],
    output_dir: str,
    llm_config: Optional[Dict[str, Any]] = None,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """端到端流程：规则抽取 → 校验 → 修正 → 图谱 → 矩阵输出。"""
    os.makedirs(output_dir, exist_ok=True)

    # 1) 基于 JSONC 先构建知识图谱
    initial_kg = build_knowledge_graph(jsonc_data)

    # 2) 基于需求文档 + 知识图谱生成规则（由大模型生成，不再硬编码）
    rules = generate_rules_with_llm(requirement_doc, initial_kg, llm_config or {})
    rule_paths = write_rule_configs(rules, os.path.join(output_dir, "rule_config"))

    corrected_jsonc = jsonc_data
    validation_report = {}
    for _ in range(max_iterations):
        validation_report = validate_jsonc_rules(corrected_jsonc, rules)
        if validation_report["conflict_count"] == 0 and validation_report["missing_count"] == 0:
            break
        corrected_jsonc = apply_validation_fixes(corrected_jsonc, validation_report)

    # 3) 修正后重新构建知识图谱
    kg = build_knowledge_graph(corrected_jsonc)
    graph_path = os.path.join(output_dir, "knowledge_graph.gpickle")
    with open(graph_path, "wb") as handle:
        pickle.dump(kg, handle)

    graph_image = os.path.join(output_dir, "knowledge_graph.png")
    visualize_knowledge_graph(kg, graph_image)

    matrices = jsonc_to_seat_matrix(corrected_jsonc, os.path.join(output_dir, "seat_matrices"))

    jsonc_path = os.path.join(output_dir, "seat_model.jsonc")
    with open(jsonc_path, "w", encoding="utf-8") as handle:
        json.dump(corrected_jsonc, handle, ensure_ascii=False, indent=2)

    report_path = os.path.join(output_dir, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(validation_report, handle, ensure_ascii=False, indent=2)

    return {
        "jsonc": corrected_jsonc,
        "jsonc_path": jsonc_path,
        "knowledge_graph_path": graph_path,
        "knowledge_graph_image": graph_image,
        "rule_configs": rule_paths,
        "validation_report": validation_report,
        "validation_report_path": report_path,
        "seat_matrices": matrices,
        "output_dir": output_dir,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
