from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import requests
import json
import re
import logging
import uuid
from datetime import datetime

from config import Config
from zipc_exporter import (
    generate_zipc_txt,
    validate_model_json,
    normalize_transitions,
    expand_wildcards,
)

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/*": {"origins": "*"}})

# ========= 日志配置 =========
logger = logging.getLogger("zipc")
logger.setLevel(logging.INFO)

if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)


@app.route('/')
def index():
    return render_template('index.html')


def _extract_json_object(text: str) -> str:
    """从模型输出中提取最外层 JSON 对象（兼容 code fence/夹杂文本）。"""
    if not text:
        return text

    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    # 尽量取最后的 {...}
    m = re.search(r"\{[\s\S]*\}\s*$", t)
    if m:
        return m.group(0).strip()
    return t


def _strip_jsonc_comments(text: str) -> str:
    """移除 JSONC 的 // 与 /* */ 注释（不处理字符串内部的注释样式）。"""
    if not text:
        return text
    no_block = re.sub(r"/\*[\s\S]*?\*/", "", text)
    return re.sub(r"//.*$", "", no_block, flags=re.MULTILINE)


def _is_seat_jsonc(model_json: dict) -> bool:
    """判定是否为 SEAT JSONC 结构（顶层包含多个表，表内含 S/E/SEAT）。"""
    if not isinstance(model_json, dict):
        return False
    for _, table in model_json.items():
        if isinstance(table, dict) and "S" in table and "E" in table and "SEAT" in table:
            return True
    return False


@app.route('/api/generate-matrix', methods=['POST'])
def generate_matrix():
    req_id = uuid.uuid4().hex[:8]
    started = datetime.now().isoformat(timespec="seconds")

    try:
        # 1) 配置
        api_key = app.config.get('DASHSCOPE_API_KEY')
        api_endpoint = app.config.get('DASHSCOPE_API_ENDPOINT')
        default_model = app.config.get('DEFAULT_MODEL')
        default_temperature = app.config.get('DEFAULT_TEMPERATURE', 0.2)

        if not api_key:
            return jsonify({"code": 400, "msg": "未配置百炼API Key"}), 400
        if not api_endpoint:
            return jsonify({"code": 400, "msg": "未配置百炼API Endpoint"}), 400

        # 2) 请求体
        data = request.get_json(silent=True) or {}
        requirement_doc = (data.get('requirementDoc') or '').strip()
        model_name = data.get('modelName', default_model)

        try:
            temperature = float(data.get('temperature', default_temperature))
        except Exception:
            temperature = float(default_temperature)

        if not requirement_doc:
            return jsonify({"code": 400, "msg": "需求文档不能为空"}), 400

        logger.info(
            f"[{req_id}] /api/generate-matrix START at {started} | model={model_name} | "
            f"temperature={temperature} | requirement_len={len(requirement_doc)}"
        )

        # 3) Prompt（SEAT JSONC 结构）
        json_prompt = f"""
你是工业控制系统的状态机建模工程师。

你的任务是：
根据需求文档，先进行 SEAT 状态迁移矩阵建模，再输出 JSONC 格式的 SEAT 模型。
该 JSONC 将作为事实源，用于前端渲染 SEAT 表格与后续分析。

【必须严格遵守】
1. 只输出 JSONC（允许 // 和 /* */ 注释）
2. 不要 Markdown
3. 不要解释
4. JSONC 必须在移除注释后可被 json.loads() 解析

【JSONC 结构要求】
- 顶层为多个表对象（可包含主表与子表），键为表名：
  "主表0", "子表0.1" 等
- 每个表包含：
  - meta: 表元数据（title/description/parent/initial/symbols 等）
  - S: 状态集合（数组，元素为 {{name, type, children?, initial?}}）
  - E: 事件集合（数组，字符串）
  - SEAT: 二维矩阵（对象）

【SEAT 单元格语义】
- SEAT 的行键为事件 E，列键为状态 S.name
- 单元格格式：{{ "A": [动作...], "T": "目标状态" }}
  - A: 动作数组，可使用 "/" 表示无操作，"×" 表示非法操作
  - T: 目标状态，"-" 表示保持当前状态

【建模要求】
- 必须先建模再分析需求问题，发现所有潜在漏洞/二义性/遗漏
- 概念内涵与外延必须严格、无歧义
- 模型需便于与概要设计、详细设计一一映射

【需求文档】
{requirement_doc}
""".strip()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8"
        }

        payload = {
            "model": model_name,
            "input": {
                "messages": [
                    {"role": "system", "content": "你是状态机建模工程师，仅输出 JSONC"},
                    {"role": "user", "content": json_prompt}
                ]
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": 3000
            }
        }

        # 4) 调用模型
        response = requests.post(
            api_endpoint,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        raw_text = (response.json().get("output", {}).get("text") or "").strip()

        # ✅ 控制台打印大模型原始输出（你要求的）
        logger.info(f"[{req_id}] LLM raw output begin >>>\n{raw_text}\n<<< LLM raw output end")

        # 5) 解析 JSON/JSONC
        extracted = _extract_json_object(raw_text)
        stm_json = json.loads(_strip_jsonc_comments(extracted))

        if _is_seat_jsonc(stm_json):
            logger.info(f"[{req_id}] parsed SEAT JSONC model with tables={len(stm_json.keys())}")
            return jsonify({
                "code": 200,
                "msg": "生成成功",
                "data": stm_json
            })

        logger.info(
            f"[{req_id}] parsed json summary | states={len(stm_json.get('states') or [])} "
            f"| events={len(stm_json.get('events') or [])} "
            f"| transitions={len(stm_json.get('transitions') or [])} "
            f"| actions={len((stm_json.get('actions') or {}))}"
        )

        # 6) ✅ 先展开通配符（修复 from='*'）
        before_expand = len(stm_json.get("transitions") or [])
        stm_json = expand_wildcards(stm_json)
        after_expand = len(stm_json.get("transitions") or [])
        if after_expand != before_expand:
            logger.warning(f"[{req_id}] transitions expanded (wildcards): {before_expand} -> {after_expand}")

        # 7) ✅ 再去重/冲突处理
        before_norm = len(stm_json.get("transitions") or [])
        stm_json = normalize_transitions(stm_json)
        after_norm = len(stm_json.get("transitions") or [])
        if before_norm != after_norm:
            logger.warning(f"[{req_id}] transitions normalized: {before_norm} -> {after_norm}")

        # 打印归一化告警
        if stm_json.get("_normalize_warnings"):
            logger.warning(f"[{req_id}] normalize warnings:")
            for w in stm_json["_normalize_warnings"]:
                logger.warning(f"[{req_id}]  - {w}")

        # 8) 校验（不会再因 '*' 报错）
        validate_model_json(stm_json)

        logger.info(f"[{req_id}] /api/generate-matrix SUCCESS")

        return jsonify({
            "code": 200,
            "msg": "生成成功",
            "data": stm_json
        })

    except json.JSONDecodeError:
        logger.exception(f"[{req_id}] JSON decode error")
        return jsonify({"code": 500, "msg": "模型返回的 JSON 无法解析"}), 500
    except requests.RequestException as e:
        logger.exception(f"[{req_id}] request model api failed")
        return jsonify({"code": 500, "msg": f"请求模型接口失败: {str(e)}"}), 500
    except Exception as e:
        logger.exception(f"[{req_id}] server error")
        return jsonify({"code": 500, "msg": str(e)}), 500


@app.route('/api/save-matrix', methods=['POST'])
def save_matrix():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"code": 400, "msg": "请求体必须是 JSON"}), 400

        if _is_seat_jsonc(data):
            with open("zipc_matrix.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return jsonify({"code": 200, "msg": "保存成功"})

        # 与生成接口同流程：expand -> normalize -> validate
        data = expand_wildcards(data)
        data = normalize_transitions(data)
        validate_model_json(data)

        with open("zipc_matrix.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return jsonify({"code": 200, "msg": "保存成功"})
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


@app.route('/api/export-zipc', methods=['POST'])
def export_zipc():
    """
    输入：状态机 JSON
    输出：ZIPC STM TEXT（txt 文件）
    """
    try:
        model_json = request.get_json(silent=True)
        if model_json is None:
            return jsonify({"code": 400, "msg": "请求体必须是 JSON"}), 400

        if _is_seat_jsonc(model_json):
            return jsonify({"code": 400, "msg": "SEAT JSONC 暂不支持 ZIPC 导出"}), 400

        model_json = expand_wildcards(model_json)
        model_json = normalize_transitions(model_json)
        validate_model_json(model_json)

        txt = generate_zipc_txt(model_json)

        return Response(
            txt,
            mimetype="text/plain; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=zipc_state_machine.txt"}
        )
    except Exception as e:
        return jsonify({"code": 500, "msg": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
