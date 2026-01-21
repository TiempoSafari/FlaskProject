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

        # 3) Prompt（保持你原思路，但明确 actions 结构）
        json_prompt = f"""
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
                    {"role": "system", "content": "你是状态机建模工程师，仅输出 JSON"},
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

        # 5) 解析 JSON
        extracted = _extract_json_object(raw_text)
        stm_json = json.loads(extracted)

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
