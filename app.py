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
【子表/复合状态规则（非常重要）】
- 只有当需求中明确存在“状态内部还有子状态/子流程/子模式”的层级结构时，才使用 type="compound" 并建立子表。
- 如果需求不需要层级状态，所有状态都应为 type="atomic"，也不要生成任何子表。
- 子表可以有多个、一个或零个；必须由需求驱动，不要因为示例而强行生成主表+子表。

【SEAT 单元格语义】
- SEAT 的行键为事件 E，列键为状态 S.name
- 单元格格式：{{ "A": [动作...], "T": "目标状态" }}
  - A: 动作数组，可使用 "/" 表示无操作，"×" 表示非法操作
  - T: 目标状态，"-" 表示保持当前状态

【建模要求】
- 必须先建模再分析需求问题，发现所有潜在漏洞/二义性/遗漏
- 概念内涵与外延必须严格、无歧义
- 模型需便于与概要设计、详细设计一一映射

【说明与示例（保留结构与内容，不要省略）】
SEAT模型结构上为一个二维表格，包括状态（State）、事件（Event）、动作（Action）和转移（Transition）四个要素组成。最上的横行为状态，最左纵列为事件，中间单元格描述在首行当前状态以及左列事件产生的条件下，对应的处理过程以及状态迁移。
以自动门需求确认示例：SEAT模型的应用
首先，拿到一份初步的自动门需求文档，该文档通常以自然语言描述系统的核心功能，但往往会因为表达的模糊性、二义性和逻辑不完整而在开发和实现中引起误解。
初步自动门需求文档示例：
自动门的电源由电源开关控制。当电源关闭时，门保持静止，所有传感器停用。
系统配有一个人感应器，当检测到有人时门会自动开启。
系统配备开门和关门的时间限制，时间到后门会自动切换到相应状态。
该文档概述了系统的主要功能，但较明显的存在问题，例如：
正确性：未定义“检测到人”在不同状态下的处理方式，例如当门在已经开启状态下，检测到人后进行开门是错误的动作;
完整性：未描述电源开机后，系统恢复到何种状态；
二义性： 对时间到后门会自动切换到相应状态，未定义“相应状态”具体内容，不同工程师会出现不同理解；
模块化缺失：电源开关、传感器和定时器之间的逻辑关系不明确。
使用SEAT模型的需求分析和模型化
通过状态迁移矩阵（SEAT）建模，将需求转化为结构化的状态-事件模型，从而清晰地揭示出二义性、逻辑缺陷及需求漏洞。
需求分析过程：
提取需求的SEAT模型：SEAT模型将需求分解为状态（State）、事件（Event）、动作（Action）、转移（Transition）四要素：
状态：包括“电源关”状态，以及“电源开”状态下的“已关门”、“开门中”、“已开门”和“关门中”子状态。
事件：例如“检测到人”、“开门时间到”、“关门时间到”、“电源开关按下”等。
动作：如“启动开门”、“启动关门”、“停止开门”、“启动定时器”等。
转移：事件和动作触发状态转换。例如，“电源开 -> 已关门”状态下，当检测到人时，系统会启动开门，进入“开门中”状态。
识别需求漏洞和不一致：
示例漏洞：通过SEAT模型分析发现，不同状态“检测到人”事件应该采用不同的处理方式。因此需补充逻辑以保证该状态下的安全响应，例如在“关门中”状态下检测到人时，系统应停止关门并重新进入“开门中”状态。
模块化需求分解与接口定义：
模块划分：SEAT建模后，需求被划分为“电源管理模块”、“门操作模块”和“传感器模块”，并通过接口实现模块间交互。传感器模块仅负责发送“检测到人”信号，而门操作模块负责接收信号并执行相应操作。这种设计清晰地划分了模块边界，减少模块间的耦合。
概要设计：定义功能的逻辑架构
基于SEAT模型分析后的需求模型，工程师进行概要设计，创建逻辑流程图和模块接口图。
状态转换逻辑图：将各模块的状态和事件关系构建为逻辑SEAT图。比如，当系统处于“开门中”状态且开门时间到时，系统自动转入“已开门”状态。
定时器与传感器的联动关系：通过SEAT模型仿真测试，进一步确认各个定时器和传感器间的逻辑依赖。例如，“开门时间到”事件会在“开门中”状态触发门进入“已开门”状态，确保定时器逻辑能正常工作。
模块接口定义：传感器模块的“检测到人”信号仅触发门操作模块的相应行为，这种模块接口定义减少直接依赖，提高系统可维护性。
详细设计：实现需求逻辑的具体规则
SEAT模型基于概要设计生成详细设计，包括状态和事件的具体操作步骤和异常处理逻辑。
状态迁移矩阵中的详细设计规则：SEAT模型在每个状态下定义具体事件和相应动作。例如，在“已开门”状态，触发“开门等待时间到”事件时，门应进入“关门中”状态并启动关门定时器。
异常处理与容错设计：可以在SEAT模型在详细设计中包含容错逻辑。例如在传感器损坏的情况下，可以在SEAT模型中引入定时器作为容错机制。例如：
如果“检测到人”传感器失效，系统在“已开门”状态保持一段时间后，自动触发“开门等待时间到”事件，进入“关门中”状态。这种设计确保系统在传感器故障时仍能安全运行。
仿真测试与逻辑确认
通过SEAT模型仿真测试详细设计的状态转换逻辑。例如，在“已关门”状态检测到人时，门自动进入“开门中”状态并启动开门定时器，确保设计逻辑无误。
SEAT模型需求映射到代码实现
可以将SEAT模型转换为代码，实现从需求到设计和代码的完整映射。
生成标准代码：根据SEAT模型自动生成C代码，将状态、事件和动作映射到代码中的函数和控制语句中。比如，“检测到人”事件的触发逻辑直接映射到开闭门控制函数中。
代码与需求的追溯：代码与需求模型同步，任何需求更改都能自动更新模型和代码。例如，若需求更改为“关门中检测到人时重新进入开门状态”，SEAT更新状态表并会相应更新代码，确保设计实现的完整映射。
容错确认：利用外部工具对SEAT模型进行容错仿真测试，系统在“开门中”状态下若未检测到人，则会通过定时器自动触发状态转换，确保在传感器失效情况下仍能实现动作。
SEAT模型建模后自动门系统的完整需求文档
1.系统状态
主状态：
电源关：所有传感器和定时器停用，自动门保持静止。
电源开：系统处于工作状态，并包含四个子状态：
已关门：门完全关闭。
开门中：门正在打开。
已开门：门完全打开。
关门中：门正在关闭。
2.电源开关控制
当自动门处于“电源关”状态时，按下“电源开关”按钮会切换到“电源开”状态。系统会恢复到上次从“电源开”状态切出时的子状态，若无记录则默认进入“已关门”状态。
当自动门处于“电源开”状态时，按下“电源开关”按钮会切换到“电源关”状态，停止所有传感器和定时器。
3.人感应检测器
在“电源关”状态下，不会触发“检测到人”事件。
当“电源开”状态下触发“检测到人”时：
若系统处于“已关门”状态，则启动开门操作并设置开门时间，开始计时，自动门进入“开门中”状态；
若系统处于“开门中”状态，忽略此事件；
若系统处于“已开门”状态，重置等待时间，并启动等待定时器；
若系统处于“关门中”状态，停止关门操作，设置开门时间，重新启动开门，系统进入“开门中”状态。
4.门开检测器
在“电源关”或“已关门”状态下，不会触发“检测门开”事件；
“已开门”或“关门中”状态下触发“检测门开”时，忽略此事件；
当“开门中”状态下触发“检测门开”时，停止开门操作，设置等待时间并启动等待定时器，自动门进入“已开门”状态。
5.门关检测器
在“电源关”或“已开门”状态下，不会触发“检测门关”事件；
在“已关门”或“开门中”状态下触发“检测门关”事件时，忽略此事件；
当“关门中”状态下触发“检测门关”事件时，停止关门操作，设置等待时间并启动等待定时器，自动门进入“已关门”状态。
6.关门定时器
在“电源关”、“已关门”、“开门中”或“已开门”状态下触发“关门时间到”事件时，忽略此事件；
在“关门中”状态下触发“关门时间到”事件时，停止关门操作，自动门进入“已关门”状态。
7.开门定时器
在“电源关”、“已关门”、“已开门”或“关门中”状态下触发“开门时间到”事件时，忽略此事件；
在“开门中”状态下触发“开门时间到”事件时，停止开门操作，设置等待时间并启动等待定时器，自动门进入“已开门”状态。
8.等待定时器
在“电源关”、“已关门”、“开门中”或“关门中”状态下触发“开门等待时间到”事件时，忽略此事件；
在“已开门”状态下触发“开门等待时间到”事件时，启动关门操作，设置关门时间并启动关门定时器，自动门进入“关门中”状态。

进行JSONC格式的SEAT建模（未做自然语言编译和变量设定）：
{{
  "主表0": {{
    "meta": {{
      "title": "设备主控状态机",
      "description": "控制电源开关及门控制子系统",
      "symbols": {{
        "A": {{
          "/": "无操作",
          "×": "非法操作（记录错误日志）",
          "null": "动作未被定义(需求遗漏)",
          "子表0.1": "激活门控制子状态机"
        }},
        "T": {{
          "-": "保持当前状态",
          "null": "动作未被定义(需求遗漏)"
        }}
      }},
      "initial": "电源关"
    }},
 // 状态集合（作为表格列名）
    "S": [
      {{
        "name": "电源关",
        "type": "atomic"
      }},
      {{
        "name": "电源开",
        "type": "compound",
        "children": "子表0.1",
        "initial": "已关门"
      }}
    ],
 // 事件集合（作为表格行名）
    "E": ["电源开关"],
// 二维核心数据：E为行键，S为列键，值为A+T
    "SEAT": {{
      "电源开关": {{
        "电源关": {{
          "A": ["@通电自检", "@系统初始化"],
          "T": "电源开 "
        }},
        "电源开": {{
          "A": ["@切断电源", "@保存状态"],
          "T": "电源关"
        }}
      }}
      }}
  }},

  // ================== 子表0.1：门控制状态机 ==================
  "子表0.1": {{
    "meta": {{
      "title": "门控制子系统",
      "parent": "主表0/电源开",
      "initial": "已关门"
    }},
 // 状态集合（作为表格列名）
    "S": [
      {{ "name": "已关门", "type": "atomic" }},
      {{ "name": "开门中", "type": "atomic" }},
      {{ "name": "已开门", "type": "atomic" }},
      {{ "name": "关门中", "type": "atomic" }}
    ],
 // 事件集合（作为表格行名）
    "E": [
      "检测到人",
      "检测到门已开",
      "检测到门已关",
      "关门时间到",
      "开门时间到",
      "开门等待时间到"
    ],
// 二维核心数据：E为行键，S为列键，值为A+T
    "SEAT": {{
      // ------ 事件：检测到人 ------
      "检测到人": {{
        "已关门": {{
          "A": ["启动开门电机", "设置开门时长(5s)", "启动开门计时"],
          "T": "开门中"
        }},
        "开门中": {{
          "A": ["/"],
          "T": "-"
        }},
        "已开门": {{
          "A": ["设置等待时长(10s)", "重置等待计时"],
          "T": "-"
        }},
        "关门中": {{
          "A": ["停止关门电机", "设置开门时长(5s)", "启动开门计时", "启动开门电机"],
          "T": "开门中"
        }}
      }},

      // ------ 事件：检测到门已开 ------
      "检测到门已开": {{
        "已关门": {{
          "A": ["×"],
          "T": "-"
        }},
        "开门中": {{
          "A": ["停止开门电机", "设置等待时长(10s)", "启动等待定时器"],
          "T": "已开门"
        }},
        "已开门": {{
          "A": ["/"],
          "T": "-"
        }},
        "关门中": {{
          "A": ["/"],
          "T": "-"
        }}
      }},

      // ------ 事件：检测到门已关 ------
      "检测到门已关": {{
        "已关门": {{
          "A": ["/"],
          "T": "-"
        }},
        "开门中": {{
          "A": ["/"],
          "T": "-"
        }},
        "已开门": {{
          "A": ["×"],
          "T": "-"
        }},
        "关门中": {{
          "A": ["停止关门电机", "上传关门完成事件"],
          "T": "已关门"
        }}
      }},

      // ------ 事件：关门时间到 ------
      "关门时间到": {{
        "已关门": {{
          "A": ["/"],
          "T": "-"
        }},
        "开门中": {{
          "A": ["/"],
          "T": "-"
        }},
        "已开门": {{
          "A": ["×"],
          "T": "-"
        }},
        "关门中": {{
          "A": ["停止关门电机"],
          "T": "已关门"
        }}
      }},

      // ------ 事件：开门时间到 ------
      "开门时间到": {{
        "已关门": {{
          "A": ["/"],
          "T": "-"
        }},
        "开门中": {{
          "A": ["停止开门电机", "设置等待时长(10s)", "启动等待定时器"],
          "T": "已开门"
        }},
        "已开门": {{
          "A": ["/"],
          "T": "-"
        }},
        "关门中": {{
          "A": ["/"],
          "T": "-"
        }}
      }},

      // ------ 事件：开门等待时间到 ------
      "开门等待时间到": {{
        "已关门": {{
          "A": ["/"],
          "T": "-"
        }},
        "开门中": {{
          "A": ["/"],
          "T": "-"
        }},
        "已开门": {{
          "A": ["启动关门电机", "设置关门时长(7s)", "启动关门定时器"],
          "T": "关门中"
        }},
        "关门中": {{
          "A": ["/"],
          "T": "-"
        }}
      }}
    }}
  }}
}}
在分析需求文档时，要先进行SEAT建模，然后再分析需求的问题。各个概念有严格的内涵和外延，从而确保：
1.概念之间关系明确，表达没有二义性；
2.需求没有遗漏，并且不存在逻辑错误；
3.需求分析描述与概要设计、详细设计存在一一映射关系，便于追溯；
4.需求分析形式化，便于机器自动检错。
分析要尽可能详细全面，要发现所有可能的问题。

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
