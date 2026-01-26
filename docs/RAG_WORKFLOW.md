# ZIPC RAG 实现与文档入库流程

本文件说明如何将 ZIPC 规范/样例文档导入本项目的 RAG，并解释运行时流程与评估方式。

## 1. 文档整理与切分策略（建议做法）

### 1.1 文档分类
建议将原始资料分为两类，方便模型优先遵循规范，再参考样例：
- **规范类**：ZIPC 状态机格式、字段定义、状态/事件/动作约束等。
- **样例类**：自动门等具体场景的需求与对应状态机结果。

### 1.2 文件存放
将所有文档统一放在：
```
data/rag_sources/
```
推荐命名方式：
- `spec_zipc_stm.txt`
- `example_auto_door.txt`
- `example_*_*.md`

### 1.3 切分策略（本项目内置）
构建索引时会进行两步切分：
1. **按标题切分**：识别“第X章/Markdown 标题/编号标题”等。
2. **定长切分**：每块约 1200 字符，保留 120 字符重叠。

这样保证：
- 规范类文档不会被截断到语义不完整
- 样例类文档能在需求召回时命中关键描述与示例结构

## 2. 构建 RAG 索引（上传）

### 2.1 运行构建脚本
```bash
python scripts/build_rag_index.py \
  --source-dir data/rag_sources \
  --index-dir data/rag_index \
  --max-chars 1200 \
  --overlap 120
```

### 2.2 构建结果
生成的索引位于：
```
data/rag_index/
  ├── chunks.jsonl
  ├── matrix.npz
  └── vectorizer.joblib
```

## 3. 运行时检索流程（后端）
1. `/api/generate-matrix` 接收需求文档。
2. 使用 `data/rag_index` 做检索，选取 Top-K 片段。
3. 将检索片段拼接进 prompt 的「参考资料」部分。
4. 调用模型生成 JSON，再执行 normalize / validate。

## 4. 评估流程（RAG vs 非 RAG）

### 4.1 评估数据准备
将评估样例写入：
```
data/eval_cases.jsonl
```
每行 JSON：
```json
{"requirementDoc": "...", "expected": {"states": [], "events": [], "actions": {}, "initial_state": "", "transitions": []}}
```

### 4.2 运行评估脚本
#### 不使用 RAG
```bash
python scripts/evaluate_rag.py --cases data/eval_cases.jsonl
```

#### 使用 RAG
```bash
python scripts/evaluate_rag.py --cases data/eval_cases.jsonl --use-rag
```

### 4.3 输出指标说明
脚本会输出：
- **valid_rate**：通过 `validate_model_json` 的比例
- **states_jaccard / events_jaccard**：与 expected 的集合相似度

## 5. 推荐的文档入库实际操作步骤（给你的资料目录）
1. 把你给的目录内容拆成多个文件：
   - 每个章节/主题单独一文件（例如「状态转移表的格式」「事件类型」「全局转移」等）。
2. 将每个文件放到 `data/rag_sources/`。
3. 运行 `scripts/build_rag_index.py` 构建索引。
4. 重启 Flask 服务即可自动启用 RAG。

这样做可以让模型在生成 ZIPC 状态机时优先遵循规范细节，并通过样例补齐语义。
