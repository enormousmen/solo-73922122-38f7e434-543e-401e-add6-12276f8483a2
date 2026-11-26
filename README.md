# 双路续写标注演示（FastAPI + NumPy 自回归小模型）

标注员同时预览两路续写。每一路是一个“订单（order）”，后台调度器按固定节拍
推进；两路都有余量时在**同一次 NumPy 批处理前向**里各走一步（合批）。每路最多
积压 **8 个未确认 token**，满额即暂停该路，标注员确认腾出额度后从**原进度**继续。
未确认片段可按序号反复重取，返回**逐字一致**的文本。

## 目录结构

```
app/
  model.py          # 纯 NumPy 字符级自回归小模型（CPU，确定性贪心解码）
  engine.py         # 订单/背压额度/合批调度器（asyncio 后台 ticker）
  main.py           # FastAPI 应用与 REST 接口
  static/index.html # 双路标注网页（停读 / 续读 / 重取 / 确认）
tests/              # pytest（模型、引擎、HTTP API 共 17 个用例）
scripts/smoke_http.py  # 真实 uvicorn 子进程的端到端冒烟脚本
Dockerfile, docker-compose.yml, requirements.txt
```

## 用 Docker Compose 启动

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

停止：`docker compose down`。

## 不用 Docker 的本地运行

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 网页演示路径（停读 → 重取 → 续读）

1. 两路各填前缀与续写长度，点“创建并开始读”，两路同步出字；
   顶栏“最近合批”显示 `2`，表示两路在一个批次里同时解码。
2. 在 **A** 路点“停读”。A 攒满 8 个未确认 token 后状态变为 `paused`、进度条变红；
   **B 路仍持续出字**（在网页持续读取/确认的前提下保持额度）。
3. 点 A 路“重取未确认片段”：紫色高亮的重取文本与此前逐字一致（页面会做一致性校验）。
4. 点 A 路“确认已读（腾额度）”，`backlog` 下降、状态回到 `active`；
   再点“续读”，A 从原来的序号继续，旧字符原样保留。

> 说明：网页只“读取/确认”时不会替你自动确认，所以一路停读且不确认，必然在 8 个
> token 处暂停——这正是背压语义。另一路要持续出字，需要持续确认（演示脚本里 B 路
> 即自动确认；网页可对 B 路反复点“确认已读”）。

## REST 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/orders` | 创建订单 `{order_id, prefix, length}` |
| `GET`  | `/api/orders` | 列出全部订单状态 |
| `GET`  | `/api/orders/{id}` | 单个订单状态 |
| `GET`  | `/api/orders/{id}/tokens?start=&end=` | 按序号取 `[start,end)` 文本，可原样重取 |
| `POST` | `/api/orders/{id}/ack` | 确认到序号 `{seq}`，腾出对应额度 |
| `GET`  | `/api/health` | 节拍数、最近合批大小等 |

### 确认（ack）语义

- `seq` 超过已连续返回的最大序号 → **HTTP 409 拒绝**，不改变任何状态。
- `seq` 高于已确认水位 → 水位前移，**腾出 `seq - 旧水位` 个额度**。
- 重复确认或迟到的较小序号 → **不多腾额度**（`credit_freed=0`）。

## 模型说明

- 字符级词表（字母/数字/标点/空格 + `<unk>`，共 76 个 token），上下文窗口 4。
- 哈希投影的 token/位置嵌入 → tanh 单隐层 MLP；叠加“上下文哈希噪声”打破贪心解码
  的吸引子循环，以及词长/元辅音交替/重复惩罚等结构偏置，使输出保持类文本形态。
- 全部参数由固定种子确定性推导，**无外部权重、仅依赖 NumPy、纯 CPU**。
- 采用贪心 argmax：相同前缀必然产生相同续写，这是“未确认片段可原样重取”的基础。
- 单流解码与合批解码对同一条流严格等价（上下文左填充到固定宽度），测试有断言覆盖。

## 测试与验证

```bash
pip install -r requirements.txt pytest pytest-asyncio httpx
pytest -q                      # 17 个用例：模型确定性/合批等价、背压、ack、重取、API
python scripts/smoke_http.py   # 真实 uvicorn + HTTP 的两路停读/重取/续读端到端演示
```

## 设计边界

- 订单状态保存在单进程内存中，单 uvicorn worker；水平扩容需要外置存储，本演示不涉及。
- 重启服务后订单清空。模型刻意做得很小、输出为“类文本”而非真实自然语言，重点在于
  演示合批调度、背压与确认/重取协议，而非生成质量。
