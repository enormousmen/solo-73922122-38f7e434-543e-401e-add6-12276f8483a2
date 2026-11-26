# 双路续写标注演示（FastAPI + NumPy 自回归小模型）

标注员同时预览两路续写。每一路是一个“订单（order）”，后台调度器按固定节拍
推进；两路都有余量时在**同一次 NumPy 批处理前向**里各走一步（合批）。每路最多
积压 **8 个未确认 token**，满额即暂停该路，标注员确认腾出额度后从**原进度**继续。
未确认片段可按序号反复重取，返回**逐字一致**的文本。

## 目录结构

```
app/
  model.py          # 纯 NumPy 字符级自回归小模型（CPU，确定性贪心解码）
  engine.py         # 订单/背压额度/交付掩码/合批调度器（asyncio 后台 ticker）
  main.py           # FastAPI 应用与 REST 接口（取片流式发送，发完才记交付）
  static/index.html # 双路标注网页（停读 / 续读 / 重取 / 确认）
tests/              # pytest（模型、引擎、HTTP API、交付连续性/并发共 29 个用例）
scripts/smoke_http.py        # 真实 uvicorn 子进程的端到端冒烟脚本
scripts/demo_back_front.py   # “先取后段、再补前段”的自校验 API 演示
scripts/requests_back_front.sh # 面向 Compose(localhost:8000) 的 curl 请求组
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
| `GET`  | `/api/orders/{id}/tokens?start=&end=` | 按序号取 `[start,end)` 文本，可原样重取；整段正文发完后登记为已交付 |
| `POST` | `/api/orders/{id}/ack` | 确认到序号 `{seq}`；前置序号未连续送达则 409，成功后腾出对应额度 |
| `GET`  | `/api/health` | 节拍数、最近合批大小等 |

### 确认（ack）语义

确认是**累计连续水位**。上次确认位置之后的每个序号，都必须由“服务端已成功发完
正文”的取片响应**连续返回**到本次 `seq`，否则：

- 中间有缺口（例如只取了后段 `[4,8)` 就想确认 `seq=7`，前段 `0..3` 未返回）
  → **HTTP 409 拒绝**，确认水位、backlog 与额度一律不变；
- 缺口补齐后（再取 `[0,4)`），先收到的后段仍记录在案，可一次确认到 `seq=7`，
  也可先确认 `3` 再确认 `7`；
- `seq` 超过已生成最大序号 → 同样 **HTTP 409**，不改变任何状态；
- `seq` 高于已确认水位且序号已连续送达 → 水位前移，
  **腾出 `seq - 旧水位` 个额度**；
- 重复确认或迟到的较小序号 → **200**，但不多腾额度（`credit_freed=0`）。

#### 什么才算“已返回 / 已交付”

- 只记录**服务端成功发完的响应正文（`text`）所含序号** `[start,end)`；
- 取片接口以流式响应在**整段正文成功发送之后**才登记交付。**空片**
  （`start==end`、`text==""`）不含序号，**发送报错、客户端断开或请求被取消**
  导致未发完的响应都**不会**登记，因而不能补缺口（重取即可，文本仍逐字一致）；
- 交付记录按单订单保存在一个 **8 bit 掩码**里（bit `i` ↔ 序号
  `acked_seq+1+i`），确认前移时移位对齐，**每单只用未确认窗口大小（8 槽）的
  常量空间**，不随 `length` 增长。原文 `tokens` 缓冲仍全量保留以支持原样重取；
- 取片在锁内组片、**发送期间不持锁**、发完后只在登记时短暂加锁，因此一路响应
  还在发送时，另一路照常取片、确认，后台节拍也照常合批续写。

订单状态里额外给出 `delivered_contig_seq`（当前已连续送达、可直接确认到的最高
序号）和 `delivered_seqs`（已送达的乱序序号，至多 8 个）。

### API 演示：先取后段，再补前段

服务启动后（`docker compose up --build`，地址 `http://localhost:8000`）：

```bash
bash scripts/requests_back_front.sh                 # 默认打 localhost:8000
# 或一组可逐条复制的 curl：
OID=DF1
curl -sS -X POST localhost:8000/api/orders -H 'Content-Type: application/json' \
  -d '{"order_id":"'$OID'","prefix":"The quick brown fox","length":40}'
sleep 2                                            # 等 8 个 token 攒满并 paused

curl -sS "localhost:8000/api/orders/$OID/tokens?start=4&end=8"   # ① 先取后段
curl -i  -X POST localhost:8000/api/orders/$OID/ack \
  -H 'Content-Type: application/json' -d '{"seq":7}'             # ② 409，前段缺失

curl -sS "localhost:8000/api/orders/$OID/tokens?start=0&end=4"   # ③ 再补前段
curl -sS -X POST localhost:8000/api/orders/$OID/ack \
  -H 'Content-Type: application/json' -d '{"seq":7}'             # ④ 200，腾出 8 额度
curl -sS "localhost:8000/api/orders/$OID/tokens?start=0&end=8"   # ⑤ 重取仍是原文
```

也可运行自校验脚本（会自行拉起真实 uvicorn，或用 `--base` 指向 Compose 服务）：

```bash
python scripts/demo_back_front.py
python scripts/demo_back_front.py --base http://127.0.0.1:8000
```

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
pytest -q                      # 29 个用例：模型、合批等价、背压、交付连续性/409、
                               # 空片/发送失败/取消不补缺口、在途响应并发、重取、API
python scripts/smoke_http.py   # 真实 uvicorn + HTTP 的两路停读/重取/续读端到端演示
python scripts/demo_back_front.py  # 真实 HTTP 的“先取后段、再补前段 + 409”演示
```

## 设计边界

- 订单状态保存在单进程内存中，单 uvicorn worker；水平扩容需要外置存储，本演示不涉及。
- 重启服务后订单清空。模型刻意做得很小、输出为“类文本”而非真实自然语言，重点在于
  演示合批调度、背压与确认/重取协议，而非生成质量。
