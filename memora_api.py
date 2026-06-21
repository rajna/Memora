#!/usr/bin/env python3
"""
Memora HTTP API — 给 pi agent (TypeScript) 用的 REST 接口
启动: python3 memora_api.py --port 5002

配置:
  MEMORA_WORKSPACE   Memora 工作区路径 (默认: ~/.nanobot/workspace)
"""
import json
import os
import re
import sys
from datetime import datetime, timedelta, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qs

MEMORA_PATH = Path(os.environ.get("MEMORA_WORKSPACE",
    os.path.expanduser("~/.nanobot/workspace")))
sys.path.insert(0, str(MEMORA_PATH))

from Memora.src.memory_system import Memora


# ── TimeRangeParser ────────────────────────────────────
# 从自然语言查询中提取精确时间范围

_TIME_PATTERNS = {
    'today_dawn':     re.compile(r'今天凌晨|今早凌晨|今日凌晨'),
    'today_morning':  re.compile(r'今天上午|今早|今日上午'),
    'today_afternoon': re.compile(r'今天下午|今日下午'),
    'today_evening':  re.compile(r'今天晚上|今晚|今日晚上'),
    'today':          re.compile(r'今天|今日(?!凌晨|上午|下午|晚上|早)'),
    'yesterday':      re.compile(r'昨天|昨日'),
    'day_before_yesterday': re.compile(r'前天|前日'),
    'this_week':      re.compile(r'本周|这周|这个星期'),
    'last_week':      re.compile(r'上周|上个星期'),
    'this_month':     re.compile(r'本月|这个月'),
    'last_month':     re.compile(r'上月|上个月'),
}


def parse_time_range(query: str, ref: Optional[datetime] = None) -> Optional[Tuple[datetime, datetime]]:
    """解析查询中的时间范围，返回 (start, end) 或 None"""
    ref = ref or datetime.now()

    if _TIME_PATTERNS['today_dawn'].search(query):
        return (datetime.combine(ref.date(), time(0, 0)),
                datetime.combine(ref.date(), time(6, 0)))
    if _TIME_PATTERNS['today_morning'].search(query):
        return (datetime.combine(ref.date(), time(6, 0)),
                datetime.combine(ref.date(), time(12, 0)))
    if _TIME_PATTERNS['today_afternoon'].search(query):
        return (datetime.combine(ref.date(), time(12, 0)),
                datetime.combine(ref.date(), time(18, 0)))
    if _TIME_PATTERNS['today_evening'].search(query):
        return (datetime.combine(ref.date(), time(18, 0)),
                datetime.combine(ref.date(), time(23, 59, 59)))
    if _TIME_PATTERNS['today'].search(query):
        return (datetime.combine(ref.date(), time(0, 0)),
                datetime.combine(ref.date(), time(23, 59, 59)))
    if _TIME_PATTERNS['yesterday'].search(query):
        d = ref.date() - timedelta(days=1)
        return (datetime.combine(d, time(0, 0)), datetime.combine(d, time(23, 59, 59)))
    if _TIME_PATTERNS['day_before_yesterday'].search(query):
        d = ref.date() - timedelta(days=2)
        return (datetime.combine(d, time(0, 0)), datetime.combine(d, time(23, 59, 59)))
    if _TIME_PATTERNS['this_week'].search(query):
        monday = ref.date() - timedelta(days=ref.weekday())
        return (datetime.combine(monday, time(0, 0)),
                datetime.combine(ref.date(), time(23, 59, 59)))
    if _TIME_PATTERNS['last_week'].search(query):
        last_monday = ref.date() - timedelta(days=ref.weekday() + 7)
        last_sunday = last_monday + timedelta(days=6)
        return (datetime.combine(last_monday, time(0, 0)),
                datetime.combine(last_sunday, time(23, 59, 59)))
    if _TIME_PATTERNS['this_month'].search(query):
        return (datetime.combine(ref.date().replace(day=1), time(0, 0)),
                datetime.combine(ref.date(), time(23, 59, 59)))
    if _TIME_PATTERNS['last_month'].search(query):
        if ref.month == 1:
            lm, yr = 12, ref.year - 1
        else:
            lm, yr = ref.month - 1, ref.year
        nm = datetime(yr + 1, 1, 1) if lm == 12 else datetime(yr, lm + 1, 1)
        last_day = (nm - timedelta(days=1)).day
        return (datetime(yr, lm, 1, 0, 0),
                datetime(yr, lm, last_day, 23, 59, 59))
    return None


class MemoraAPI(BaseHTTPRequestHandler):
    _ms = None

    @classmethod
    def get_ms(cls):
        if cls._ms is None:
            cls._ms = Memora()
        return cls._ms

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            self._json({"status": "ok", "memories": len(self.get_ms().list_all(limit=10000))})

        elif path == "/search":
            qs = parse_qs(urlparse(self.path).query)
            query = qs.get("q", [""])[0]
            top_k = int(qs.get("k", ["5"])[0])
            if not query:
                self._json({"error": "missing q"}, 400)
                return
            results = self.get_ms().search(query, top_k=top_k)
            self._json([
                {
                    "id": r.node.id,
                    "title": r.node.title,
                    "content": r.node.content[:300],
                    "tags": r.node.tags,
                    "pagerank": r.node.pagerank,
                    "score": r.final_score,
                    "created": r.node.created.isoformat(),
                }
                for r in results
            ])

        elif path == "/search-hybrid":
            qs = parse_qs(urlparse(self.path).query)
            query = qs.get("q", [""])[0]
            top_k = int(qs.get("k", ["5"])[0])
            if not query:
                self._json({"error": "missing q"}, 400)
                return

            # 标签过滤
            filter_tags = None
            if "tags" in qs:
                filter_tags = [t.strip() for t in qs["tags"][0].split(",") if t.strip()]

            # 时间范围：优先精确时间解析，其次 days 参数
            time_range_days = int(qs["days"][0]) if "days" in qs else None
            time_range = parse_time_range(query)
            if time_range:
                # 自然语言时间覆盖 days 参数
                time_range_days = None

            ms = self.get_ms()
            results = ms.retrieval.search_with_graph_expansion(
                query=query, top_k=top_k,
                filter_tags=filter_tags,
                time_range_days=time_range_days,
                time_range=time_range,
            )

            self._json([
                {
                    "id": r.node.id,
                    "title": r.node.title,
                    "content": r.node.content[:600],
                    "tags": r.node.tags,
                    "pagerank": round(r.node.pagerank, 6),
                    "score": round(r.final_score, 4),
                    "created": r.node.created.isoformat(),
                }
                for r in results
            ])

        elif path == "/memories":
            limit = int(parse_qs(urlparse(self.path).query).get("limit", ["20"])[0])
            nodes = self.get_ms().list_all(limit=limit)
            self._json([
                {
                    "id": n.id,
                    "title": n.title,
                    "content": n.content[:200],
                    "tags": n.tags,
                    "pagerank": n.pagerank,
                    "created": n.created.isoformat(),
                }
                for n in nodes
            ])

        elif path == "/important":
            n = int(parse_qs(urlparse(self.path).query).get("n", ["10"])[0])
            nodes = list(self.get_ms().list_all(limit=10000))
            nodes.sort(key=lambda x: x.pagerank, reverse=True)
            self._json([
                {"id": n.id, "title": n.title, "pagerank": n.pagerank}
                for n in nodes[:n]
            ])

        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path == "/save":
            node = self.get_ms().add_memory(
                content=body.get("content", ""),
                title=body.get("title"),
                tags=body.get("tags", []),
            )
            self._json({"id": node.id, "title": node.title, "status": "saved"})

        elif path == "/save-conversation":
            messages = body.get("messages", [])
            if not messages:
                self._json({"error": "missing messages"}, 400)
                return
            node = self.get_ms().add_memory_from_messages(
                messages=messages,
                source=body.get("source", "pi-agent"),
                base_tags=body.get("tags", ["pi-conversation"]),
            )
            if node:
                self._json({"id": node.id, "title": node.title, "status": "saved"})
            else:
                self._json({"status": "skipped", "reason": "content too short or not worth saving"})

        elif path == "/build-graph":
            self.get_ms().build_graph()
            self._json({"status": "graph rebuilt"})

        else:
            self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    port = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else 5002
    server = HTTPServer(("127.0.0.1", port), MemoraAPI)
    print(f"🧠 Memora API → http://localhost:{port}")
    print(f"   GET  /health")
    print(f"   GET  /search?q=...&k=5")
    print(f"   GET  /search-hybrid?q=...&k=5&tags=...&days=7")
    print(f"   GET  /memories?limit=20")
    print(f"   GET  /important?n=10")
    print(f"   POST /save          {{\"content\":\"...\",\"title\":\"...\",\"tags\":[...]}}")
    print(f"   POST /save-conversation  {{\"messages\":[...],\"tags\":[...]}}")
    print(f"   POST /build-graph")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋")
