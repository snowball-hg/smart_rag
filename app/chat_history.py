"""对话历史持久化模块。

使用 SQLite 存储会话和消息记录，支持：
- 会话的创建、列表、删除、重命名
- 消息的追加和按会话查询
- 自动从首条消息生成会话标题
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.logger import logger

# 数据库文件路径（项目根目录下的 data/chat_history.db）
_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_PATH = _DB_DIR / "chat_history.db"

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """获取当前线程的数据库连接。"""
    if not hasattr(_local, "conn") or _local.conn is None:
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


def init_db():
    """初始化数据库表结构。"""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '新对话',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL DEFAULT '',
            sources TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);
    """)
    conn.commit()


# ==================== 会话管理 ====================


def create_session(session_id: Optional[str] = None, title: str = "新对话") -> dict:
    """创建新会话。

    Args:
        session_id: 可选，指定会话 ID（用于恢复已有会话）。
        title: 会话标题。

    Returns:
        会话字典。
    """
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    sid = session_id or f"session-{uuid.uuid4().hex[:12]}"

    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (sid, title, now, now),
    )
    conn.commit()

    return {"id": sid, "title": title, "created_at": now, "updated_at": now}


def list_sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    """获取会话列表，按更新时间倒序。

    Args:
        limit: 返回条数上限。
        offset: 偏移量。

    Returns:
        会话字典列表。
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[dict]:
    """获取单个会话信息。

    Args:
        session_id: 会话 ID。

    Returns:
        会话字典，不存在则返回 None。
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def rename_session(session_id: str, title: str) -> bool:
    """重命名会话。

    Args:
        session_id: 会话 ID。
        title: 新标题。

    Returns:
        是否成功。
    """
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
        (title, now, session_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def delete_session(session_id: str) -> bool:
    """删除会话及其所有消息。

    Args:
        session_id: 会话 ID。

    Returns:
        是否成功。
    """
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    return cursor.rowcount > 0


def _auto_title(content: str, max_len: int = 30) -> str:
    """从消息内容自动生成会话标题。

    Args:
        content: 消息文本。
        max_len: 标题最大长度。

    Returns:
        截断后的标题。
    """
    # 移除多余空白，取第一行
    title = content.strip().replace("\n", " ").replace("\r", "")
    if len(title) > max_len:
        title = title[:max_len] + "…"
    return title if title else "新对话"


# ==================== 消息管理 ====================


def add_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[list[dict]] = None,
) -> dict:
    """添加一条消息到会话。

    Args:
        session_id: 会话 ID。
        role: 角色（user/assistant）。
        content: 消息内容。
        sources: 来源信息列表（仅 assistant 消息有）。

    Returns:
        消息字典。
    """
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()

    # 插入消息
    cursor = conn.execute(
        "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, json.dumps(sources or [], ensure_ascii=False), now),
    )
    msg_id = cursor.lastrowid

    # 更新会话的 updated_at
    conn.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?",
        (now, session_id),
    )

    # 如果是首条用户消息，自动生成标题
    if role == "user":
        msg_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        if msg_count == 1:
            title = _auto_title(content)
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )

    conn.commit()

    return {
        "id": msg_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "created_at": now,
    }


def get_messages(session_id: str, limit: int = 200) -> list[dict]:
    """获取会话的所有消息，按创建时间正序。

    Args:
        session_id: 会话 ID。
        limit: 返回条数上限。

    Returns:
        消息字典列表。
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, session_id, role, content, sources, created_at "
        "FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
        (session_id, limit),
    ).fetchall()

    result = []
    for r in rows:
        msg = dict(r)
        try:
            msg["sources"] = json.loads(msg["sources"])
        except (json.JSONDecodeError, TypeError):
            msg["sources"] = []
        result.append(msg)

    return result


def get_session_messages_count(session_id: str) -> int:
    """获取会话的消息总数。

    Args:
        session_id: 会话 ID。

    Returns:
        消息数量。
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
    ).fetchone()
    return row[0] if row else 0


# ==================== 初始化 ====================

init_db()
