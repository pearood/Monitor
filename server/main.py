import hashlib
import json
import os
import secrets
import sqlite3
import string
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


DB_PATH = os.environ.get("FOCUS_SERVER_DB", os.path.join(os.path.dirname(__file__), "focus_server.db"))
DEFAULT_CLASSES = ["计算机1班", "计算机2班"]


app = FastAPI(title="Focus Desktop Cloud API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuthPayload(BaseModel):
    username: str
    password: str
    user_type: str
    name: str = ""
    email: str = ""
    class_name: Optional[str] = None


class AttentionRecordPayload(BaseModel):
    user_id: int
    attention_score: float
    status: str
    metrics: Optional[Dict[str, Any]] = None


class CreateClassPayload(BaseModel):
    teacher_id: int
    class_name: str


class JoinClassPayload(BaseModel):
    student_id: int
    class_code: str


class PresencePayload(BaseModel):
    user_id: int
    online: bool = True


class BindParentPayload(BaseModel):
    parent_id: int
    student_username: str


class AssistantLogPayload(BaseModel):
    user_id: int
    conversation_id: Optional[int] = None
    query_text: str
    reply_text: str = ""
    source: str = "text"
    model_name: str = ""


class AssistantConversationPayload(BaseModel):
    user_id: int
    title: str = "新对话"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(cursor, table_name: str, column_name: str, definition: str):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {row["name"] for row in cursor.fetchall()}
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 160000)
    return f"pbkdf2_sha256${salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    if stored_hash.startswith("pbkdf2_sha256$"):
        _, salt, expected = stored_hash.split("$", 2)
        return secrets.compare_digest(hash_password(password, salt), f"pbkdf2_sha256${salt}${expected}")

    # Compatibility with early local demo users that used plain SHA-256.
    return secrets.compare_digest(hashlib.sha256(password.encode("utf-8")).hexdigest(), stored_hash)


def init_database():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            user_type TEXT NOT NULL,
            name TEXT,
            email TEXT,
            class_name TEXT,
            class_joined INTEGER DEFAULT 0,
            last_seen_at TEXT,
            presence_state TEXT DEFAULT 'offline',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(username, user_type)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS attention_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            attention_score REAL NOT NULL,
            status TEXT NOT NULL,
            metrics_json TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL UNIQUE,
            class_code TEXT NOT NULL UNIQUE,
            teacher_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (teacher_id) REFERENCES users (id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS parent_student_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER NOT NULL UNIQUE,
            student_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_id) REFERENCES users (id),
            FOREIGN KEY (student_id) REFERENCES users (id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS assistant_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS assistant_query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            conversation_id INTEGER,
            query_text TEXT NOT NULL,
            reply_text TEXT,
            source TEXT DEFAULT 'text',
            model_name TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (conversation_id) REFERENCES assistant_conversations (id)
        )
        """
    )
    ensure_column(cursor, "users", "class_name", "TEXT")
    class_joined_added = ensure_column(cursor, "users", "class_joined", "INTEGER DEFAULT 0")
    ensure_column(cursor, "users", "last_seen_at", "TEXT")
    ensure_column(cursor, "users", "presence_state", "TEXT DEFAULT 'offline'")
    ensure_column(cursor, "assistant_query_logs", "conversation_id", "INTEGER")
    conn.commit()

    default_users = [
        ("student1", "123456", "student", "测试学生", "student1@test.com", DEFAULT_CLASSES[0], 1),
        ("teacher1", "123456", "teacher", "测试教师", "teacher1@test.com", None, 0),
        ("parent1", "123456", "parent", "测试家长", "parent1@test.com", None, 0),
    ]
    for username, password, user_type, name, email, class_name, class_joined in default_users:
        try:
            cursor.execute(
                """
                INSERT INTO users (username, password, user_type, name, email, class_name, class_joined)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (username, hash_password(password), user_type, name, email, class_name, class_joined),
            )
        except sqlite3.IntegrityError:
            pass

    if class_joined_added:
        cursor.execute(
            """
            UPDATE users
            SET class_joined = CASE
                WHEN user_type = 'student' AND username = 'student1' THEN 1
                WHEN user_type = 'student' THEN 0
                ELSE COALESCE(class_joined, 0)
            END
            """
        )
        cursor.execute(
            """
            UPDATE users
            SET class_name = NULL
            WHERE user_type = 'student'
              AND COALESCE(class_joined, 0) = 0
              AND username != 'student1'
            """
        )

    ensure_default_classes(cursor)
    sync_classes_from_students(cursor)
    ensure_default_parent_binding(cursor)
    ensure_legacy_assistant_conversations(cursor)
    conn.commit()
    conn.close()


@app.on_event("startup")
def on_startup():
    init_database()


@app.get("/health")
def health():
    return {"ok": True, "service": "focus-cloud-api"}


def user_to_dict(row):
    if row is None:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "user_type": row["user_type"],
        "name": row["name"] or row["username"],
        "email": row["email"] or "",
        "class_name": row["class_name"],
        "last_seen_at": row["last_seen_at"] if "last_seen_at" in row.keys() else None,
        "online": is_user_online(
            row["last_seen_at"] if "last_seen_at" in row.keys() else None,
            row["presence_state"] if "presence_state" in row.keys() else None,
        ),
    }


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_user_online(last_seen_at: Optional[str], presence_state: Optional[str]) -> bool:
    if presence_state != "online" or not last_seen_at:
        return False
    try:
        last_seen = datetime.strptime(last_seen_at, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return (datetime.now() - last_seen).total_seconds() <= 20


def pick_default_class() -> str:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT c.class_name, COUNT(u.id) AS total
        FROM classes c
        LEFT JOIN users u
          ON u.user_type = 'student' AND u.class_name = c.class_name
        GROUP BY c.class_name
        """
    )
    counts = {row["class_name"]: row["total"] for row in cursor.fetchall() if row["class_name"]}
    conn.close()
    return min(counts, key=lambda item: counts.get(item, 0)) if counts else DEFAULT_CLASSES[0]


def generate_class_code(cursor) -> str:
    alphabet = string.ascii_uppercase + string.digits
    while True:
        class_code = "XR" + "".join(secrets.choice(alphabet) for _ in range(6))
        cursor.execute("SELECT 1 FROM classes WHERE class_code = ?", (class_code,))
        if not cursor.fetchone():
            return class_code


def ensure_default_classes(cursor):
    for class_name in DEFAULT_CLASSES:
        cursor.execute("SELECT id FROM classes WHERE class_name = ?", (class_name,))
        if cursor.fetchone():
            continue
        cursor.execute(
            """
            INSERT INTO classes (class_name, class_code, teacher_id)
            VALUES (?, ?, NULL)
            """,
            (class_name, generate_class_code(cursor)),
        )


def ensure_default_parent_binding(cursor):
    cursor.execute("SELECT id FROM users WHERE username = 'parent1' AND user_type = 'parent'")
    parent_row = cursor.fetchone()
    cursor.execute("SELECT id FROM users WHERE username = 'student1' AND user_type = 'student'")
    student_row = cursor.fetchone()
    if not parent_row or not student_row:
        return
    cursor.execute(
        """
        INSERT OR IGNORE INTO parent_student_links (parent_id, student_id)
        VALUES (?, ?)
        """,
        (parent_row["id"], student_row["id"]),
    )


def sync_classes_from_students(cursor):
    cursor.execute(
        """
        SELECT DISTINCT class_name
        FROM users
        WHERE user_type = 'student' AND class_name IS NOT NULL AND TRIM(class_name) != ''
        """
    )
    class_names = [row["class_name"] for row in cursor.fetchall()]
    for class_name in class_names:
        cursor.execute("SELECT id FROM classes WHERE class_name = ?", (class_name,))
        if cursor.fetchone():
            continue
        cursor.execute(
            """
            INSERT INTO classes (class_name, class_code, teacher_id)
            VALUES (?, ?, NULL)
            """,
            (class_name, generate_class_code(cursor)),
        )


def get_class_info_by_name(class_name: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT c.class_name, c.class_code, c.teacher_id, COUNT(u.id) AS student_count
        FROM classes c
        LEFT JOIN users u
          ON u.user_type = 'student' AND u.class_name = c.class_name
        WHERE c.class_name = ?
        GROUP BY c.id, c.class_name, c.class_code, c.teacher_id
        """,
        (class_name,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "class_name": row["class_name"],
        "class_code": row["class_code"],
        "teacher_id": row["teacher_id"],
        "student_count": row["student_count"],
    }


def get_parent_child(parent_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT u.id, u.username, u.name, u.email, u.class_name, u.last_seen_at, u.presence_state
        FROM parent_student_links p
        INNER JOIN users u ON u.id = p.student_id
        WHERE p.parent_id = ? AND u.user_type = 'student'
        """,
        (parent_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "name": row["name"] or row["username"],
        "email": row["email"] or "",
        "class_name": row["class_name"],
        "last_seen_at": row["last_seen_at"],
        "online": is_user_online(row["last_seen_at"], row["presence_state"]),
    }


def get_assistant_query_logs(user_id: int, limit: int = 30):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT query_text, reply_text, source, model_name, timestamp
        FROM assistant_query_logs
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "query_text": row["query_text"],
            "reply_text": row["reply_text"] or "",
            "source": row["source"] or "text",
            "model_name": row["model_name"] or "",
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]


def ensure_legacy_assistant_conversations(cursor):
    cursor.execute(
        """
        SELECT DISTINCT user_id
        FROM assistant_query_logs
        WHERE conversation_id IS NULL
        """
    )
    user_ids = [row["user_id"] for row in cursor.fetchall()]
    for user_id in user_ids:
        cursor.execute(
            """
            SELECT id
            FROM assistant_conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            conversation_id = row["id"]
        else:
            cursor.execute(
                """
                INSERT INTO assistant_conversations (user_id, title)
                VALUES (?, ?)
                """,
                (user_id, "历史对话"),
            )
            conversation_id = cursor.lastrowid
        cursor.execute(
            """
            UPDATE assistant_query_logs
            SET conversation_id = ?
            WHERE user_id = ? AND conversation_id IS NULL
            """,
            (conversation_id, user_id),
        )


def create_assistant_conversation(user_id: int, title: str = "新对话"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO assistant_conversations (user_id, title)
        VALUES (?, ?)
        """,
        (user_id, (title or "新对话").strip()[:60] or "新对话"),
    )
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_assistant_conversation(user_id, conversation_id)


def get_assistant_conversation(user_id: int, conversation_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT c.id, c.title, c.created_at,
               COALESCE(MAX(l.timestamp), c.updated_at, c.created_at) AS latest_at,
               COUNT(l.id) AS message_count
        FROM assistant_conversations c
        LEFT JOIN assistant_query_logs l ON l.conversation_id = c.id
        WHERE c.user_id = ? AND c.id = ?
        GROUP BY c.id, c.title, c.created_at, c.updated_at
        """,
        (user_id, conversation_id),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "title": row["title"] or "新对话",
        "created_at": row["created_at"],
        "updated_at": row["latest_at"],
        "message_count": int(row["message_count"] or 0),
    }


def list_assistant_conversations(user_id: int, limit: int = 40):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT c.id, c.title, c.created_at,
               COALESCE(MAX(l.timestamp), c.updated_at, c.created_at) AS latest_at,
               COUNT(l.id) AS message_count,
               COALESCE((
                   SELECT l2.query_text
                   FROM assistant_query_logs l2
                   WHERE l2.conversation_id = c.id
                     AND l2.user_id = c.user_id
                     AND l2.query_text IS NOT NULL
                     AND TRIM(l2.query_text) != ''
                   ORDER BY l2.id DESC
                   LIMIT 1
               ), '') AS preview
        FROM assistant_conversations c
        LEFT JOIN assistant_query_logs l ON l.conversation_id = c.id
        WHERE c.user_id = ?
        GROUP BY c.id, c.title, c.created_at, c.updated_at
        ORDER BY latest_at DESC, c.id DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": row["id"],
            "title": row["title"] or "新对话",
            "created_at": row["created_at"],
            "updated_at": row["latest_at"],
            "message_count": int(row["message_count"] or 0),
            "preview": (row["preview"] or "").strip(),
        }
        for row in rows
    ]


def get_assistant_conversation_messages(user_id: int, conversation_id: int, limit: int = 80):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT query_text, reply_text, source, model_name, timestamp
        FROM (
            SELECT id, query_text, reply_text, source, model_name, timestamp
            FROM assistant_query_logs
            WHERE user_id = ? AND conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
        ) recent_logs
        ORDER BY id ASC
        """,
        (user_id, conversation_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "query_text": row["query_text"],
            "reply_text": row["reply_text"] or "",
            "source": row["source"] or "text",
            "model_name": row["model_name"] or "",
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]


def delete_assistant_conversation(user_id: int, conversation_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM assistant_query_logs WHERE user_id = ? AND conversation_id = ?", (user_id, conversation_id))
    cursor.execute("DELETE FROM assistant_conversations WHERE user_id = ? AND id = ?", (user_id, conversation_id))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


@app.post("/api/auth/register")
def register_user(payload: AuthPayload):
    username = payload.username.strip()
    password = payload.password.strip()
    user_type = payload.user_type.strip()
    if not username or not password:
        return {"success": False, "message": "用户名和密码不能为空"}
    if user_type not in {"student", "teacher", "parent"}:
        return {"success": False, "message": "用户类型不正确"}
    if len(password) < 6:
        return {"success": False, "message": "密码长度至少6位"}

    class_name = payload.class_name.strip() if payload.class_name else None
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO users (username, password, user_type, name, email, class_name, class_joined)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                hash_password(password),
                user_type,
                payload.name.strip(),
                payload.email.strip(),
                class_name,
                1 if class_name else 0,
            ),
        )
        conn.commit()
        return {"success": True, "message": "注册成功"}
    except sqlite3.IntegrityError:
        user_type_text = {"student": "学生", "teacher": "教师", "parent": "家长"}.get(user_type, "用户")
        return {"success": False, "message": f"该用户名已注册为{user_type_text}用户"}
    finally:
        conn.close()


@app.post("/api/auth/login")
def login_user(payload: AuthPayload):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, password, user_type, name, email, class_name
        FROM users
        WHERE username = ? AND user_type = ?
        """,
        (payload.username.strip(), payload.user_type.strip()),
    )
    row = cursor.fetchone()
    if row and verify_password(payload.password.strip(), row["password"]):
        cursor.execute(
            """
            UPDATE users
            SET last_seen_at = ?, presence_state = 'online'
            WHERE id = ?
            """,
            (now_string(), row["id"]),
        )
        conn.commit()
        cursor.execute(
            """
            SELECT id, username, password, user_type, name, email, class_name, last_seen_at, presence_state
            FROM users
            WHERE id = ?
            """,
            (row["id"],),
        )
        refreshed = cursor.fetchone()
        conn.close()
        return {"success": True, "user": user_to_dict(refreshed)}
    conn.close()
    return {"success": False, "message": "用户名、密码或用户类型错误"}


@app.post("/api/attention/records")
def save_attention_record(payload: AttentionRecordPayload):
    conn = get_connection()
    cursor = conn.cursor()
    metrics_json = json.dumps(payload.metrics or {}, ensure_ascii=False)
    cursor.execute(
        """
        INSERT INTO attention_records (user_id, attention_score, status, metrics_json)
        VALUES (?, ?, ?, ?)
        """,
        (payload.user_id, payload.attention_score, payload.status, metrics_json),
    )
    conn.commit()
    conn.close()
    return {"success": True, "message": "记录已保存"}


@app.post("/api/presence")
def update_presence(payload: PresencePayload):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE users
        SET last_seen_at = ?, presence_state = ?
        WHERE id = ?
        """,
        (now_string(), "online" if payload.online else "offline", payload.user_id),
    )
    conn.commit()
    conn.close()
    return {"success": True}


@app.post("/api/assistant/logs")
def save_assistant_log(payload: AssistantLogPayload):
    query_text = payload.query_text.strip()
    if not query_text:
        return {"success": False, "message": "提问内容不能为空"}

    conversation_id = payload.conversation_id
    if conversation_id is None:
        conversation = create_assistant_conversation(payload.user_id, "新对话")
        conversation_id = conversation["id"] if conversation else None

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO assistant_query_logs (user_id, conversation_id, query_text, reply_text, source, model_name)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            payload.user_id,
            conversation_id,
            query_text,
            payload.reply_text.strip(),
            payload.source.strip() or "text",
            payload.model_name.strip(),
        ),
    )
    cursor.execute(
        """
        UPDATE assistant_conversations
        SET title = CASE
                WHEN TRIM(COALESCE(title, '')) = '' OR title = '新对话' THEN ?
                ELSE title
            END,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND user_id = ?
        """,
        (query_text[:24], conversation_id, payload.user_id),
    )
    conn.commit()
    conn.close()
    return {"success": True}


@app.post("/api/assistant/conversations")
def create_assistant_conversation_api(payload: AssistantConversationPayload):
    conversation = create_assistant_conversation(payload.user_id, payload.title)
    return {"success": True, "conversation": conversation}


@app.get("/api/users/{user_id}/assistant-conversations")
def list_assistant_conversations_api(user_id: int, limit: int = 40):
    return {"conversations": list_assistant_conversations(user_id, limit)}


@app.get("/api/users/{user_id}/assistant-conversations/{conversation_id}")
def get_assistant_conversation_api(user_id: int, conversation_id: int, limit: int = 80):
    return {
        "conversation": get_assistant_conversation(user_id, conversation_id),
        "messages": get_assistant_conversation_messages(user_id, conversation_id, limit),
    }


@app.delete("/api/users/{user_id}/assistant-conversations/{conversation_id}")
def delete_assistant_conversation_api(user_id: int, conversation_id: int):
    return {"success": delete_assistant_conversation(user_id, conversation_id)}


def rows_to_records(rows) -> List[Dict[str, Any]]:
    records = []
    for row in rows:
        records.append(
            {
                "attention_score": float(row["attention_score"]),
                "status": row["status"],
                "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else {},
                "timestamp": row["timestamp"],
            }
        )
    return records


def get_attention_records(user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT attention_score, status, metrics_json, timestamp
        FROM attention_records
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows_to_records(rows)


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "record_count": 0,
            "focused_rate": 0.0,
            "latest_status": "暂无数据",
            "latest_score": 0.0,
            "latest_timestamp": "",
            "status_distribution": {},
        }

    scores = [item["attention_score"] for item in records]
    status_counts = defaultdict(int)
    for item in records:
        status_counts[item["status"]] += 1

    latest_record = records[0]
    focused_count = status_counts.get("专注", 0) + status_counts.get("focused", 0)
    return {
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
        "record_count": len(records),
        "focused_rate": focused_count / len(records),
        "latest_status": latest_record["status"],
        "latest_score": latest_record["attention_score"],
        "latest_timestamp": latest_record["timestamp"],
        "status_distribution": dict(status_counts),
    }


@app.get("/api/classes")
def get_class_options():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT class_name
        FROM classes
        ORDER BY class_name ASC
        """
    )
    options = [row["class_name"] for row in cursor.fetchall()]
    conn.close()
    return {"classes": options}


@app.get("/api/classes/{class_name}/info")
def api_class_info(class_name: str):
    return {"class_info": get_class_info_by_name(class_name)}


@app.post("/api/classes/create")
def api_create_class(payload: CreateClassPayload):
    class_name = payload.class_name.strip()
    if not class_name:
        return {"success": False, "message": "班级名称不能为空"}

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM classes WHERE class_name = ?", (class_name,))
    if cursor.fetchone():
        conn.close()
        return {"success": False, "message": "该班级名称已存在，请更换一个名称。"}

    class_code = generate_class_code(cursor)
    cursor.execute(
        """
        INSERT INTO classes (class_name, class_code, teacher_id)
        VALUES (?, ?, ?)
        """,
        (class_name, class_code, payload.teacher_id),
    )
    conn.commit()
    conn.close()
    return {
        "success": True,
        "message": f"班级已创建，班级码为 {class_code}",
        "class_info": {
            "class_name": class_name,
            "class_code": class_code,
            "teacher_id": payload.teacher_id,
            "student_count": 0,
        },
    }


@app.post("/api/classes/join")
def api_join_class(payload: JoinClassPayload):
    class_code = payload.class_code.strip().upper()
    if not class_code:
        return {"success": False, "message": "请输入班级码"}

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT class_name, class_code
        FROM classes
        WHERE UPPER(class_code) = ?
        """,
        (class_code,),
    )
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {"success": False, "message": "未找到对应班级，请检查班级码是否正确。"}

    cursor.execute(
        """
        UPDATE users
        SET class_name = ?, class_joined = 1
        WHERE id = ? AND user_type = 'student'
        """,
        (row["class_name"], payload.student_id),
    )
    conn.commit()
    conn.close()
    return {
        "success": True,
        "message": f"已加入 {row['class_name']}",
        "class_info": {
            "class_name": row["class_name"],
            "class_code": row["class_code"],
        },
    }


@app.get("/api/users/{user_id}")
def get_user_by_id(user_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, password, user_type, name, email, class_name, last_seen_at, presence_state
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return {"user": user_to_dict(row)}


@app.post("/api/parents/bind-child")
def api_bind_parent_child(payload: BindParentPayload):
    username = payload.student_username.strip()
    if not username:
        return {"success": False, "message": "请输入孩子的学生账号"}

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, name, email, class_name, last_seen_at, presence_state
        FROM users
        WHERE username = ? AND user_type = 'student'
        """,
        (username,),
    )
    student = cursor.fetchone()
    if not student:
        conn.close()
        return {"success": False, "message": "未找到对应学生账号，请确认孩子的用户名是否正确。"}

    cursor.execute(
        """
        INSERT INTO parent_student_links (parent_id, student_id)
        VALUES (?, ?)
        ON CONFLICT(parent_id) DO UPDATE SET
            student_id = excluded.student_id,
            created_at = CURRENT_TIMESTAMP
        """,
        (payload.parent_id, student["id"]),
    )
    conn.commit()
    conn.close()

    child = {
        "id": student["id"],
        "username": student["username"],
        "name": student["name"] or student["username"],
        "email": student["email"] or "",
        "class_name": student["class_name"],
        "last_seen_at": student["last_seen_at"],
        "online": is_user_online(student["last_seen_at"], student["presence_state"]),
    }
    return {"success": True, "message": f"已绑定孩子账号：{child['name']}", "child": child}


@app.get("/api/parents/{parent_id}/child")
def api_parent_child(parent_id: int):
    return {"child": get_parent_child(parent_id)}


@app.get("/api/parents/{parent_id}/dashboard")
def api_parent_dashboard(parent_id: int, limit_days: int = 14, limit_logs: int = 30):
    child = get_parent_child(parent_id)
    if not child:
        return {
            "child": None,
            "summary": summarize_records([]),
            "daily_rows": [],
            "assistant_logs": [],
            "today_query_count": 0,
        }

    summary = summarize_records(get_attention_records(child["id"], 300))
    grouped_records = defaultdict(list)
    for item in get_attention_records(child["id"], 600):
        grouped_records[str(item["timestamp"])[:10]].append(item)
    daily_rows = []
    for date_key, items in sorted(grouped_records.items(), reverse=True)[:limit_days]:
        day_summary = summarize_records(items)
        daily_rows.append(
            {
                "date": date_key,
                "avg_score": day_summary["avg_score"],
                "record_count": day_summary["record_count"],
                "focused_rate": day_summary["focused_rate"],
            }
        )

    assistant_logs = get_assistant_query_logs(child["id"], limit_logs)
    today = datetime.now().strftime("%Y-%m-%d")
    today_query_count = sum(1 for item in assistant_logs if str(item.get("timestamp", "")).startswith(today))
    return {
        "child": child,
        "summary": summary,
        "daily_rows": daily_rows,
        "assistant_logs": assistant_logs,
        "today_query_count": today_query_count,
    }


@app.get("/api/students")
def get_all_students(class_name: Optional[str] = None):
    conn = get_connection()
    cursor = conn.cursor()
    if class_name:
        cursor.execute(
            """
            SELECT id, username, name, email, class_name, last_seen_at, presence_state
            FROM users
            WHERE user_type = 'student' AND class_name = ? AND COALESCE(class_joined, 0) = 1
            ORDER BY id ASC
            """,
            (class_name,),
        )
    else:
        cursor.execute(
            """
            SELECT id, username, name, email, class_name, last_seen_at, presence_state
            FROM users
            WHERE user_type = 'student' AND COALESCE(class_joined, 0) = 1
            ORDER BY id ASC
            """
        )
    rows = cursor.fetchall()
    conn.close()
    return {
        "students": [
            [
                row["id"],
                row["username"],
                row["name"] or row["username"],
                row["email"] or "",
                row["class_name"],
                row["last_seen_at"],
                is_user_online(row["last_seen_at"], row["presence_state"]),
            ]
            for row in rows
        ]
    }


@app.get("/api/students/{user_id}/records")
def api_attention_records(user_id: int, limit: int = 100):
    return {"records": get_attention_records(user_id, limit)}


@app.get("/api/students/{user_id}/trend")
def api_recent_student_trend(user_id: int, limit: int = 60):
    records = get_attention_records(user_id, limit)
    records.reverse()
    return {"trend": records}


@app.get("/api/students/{user_id}/summary")
def api_student_summary(user_id: int, limit: int = 200):
    return {"summary": summarize_records(get_attention_records(user_id, limit))}


@app.get("/api/students/{user_id}/assistant-logs")
def api_student_assistant_logs(user_id: int, limit: int = 30):
    return {"logs": get_assistant_query_logs(user_id, limit)}


def class_student_summaries(class_name: str) -> List[Dict[str, Any]]:
    students = get_all_students(class_name)["students"]
    summaries = []
    for student_id, username, name, email, student_class, last_seen_at, online in students:
        summary = summarize_records(get_attention_records(student_id, 200))
        summary.update(
            {
                "id": student_id,
                "username": username,
                "name": name or username,
                "email": email,
                "class_name": student_class,
                "last_seen_at": last_seen_at,
                "online": online,
            }
        )
        summaries.append(summary)
    summaries.sort(key=lambda item: (-item["avg_score"], item["name"]))
    return summaries


@app.get("/api/classes/{class_name}/students")
def api_class_student_summaries(class_name: str):
    return {"students": class_student_summaries(class_name)}


@app.get("/api/classes/{class_name}/summary")
def api_class_summary(class_name: str):
    summaries = class_student_summaries(class_name)
    active_summaries = [item for item in summaries if item["record_count"] > 0]
    if not active_summaries:
        return {
            "summary": {
                "class_name": class_name,
                "student_count": len(get_all_students(class_name)["students"]),
                "online_count": 0,
                "record_count": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "focused_rate": 0.0,
            }
        }

    return {
        "summary": {
            "class_name": class_name,
            "student_count": len(get_all_students(class_name)["students"]),
            "online_count": sum(1 for item in summaries if item.get("online")),
            "record_count": sum(item["record_count"] for item in active_summaries),
            "avg_score": sum(item["avg_score"] for item in active_summaries) / len(active_summaries),
            "max_score": max(item["max_score"] for item in active_summaries),
            "min_score": min(item["min_score"] for item in active_summaries),
            "focused_rate": sum(item["focused_rate"] for item in active_summaries) / len(active_summaries),
        }
    }


@app.get("/api/classes/{class_name}/trend")
def api_class_trend(class_name: str, limit: int = 120):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT ar.attention_score, ar.status, ar.timestamp, u.name, u.username
        FROM attention_records ar
        INNER JOIN users u ON u.id = ar.user_id
        WHERE u.user_type = 'student' AND u.class_name = ?
        ORDER BY ar.timestamp DESC
        LIMIT ?
        """,
        (class_name, limit),
    )
    rows = cursor.fetchall()
    conn.close()

    trend = [
        {
            "attention_score": float(row["attention_score"]),
            "status": row["status"],
            "timestamp": row["timestamp"],
            "student_name": row["name"] or row["username"],
        }
        for row in rows
    ]
    trend.reverse()
    return {"trend": trend}
