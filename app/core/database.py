import hashlib
import json
import secrets
import sqlite3
import string
from collections import defaultdict
from datetime import datetime

from config.config import DB_PATH, DEFAULT_CLASSES


class Database:
    def __init__(self):
        self.init_database()

    def get_connection(self):
        return sqlite3.connect(DB_PATH)

    def init_database(self):
        conn = self.get_connection()
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

        self._ensure_column(cursor, "users", "class_name", "TEXT")
        class_joined_added = self._ensure_column(cursor, "users", "class_joined", "INTEGER DEFAULT 0")
        self._ensure_column(cursor, "users", "last_seen_at", "TEXT")
        self._ensure_column(cursor, "users", "presence_state", "TEXT DEFAULT 'offline'")
        self._ensure_column(cursor, "attention_records", "metrics_json", "TEXT")
        self._ensure_column(cursor, "classes", "teacher_id", "INTEGER")
        self._ensure_column(cursor, "assistant_query_logs", "conversation_id", "INTEGER")

        self._create_default_users(cursor)
        self._ensure_default_classes(cursor)
        if class_joined_added:
            self._initialize_student_class_membership(cursor)
        self._sync_classes_from_students(cursor)
        self._ensure_default_parent_binding(cursor)
        self._ensure_legacy_assistant_conversations(cursor)

        conn.commit()
        conn.close()

    def _ensure_column(self, cursor, table_name, column_name, definition):
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1] for row in cursor.fetchall()}
        if column_name not in columns:
            cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}"
            )
            return True
        return False

    def _create_default_users(self, cursor):
        default_users = [
            (
                "student1",
                self._hash_password("123456"),
                "student",
                "测试学生",
                "student1@test.com",
                DEFAULT_CLASSES[0],
                1,
            ),
            (
                "teacher1",
                self._hash_password("123456"),
                "teacher",
                "测试教师",
                "teacher1@test.com",
                None,
                0,
            ),
            (
                "parent1",
                self._hash_password("123456"),
                "parent",
                "测试家长",
                "parent1@test.com",
                None,
                0,
            ),
        ]

        for user in default_users:
            try:
                cursor.execute(
                    """
                    INSERT INTO users (username, password, user_type, name, email, class_name, class_joined)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    user,
                )
            except sqlite3.IntegrityError:
                pass

    def _initialize_student_class_membership(self, cursor):
        # Older versions auto-assigned students into default classes at registration time.
        # When migrating to explicit "join class by code", clear those non-demo assignments once.
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

    def _ensure_default_classes(self, cursor):
        for class_name in DEFAULT_CLASSES:
            cursor.execute(
                "SELECT id FROM classes WHERE class_name = ?",
                (class_name,),
            )
            if cursor.fetchone():
                continue
            cursor.execute(
                """
                INSERT INTO classes (class_name, class_code, teacher_id)
                VALUES (?, ?, NULL)
                """,
                (class_name, self._generate_class_code(cursor)),
            )

    def _ensure_default_parent_binding(self, cursor):
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
            (parent_row[0], student_row[0]),
        )

    def _ensure_legacy_assistant_conversations(self, cursor):
        cursor.execute(
            """
            SELECT DISTINCT user_id
            FROM assistant_query_logs
            WHERE conversation_id IS NULL
            """
        )
        user_ids = [row[0] for row in cursor.fetchall()]
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
                conversation_id = row[0]
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

    def _sync_classes_from_students(self, cursor):
        cursor.execute(
            """
            SELECT DISTINCT class_name
            FROM users
            WHERE user_type = 'student' AND class_name IS NOT NULL AND TRIM(class_name) != ''
            """
        )
        class_names = [row[0] for row in cursor.fetchall()]
        for class_name in class_names:
            cursor.execute("SELECT id FROM classes WHERE class_name = ?", (class_name,))
            if cursor.fetchone():
                continue
            cursor.execute(
                """
                INSERT INTO classes (class_name, class_code, teacher_id)
                VALUES (?, ?, NULL)
                """,
                (class_name, self._generate_class_code(cursor)),
            )

    def _hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def _now_string(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _is_user_online(self, last_seen_at, presence_state):
        if presence_state != "online" or not last_seen_at:
            return False
        try:
            last_seen = datetime.strptime(last_seen_at, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return False
        return (datetime.now() - last_seen).total_seconds() <= 20

    def _generate_class_code(self, cursor):
        alphabet = string.ascii_uppercase + string.digits
        while True:
            class_code = "XR" + "".join(secrets.choice(alphabet) for _ in range(6))
            cursor.execute("SELECT 1 FROM classes WHERE class_code = ?", (class_code,))
            if not cursor.fetchone():
                return class_code

    def _pick_default_class(self):
        conn = self.get_connection()
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
        counts = {row[0]: row[1] for row in cursor.fetchall() if row[0]}
        conn.close()

        if not counts:
            return DEFAULT_CLASSES[0]

        return min(counts, key=lambda item: counts.get(item, 0))

    def register_user(self, username, password, user_type, name="", email="", class_name=None):
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            hashed_password = self._hash_password(password)
            cursor.execute(
                """
                INSERT INTO users (username, password, user_type, name, email, class_name, class_joined)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (username, hashed_password, user_type, name, email, class_name, 1 if class_name else 0),
            )
            conn.commit()
            return True, "注册成功"
        except sqlite3.IntegrityError as exc:
            cursor.execute(
                """
                SELECT username, user_type FROM users
                WHERE username = ? AND user_type = ?
                """,
                (username, user_type),
            )
            existing_user = cursor.fetchone()
            if existing_user:
                user_type_text = {"student": "学生", "teacher": "教师", "parent": "家长"}.get(user_type, "用户")
                return False, f"该用户名已注册为{user_type_text}用户"
            return False, f"注册失败: {exc}"
        finally:
            conn.close()

    def login_user(self, username, password, user_type):
        conn = self.get_connection()
        cursor = conn.cursor()

        hashed_password = self._hash_password(password)
        cursor.execute(
            """
            SELECT id, username, user_type, name, class_name
            FROM users
            WHERE username = ? AND password = ? AND user_type = ?
            """,
            (username, hashed_password, user_type),
        )

        user = cursor.fetchone()

        if user:
            cursor.execute(
                """
                UPDATE users
                SET last_seen_at = ?, presence_state = 'online'
                WHERE id = ?
                """,
                (self._now_string(), user[0]),
            )
            conn.commit()
            result = {
                "id": user[0],
                "username": user[1],
                "user_type": user[2],
                "name": user[3],
                "class_name": user[4],
            }
            conn.close()
            return True, result
        conn.close()
        return False, "用户名、密码或用户类型错误"

    def update_presence(self, user_id, online=True):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE users
            SET last_seen_at = ?, presence_state = ?
            WHERE id = ?
            """,
            (self._now_string(), "online" if online else "offline", user_id),
        )
        conn.commit()
        conn.close()
        return True

    def save_attention_record(self, user_id, attention_score, status, metrics=None):
        conn = self.get_connection()
        cursor = conn.cursor()

        metrics_json = json.dumps(metrics, ensure_ascii=False) if metrics else None
        cursor.execute(
            """
            INSERT INTO attention_records (user_id, attention_score, status, metrics_json)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, attention_score, status, metrics_json),
        )

        conn.commit()
        conn.close()

    def get_attention_records(self, user_id, limit=100):
        conn = self.get_connection()
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

        records = []
        for score, status, metrics_json, timestamp in rows:
            records.append(
                {
                    "attention_score": float(score),
                    "status": status,
                    "metrics": json.loads(metrics_json) if metrics_json else {},
                    "timestamp": timestamp,
                }
            )
        return records

    def get_recent_student_trend(self, user_id, limit=60):
        records = self.get_attention_records(user_id, limit=limit)
        records.reverse()
        return records

    def get_student_summary(self, user_id, limit=200):
        records = self.get_attention_records(user_id, limit=limit)
        return self._summarize_records(records)

    def get_all_students(self, class_name=None):
        conn = self.get_connection()
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

        students = cursor.fetchall()
        conn.close()
        return students

    def get_class_options(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT class_name
            FROM classes
            ORDER BY class_name ASC
            """
        )
        options = [row[0] for row in cursor.fetchall()]
        conn.close()

        return options

    def get_class_info(self, class_name):
        if not class_name:
            return None

        conn = self.get_connection()
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
            "class_name": row[0],
            "class_code": row[1],
            "teacher_id": row[2],
            "student_count": row[3],
        }

    def create_class(self, teacher_id, class_name):
        class_name = (class_name or "").strip()
        if not class_name:
            return False, "班级名称不能为空", None

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT class_code FROM classes WHERE class_name = ?", (class_name,))
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return False, "该班级名称已存在，请更换一个名称。", None

        class_code = self._generate_class_code(cursor)
        cursor.execute(
            """
            INSERT INTO classes (class_name, class_code, teacher_id)
            VALUES (?, ?, ?)
            """,
            (class_name, class_code, teacher_id),
        )
        conn.commit()
        conn.close()
        return True, f"班级已创建，班级码为 {class_code}", {
            "class_name": class_name,
            "class_code": class_code,
            "teacher_id": teacher_id,
            "student_count": 0,
        }

    def join_class(self, student_id, class_code):
        class_code = (class_code or "").strip().upper()
        if not class_code:
            return False, "请输入班级码", None

        conn = self.get_connection()
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
            return False, "未找到对应班级，请检查班级码是否正确。", None

        class_name, normalized_code = row
        cursor.execute(
            """
            UPDATE users
            SET class_name = ?, class_joined = 1
            WHERE id = ? AND user_type = 'student'
            """,
            (class_name, student_id),
        )
        conn.commit()
        conn.close()
        return True, f"已加入 {class_name}", {
            "class_name": class_name,
            "class_code": normalized_code,
        }

    def get_user_by_id(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, username, user_type, name, email, class_name, last_seen_at, presence_state
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        )

        user = cursor.fetchone()
        conn.close()

        if not user:
            return None

        return {
            "id": user[0],
            "username": user[1],
            "user_type": user[2],
            "name": user[3],
            "email": user[4],
            "class_name": user[5],
            "last_seen_at": user[6],
            "online": self._is_user_online(user[6], user[7]),
        }

    def bind_parent_to_student(self, parent_id, student_username):
        username = (student_username or "").strip()
        if not username:
            return False, "请输入孩子的学生账号", None

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, username, name, class_name, last_seen_at, presence_state
            FROM users
            WHERE username = ? AND user_type = 'student'
            """,
            (username,),
        )
        student = cursor.fetchone()
        if not student:
            conn.close()
            return False, "未找到对应学生账号，请确认孩子的用户名是否正确。", None

        cursor.execute(
            """
            INSERT INTO parent_student_links (parent_id, student_id)
            VALUES (?, ?)
            ON CONFLICT(parent_id) DO UPDATE SET
                student_id = excluded.student_id,
                created_at = CURRENT_TIMESTAMP
            """,
            (parent_id, student[0]),
        )
        conn.commit()
        conn.close()

        child = {
            "id": student[0],
            "username": student[1],
            "name": student[2] or student[1],
            "class_name": student[3],
            "last_seen_at": student[4],
            "online": self._is_user_online(student[4], student[5]),
        }
        return True, f"已绑定孩子账号：{child['name']}", child

    def get_parent_child(self, parent_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT u.id, u.username, u.name, u.class_name, u.last_seen_at, u.presence_state
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
            "id": row[0],
            "username": row[1],
            "name": row[2] or row[1],
            "class_name": row[3],
            "last_seen_at": row[4],
            "online": self._is_user_online(row[4], row[5]),
        }

    def create_assistant_conversation(self, user_id, title="新对话"):
        title = (title or "").strip() or "新对话"
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO assistant_conversations (user_id, title)
            VALUES (?, ?)
            """,
            (user_id, title[:60]),
        )
        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return self.get_assistant_conversation(user_id, conversation_id)

    def get_assistant_conversation(self, user_id, conversation_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(l.id) AS message_count,
                   COALESCE(MAX(l.timestamp), c.updated_at) AS latest_at
            FROM assistant_conversations c
            LEFT JOIN assistant_query_logs l
              ON l.conversation_id = c.id
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
            "id": row[0],
            "title": row[1] or "新对话",
            "created_at": row[2],
            "updated_at": row[5] or row[3],
            "message_count": int(row[4] or 0),
        }

    def get_assistant_conversations(self, user_id, limit=40):
        conn = self.get_connection()
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
                   ), '') AS preview_query
            FROM assistant_conversations c
            LEFT JOIN assistant_query_logs l
              ON l.conversation_id = c.id
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
                "id": row[0],
                "title": row[1] or "新对话",
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": int(row[4] or 0),
                "preview": (row[5] or "").strip(),
            }
            for row in rows
        ]

    def get_assistant_conversation_messages(self, user_id, conversation_id, limit=80):
        conn = self.get_connection()
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
                "query_text": row[0],
                "reply_text": row[1] or "",
                "source": row[2] or "text",
                "model_name": row[3] or "",
                "timestamp": row[4],
            }
            for row in rows
        ]

    def delete_assistant_conversation(self, user_id, conversation_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM assistant_query_logs WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        )
        cursor.execute(
            "DELETE FROM assistant_conversations WHERE user_id = ? AND id = ?",
            (user_id, conversation_id),
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def save_assistant_query_log(self, user_id, query_text, reply_text="", source="text", model_name="", conversation_id=None):
        query_text = (query_text or "").strip()
        if not query_text:
            return False
        if conversation_id is None:
            conversation = self.create_assistant_conversation(user_id, "新对话")
            conversation_id = conversation["id"] if conversation else None

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO assistant_query_logs (user_id, conversation_id, query_text, reply_text, source, model_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                conversation_id,
                query_text,
                (reply_text or "").strip(),
                (source or "text").strip(),
                (model_name or "").strip(),
            ),
        )
        auto_title = query_text[:24]
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
            (auto_title, conversation_id, user_id),
        )
        conn.commit()
        conn.close()
        return True

    def get_assistant_query_logs(self, user_id, limit=30):
        conn = self.get_connection()
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
                "query_text": row[0],
                "reply_text": row[1] or "",
                "source": row[2] or "text",
                "model_name": row[3] or "",
                "timestamp": row[4],
            }
            for row in rows
        ]

    def get_parent_dashboard(self, parent_id, limit_days=14, limit_logs=30):
        child = self.get_parent_child(parent_id)
        if not child:
            return {
                "child": None,
                "summary": self._summarize_records([]),
                "daily_rows": [],
                "assistant_logs": [],
                "today_query_count": 0,
            }

        summary = self.get_student_summary(child["id"], limit=300)
        daily_rows = self.get_student_records_grouped_by_day(child["id"], limit=600)[:limit_days]
        assistant_logs = self.get_assistant_query_logs(child["id"], limit=limit_logs)
        today = datetime.now().strftime("%Y-%m-%d")
        today_query_count = sum(1 for item in assistant_logs if str(item.get("timestamp", "")).startswith(today))

        return {
            "child": child,
            "summary": summary,
            "daily_rows": daily_rows,
            "assistant_logs": assistant_logs,
            "today_query_count": today_query_count,
        }

    def get_class_summary(self, class_name):
        summaries = self.get_class_student_summaries(class_name)
        active_summaries = [item for item in summaries if item["record_count"] > 0]

        if not active_summaries:
            return {
                "class_name": class_name,
                "student_count": len(self.get_all_students(class_name)),
                "online_count": 0,
                "record_count": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "focused_rate": 0.0,
            }

        avg_score = sum(item["avg_score"] for item in active_summaries) / len(active_summaries)
        max_score = max(item["max_score"] for item in active_summaries)
        min_score = min(item["min_score"] for item in active_summaries)
        record_count = sum(item["record_count"] for item in active_summaries)
        focused_rate = sum(item["focused_rate"] for item in active_summaries) / len(active_summaries)

        return {
            "class_name": class_name,
            "student_count": len(self.get_all_students(class_name)),
            "online_count": sum(1 for item in summaries if item.get("online")),
            "record_count": record_count,
            "avg_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "focused_rate": focused_rate,
        }

    def get_class_student_summaries(self, class_name):
        students = self.get_all_students(class_name)
        summaries = []
        for student_id, username, name, email, student_class, last_seen_at, presence_state in students:
            student_info = {
                "id": student_id,
                "username": username,
                "name": name or username,
                "email": email,
                "class_name": student_class,
                "last_seen_at": last_seen_at,
                "online": self._is_user_online(last_seen_at, presence_state),
            }
            summary = self.get_student_summary(student_id)
            summary.update(student_info)
            summaries.append(summary)

        summaries.sort(key=lambda item: (-item["avg_score"], item["name"]))
        return summaries

    def get_class_trend(self, class_name, limit=120):
        conn = self.get_connection()
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

        rows.reverse()
        trend = []
        for score, status, timestamp, name, username in rows:
            trend.append(
                {
                    "attention_score": float(score),
                    "status": status,
                    "timestamp": timestamp,
                    "student_name": name or username,
                }
            )
        return trend

    def _summarize_records(self, records):
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
            "focused_rate": focused_count / len(records) if records else 0.0,
            "latest_status": latest_record["status"],
            "latest_score": latest_record["attention_score"],
            "latest_timestamp": latest_record["timestamp"],
            "status_distribution": dict(status_counts),
        }

    def get_student_records_grouped_by_day(self, user_id, limit=500):
        records = self.get_attention_records(user_id, limit=limit)
        grouped = defaultdict(list)
        for record in records:
            date_key = record["timestamp"][:10]
            grouped[date_key].append(record)

        rows = []
        for date_key, items in sorted(grouped.items(), reverse=True):
            summary = self._summarize_records(items)
            rows.append(
                {
                    "date": date_key,
                    "avg_score": summary["avg_score"],
                    "record_count": summary["record_count"],
                    "focused_rate": summary["focused_rate"],
                }
            )
        return rows
