import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

from config.config import APP_USER_AGENT, DEFAULT_CLASSES


class RemoteDatabaseError(Exception):
    pass


class RemoteDatabase:
    def __init__(self, base_url, timeout=5, retries=0, retry_delay=0.25):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay

    def _request(self, method, path, payload=None, query=None, timeout=None, retries=None):
        url = f"{self.base_url}{path}"
        if query:
            query = {key: value for key, value in query.items() if value is not None}
            url = f"{url}?{urllib.parse.urlencode(query)}"

        data = None
        headers = {
            "Accept": "application/json",
            "User-Agent": APP_USER_AGENT,
            "Connection": "close",
        }
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        timeout = self.timeout if timeout is None else timeout
        retries = self.retries if retries is None else retries

        last_exc = None
        for attempt in range(retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read().decode("utf-8")
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as exc:
                message = exc.reason
                try:
                    body = exc.read().decode("utf-8")
                    detail = json.loads(body).get("detail")
                    if detail:
                        message = detail
                except Exception:
                    pass
                raise RemoteDatabaseError(f"服务器返回错误: {message}") from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
                last_exc = exc
                if attempt < retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise RemoteDatabaseError(f"无法连接服务器 {self.base_url}: {exc}") from exc
            except Exception as exc:
                last_exc = exc
                break

        raise RemoteDatabaseError(f"无法连接服务器 {self.base_url}: {last_exc}")

    def register_user(self, username, password, user_type, name="", email="", class_name=None):
        try:
            response = self._request(
                "POST",
                "/api/auth/register",
                {
                    "username": username,
                    "password": password,
                    "user_type": user_type,
                    "name": name,
                    "email": email,
                    "class_name": class_name,
                },
                timeout=6.0,
                retries=0,
            )
            return bool(response.get("success")), response.get("message", "注册完成")
        except RemoteDatabaseError as exc:
            return False, str(exc)

    def _friendly_class_feature_error(self, error):
        message = str(error)
        if "Not Found" in message:
            return (
                "云端服务器还没有同步班级功能接口。"
                "请先把最新版 /server/main.py 部署到阿里云后端，再重试创建或加入班级。"
            )
        return message

    def login_user(self, username, password, user_type):
        try:
            response = self._request(
                "POST",
                "/api/auth/login",
                {"username": username, "password": password, "user_type": user_type},
                timeout=6.0,
                retries=0,
            )
            if response.get("success"):
                return True, response["user"]
            return False, response.get("message", "用户名、密码或用户类型错误")
        except RemoteDatabaseError as exc:
            return False, str(exc)

    def save_attention_record(self, user_id, attention_score, status, metrics=None):
        try:
            self._request(
                "POST",
                "/api/attention/records",
                {
                    "user_id": user_id,
                    "attention_score": attention_score,
                    "status": status,
                    "metrics": metrics or {},
                },
            )
            return True
        except RemoteDatabaseError as exc:
            print(f"Save remote attention record failed: {exc}")
            return False

    def update_presence(self, user_id, online=True):
        try:
            self._request(
                "POST",
                "/api/presence",
                {"user_id": user_id, "online": bool(online)},
                timeout=2.0,
                retries=0,
            )
            return True
        except RemoteDatabaseError:
            return False

    def create_assistant_conversation(self, user_id, title="新对话"):
        try:
            return self._request(
                "POST",
                "/api/assistant/conversations",
                {"user_id": user_id, "title": title},
            ).get("conversation")
        except RemoteDatabaseError:
            return None

    def get_assistant_conversations(self, user_id, limit=40):
        try:
            return self._request(
                "GET",
                f"/api/users/{user_id}/assistant-conversations",
                query={"limit": limit},
            ).get("conversations", [])
        except RemoteDatabaseError:
            return []

    def get_assistant_conversation_messages(self, user_id, conversation_id, limit=80):
        try:
            return self._request(
                "GET",
                f"/api/users/{user_id}/assistant-conversations/{conversation_id}",
                query={"limit": limit},
            ).get("messages", [])
        except RemoteDatabaseError:
            return []

    def delete_assistant_conversation(self, user_id, conversation_id):
        try:
            return bool(
                self._request(
                    "DELETE",
                    f"/api/users/{user_id}/assistant-conversations/{conversation_id}",
                ).get("success")
            )
        except RemoteDatabaseError:
            return False

    def save_assistant_query_log(self, user_id, query_text, reply_text="", source="text", model_name="", conversation_id=None):
        try:
            self._request(
                "POST",
                "/api/assistant/logs",
                {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "query_text": query_text,
                    "reply_text": reply_text,
                    "source": source,
                    "model_name": model_name,
                },
                timeout=3.0,
                retries=0,
            )
            return True
        except RemoteDatabaseError:
            return False

    def get_attention_records(self, user_id, limit=100):
        try:
            return self._request("GET", f"/api/students/{user_id}/records", query={"limit": limit}).get("records", [])
        except RemoteDatabaseError:
            return []

    def get_recent_student_trend(self, user_id, limit=60):
        try:
            return self._request("GET", f"/api/students/{user_id}/trend", query={"limit": limit}).get("trend", [])
        except RemoteDatabaseError:
            return []

    def get_student_summary(self, user_id, limit=200):
        try:
            return self._request("GET", f"/api/students/{user_id}/summary", query={"limit": limit}).get(
                "summary", self._empty_student_summary()
            )
        except RemoteDatabaseError:
            return self._empty_student_summary()

    def get_all_students(self, class_name=None):
        try:
            rows = self._request("GET", "/api/students", query={"class_name": class_name}).get("students", [])
            return [tuple(row) for row in rows]
        except RemoteDatabaseError:
            return []

    def get_class_options(self):
        try:
            classes = self._request("GET", "/api/classes").get("classes", [])
            return classes or DEFAULT_CLASSES
        except RemoteDatabaseError:
            return DEFAULT_CLASSES

    def get_class_info(self, class_name):
        if not class_name:
            return None
        try:
            return self._request("GET", f"/api/classes/{urllib.parse.quote(class_name)}/info").get("class_info")
        except RemoteDatabaseError:
            return None

    def create_class(self, teacher_id, class_name):
        try:
            response = self._request(
                "POST",
                "/api/classes/create",
                {"teacher_id": teacher_id, "class_name": class_name},
            )
            return bool(response.get("success")), response.get("message", "创建完成"), response.get("class_info")
        except RemoteDatabaseError as exc:
            return False, self._friendly_class_feature_error(exc), None

    def join_class(self, student_id, class_code):
        try:
            response = self._request(
                "POST",
                "/api/classes/join",
                {"student_id": student_id, "class_code": class_code},
            )
            return bool(response.get("success")), response.get("message", "加入完成"), response.get("class_info")
        except RemoteDatabaseError as exc:
            return False, self._friendly_class_feature_error(exc), None

    def bind_parent_to_student(self, parent_id, student_username):
        try:
            response = self._request(
                "POST",
                "/api/parents/bind-child",
                {"parent_id": parent_id, "student_username": student_username},
            )
            return bool(response.get("success")), response.get("message", "绑定完成"), response.get("child")
        except RemoteDatabaseError as exc:
            return False, str(exc), None

    def get_parent_child(self, parent_id):
        try:
            return self._request("GET", f"/api/parents/{parent_id}/child").get("child")
        except RemoteDatabaseError:
            return None

    def get_parent_dashboard(self, parent_id, limit_days=14, limit_logs=30):
        try:
            return self._request(
                "GET",
                f"/api/parents/{parent_id}/dashboard",
                query={"limit_days": limit_days, "limit_logs": limit_logs},
            )
        except RemoteDatabaseError as exc:
            return {
                "_fetch_failed": True,
                "_error_message": str(exc),
                "child": None,
                "summary": self._empty_student_summary(),
                "daily_rows": [],
                "assistant_logs": [],
                "today_query_count": 0,
            }

    def get_user_by_id(self, user_id):
        try:
            return self._request("GET", f"/api/users/{user_id}").get("user")
        except RemoteDatabaseError:
            return None

    def get_assistant_query_logs(self, user_id, limit=30):
        try:
            return self._request("GET", f"/api/students/{user_id}/assistant-logs", query={"limit": limit}).get("logs", [])
        except RemoteDatabaseError:
            return []

    def get_class_summary(self, class_name):
        try:
            return self._request("GET", f"/api/classes/{urllib.parse.quote(class_name)}" + "/summary").get(
                "summary", self._empty_class_summary(class_name)
            )
        except RemoteDatabaseError:
            return self._empty_class_summary(class_name)

    def get_class_student_summaries(self, class_name):
        try:
            return self._request("GET", f"/api/classes/{urllib.parse.quote(class_name)}" + "/students").get(
                "students", []
            )
        except RemoteDatabaseError:
            return []

    def get_class_trend(self, class_name, limit=120):
        try:
            return self._request(
                "GET",
                f"/api/classes/{urllib.parse.quote(class_name)}" + "/trend",
                query={"limit": limit},
            ).get("trend", [])
        except RemoteDatabaseError:
            return []

    def get_student_records_grouped_by_day(self, user_id, limit=500):
        records = self.get_attention_records(user_id, limit=limit)
        grouped = {}
        for record in records:
            date_key = record["timestamp"][:10]
            grouped.setdefault(date_key, []).append(record)

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

    def _summarize_records(self, records):
        if not records:
            return self._empty_student_summary()

        scores = [item["attention_score"] for item in records]
        status_distribution = {}
        for item in records:
            status_distribution[item["status"]] = status_distribution.get(item["status"], 0) + 1
        focused_count = status_distribution.get("专注", 0) + status_distribution.get("focused", 0)
        latest = records[0]
        return {
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "record_count": len(records),
            "focused_rate": focused_count / len(records),
            "latest_status": latest["status"],
            "latest_score": latest["attention_score"],
            "latest_timestamp": latest["timestamp"],
            "status_distribution": status_distribution,
        }

    def _empty_student_summary(self):
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

    def _empty_class_summary(self, class_name):
        return {
            "class_name": class_name,
            "student_count": 0,
            "online_count": 0,
            "record_count": 0,
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "focused_rate": 0.0,
        }
