import base64
import json
import os
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
import zipfile
from xml.etree import ElementTree as ET

from PyQt5.QtCore import QBuffer, QByteArray, Qt
from PyQt5.QtGui import QImage

from config.config import (
    APP_USER_AGENT,
    DEEPSEEK_API_BASE_URL,
    DEEPSEEK_LLM_API_KEY,
    DOUBAO_API_BASE_URL,
    DOUBAO_LLM_API_KEY,
    LOCAL_LLM_API_KEY,
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_ENABLED,
    LOCAL_LLM_MODEL,
    LOCAL_LLM_PROVIDER,
    LOCAL_LLM_TIMEOUT,
)


class VoiceLLMControllerError(Exception):
    pass


class VoiceLLMController:
    """Interpret XiaoRui commands with remote LLM providers."""
    REQUEST_RETRIES = 2
    DOUBAO_REQUEST_RETRIES = 3
    REASONING_MODEL_NAMES = {"deepseek-reasoner"}
    MULTIMODAL_MODELS = {"doubao-seed-1-6-vision-250815"}
    IMAGE_GENERATION_MODELS = {"doubao-seedream-5-0-lite-260128"}
    FINAL_ANSWER_FALLBACK_MODELS = {
        "deepseek-reasoner": "deepseek-chat",
    }
    DOUBAO_VISION_MODELS = {"doubao-seed-1-6-vision-250815"}
    TEXT_FILE_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".json",
        ".log",
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".xml",
        ".toml",
        ".yml",
        ".yaml",
        ".ini",
        ".cfg",
    }

    def __init__(
        self,
        local_enabled=None,
        local_provider=None,
        local_base_url=None,
        local_model=None,
        local_api_key=None,
        local_timeout=None,
    ):
        self.local_enabled = LOCAL_LLM_ENABLED if local_enabled is None else bool(local_enabled)
        self.local_provider = (
            LOCAL_LLM_PROVIDER if local_provider is None else str(local_provider).strip().lower()
        ) or "deepseek"
        self.local_base_url = (
            LOCAL_LLM_BASE_URL if local_base_url is None else str(local_base_url).strip()
        ).rstrip("/")
        self.local_model = (LOCAL_LLM_MODEL if local_model is None else str(local_model).strip()) or "deepseek-chat"
        self.local_api_key = (
            LOCAL_LLM_API_KEY if local_api_key is None else str(local_api_key).strip()
        )
        self.local_timeout = float(LOCAL_LLM_TIMEOUT if local_timeout is None else local_timeout)
        self._local_probe_ok = None

    @property
    def local_available(self):
        return (
            self.local_enabled
            and bool(self.local_model)
            and bool(self.local_base_url)
            and bool(self.local_api_key)
        )

    @property
    def available(self):
        return self.local_available

    def get_status(self):
        if self.local_available:
            return f"当前问答模型：{self.local_model}（{self._provider_label()}）"
        return f"未配置{self._provider_label()} API Key，当前使用内置指令处理"

    def shutdown(self):
        return None

    def interpret(self, command, context):
        if self.local_available:
            try:
                result = self._interpret_with_local(command, context)
                result["source"] = "local"
                return result
            except VoiceLLMControllerError:
                raise

        raise VoiceLLMControllerError(f"未配置{self._provider_label()} API Key")

    def chat(self, message, context, max_tokens_override=None):
        if self.local_available:
            try:
                return self._chat_with_local(message, context, max_tokens_override=max_tokens_override)
            except VoiceLLMControllerError:
                raise

        raise VoiceLLMControllerError(f"未配置{self._provider_label()} API Key")

    def assist_text(self, message, context):
        if self.local_available:
            try:
                return self._assist_text_with_local(message, context)
            except VoiceLLMControllerError:
                raise

        raise VoiceLLMControllerError(f"未配置{self._provider_label()} API Key")

    def chat_with_attachment(self, message, context, attachment, max_tokens_override=None):
        if self.local_available:
            try:
                return self._chat_with_attachment_local(message, context, attachment, max_tokens_override=max_tokens_override)
            except VoiceLLMControllerError:
                raise

        raise VoiceLLMControllerError(f"未配置{self._provider_label()} API Key")

    def generate_image(self, prompt, size="2K"):
        if self.local_available:
            try:
                return self._generate_image_local(prompt, size=size)
            except VoiceLLMControllerError:
                raise
        raise VoiceLLMControllerError(f"未配置{self._provider_label()} API Key")

    def _interpret_with_local(self, command, context):
        self._ensure_local_endpoint_ready()
        provider = self.local_provider
        if provider in {"deepseek", "doubao"}:
            return self._interpret_with_openai_compatible(command, context)
        raise VoiceLLMControllerError(f"当前不支持的模型提供方：{provider}")

    def _chat_with_local(self, message, context, max_tokens_override=None):
        self._ensure_local_endpoint_ready()
        provider = self.local_provider
        if provider in {"deepseek", "doubao"}:
            return self._chat_with_openai_compatible(message, context, max_tokens_override=max_tokens_override)
        raise VoiceLLMControllerError(f"当前不支持的模型提供方：{provider}")

    def _assist_text_with_local(self, message, context):
        self._ensure_local_endpoint_ready()
        provider = self.local_provider
        if provider in {"deepseek", "doubao"}:
            return self._assist_text_with_openai_compatible(message, context)
        raise VoiceLLMControllerError(f"当前不支持的模型提供方：{provider}")

    def _chat_with_attachment_local(self, message, context, attachment, max_tokens_override=None):
        self._ensure_local_endpoint_ready()
        attachment = dict(attachment or {})
        kind = str(attachment.get("kind") or "").strip().lower()
        if kind == "file":
            return self._chat_with_file_attachment_local(message, context, attachment, max_tokens_override=max_tokens_override)
        preferred_model = self._preferred_model_for_image_attachment()
        if not preferred_model:
            raise VoiceLLMControllerError("当前还没有可用的图片理解模型，请先配置 Doubao API Key。")
        return self._run_with_provider_and_model_override(
            preferred_model,
            lambda: self._chat_with_openai_multimodal_attachment(message, context, attachment, max_tokens_override=max_tokens_override),
        )

    def _generate_image_local(self, prompt, size="2K"):
        self._ensure_local_endpoint_ready()
        preferred_model = self._preferred_model_for_image_generation()
        if not preferred_model:
            raise VoiceLLMControllerError("当前还没有可用的图片生成模型，请先配置 Doubao API Key。")
        return self._run_with_provider_and_model_override(
            preferred_model,
            lambda: self._generate_image_with_doubao(prompt, size=size),
        )

    def _chat_with_file_attachment_local(self, message, context, attachment, max_tokens_override=None):
        attachment = dict(attachment or {})
        path = str(attachment.get("path") or "").strip()
        if not path or not os.path.exists(path):
            raise VoiceLLMControllerError("上传内容不存在，请重新选择文件。")

        file_text = self._extract_file_text(path)
        prompt = self._file_prompt_text(message, context, os.path.basename(path), file_text)
        preferred_model = self._preferred_model_for_file_attachment()
        return self._run_with_provider_and_model_override(
            preferred_model,
            lambda: self._chat_with_local(
                prompt,
                {
                    **(context or {}),
                    "attachment_kind": "file",
                    "attachment_name": os.path.basename(path),
                },
                max_tokens_override=max_tokens_override or 520,
            ),
        )

    def _chat_with_openai_multimodal_attachment(self, message, context, attachment, max_tokens_override=None):
        attachment = dict(attachment or {})
        path = str(attachment.get("path") or "").strip()
        kind = str(attachment.get("kind") or "").strip().lower()
        if not path or not os.path.exists(path):
            raise VoiceLLMControllerError("上传内容不存在，请重新选择图片或文件。")
        if kind not in {"image", "file"}:
            raise VoiceLLMControllerError("当前上传内容类型暂不支持，请重新选择。")

        endpoint = self._resolve_chat_endpoint()
        if self.local_provider == "doubao":
            request_timeout = max(self.local_timeout + 20, 40)
            default_tokens = 180
        else:
            request_timeout = self.local_timeout + (10 if self._is_reasoning_model() else 4)
            default_tokens = 800 if self._is_reasoning_model() else 380
        payload = {
            "model": self.local_model,
            "temperature": 0.35,
            "top_p": 0.8,
            "max_tokens": int(max_tokens_override or default_tokens),
            "messages": self._attachment_messages(message, context, path, kind),
        }
        if self._is_reasoning_model():
            payload["thinking"] = {"type": "enabled"}

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": APP_USER_AGENT,
        }
        if self.local_api_key:
            headers["Authorization"] = f"Bearer {self.local_api_key}"

        parsed = self._post_json(endpoint, payload, timeout=request_timeout, headers=headers)
        try:
            message_payload = parsed["choices"][0]["message"]
            content = message_payload.get("content")
        except Exception as exc:
            raise VoiceLLMControllerError(f"{self._provider_label()} 接口响应格式异常：{exc}") from exc

        if not self._normalize_chat_content(content) and self._has_reasoning_without_answer(message_payload):
            raise VoiceLLMControllerError(f"{self._provider_label()} 深度思考尚未生成最终答案，请再试一次或换一个更短的问题。")

        reply = self._normalize_chat_content(content)
        if not reply:
            raise VoiceLLMControllerError(f"{self._provider_label()} 没有返回可用内容，请稍后再试。")
        return reply

    def _generate_image_with_doubao(self, prompt, size="2K"):
        endpoint = self._resolve_image_generation_endpoint()
        payload = {
            "model": self.local_model,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
            "stream": False,
            "watermark": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": APP_USER_AGENT,
        }
        if self.local_api_key:
            headers["Authorization"] = f"Bearer {self.local_api_key}"
        parsed = self._post_json(endpoint, payload, timeout=max(self.local_timeout + 18, 30), headers=headers)
        try:
            data = parsed["data"][0]
        except Exception as exc:
            raise VoiceLLMControllerError(f"{self._provider_label()} 图片生成接口响应格式异常：{exc}") from exc

        b64_json = str(data.get("b64_json") or "").strip()
        url = str(data.get("url") or "").strip()
        if b64_json:
            return {"type": "b64_json", "data": b64_json}
        if url:
            return {"type": "url", "data": url}
        raise VoiceLLMControllerError(f"{self._provider_label()} 图片生成没有返回可用图片数据，请稍后再试。")

    def _ensure_local_endpoint_ready(self):
        if self._local_probe_ok is True:
            return

        if self.local_provider in {"deepseek", "doubao"}:
            self._local_probe_ok = True
            return

        try:
            parsed = urlparse(self.local_base_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=0.35):
                self._local_probe_ok = True
                return
        except OSError as exc:
            self._local_probe_ok = False
            raise VoiceLLMControllerError(self._endpoint_unreachable_message()) from exc

    def _interpret_with_openai_compatible(self, command, context):
        endpoint = self._resolve_chat_endpoint()

        request_timeout = self.local_timeout + (8 if self._is_reasoning_model() else 0)
        payload = {
            "model": self.local_model,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 360 if self._is_reasoning_model() else 180,
            "messages": self._messages(command, context),
        }

        if self.local_provider == "deepseek":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": APP_USER_AGENT,
        }
        if self.local_api_key:
            headers["Authorization"] = f"Bearer {self.local_api_key}"

        parsed = self._post_json(endpoint, payload, timeout=request_timeout, headers=headers)
        try:
            message = parsed["choices"][0]["message"]
            content = message.get("content")
        except Exception as exc:
            raise VoiceLLMControllerError(f"{self._provider_label()} 接口响应格式异常：{exc}") from exc
        if not self._normalize_chat_content(content) and self._has_reasoning_without_answer(message):
            fallback_model = self._fallback_model_for_final_answer()
            if fallback_model:
                return self._run_with_model_override(
                    fallback_model,
                    lambda: self._interpret_with_openai_compatible(command, context),
                )
            raise VoiceLLMControllerError(f"{self._provider_label()} 深度思考尚未生成最终答案，请再试一次或换一个更短的问题。")
        return self._parse_result_content(content, f"{self._provider_label()} 接口")

    def _chat_with_openai_compatible(self, message, context, max_tokens_override=None):
        endpoint = self._resolve_chat_endpoint()

        request_timeout = self.local_timeout + (8 if self._is_reasoning_model() else 0)
        payload = {
            "model": self.local_model,
            "temperature": 0.45,
            "top_p": 0.8,
            "max_tokens": int(max_tokens_override or (700 if self._is_reasoning_model() else 220)),
            "messages": self._chat_messages(message, context),
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": APP_USER_AGENT,
        }
        if self.local_api_key:
            headers["Authorization"] = f"Bearer {self.local_api_key}"

        parsed = self._post_json(endpoint, payload, timeout=request_timeout, headers=headers)
        try:
            parsed_message = parsed["choices"][0]["message"]
            content = parsed_message.get("content")
        except Exception as exc:
            raise VoiceLLMControllerError(f"{self._provider_label()} 接口响应格式异常：{exc}") from exc
        if not self._normalize_chat_content(content) and self._has_reasoning_without_answer(parsed_message):
            fallback_model = self._fallback_model_for_final_answer()
            if fallback_model:
                return self._run_with_model_override(
                    fallback_model,
                    lambda: self._chat_with_openai_compatible(message, context, max_tokens_override=max_tokens_override),
                )
            raise VoiceLLMControllerError(f"{self._provider_label()} 深度思考已完成推理，但这次没有生成最终答案。你可以重试，或先关闭深度思考。")
        reply = self._normalize_chat_content(content)
        if not reply:
            raise VoiceLLMControllerError(f"{self._provider_label()} 没有返回可用内容，请稍后再试。")
        return reply

    def _assist_text_with_openai_compatible(self, message, context):
        endpoint = self._resolve_chat_endpoint()

        request_timeout = self.local_timeout + (8 if self._is_reasoning_model() else 0)
        payload = {
            "model": self.local_model,
            "temperature": 0.25,
            "top_p": 0.8,
            "max_tokens": 480 if self._is_reasoning_model() else 220,
            "messages": self._text_messages(message, context),
        }
        if self.local_provider == "deepseek":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": APP_USER_AGENT,
        }
        if self.local_api_key:
            headers["Authorization"] = f"Bearer {self.local_api_key}"

        parsed = self._post_json(endpoint, payload, timeout=request_timeout, headers=headers)
        try:
            parsed_message = parsed["choices"][0]["message"]
            content = parsed_message.get("content")
        except Exception as exc:
            raise VoiceLLMControllerError(f"{self._provider_label()} 接口响应格式异常：{exc}") from exc
        if not self._normalize_chat_content(content) and self._has_reasoning_without_answer(parsed_message):
            fallback_model = self._fallback_model_for_final_answer()
            if fallback_model:
                return self._run_with_model_override(
                    fallback_model,
                    lambda: self._assist_text_with_openai_compatible(message, context),
                )
            raise VoiceLLMControllerError(f"{self._provider_label()} 深度思考只返回了推理过程，没有给出最终结果。建议重试或关闭深度思考。")
        return self._parse_result_content(content, f"{self._provider_label()} 接口")

    def _post_json(self, url, payload, timeout, headers=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers or {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": APP_USER_AGENT,
            },
            method="POST",
        )

        raw = None
        last_error = None
        retries = self.DOUBAO_REQUEST_RETRIES if self.local_provider == "doubao" else self.REQUEST_RETRIES
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read().decode("utf-8")
                    break
            except urllib.error.HTTPError as exc:
                detail = exc.reason
                try:
                    body_text = exc.read().decode("utf-8")
                    parsed = json.loads(body_text)
                    detail = (
                        parsed.get("error", {}).get("message")
                        or parsed.get("message")
                        or parsed.get("error")
                        or detail
                    )
                except Exception:
                    pass
                if exc.code == 429:
                    user_message = "你的提问有点频繁，请稍等两秒再问我。"
                    if attempt < retries - 1:
                        last_error = user_message
                        time.sleep((0.9 if self.local_provider == "doubao" else 0.7) * (attempt + 1))
                        continue
                    raise VoiceLLMControllerError(user_message) from exc
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < retries - 1:
                    last_error = f"接口返回错误：{detail}"
                    time.sleep((0.75 if self.local_provider == "doubao" else 0.45) * (attempt + 1))
                    continue
                raise VoiceLLMControllerError(f"接口返回错误：{detail}") from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep((0.75 if self.local_provider == "doubao" else 0.45) * (attempt + 1))
                    continue
                if "timed out" in str(exc).lower():
                    raise VoiceLLMControllerError(f"{self._provider_label()} 响应超时，请稍后重试或换一个更短的问题。") from exc
                raise VoiceLLMControllerError(f"连接失败：{exc}") from exc
            except Exception as exc:
                raise VoiceLLMControllerError(f"请求异常：{exc}") from exc

        if raw is None:
            if last_error and "timed out" in str(last_error).lower():
                raise VoiceLLMControllerError(f"{self._provider_label()} 响应超时，请稍后重试或换一个更短的问题。")
            raise VoiceLLMControllerError(f"连接失败：{last_error}")

        try:
            return json.loads(raw)
        except Exception as exc:
            raise VoiceLLMControllerError(f"响应不是有效 JSON：{raw[:300]}") from exc

    def _parse_result_content(self, content, source_name):
        normalized_content = self._normalize_result_content(content)
        try:
            result = json.loads(normalized_content)
        except Exception as exc:
            raise VoiceLLMControllerError(f"{source_name} 返回内容不是有效 JSON：{normalized_content}") from exc

        intent = str(result.get("intent") or "none").strip()
        reply = str(result.get("reply") or "").strip()
        confidence = result.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        if intent not in {"query_focus", "open_analytics", "give_advice", "start_detection", "stop_detection", "minimize_window", "show_detection", "introduce_self", "general_chat", "none"}:
            intent = "none"

        if not reply:
            reply = "我听到了，但暂时没有生成合适的回复。"

        return {
            "intent": intent,
            "reply": reply,
            "confidence": max(0.0, min(1.0, confidence)),
            "raw": result,
        }

    def _normalize_result_content(self, content):
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            content = "".join(parts)

        text = str(content or "").strip()
        if not text:
            return "{}"

        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        if "```" in text:
            text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1].strip()

        return text

    def _provider_label(self):
        if self.local_provider == "doubao":
            return "Doubao"
        return "DeepSeek"

    def _endpoint_unreachable_message(self):
        return f"{self._provider_label()} 接口不可达，请检查网络、API 地址或 API Key。"

    def _resolve_chat_endpoint(self):
        endpoint = self.local_base_url.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            return endpoint
        if self.local_provider == "doubao":
            if endpoint.endswith("/v4"):
                return f"{endpoint}/chat/completions"
            if endpoint.endswith("/api/v3"):
                return f"{endpoint}/chat/completions"
            return f"{endpoint}/chat/completions"
        if endpoint.endswith("/v1"):
            return f"{endpoint}/chat/completions"
        return f"{endpoint}/v1/chat/completions"

    def _resolve_image_generation_endpoint(self):
        endpoint = self.local_base_url.rstrip("/")
        if endpoint.endswith("/images/generations"):
            return endpoint
        if endpoint.endswith("/api/v3"):
            return f"{endpoint}/images/generations"
        if endpoint.endswith("/v1"):
            return f"{endpoint}/images/generations"
        return f"{endpoint}/images/generations"

    def _is_reasoning_model(self):
        return (self.local_model or "").strip().lower() in self.REASONING_MODEL_NAMES

    def _fallback_model_for_final_answer(self):
        current = (self.local_model or "").strip().lower()
        fallback = self.FINAL_ANSWER_FALLBACK_MODELS.get(current)
        if not fallback or fallback == current:
            return None
        return fallback

    def _preferred_model_for_file_attachment(self):
        provider = (self.local_provider or "").strip().lower()
        model = (self.local_model or "").strip().lower()
        if DEEPSEEK_LLM_API_KEY:
            return "deepseek:deepseek-chat"
        if provider == "doubao" and DOUBAO_LLM_API_KEY:
            return f"doubao:{model or 'doubao-seed-1-6-vision-250815'}"
        if DOUBAO_LLM_API_KEY:
            return "doubao:doubao-seed-1-6-vision-250815"
        return f"{provider}:{model}" if provider and model else "deepseek:deepseek-chat"

    def _preferred_model_for_image_attachment(self):
        provider = (self.local_provider or "").strip().lower()
        model = (self.local_model or "").strip().lower()
        if DOUBAO_LLM_API_KEY:
            return "doubao:doubao-seed-1-6-vision-250815"
        if provider == "doubao" and model in self.MULTIMODAL_MODELS:
            return f"{provider}:{model}"
        return ""

    def _preferred_model_for_image_generation(self):
        provider = (self.local_provider or "").strip().lower()
        model = (self.local_model or "").strip().lower()
        if DOUBAO_LLM_API_KEY:
            return "doubao:doubao-seedream-5-0-lite-260128"
        if provider == "doubao" and model in self.IMAGE_GENERATION_MODELS:
            return f"{provider}:{model}"
        return ""

    def _run_with_model_override(self, model_name, callback):
        original_model = self.local_model
        try:
            self.local_model = model_name
            return callback()
        finally:
            self.local_model = original_model

    def _run_with_provider_and_model_override(self, spec, callback):
        spec = (spec or "").strip().lower()
        if not spec or ":" not in spec:
            return callback()
        provider, model_name = spec.split(":", 1)
        original_provider = self.local_provider
        original_model = self.local_model
        original_base_url = self.local_base_url
        original_api_key = self.local_api_key
        try:
            self.local_provider = provider
            self.local_model = model_name
            if provider == "doubao":
                self.local_base_url = DOUBAO_API_BASE_URL
                self.local_api_key = DOUBAO_LLM_API_KEY
            elif provider == "deepseek":
                self.local_base_url = DEEPSEEK_API_BASE_URL
                self.local_api_key = DEEPSEEK_LLM_API_KEY
            return callback()
        finally:
            self.local_provider = original_provider
            self.local_model = original_model
            self.local_base_url = original_base_url
            self.local_api_key = original_api_key

    def _has_reasoning_without_answer(self, message):
        if not isinstance(message, dict):
            return False
        reasoning = self._normalize_chat_content(message.get("reasoning_content"))
        answer = self._normalize_chat_content(message.get("content"))
        return bool(reasoning) and not bool(answer)

    def _messages(self, command, context):
        return [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "voice_command": command,
                        "runtime_context": context,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    def _chat_messages(self, message, context):
        return [
            {"role": "system", "content": self._chat_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_message": message,
                        "runtime_context": context,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    def _attachment_messages(self, message, context, path, kind):
        if kind == "image":
            return [
                {"role": "system", "content": self._multimodal_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_to_base64(path),
                            },
                        },
                        {
                            "type": "text",
                            "text": self._image_prompt_text(message, context, os.path.basename(path)),
                        },
                    ],
                },
            ]

        file_text = self._extract_file_text(path)
        return [
            {"role": "system", "content": self._multimodal_system_prompt()},
            {
                "role": "user",
                "content": self._file_prompt_text(message, context, os.path.basename(path), file_text),
            },
        ]

    def _text_messages(self, message, context):
        return [
            {"role": "system", "content": self._text_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_message": message,
                        "runtime_context": context,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    def _system_prompt(self):
        return (
            "你是桌面学习助手“小睿”。"
            "你是一名中文学习陪伴助手，语气自然、简洁、友好。"
            "你只负责理解语音命令并返回 JSON，不要输出任何额外解释。"
            "你必须严格返回一个 JSON 对象，字段包括 intent、reply、confidence。"
            "intent 只能是 query_focus、open_analytics、give_advice、start_detection、stop_detection、minimize_window、show_detection、introduce_self、general_chat、none 十选一。"
            "当用户询问当前状态、当前专注度、分数时，使用 query_focus。"
            "当用户要求打开详细数据、分析页、数据页、统计页时，使用 open_analytics。"
            "当用户询问为什么分数低、怎么提高、给学习建议、该怎么办时，使用 give_advice。"
            "当用户要求开始检测、打开摄像头、启动检测时，使用 start_detection。"
            "当用户要求停止检测、结束检测、关闭摄像头时，使用 stop_detection。"
            "当用户要求最小化窗口、隐藏程序、缩小窗口时，使用 minimize_window。"
            "当用户要求返回检测界面、回到主界面、恢复窗口时，使用 show_detection。"
            "当用户要求介绍你自己、问你是谁、你能做什么时，使用 introduce_self。"
            "当用户是在闲聊、一般提问、让你解释某个概念或进行简短对话时，使用 general_chat。"
            "完全无法理解且不适合回答时返回 none。"
            "reply 要简洁自然，优先只用 1 句，最多 2 句，适合语音播报。"
            "如果 runtime_context 里表明检测尚未开始，就在 reply 中说明需要先开始检测。"
            "如果 intent=query_focus，可以结合 current_score、avg_score、status_text、main_reason 生成回复。"
            "如果 intent=give_advice，可以结合 current_score、status_text、main_reason、all_reasons、face_detected、phone_detected、pose 生成具体建议。"
            "如果 intent=introduce_self，请只用一句话介绍你是“小睿伴学”里的语音助手，核心功能是检测学习状态、查看数据和语音控制。"
            "如果 intent=general_chat，请直接自然回答用户的问题，但保持简短，不要编造项目里不存在的功能。"
            "confidence 返回 0 到 1 的小数。"
        )

    def _chat_system_prompt(self):
        return (
            "你是“小睿伴学”里的中文学习陪伴助手“小睿”。"
            "现在处于文本聊天模式，请直接自然回答用户，不要输出 JSON，不要解释你的内部规则。"
            "回答风格要简洁、友好、像真实助手，通常 1 到 3 句即可。"
            "你可以结合 runtime_context 回答学习状态、功能使用、项目介绍、教师端和学生端能力。"
            "如果用户问的是开始检测、停止检测、打开详细数据、最小化之类操作，也先用自然语言回答，不要假装已经执行。"
            "如果涉及数学公式或符号，请优先使用纯文本可读写法，例如 x^2、sqrt(x)、(a+b)/c、f(x)=ax^2+bx+c；除非用户明确要求，否则不要使用 LaTeX 公式块。"
            "不要编造项目里不存在的功能。"
        )

    def _text_system_prompt(self):
        return (
            "你是“小睿伴学”里的中文学习陪伴助手“小睿”，当前处于文本助手模式。"
            "你必须严格返回一个 JSON 对象，字段包括 intent、reply、confidence。"
            "intent 只能是 query_focus、open_analytics、give_advice、start_detection、stop_detection、minimize_window、show_detection、introduce_self、general_chat、none 十选一。"
            "如果用户输入的是操作命令，比如开始检测、停止检测、打开详细数据、返回主页面、最小化窗口，就返回对应 intent。"
            "如果用户是在提问、闲聊、解释概念或询问项目功能，就使用 general_chat，并自然回答。"
            "reply 要像真实助手一样自然，不要太机械。"
            "对于 general_chat，优先使用 1 到 3 句回答。"
            "对于 introduce_self，只用一句话介绍自己。"
            "如果涉及数学公式或符号，请优先使用纯文本可读写法，例如 x^2、sqrt(x)、(a+b)/c；除非用户明确要求，否则不要使用 LaTeX 公式块。"
            "不要编造项目中不存在的功能，也不要假装已经执行未触发的动作。"
            "confidence 返回 0 到 1 的小数。"
        )

    def _multimodal_system_prompt(self):
        return (
            "你是“小睿伴学”里的中文学习陪伴助手“小睿”。"
            "当前是上传图片/文件后的多模态问答模式。"
            "请优先依据用户上传的内容回答，语气自然、简洁、友好。"
            "回答通常控制在 2 到 5 句。"
            "如果涉及数学公式或符号，请尽量用纯文本可读写法表达，不要默认输出 LaTeX 公式。"
            "如果上传内容不足以支撑结论，就明确说明你看到的信息有限，不要编造。"
        )

    def _image_prompt_text(self, message, context, filename):
        user_message = (message or "").strip() or "请帮我看看这张图片的内容。"
        project_profile = str((context or {}).get("project_profile") or "").strip()
        return (
            f"用户上传了一张图片《{filename}》。"
            f"用户问题：{user_message}。"
            f"{('项目背景：' + project_profile + '。') if project_profile else ''}"
            "请结合图片内容直接回答。"
        )

    def _file_prompt_text(self, message, context, filename, file_text):
        user_message = (message or "").strip() or "请概括这个文件的主要内容。"
        project_profile = str((context or {}).get("project_profile") or "").strip()
        clipped_text = file_text[:12000]
        project_prefix = f"项目背景：{project_profile}。\n" if project_profile else ""
        return (
            f"用户上传了一个文件《{filename}》。"
            f"用户问题：{user_message}。\n"
            f"{project_prefix}"
            "下面是该文件提取出的可读内容节选，请基于这些内容回答；如果节选不足以支撑结论，请明确说明。\n"
            f"文件内容节选：\n{clipped_text}"
        )

    def _image_to_base64(self, path):
        ext = os.path.splitext(path)[1].lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
        try:
            image = QImage(path)
            if not image.isNull():
                max_dim = 1600
                if image.width() > max_dim or image.height() > max_dim:
                    image = image.scaled(
                        max_dim,
                        max_dim,
                        aspectRatioMode=Qt.KeepAspectRatio,
                        transformMode=Qt.SmoothTransformation,
                    )
                byte_array = QByteArray()
                buffer = QBuffer(byte_array)
                buffer.open(QBuffer.WriteOnly)
                fmt = b"PNG" if mime == "image/png" else b"JPEG"
                quality = -1 if fmt == b"PNG" else 82
                image.save(buffer, fmt.decode("ascii"), quality)
                encoded = base64.b64encode(bytes(byte_array)).decode("utf-8")
                return f"data:{mime};base64,{encoded}"
            with open(path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("utf-8")
                return f"data:{mime};base64,{encoded}"
        except OSError as exc:
            raise VoiceLLMControllerError(f"读取图片失败：{exc}") from exc

    def _extract_file_text(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in self.TEXT_FILE_EXTENSIONS:
            return self._read_text_file(path)
        if ext == ".docx":
            return self._read_docx_file(path)
        if ext == ".pdf":
            return self._read_pdf_file(path)
        if ext == ".xlsx":
            return self._read_xlsx_file(path)
        if ext == ".pptx":
            return self._read_pptx_file(path)
        raise VoiceLLMControllerError("当前文件问答支持常见文本/代码文件、docx、xlsx、pptx 和部分 pdf。")

    def _read_text_file(self, path):
        for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"):
            try:
                with open(path, "r", encoding=encoding) as handle:
                    text = handle.read()
                return text.strip() or "文件内容为空。"
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                raise VoiceLLMControllerError(f"读取文件失败：{exc}") from exc
        raise VoiceLLMControllerError("文件编码暂不支持，请优先上传 UTF-8、GBK 或常见文本文件。")

    def _read_docx_file(self, path):
        try:
            with zipfile.ZipFile(path) as archive:
                xml_bytes = archive.read("word/document.xml")
        except OSError as exc:
            raise VoiceLLMControllerError(f"读取 docx 文件失败：{exc}") from exc
        except KeyError as exc:
            raise VoiceLLMControllerError("这个 docx 文件结构不完整，暂时无法解析。") from exc

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise VoiceLLMControllerError("这个 docx 文件内容解析失败。") from exc

        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        texts = [node.text for node in root.findall(".//w:t", namespace) if node.text]
        content = "".join(texts).strip()
        if not content:
            raise VoiceLLMControllerError("这个 docx 文件里没有提取到可读文本。")
        return content

    def _read_pdf_file(self, path):
        for module_name in ("pypdf", "PyPDF2"):
            try:
                module = __import__(module_name)
                reader_cls = getattr(module, "PdfReader", None)
                if reader_cls is None:
                    continue
                reader = reader_cls(path)
                parts = []
                for page in reader.pages[:8]:
                    try:
                        parts.append(page.extract_text() or "")
                    except Exception:
                        continue
                text = "\n".join(parts).strip()
                if text:
                    return text
            except ImportError:
                continue
            except Exception as exc:
                raise VoiceLLMControllerError(f"PDF 解析失败：{exc}") from exc
        raise VoiceLLMControllerError("当前环境还没有 PDF 解析能力，先支持 txt、md、csv、json、docx、xlsx、pptx 等文件。")

    def _read_xlsx_file(self, path):
        try:
            with zipfile.ZipFile(path) as archive:
                shared_strings = self._xlsx_shared_strings(archive)
                sheet_texts = []
                worksheet_names = sorted(
                    [name for name in archive.namelist() if name.startswith("xl/worksheets/") and name.endswith(".xml")]
                )
                for index, worksheet_name in enumerate(worksheet_names[:5], start=1):
                    try:
                        xml_bytes = archive.read(worksheet_name)
                        parsed = self._extract_xlsx_sheet_text(xml_bytes, shared_strings)
                    except Exception:
                        continue
                    if parsed:
                        sheet_texts.append(f"[工作表{index}]\n{parsed}")
                content = "\n\n".join(sheet_texts).strip()
                if not content:
                    raise VoiceLLMControllerError("这个 Excel 文件里没有提取到可读文本。")
                return content
        except OSError as exc:
            raise VoiceLLMControllerError(f"读取 Excel 文件失败：{exc}") from exc
        except zipfile.BadZipFile as exc:
            raise VoiceLLMControllerError("这个 Excel 文件结构不完整，暂时无法解析。") from exc

    def _xlsx_shared_strings(self, archive):
        try:
            xml_bytes = archive.read("xl/sharedStrings.xml")
        except KeyError:
            return []
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return []
        namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        strings = []
        for item in root.findall(".//s:si", namespace):
            parts = [node.text for node in item.findall(".//s:t", namespace) if node.text]
            strings.append("".join(parts))
        return strings

    def _extract_xlsx_sheet_text(self, xml_bytes, shared_strings):
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return ""
        namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        lines = []
        for row in root.findall(".//s:sheetData/s:row", namespace)[:120]:
            row_values = []
            for cell in row.findall("s:c", namespace):
                cell_type = cell.attrib.get("t", "")
                value_node = cell.find("s:v", namespace)
                if cell_type == "inlineStr":
                    text_parts = [node.text for node in cell.findall(".//s:t", namespace) if node.text]
                    cell_text = "".join(text_parts).strip()
                elif cell_type == "s" and value_node is not None and value_node.text:
                    try:
                        cell_text = shared_strings[int(value_node.text)]
                    except Exception:
                        cell_text = value_node.text.strip()
                elif value_node is not None and value_node.text:
                    cell_text = value_node.text.strip()
                else:
                    cell_text = ""
                if cell_text:
                    row_values.append(cell_text)
            if row_values:
                lines.append(" | ".join(row_values))
        return "\n".join(lines).strip()

    def _read_pptx_file(self, path):
        try:
            with zipfile.ZipFile(path) as archive:
                slide_names = sorted(
                    [name for name in archive.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml")]
                )
                slide_texts = []
                for index, slide_name in enumerate(slide_names[:20], start=1):
                    try:
                        xml_bytes = archive.read(slide_name)
                    except Exception:
                        continue
                    text = self._extract_pptx_slide_text(xml_bytes)
                    if text:
                        slide_texts.append(f"[第{index}页]\n{text}")
                content = "\n\n".join(slide_texts).strip()
                if not content:
                    raise VoiceLLMControllerError("这个 PPT 文件里没有提取到可读文本。")
                return content
        except OSError as exc:
            raise VoiceLLMControllerError(f"读取 PPT 文件失败：{exc}") from exc
        except zipfile.BadZipFile as exc:
            raise VoiceLLMControllerError("这个 PPT 文件结构不完整，暂时无法解析。") from exc

    def _extract_pptx_slide_text(self, xml_bytes):
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return ""
        namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        parts = [node.text.strip() for node in root.findall(".//a:t", namespace) if node.text and node.text.strip()]
        return "\n".join(parts).strip()

    def _normalize_chat_content(self, content):
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            content = "".join(parts)

        text = str(content or "").strip()
        if not text:
            return ""

        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2:
                text = "\n".join(lines[1:-1]).strip()

        return text.strip()
