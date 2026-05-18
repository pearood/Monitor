import html
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone


class StudyAssistantError(Exception):
    pass


SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DOCUMENT_FILE_EXTENSIONS = {
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
    ".docx",
    ".pdf",
    ".xlsx",
    ".pptx",
}
SITE_SEARCH_CONFIGS = (
    {
        "key": "bilibili",
        "label": "哔哩哔哩",
        "aliases": ("bilibili", "b站", "哔哩", "哔哩哔哩", "bili"),
        "home_url": "https://www.bilibili.com",
        "search_url": "https://search.bilibili.com/all?keyword={query}",
    },
    {
        "key": "baidu",
        "label": "百度",
        "aliases": ("百度", "baidu"),
        "home_url": "https://www.baidu.com",
        "search_url": "https://www.baidu.com/s?wd={query}",
    },
    {
        "key": "bing",
        "label": "必应",
        "aliases": ("必应", "bing", "必应搜索"),
        "home_url": "https://www.bing.com",
        "search_url": "https://www.bing.com/search?q={query}&setlang=zh-Hans",
    },
    {
        "key": "zhihu",
        "label": "知乎",
        "aliases": ("知乎", "zhihu"),
        "home_url": "https://www.zhihu.com",
        "search_url": "https://www.zhihu.com/search?type=content&q={query}",
    },
    {
        "key": "mooc",
        "label": "中国大学 MOOC",
        "aliases": ("中国大学mooc", "中国大学 mooc", "mooc", "慕课"),
        "home_url": "https://www.icourse163.org",
        "search_url": "https://www.icourse163.org/search.htm?search={query}",
    },
    {
        "key": "csdn",
        "label": "CSDN",
        "aliases": ("csdn", "CSDN"),
        "home_url": "https://www.csdn.net",
        "search_url": "https://so.csdn.net/so/search?q={query}",
    },
    {
        "key": "github",
        "label": "GitHub",
        "aliases": ("github", "GitHub"),
        "home_url": "https://github.com",
        "search_url": "https://github.com/search?q={query}",
    },
)


def extract_urls(text):
    return re.findall(r"https?://[^\s)>\u3002\uff0c\uff1b]+", text or "")


def detect_web_search_command(command):
    command = (command or "").strip()
    if not command:
        return None

    explicit_urls = extract_urls(command)
    if explicit_urls and any(token in command for token in ("打开", "访问", "进入")):
        return {
            "kind": "open_web_search",
            "site_key": "url",
            "site_label": "指定网页",
            "query": "",
            "url": explicit_urls[0],
            "mode": "open",
        }

    has_open_intent = any(token in command for token in ("打开", "访问", "进入", "跳转到"))
    has_search_intent = any(token in command for token in ("搜索", "查找", "搜一下", "检索"))
    if not has_open_intent and not has_search_intent:
        return None

    site = _match_search_site(command)
    query = _extract_web_search_query(command, site)
    if not site and has_open_intent and not has_search_intent:
        return None

    if not site:
        site = _site_config_by_key("baidu")

    encoded_query = urllib.parse.quote(query)
    if query:
        target_url = site["search_url"].format(query=encoded_query)
        mode = "search"
    else:
        target_url = site["home_url"]
        mode = "open"

    return {
        "kind": "open_web_search",
        "site_key": site["key"],
        "site_label": site["label"],
        "query": query,
        "url": target_url,
        "mode": mode,
    }


def _match_search_site(command):
    command_lower = command.lower()
    normalized_command = re.sub(r"\s+", "", command_lower)
    for site in SITE_SEARCH_CONFIGS:
        for alias in site["aliases"]:
            alias_norm = re.sub(r"\s+", "", str(alias).lower())
            if alias_norm and alias_norm in normalized_command:
                return site
    return None


def _site_config_by_key(key):
    for site in SITE_SEARCH_CONFIGS:
        if site["key"] == key:
            return site
    return SITE_SEARCH_CONFIGS[1]


def _extract_web_search_query(command, site):
    text = command or ""
    quoted = re.findall(r"[“\"']([^\"'”]+)[”\"']", text)
    if quoted:
        return quoted[0].strip()

    patterns = [
        r"(?:搜索|查找|检索|搜一下)(.+?)(?:资料|内容|教程|课程)?(?:并|，|。|$)",
        r"(?:关于|有关)(.+?)(?:的资料|的内容|并|，|。|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            query = _clean_web_query(match.group(1), site)
            if query:
                return query

    query = text
    if site:
        for alias in site["aliases"]:
            query = re.sub(re.escape(str(alias)), " ", query, flags=re.IGNORECASE)
    for token in (
        "帮我",
        "请",
        "打开",
        "访问",
        "进入",
        "跳转到",
        "网站",
        "网页",
        "然后",
        "并且",
        "并",
        "搜索",
        "查找",
        "检索",
        "搜一下",
        "内容",
        "资料",
        "学习资料",
        "帮忙",
    ):
        query = query.replace(token, " ")
    return _clean_web_query(query, site)


def _clean_web_query(query, site=None):
    query = re.sub(r"https?://\S+", " ", query or "")
    if site:
        for alias in site["aliases"]:
            query = re.sub(re.escape(str(alias)), " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\s+", " ", query)
    return query.strip(" ：:，,。.!！?、")


def infer_save_mode(command):
    command = command or ""
    if "桌面" in command and "不保存到桌面" not in command:
        return "desktop"
    return "dialog"


def infer_output_format(command):
    command = (command or "").lower()
    if any(token in command for token in ("excel", ".xlsx", "xlsx", "表格", "电子表格")):
        return "xlsx"
    if any(token in command for token in ("word", ".docx", "docx", "word文档", "文档格式")):
        return "docx"
    return "md"


def infer_learning_outputs(command):
    command = command or ""
    outputs = []
    if any(token in command.lower() for token in ("excel", "xlsx")) or any(
        token in command for token in ("表格", "电子表格", "知识点表", "错题表", "计划表", "整理成表")
    ):
        outputs.append("table")
    if any(token in command for token in ("学习笔记", "笔记整理", "课堂笔记", "总结笔记")):
        outputs.append("notes")
    if any(token in command for token in ("复习提纲", "复习大纲", "知识提纲")):
        outputs.append("outline")
    if any(token in command for token in ("思维导图", "导图", "脑图")):
        outputs.append("mindmap")
    if not outputs and any(token in command for token in ("学习化处理", "整理当前页面", "整理这个文档", "整理这个文件", "总结当前页面", "总结这个文档", "总结这个文件")):
        outputs = ["notes", "outline", "mindmap"]
    deduped = []
    for item in outputs:
        if item not in deduped:
            deduped.append(item)
    return deduped


def is_current_page_command(command):
    command = command or ""
    return any(
        token in command
        for token in (
            "当前页面",
            "当前网页",
            "这个网页",
            "本页面",
            "当前标签页",
            "当前内容",
            "当前图片",
            "当前文档",
            "当前文件",
            "这个文档",
            "这个文件",
        )
    )


def is_current_tabs_command(command):
    command = command or ""
    return any(
        token in command
        for token in (
            "当前打开的页面",
            "当前打开的网页",
            "当前打开的标签页",
            "当前标签页们",
            "这些页面",
            "这些网页",
            "这些标签页",
            "所有标签页",
            "当前窗口页面",
            "当前窗口标签页",
        )
    )


def is_material_collection_command(command):
    command = command or ""
    has_material = any(token in command.lower() for token in ("excel", "xlsx")) or any(
        token in command for token in ("学习资料", "资料包", "资料整理", "学习网站", "学习文章", "表格", "电子表格")
    )
    has_collect = any(token in command for token in ("自动采集", "采集", "收集", "搜集", "抓取", "搜索"))
    return has_material and has_collect


def extract_topic(command):
    command = (command or "").strip()
    if not command:
        return ""

    quoted = re.findall(r"[“\"']([^\"'”]+)[”\"']", command)
    if quoted:
        return quoted[0].strip()

    patterns = [
        r"(?:搜索|查找)(.+?)(?:内容|资料|教程|课程|并|，|。|$)",
        r"(?:关于|围绕|主题是)(.+?)(?:的学习资料|学习资料|资料包|并|，|。|$)",
        r"(?:收集|搜集|采集|整理)(.+?)(?:的学习资料|学习资料|资料包|并|，|。|$)",
        r"(.+?)(?:学习资料|资料包)(?:并|，|。|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, command)
        if match:
            topic = match.group(1).strip(" ：:，,。.!！?")
            if topic:
                return topic

    cleaned = command
    for token in (
        "帮我",
        "请",
        "自动",
        "采集",
        "收集",
        "搜集",
        "整理",
        "生成",
        "保存到桌面",
        "保存成word",
        "保存成word文档",
        "保存为word",
        "word保存到桌面",
        "保存成excel",
        "保存成excel表格",
        "保存为excel",
        "excel保存到桌面",
        "xlsx",
        "excel",
        "表格",
        "电子表格",
        "转成",
        "学习资料",
        "资料包",
        "并保存",
        "保存",
        "网站",
        "打开",
        "内容",
    ):
        cleaned = cleaned.replace(token, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ：:，,。.!！?")
    return cleaned


def sanitize_filename(name, default="学习资料"):
    text = re.sub(r"[\\\\/:*?\"<>|]+", "_", (name or "").strip())
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text[:80] or default


def desktop_dir():
    return os.path.join(os.path.expanduser("~"), "Desktop")


def build_default_study_filename(prefix, topic_or_title, extension="md"):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = sanitize_filename(topic_or_title or prefix)
    ext = extension.lstrip(".") or "md"
    return f"{prefix}_{safe}_{stamp}.{ext}"


def build_default_sources_filename(prefix, topic_or_title):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe = sanitize_filename(topic_or_title or prefix)
    return f"{prefix}_{safe}_{stamp}_来源.txt"


def save_text(path, content):
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(content)
    return target


def save_document(path, content):
    target = os.path.abspath(path)
    ext = os.path.splitext(target)[1].lower()
    if ext == ".xlsx":
        return _save_xlsx(target, content)
    if ext == ".docx":
        return _save_docx(target, content)
    return save_text(target, content)


def search_learning_sources(topic, command="", limit=4, timeout=10):
    query = _build_search_query(topic, command)
    site_hints = extract_site_preferences(command)
    results = _filter_search_results(_search_bing_rss(query, limit=limit, timeout=timeout), site_hints)
    if results:
        return results
    results = _filter_search_results(_search_bing(query, limit=limit, timeout=timeout), site_hints)
    if results:
        return results
    return _filter_search_results(_search_duckduckgo(query, limit=limit, timeout=timeout), site_hints)


def build_curated_learning_sources(topic, command="", limit=4):
    topic = (topic or "").strip()
    encoded = urllib.parse.quote(topic or "学习资料")
    site_hints = extract_site_preferences(command)
    curated = []

    if "bilibili" in site_hints:
        curated.append(
            {
                "title": f"Bilibili 搜索：{topic}",
                "url": f"https://search.bilibili.com/all?keyword={encoded}",
                "snippet": "适合找课程讲解、例题视频、老师专题合集。",
            }
        )
    if "zhihu" in site_hints:
        curated.append(
            {
                "title": f"知乎搜索：{topic}",
                "url": f"https://www.zhihu.com/search?type=content&q={encoded}",
                "snippet": "适合找知识点解释、答疑讨论与经验总结。",
            }
        )
    if "mooc" in site_hints:
        curated.append(
            {
                "title": f"中国大学 MOOC 搜索：{topic}",
                "url": f"https://www.icourse163.org/search.htm?search={encoded}",
                "snippet": "适合找系统课程和章节式学习资源。",
            }
        )

    defaults = [
        {
            "title": f"Bilibili 搜索：{topic}",
            "url": f"https://search.bilibili.com/all?keyword={encoded}",
            "snippet": "适合找课程视频、专题讲解、例题解析。",
        },
        {
            "title": f"知乎搜索：{topic}",
            "url": f"https://www.zhihu.com/search?type=content&q={encoded}",
            "snippet": "适合找概念解释、学习路线和常见问题。",
        },
        {
            "title": f"百度搜索：{topic} 学习资料",
            "url": f"https://www.baidu.com/s?wd={urllib.parse.quote((topic or '学习资料') + ' 学习资料')}",
            "snippet": "适合补充网页文章、课件和资料入口。",
        },
        {
            "title": f"必应搜索：{topic} 教程",
            "url": f"https://www.bing.com/search?q={urllib.parse.quote((topic or '学习资料') + ' 教程')}",
            "snippet": "适合找不同网站的教程和知识点总结。",
        },
    ]

    seen = set()
    results = []
    for item in curated + defaults:
        key = item["url"]
        if key in seen:
            continue
        seen.add(key)
        results.append(item)
        if len(results) >= limit:
            break
    return results


def extract_site_preferences(command):
    command = (command or "").lower()
    sites = []
    if any(token in command for token in ("bilibili", "b站", "哔哩", "哔哩哔哩")):
        sites.append("bilibili")
    if "知乎" in command or "zhihu" in command:
        sites.append("zhihu")
    if any(token in command for token in ("mooc", "中国大学mooc", "慕课")):
        sites.append("mooc")
    return sites


def _build_search_query(topic, command=""):
    topic = (topic or "").strip()
    query = f"{topic} 学习资料 教程 知识点 例题"
    sites = extract_site_preferences(command)
    if "bilibili" in sites:
        return f"site:bilibili.com {topic} 课程 讲解"
    if "zhihu" in sites:
        return f"site:zhihu.com {topic} 知识点 学习"
    if "mooc" in sites:
        return f"site:icourse163.org {topic} 课程"
    return query


def _filter_search_results(results, site_hints):
    results = list(results or [])
    if not results:
        return []
    if not site_hints:
        return results

    allowed_domains = []
    if "bilibili" in site_hints:
        allowed_domains.extend(["bilibili.com", "b23.tv"])
    if "zhihu" in site_hints:
        allowed_domains.append("zhihu.com")
    if "mooc" in site_hints:
        allowed_domains.extend(["icourse163.org", "coursera.org", "edx.org"])

    if not allowed_domains:
        return results

    filtered = []
    for item in results:
        url = (item or {}).get("url", "")
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        if any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains):
            filtered.append(item)
    return filtered


def _search_bing_rss(query, limit=4, timeout=10):
    encoded = urllib.parse.quote(query)
    url = f"https://www.bing.com/search?format=rss&q={encoded}&setlang=zh-Hans"
    request = urllib.request.Request(url, headers={"User-Agent": SEARCH_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            xml_text = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    items = re.findall(r"<item>(.*?)</item>", xml_text, re.S)
    results = []
    for item in items:
        title_match = re.search(r"<title>(.*?)</title>", item, re.S)
        link_match = re.search(r"<link>(.*?)</link>", item, re.S)
        desc_match = re.search(r"<description>(.*?)</description>", item, re.S)
        title = _clean_html_text(title_match.group(1) if title_match else "")
        link = (link_match.group(1).strip() if link_match else "")
        snippet = _clean_html_text(desc_match.group(1) if desc_match else "")
        if not title or not link:
            continue
        results.append({"title": title, "url": link, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def _search_bing(query, limit=4, timeout=10):
    encoded = urllib.parse.quote(query)
    url = f"https://www.bing.com/search?q={encoded}&setlang=zh-Hans"
    request = urllib.request.Request(url, headers={"User-Agent": SEARCH_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            html_text = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    pattern = re.compile(
        r'<li class="b_algo".*?<h2><a href="(https?://[^"]+)"[^>]*>(.*?)</a>.*?(?:<p>(.*?)</p>)?',
        re.S,
    )
    results = []
    for link, raw_title, raw_snippet in pattern.findall(html_text):
        title = _clean_html_text(raw_title)
        snippet = _clean_html_text(raw_snippet)
        if not title or not link:
            continue
        results.append({"title": title, "url": link, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def _search_duckduckgo(query, limit=4, timeout=10):
    encoded = urllib.parse.quote(query)
    url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
    request = urllib.request.Request(url, headers={"User-Agent": SEARCH_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            html_text = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    pattern = re.compile(
        r'<a[^>]+href="(https?://[^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>',
        re.S,
    )
    results = []
    for link, raw_title in pattern.findall(html_text):
        title = _clean_html_text(raw_title)
        if not title or not link:
            continue
        results.append({"title": title, "url": link, "snippet": ""})
        if len(results) >= limit:
            break
    return results


def fetch_page_content(url, timeout=10, max_chars=5000):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": SEARCH_USER_AGENT,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        html_bytes = response.read()
    html_text = html_bytes.decode(charset, errors="ignore")
    title = _extract_title(html_text) or url
    content = _html_to_text(html_text)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    if not content:
        raise StudyAssistantError("网页正文为空或暂时无法提取。")
    return {
        "title": title,
        "url": url,
        "content": content[:max_chars],
    }


def _extract_title(html_text):
    match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.I | re.S)
    if not match:
        return ""
    return _clean_html_text(match.group(1))


def _html_to_text(html_text):
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", text)
    text = re.sub(r"(?is)<br\\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p>", "\n", text)
    text = re.sub(r"(?is)</div>", "\n", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text


def _clean_html_text(text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<.*?>", " ", text or ""))).strip()


def get_active_browser_tabs(timeout=4, limit=6):
    scripts = [
        """
        tell application "Google Chrome"
            if it is running then
                if (count of windows) > 0 then
                    set outLines to {}
                    repeat with theTab in tabs of front window
                        set end of outLines to ((title of theTab) & tab & (URL of theTab))
                    end repeat
                    return outLines as text
                end if
            end if
        end tell
        """,
        """
        tell application "Safari"
            if it is running then
                if (count of windows) > 0 then
                    set outLines to {}
                    repeat with theTab in tabs of front window
                        set end of outLines to ((name of theTab) & tab & (URL of theTab))
                    end repeat
                    return outLines as text
                end if
            end if
        end tell
        """,
    ]
    parsed_tabs = []
    for script in scripts:
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except Exception:
            continue
        if result.returncode != 0:
            continue
        for line in (result.stdout or "").splitlines():
            raw = line.strip()
            if not raw:
                continue
            if "\t" in raw:
                title, url = raw.split("\t", 1)
            else:
                title, url = "", raw
            url = (url or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            parsed_tabs.append({"title": (title or "").strip() or "未命名页面", "url": url})
            if len(parsed_tabs) >= max(1, int(limit)):
                return parsed_tabs
        if parsed_tabs:
            return parsed_tabs
    raise StudyAssistantError("暂时没有读取到当前浏览器标签页，请先把浏览器切到目标页面。")


def _column_name(index):
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name or "A"


def _clean_cell_text(value):
    text = str(value or "")
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)
    return text.strip()


def _split_markdown_table_line(line):
    text = line.strip()
    if not text.startswith("|") or text.count("|") < 2:
        return []
    cells = [cell.strip() for cell in text.strip("|").split("|")]
    return [_clean_cell_text(cell) for cell in cells]


def _is_markdown_separator_row(cells):
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def _extract_markdown_tables(content):
    tables = []
    active_rows = []
    active_title = ""
    latest_heading = ""

    def finish_table():
        nonlocal active_rows, active_title
        if active_rows:
            tables.append((active_title or latest_heading, active_rows))
        active_rows = []
        active_title = ""

    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line:
            finish_table()
            continue
        heading = re.sub(r"^#{1,6}\s*", "", line).strip()
        if heading != line:
            finish_table()
            latest_heading = heading
            continue
        cells = _split_markdown_table_line(line)
        if cells:
            if _is_markdown_separator_row(cells):
                continue
            if not active_rows:
                active_title = latest_heading
            active_rows.append(cells)
        else:
            finish_table()
    finish_table()
    return tables


def _content_to_xlsx_rows(content):
    tables = _extract_markdown_tables(content)
    if tables:
        rows = []
        for table_index, (title, table_rows) in enumerate(tables):
            if table_index > 0:
                rows.append([""])
            if title:
                rows.append([title])
            rows.extend(table_rows)
        return rows

    rows = [["模块", "项目", "内容"]]
    current_section = "学习资料"
    in_code_block = False
    for raw_line in (content or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        heading = re.sub(r"^#{1,6}\s*", "", line).strip()
        if heading != line and heading:
            current_section = heading
            continue
        cleaned = re.sub(r"^[\-\*\d\.\)\(、\s]+", "", line).strip()
        if not cleaned:
            continue
        key = ""
        value = cleaned
        split_match = re.match(r"^([^：:]{1,24})[：:]\s*(.+)$", cleaned)
        if split_match:
            key = split_match.group(1).strip()
            value = split_match.group(2).strip()
        rows.append([current_section, key or cleaned[:24], value])
        if len(rows) >= 301:
            break

    if len(rows) == 1:
        rows.append(["学习资料", "内容", _clean_cell_text(content or "暂无可写入的内容")])
    return rows


def _xlsx_cell_xml(row_index, col_index, value, style_id):
    cell_ref = f"{_column_name(col_index)}{row_index}"
    safe_value = html.escape(_clean_cell_text(value))
    return (
        f'<c r="{cell_ref}" t="inlineStr" s="{style_id}">'
        f'<is><t xml:space="preserve">{safe_value}</t></is>'
        "</c>"
    )


def _save_xlsx(path, content):
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    rows = _content_to_xlsx_rows(content)
    max_cols = max((len(row) for row in rows), default=1)
    max_cols = max(1, min(max_cols, 12))
    normalized_rows = []
    for row in rows[:500]:
        normalized = list(row[:max_cols])
        normalized.extend([""] * (max_cols - len(normalized)))
        normalized_rows.append(normalized)
    if not normalized_rows:
        normalized_rows = [["内容"], ["暂无可写入的内容"]]
        max_cols = 1

    col_widths = []
    for col_index in range(max_cols):
        max_len = 0
        for row in normalized_rows[:120]:
            text = _clean_cell_text(row[col_index])
            weighted_len = sum(2 if ord(ch) > 127 else 1 for ch in text)
            max_len = max(max_len, min(weighted_len, 80))
        col_widths.append(max(10, min(42, int(max_len * 0.7) + 4)))

    dimension = f"A1:{_column_name(max_cols)}{len(normalized_rows)}"
    header_row_index = 1
    for index, row in enumerate(normalized_rows, start=1):
        row_text = "|".join(_clean_cell_text(value) for value in row)
        if "模块" in row_text and ("知识点" in row_text or "内容" in row_text):
            header_row_index = index
            break
    filter_ref = f"A{header_row_index}:{_column_name(max_cols)}{len(normalized_rows)}"
    cols_xml = "".join(
        f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>'
        for idx, width in enumerate(col_widths, start=1)
    )
    row_xml_parts = []
    for row_index, row in enumerate(normalized_rows, start=1):
        row_text = "|".join(_clean_cell_text(value) for value in row)
        is_header_like = row_index == 1 or ("模块" in row_text and ("知识点" in row_text or "内容" in row_text))
        style_id = "1" if is_header_like else "2"
        cells_xml = "".join(_xlsx_cell_xml(row_index, col_index, value, style_id) for col_index, value in enumerate(row, start=1))
        row_xml_parts.append(f'<row r="{row_index}" ht="24" customHeight="1">{cells_xml}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '<selection pane="bottomLeft"/>'
        '</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="18"/>'
        f"<cols>{cols_xml}</cols>"
        f"<sheetData>{''.join(row_xml_parts)}</sheetData>"
        '<autoFilter ref="' + filter_ref + '"/>'
        "</worksheet>"
    )
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""",
        )
        workbook.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        )
        workbook.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="学习表格" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""",
        )
        workbook.writestr(
            "xl/styles.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2"><font><sz val="11"/><name val="Microsoft YaHei"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Microsoft YaHei"/></font></fonts>
<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF2C3E50"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border><border><left style="thin"><color rgb="FFD9E1E8"/></left><right style="thin"><color rgb="FFD9E1E8"/></right><top style="thin"><color rgb="FFD9E1E8"/></top><bottom style="thin"><color rgb="FFD9E1E8"/></bottom><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf></cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>""",
        )
        workbook.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        workbook.writestr(
            "docProps/core.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>小睿伴学学习表格</dc:title>
<dc:creator>小睿伴学</dc:creator>
<cp:lastModifiedBy>小睿伴学</cp:lastModifiedBy>
<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""",
        )
        workbook.writestr(
            "docProps/app.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
<Application>小睿伴学</Application>
</Properties>""",
        )
    return target


def infer_attachment_kind_from_path(path):
    ext = os.path.splitext((path or "").strip())[1].lower()
    if ext in IMAGE_FILE_EXTENSIONS:
        return "image"
    if ext in DOCUMENT_FILE_EXTENSIONS:
        return "file"
    return None


def _save_docx(path, content):
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    paragraphs = [line.rstrip() for line in (content or "").splitlines()]
    if not paragraphs:
        paragraphs = [""]

    doc_lines = []
    for line in paragraphs:
        safe = (
            html.escape(line or "")
            .replace("\n", "")
            .replace("\r", "")
        )
        if not safe:
            doc_lines.append("<w:p/>")
        else:
            doc_lines.append(
                "<w:p><w:r><w:t xml:space=\"preserve\">"
                f"{safe}"
                "</w:t></w:r></w:p>"
            )
    document_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 wp14\">"
        "<w:body>"
        + "".join(doc_lines)
        + "<w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/><w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/></w:sectPr>"
        "</w:body></w:document>"
    )

    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""",
        )
        docx.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        )
        docx.writestr(
            "docProps/core.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>小睿伴学学习文档</dc:title>
<dc:creator>小睿伴学</dc:creator>
<cp:lastModifiedBy>小睿伴学</cp:lastModifiedBy>
<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""",
        )
        docx.writestr(
            "docProps/app.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
<Application>小睿伴学</Application>
</Properties>""",
        )
        docx.writestr("word/document.xml", document_xml)
    return target
