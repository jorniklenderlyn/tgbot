"""
Kaiten import script — создаёт доску, эпики и задачи для ЛР3.

Запуск:
    pip install requests
    python kaiten_import.py
"""

import requests
import sys
import json

# ── Конфигурация ──────────────────────────────────────────────────────────────
API_TOKEN = "974a2f3a-2666-4e68-90e3-d0a9b8ed88ae"
BASE_URL   = "https://aissistant.kaiten.ru/api/latest"
SPACE_ID   = 790412

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}

# ── Вспомогательные функции ──────────────────────────────────────────────────
def api(method, path, **kwargs):
    r = requests.request(method, f"{BASE_URL}{path}", headers=HEADERS, **kwargs)
    if not r.ok:
        print(f"  ✗ {method} {path} → {r.status_code}: {r.text[:300]}")
        return None
    return r.json()

def get(path, **kw):    return api("GET",    path, **kw)
def post(path, **kw):   return api("POST",   path, json=kw.get("data"), **{k:v for k,v in kw.items() if k!="data"})
def put(path, **kw):    return api("PUT",    path, json=kw.get("data"), **{k:v for k,v in kw.items() if k!="data"})

def attach_tag(card_id: int, tag_id: int, tag_name: str):
    """Привязывает тег к карточке через обновление карточки (надёжный способ)."""
    card = get(f"/cards/{card_id}")
    if not card:
        return
    existing_ids = {t["id"] for t in card.get("tags", [])}
    if tag_id in existing_ids:
        return  # уже есть
    new_tags = card.get("tags", []) + [{"id": tag_id, "name": tag_name}]
    r = put(f"/cards/{card_id}", data={"tags": new_tags})
    if not r:
        # fallback: попробуем POST с обоими полями
        api("POST", f"/cards/{card_id}/tags",
            json={"id": tag_id, "name": tag_name})

# ── Данные ────────────────────────────────────────────────────────────────────
COLUMNS = [
    "Бэклог",
    "К работе / To Do",
    "В работе",
    "На проверке",
    "Готово",
]

EPICS = [
    {
        "name": "E1 — Сбор и обработка данных",
        "description": (
            "Всё, что связано с подготовкой обучающих данных: импорт JSON-выгрузок из Telegram Desktop, "
            "анонимизация переписки (замена имён/дат/мест на плейсхолдеры), извлечение пар "
            "«входящее → ответ пользователя» и построение векторного few-shot индекса в Qdrant.\n\n"
            "Необходим для: E2 (Persona Extraction), E4 (Response Pipeline)."
        ),
    },
    {
        "name": "E2 — Профиль стиля (Persona Extraction)",
        "description": (
            "CLI-инструмент extract-persona, анализирующий историю переписки и формирующий JSON-профиль стиля "
            "пользователя (persona.json). Детерминированные признаки — через regex; речевые паттерны — через LLM. "
            "Результат используется как системная часть промпта при генерации ответов.\n\n"
            "Зависит от: E1. Необходим для: E4."
        ),
    },
    {
        "name": "E3 — Долговременная память (Mem0 Self-hosted)",
        "description": (
            "Локальная система памяти ассистента на базе Mem0 (self-hosted): Qdrant (Docker, векторный поиск), "
            "Neo4j (Docker, граф связей), SQLite (структурированные факты). "
            "Хранит факты о собеседниках и пользователе, автоматически пополняется после каждого ответа. "
            "Все данные остаются на устройстве пользователя.\n\n"
            "Необходим для: E4."
        ),
    },
    {
        "name": "E4 — Конвейер генерации ответов (Response Pipeline)",
        "description": (
            "Центральный pipeline: детекция чувствительных тем → few-shot поиск в Qdrant → "
            "извлечение памяти из Mem0 → сборка промпта (persona + few-shot + контекст + воспоминания) → "
            "вызов LLM API → формирование черновика. "
            "Устойчивость к сбоям: очередь в SQLite, retry с exponential backoff (max 5 мин).\n\n"
            "Зависит от: E1, E2, E3. Необходим для: E5.\n\n"
            "Состав: T4.1 MessageClassifier, T4.2 FewShotRetriever, T4.3 PromptBuilder, "
            "T4.4 LLMClient, T4.5 DraftManager, T4.6 MessageQueue, T4.6.1 BackoffRetry."
        ),
    },
    {
        "name": "E5 — Telegram-интеграция (Telethon + Aiogram)",
        "description": (
            "Два компонента: MTProto-клиент (Telethon) слушает входящие из whitelist-чатов и отправляет ответы; "
            "управляющий Aiogram-бот доставляет черновики владельцу с кнопками «Отправить / Редактировать / Отклонить», "
            "управляет whitelist'ом, режимами auto/review, уведомляет о сбоях и чувствительных сообщениях.\n\n"
            "Зависит от: E4."
        ),
    },
]

TASKS = [
    {
        "title": "T4.1 — MessageClassifier: детекция чувствительных тем",
        "epic_name": "E4 — Конвейер генерации ответов (Response Pipeline)",
        "points": 3,
        "tech_tags": ["Python", "Regex", "LLM"],
        "description": (
            "Реализовать модуль `MessageClassifier` в pipeline/classifier.py.\n\n"
            "Анализирует входящее сообщение и определяет чувствительные темы: "
            "финансовые обязательства, даты встреч, обещания/договорённости. "
            "При is_sensitive=True черновик получает requires_confirmation=True и никогда не отправляется автоматически.\n\n"
            "Двухуровневая логика:\n"
            "1. Regex-проверка (без LLM) — 90% случаев\n"
            "2. LLM-fallback — только для пограничных случаев\n\n"
            "Что разработать:\n"
            "- Датакласс ClassificationResult(is_sensitive: bool, categories: list[str])\n"
            "- Словарь PATTERNS с категориями: finance, commitment, datetime\n"
            "- Функцию classify(text: str) -> ClassificationResult\n\n"
            "Критерий приёмки:\n"
            "- Regex покрывает: суммы денег, даты/время, слова-обязательства\n"
            "- Тест: 10 чувствительных → все True, 10 нейтральных → все False\n"
            "- Без вызова LLM при явном совпадении regex\n\n"
            "Зависит от: —\n"
            "Влияет на: T4.3 (PromptBuilder использует ClassificationResult)"
        ),
    },
    {
        "title": "T4.2 — FewShotRetriever: поиск примеров в Qdrant",
        "epic_name": "E4 — Конвейер генерации ответов (Response Pipeline)",
        "points": 5,
        "tech_tags": ["Qdrant", "sentence-transformers", "Python"],
        "description": (
            "Реализовать FewShotRetriever в pipeline/retriever.py.\n\n"
            "По тексту входящего сообщения ищет top-K семантически похожих пар «входящее → ответ» "
            "в Qdrant (коллекция fewshot_pairs, проиндексированная в E1).\n\n"
            "Ключевое поведение: если max(score) < MIN_SCORE (0.65) → low_confidence=True, "
            "pipeline не генерирует черновик, владелец получает уведомление.\n\n"
            "Что разработать:\n"
            "- Датакласс FewShotExample(incoming, reply, score)\n"
            "- Датакласс RetrievalResult(examples: list, low_confidence: bool)\n"
            "- Класс FewShotRetriever с методом retrieve(query, top_k=5)\n"
            "- Векторизация через sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\n\n"
            "Критерий приёмки:\n"
            "- При score < 0.65 → RetrievalResult(examples=[], low_confidence=True)\n"
            "- Время поиска < 300 мс на коллекции до 10 000 пар\n"
            "- Подключение через env: QDRANT_HOST, QDRANT_PORT\n\n"
            "Зависит от: E1 (индекс в Qdrant)\n"
            "Влияет на: T4.3 (PromptBuilder использует примеры)"
        ),
    },
    {
        "title": "T4.3 — PromptBuilder: сборка промпта",
        "epic_name": "E4 — Конвейер генерации ответов (Response Pipeline)",
        "points": 3,
        "tech_tags": ["Python", "tiktoken", "LLM"],
        "description": (
            "Реализовать PromptBuilder в pipeline/prompt_builder.py.\n\n"
            "Собирает финальный промпт для LLM из 4 блоков:\n"
            "1. Persona — системная инструкция из persona.json (E2)\n"
            "2. Few-shot примеры — пары из Qdrant (T4.2)\n"
            "3. Контекст диалога — последние N сообщений\n"
            "4. Воспоминания из Mem0 (E3) о собеседнике и пользователе\n\n"
            "Ограничение по токенам: промпт ≤ max_tokens (6000). "
            "При превышении — обрезать few-shot (старые первыми), затем контекст. "
            "Persona и воспоминания не обрезаются.\n\n"
            "При is_sensitive=True — добавить в промпт: «Не бери на себя финансовых или временных обязательств».\n\n"
            "Метод: build(persona, examples, context, memories, classification, max_tokens=6000) -> str\n\n"
            "Критерий приёмки:\n"
            "- Промпт ≤ max_tokens (проверяется через tiktoken)\n"
            "- Few-shot форматируются как «Входящее: ...\nОтвет: ...»\n"
            "- Воспоминания добавляются только если len(memories) > 0\n\n"
            "Зависит от: T4.1 (ClassificationResult), T4.2 (примеры)\n"
            "Влияет на: T4.4 (LLMClient получает готовый промпт)"
        ),
    },
    {
        "title": "T4.4 — LLMClient: вызов LLM с retry/backoff",
        "epic_name": "E4 — Конвейер генерации ответов (Response Pipeline)",
        "points": 5,
        "tech_tags": ["OpenRouter", "OpenAI SDK", "Python"],
        "description": (
            "Реализовать LLMClient в pipeline/llm_client.py — обёртку над OpenRouter (OpenAI-совместимый API).\n\n"
            "Принимает промпт из PromptBuilder (T4.3) и возвращает текст черновика.\n\n"
            "Поведение при сбоях:\n"
            "- Retry с exponential backoff: 1с → 2с → 4с → 8с → 16с (max 5 попыток)\n"
            "- После 5 неудач → raise LLMUnavailableError → MessageQueue (T4.6) ставит в очередь\n"
            "- При latency > 10 сек → callback on_slow() → уведомление владельца\n\n"
            "Что разработать:\n"
            "- Датакласс LLMResponse(text, latency_ms, model)\n"
            "- Исключение LLMUnavailableError\n"
            "- Класс LLMClient с методом complete(prompt, model) -> LLMResponse\n"
            "- Конфигурация через .env: OPENROUTER_API_KEY, LLM_MODEL\n\n"
            "Критерий приёмки:\n"
            "- В API уходит только анонимизированный промпт (без сырой переписки)\n"
            "- Latency логируется при каждом вызове\n"
            "- API-ключ из env, не хардкодится\n\n"
            "Зависит от: T4.3 (готовый промпт)\n"
            "Влияет на: T4.5 (DraftManager), T4.6 (MessageQueue)"
        ),
    },
    {
        "title": "T4.5 — DraftManager: хранение черновиков в SQLite",
        "epic_name": "E4 — Конвейер генерации ответов (Response Pipeline)",
        "points": 3,
        "tech_tags": ["SQLite", "Python"],
        "description": (
            "Реализовать DraftManager в pipeline/draft_manager.py — хранилище черновиков на базе SQLite.\n\n"
            "Связующее звено между LLMClient (T4.4) и Aiogram-ботом (E5): "
            "pipeline сохраняет черновик, бот читает pending-черновики и доставляет владельцу.\n\n"
            "Схема таблицы drafts:\n"
            "- draft_id (UUID), chat_id, incoming_text, draft_text\n"
            "- requires_confirmation (bool), low_confidence (bool)\n"
            "- status: pending / sent / edited / rejected\n"
            "- created_at (ISO 8601)\n\n"
            "Методы: save(), get(draft_id), update_status(draft_id, status), get_pending()\n\n"
            "Критерий приёмки:\n"
            "- Черновик не теряется при перезапуске (персистентность)\n"
            "- get_pending() возвращает только status=pending\n"
            "- requires_confirmation проставляется из ClassificationResult.is_sensitive\n"
            "- Таблица создаётся автоматически (CREATE TABLE IF NOT EXISTS)\n\n"
            "Зависит от: T4.4 (LLMResponse)\n"
            "Влияет на: E5 (Aiogram-бот доставляет черновики)"
        ),
    },
    {
        "title": "T4.6 — MessageQueue: устойчивость к сбоям",
        "epic_name": "E4 — Конвейер генерации ответов (Response Pipeline)",
        "points": 8,
        "tech_tags": ["SQLite", "asyncio", "Python"],
        "description": (
            "Реализовать MessageQueue в pipeline/message_queue.py — очередь входящих сообщений на SQLite.\n\n"
            "Логика:\n"
            "1. Входящее сообщение сразу попадает в очередь (status=queued)\n"
            "2. Pipeline пытается обработать немедленно\n"
            "3. Успех → status=done\n"
            "4. LLMUnavailableError → status=failed, retry_count+1, next_retry_at = now + backoff\n"
            "5. Фоновый воркер каждые 30 сек повторяет обработку готовых к retry\n"
            "6. После 5 неудач → status=dead, уведомление владельцу\n\n"
            "Задержки retry: 15с → 30с → 60с → 120с → 300с (max 5 мин)\n\n"
            "Схема таблицы message_queue:\n"
            "- msg_id (UUID), chat_id, text, status, retry_count, next_retry_at, received_at\n\n"
            "Методы: enqueue(), mark_done(), mark_failed(), get_ready_for_retry()\n\n"
            "Дочерняя задача: T4.6.1 ExponentialBackoffRetry\n\n"
            "Зависит от: T4.4 (LLMUnavailableError как сигнал сбоя)\n"
            "Влияет на: —"
        ),
    },
    {
        "title": "T4.6.1 — ExponentialBackoffRetry [дочерняя к T4.6]",
        "epic_name": "E4 — Конвейер генерации ответов (Response Pipeline)",
        "points": 2,
        "tech_tags": ["Python", "asyncio"],
        "description": (
            "Реализовать утилиту with_retry в utils/retry.py — декоратор с exponential backoff.\n\n"
            "Используется в:\n"
            "- LLMClient (T4.4) — retry при вызовах к OpenRouter API\n"
            "- MessageQueue (T4.6) — расчёт задержки next_retry_at\n\n"
            "Поведение:\n"
            "- Задержки: base * 2^(attempt-1), ограничены max_delay\n"
            "- Jitter ±20% для предотвращения thundering herd\n"
            "- Параметр exceptions — tuple исключений для retry (остальные пробрасываются)\n"
            "- Работает для sync и async функций\n\n"
            "Функции:\n"
            "- compute_delay(attempt, base=1.0, max_delay=30.0) -> float\n"
            "- Декоратор @with_retry(max_attempts, base_delay, max_delay, exceptions)\n\n"
            "Критерий приёмки:\n"
            "- При исчерпании попыток — пробрасывается исходное исключение (не обёртка)\n"
            "- Каждая попытка логируется: attempt N/M failed: <error>. Retry in X.Xs\n"
            "- Jitter: задержки не строго детерминированы\n\n"
            "Зависит от: —\n"
            "Влияет на: T4.4 (LLMClient), T4.6 (MessageQueue)"
        ),
    },
]

# ── Основной скрипт ──────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-epics", action="store_true",
                        help="Пропустить создание карточек эпиков (если уже созданы)")
    parser.add_argument("--skip-tasks", action="store_true",
                        help="Пропустить создание задач")
    parser.add_argument("--patch-tags", action="store_true",
                        help="Только добавить теги к уже созданным карточкам (по ID из --epic-ids и --task-ids)")
    parser.add_argument("--epic-ids", nargs="+", type=int, default=[65498531,65498532,65498533,65498534,65498535],
                        help="ID карточек эпиков (по порядку E1..E5)")
    parser.add_argument("--task-ids", nargs="+", type=int, default=[65498536,65498537,65498538,65498539,65498541,65498542],
                        help="ID карточек задач (по порядку T4.1..T4.6)")
    args = parser.parse_args()

    print("🚀 Kaiten import — reply-as-me ЛР3\n")

    # 1. Найти борд в пространстве
    print("1. Ищем доску в пространстве...")
    boards = get(f"/spaces/{SPACE_ID}/boards")
    if not boards:
        boards = get("/boards", params={"space_id": SPACE_ID})

    if not boards:
        print("  ✗ Не удалось получить список досок. Проверь токен и space_id.")
        sys.exit(1)

    if isinstance(boards, list) and boards:
        board = boards[0]
    elif isinstance(boards, dict) and "data" in boards:
        board = boards["data"][0]
    else:
        print(f"  ✗ Неожиданный формат ответа: {str(boards)[:200]}")
        sys.exit(1)

    board_id = board["id"]
    print(f"  ✓ Доска найдена: «{board.get('title', board_id)}» (id={board_id})")

    # 2. Получить существующие колонки
    print("\n2. Получаем колонки...")
    cols_resp = get(f"/boards/{board_id}/columns")
    existing_cols = {}
    if cols_resp:
        col_list = cols_resp if isinstance(cols_resp, list) else cols_resp.get("data", [])
        for c in col_list:
            existing_cols[c["title"]] = c["id"]
    print(f"  Существующие: {list(existing_cols.keys()) or '(нет)'}")

    col_ids = {}
    for col_name in COLUMNS:
        if col_name in existing_cols:
            col_ids[col_name] = existing_cols[col_name]
            print(f"  ✓ Колонка уже есть: «{col_name}»")
        else:
            r = post(f"/boards/{board_id}/columns", data={"title": col_name})
            if r:
                col_ids[col_name] = r["id"]
                print(f"  ✓ Создана колонка: «{col_name}»")
            else:
                print(f"  ✗ Не удалось создать колонку «{col_name}»")

    backlog_col_id = col_ids.get("Бэклог")
    if not backlog_col_id:
        print("  ✗ Колонка «Бэклог» не найдена, используем первую доступную")
        backlog_col_id = list(col_ids.values())[0]

    # 3. Создать теги для эпиков
    print("\n3. Создаём теги для эпиков...")
    existing_tags = get("/tags") or []
    if isinstance(existing_tags, dict):
        existing_tags = existing_tags.get("data", [])
    existing_tag_names = {t["name"]: t["id"] for t in existing_tags}

    tag_ids = {}
    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
    for i, epic in enumerate(EPICS):
        name = epic["name"]
        if name in existing_tag_names:
            tag_ids[name] = existing_tag_names[name]
            print(f"  ✓ Тег уже есть: «{name}»")
        else:
            r = post("/tags", data={"name": name, "color": colors[i % len(colors)]})
            if r:
                tag_ids[name] = r["id"]
                print(f"  ✓ Создан тег-эпик: «{name}»")
            else:
                print(f"  ✗ Не удалось создать тег «{name}»")

    # 4. Создать карточки-эпики
    print("\n4. Создаём карточки эпиков...")
    epic_card_ids = {}
    for epic in EPICS:
        r = post("/cards", data={
            "board_id": board_id,
            "column_id": backlog_col_id,
            "title": epic["name"],
            "description": epic["description"],
        })
        if r:
            card_id = r["id"]
            epic_card_ids[epic["name"]] = card_id
            print(f"  ✓ Карточка эпика: «{epic['name']}» (id={card_id})")
            # Добавляем тег к карточке
            if epic["name"] in tag_ids:
                attach_tag(card_id, tag_ids[epic["name"]], epic["name"])
        else:
            print(f"  ✗ Не удалось создать карточку «{epic['name']}»")

    # 5. Создать теги технологий
    print("\n5. Создаём теги технологий...")
    all_tech = sorted({t for task in TASKS for t in task.get("tech_tags", [])})
    tech_colors = {
        "Python":               "#3b82f6",
        "Regex":                "#64748b",
        "LLM":                  "#a855f7",
        "Qdrant":               "#06b6d4",
        "sentence-transformers":"#0ea5e9",
        "tiktoken":             "#f97316",
        "OpenRouter":           "#10b981",
        "OpenAI SDK":           "#059669",
        "SQLite":               "#78716c",
        "asyncio":              "#ec4899",
    }
    tech_tag_ids = {}
    for tech in all_tech:
        if tech in existing_tag_names:
            tech_tag_ids[tech] = existing_tag_names[tech]
            print(f"  ✓ Тег уже есть: «{tech}»")
        else:
            color = tech_colors.get(tech, "#94a3b8")
            r = post("/tags", data={"name": tech, "color": color})
            if r:
                tech_tag_ids[tech] = r["id"]
                print(f"  ✓ Создан тег технологии: «{tech}»")
            else:
                print(f"  ✗ Не удалось создать тег «{tech}»")

    # 6. Создать задачи
    print("\n6. Создаём задачи...")
    task_card_ids = {}
    for task in TASKS:
        epic_tag_id = tag_ids.get(task["epic_name"])
        r = post("/cards", data={
            "board_id": board_id,
            "column_id": backlog_col_id,
            "title": task["title"],
            "description": task["description"],
            "size": task.get("points", 1),          # story points
        })
        if r:
            card_id = r["id"]
            task_card_ids[task["title"]] = card_id
            pts = task.get("points", "?")
            print(f"  ✓ Задача [{pts}sp]: «{task['title']}» (id={card_id})")
            # Эпик-тег
            if epic_tag_id:
                attach_tag(card_id, epic_tag_id, task["epic_name"])
            # Теги технологий
            for tech in task.get("tech_tags", []):
                if tech in tech_tag_ids:
                    attach_tag(card_id, tech_tag_ids[tech], tech)
        else:
            print(f"  ✗ Не удалось создать задачу «{task['title']}»")

    # patch-tags: добавить теги к уже существующим карточкам
    if args.patch_tags:
        print("\n📌 Режим patch-tags: добавляем теги к существующим карточкам...")
        for i, (epic, card_id) in enumerate(zip(EPICS, args.epic_ids)):
            if epic["name"] in tag_ids:
                attach_tag(card_id, tag_ids[epic["name"]], epic["name"])
                print(f"  ✓ Эпик-тег на карточке {card_id}: «{epic['name']}»")
        for i, (task, card_id) in enumerate(zip(TASKS, args.task_ids)):
            if task["epic_name"] in tag_ids:
                attach_tag(card_id, tag_ids[task["epic_name"]], task["epic_name"])
            for tech in task.get("tech_tags", []):
                if tech in tech_tag_ids:
                    attach_tag(card_id, tech_tag_ids[tech], tech)
            techs = ", ".join(task.get("tech_tags", []))
            print(f"  ✓ Теги на карточке {card_id}: E4-тег + [{techs}]")
        print("✅ Теги проставлены!")
        return

    # 7. Связи между задачами (через custom fields / relations если доступны)
    print("\n7. Устанавливаем связи между задачами...")
    relations = [
        ("T4.1 — MessageClassifier: детекция чувствительных тем",
         "T4.3 — PromptBuilder: сборка промпта"),
        ("T4.2 — FewShotRetriever: поиск примеров в Qdrant",
         "T4.3 — PromptBuilder: сборка промпта"),
        ("T4.3 — PromptBuilder: сборка промпта",
         "T4.4 — LLMClient: вызов LLM с retry/backoff"),
        ("T4.4 — LLMClient: вызов LLM с retry/backoff",
         "T4.5 — DraftManager: хранение черновиков в SQLite"),
        ("T4.4 — LLMClient: вызов LLM с retry/backoff",
         "T4.6 — MessageQueue: устойчивость к сбоям"),
    ]

    for blocker_title, blocked_title in relations:
        blocker_id = next((v for k,v in task_card_ids.items() if blocker_title[:10] in k), None)
        blocked_id  = next((v for k,v in task_card_ids.items() if blocked_title[:10] in k), None)
        if blocker_id and blocked_id:
            r = post(f"/cards/{blocker_id}/relations", data={
                "card_id": blocked_id,
                "relation_type": "blocks",  # «блокирует»
            })
            status = "✓" if r else "✗ (API может не поддерживать, добавь вручную)"
            short_b  = blocker_title.split("—")[0].strip()
            short_bl = blocked_title.split("—")[0].strip()
            print(f"  {status} {short_b} → {short_bl}")

    print("\n✅ Готово! Открой доску: https://aissistant.kaiten.ru/space/790412/boards")

if __name__ == "__main__":
    main()
