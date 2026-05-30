# ЛР3 — Декомпозиция проекта «reply-as-me»

## Структура доски

**Колонки:** Бэклог → К работе / To Do → В работе → На проверке → Готово

## Эпики (5 штук)

| ID | Название | Файл |
|----|----------|------|
| E1 | Сбор и обработка данных | `epics/E1_data_collection.md` |
| E2 | Профиль стиля (Persona Extraction) | `epics/E2_persona_extraction.md` |
| E3 | Долговременная память (Mem0) | `epics/E3_memory_mem0.md` |
| E4 | Конвейер генерации ответов | `epics/E4_response_pipeline.md` |
| E5 | Telegram-интеграция (Telethon + Aiogram) | `epics/E5_telegram_integration.md` |

## Задачи Эпика 4 (детальный разбор, 7 штук)

| ID | Название | Зависит от | Влияет на | Файл |
|----|----------|------------|-----------|------|
| 4.1 | MessageClassifier | — | 4.3 | `tasks/T4.1_message_classifier.md` |
| 4.2 | FewShotRetriever | — | 4.3 | `tasks/T4.2_fewshot_retriever.md` |
| 4.3 | PromptBuilder | 4.1, 4.2 | 4.4 | `tasks/T4.3_prompt_builder.md` |
| 4.4 | LLMClient | 4.3 | 4.5, 4.6 | `tasks/T4.4_llm_client.md` |
| 4.5 | DraftManager | 4.4 | — | `tasks/T4.5_draft_manager.md` |
| 4.6 | MessageQueue | 4.4 | — | `tasks/T4.6_message_queue.md` |
| 4.6.1 | ExponentialBackoffRetry *(дочерняя к 4.6)* | — | 4.4 | `tasks/T4.6.1_backoff_retry.md` |

## Зависимости (граф)

```
4.1 ──┐
      ├──► 4.3 ──► 4.4 ──► 4.5
4.2 ──┘             │
                    └──► 4.6
4.6.1 ──────────────────► 4.4
```
