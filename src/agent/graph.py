"""LangGraph reply-generation pipeline with triage and message classification.

Flow:
  START → load_chat_prompt → load_working_memory → classify_messages → triage → generate → END

classify_messages: determines if batched messages are story/questions/mixed/single.
triage: decides auto_reply vs needs_attention.
generate: produces the reply, adapting style to message_type.
"""

import json
import re
import sys
import time

from langgraph.graph import END, StateGraph
from openai import OpenAI

MAX_RETRIES = 5


def _llm_call_with_retry(llm, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return llm.chat.completions.create(**kwargs)
        except Exception as e:
            is_retryable = "429" in str(e) or "502" in str(e) or "503" in str(e)
            if attempt == MAX_RETRIES or not is_retryable:
                raise
            delay = 0.5 * attempt
            print(f"[llm] attempt {attempt}/{MAX_RETRIES} failed, retrying in {delay}s...",
                  file=sys.stderr)
            time.sleep(delay)

from src.agent.prompts import CLASSIFY_PROMPT, SYSTEM_PROMPT, TRIAGE_PROMPT
from src.agent.state import AgentState
from src.config import ASSISTANT_LLM_MODEL, OPENROUTER_API_KEY


RISK_KEYWORDS = re.compile(
    r"\b(завтра|сегодня|вечером|утром|днём|днем|встреч|деньги|оплат|перевед|"
    r"когда|во\s*сколько|давай|план|съезд|прие[ху]|поедем|купи|закаж|подпиш|"
    r"договор|адрес|пришли\s+(телефон|номер|карт)|кредит)",
    re.IGNORECASE,
)


class AgentDeps:
    def __init__(self, llm: OpenAI, chat_prompts: dict[str, str]):
        self.llm = llm
        self.chat_prompts = chat_prompts


def _load_chat_prompt_node(deps: AgentDeps):
    from src.chat_prompts import resolve_chat_prompt

    def _fn(state: AgentState) -> AgentState:
        md = resolve_chat_prompt(
            deps.chat_prompts,
            chat_id=state["chat_id"],
            username=state.get("chat_username"),
        )
        return {"chat_prompt": md}
    return _fn


def _load_working_memory_node(deps: AgentDeps):
    from src.working_memory import load_memory

    def _fn(state: AgentState) -> AgentState:
        memory = load_memory(state["chat_id"])
        return {"working_memory": memory}
    return _fn


def _classify_messages_node(deps: AgentDeps):
    def _fn(state: AgentState) -> AgentState:
        incoming = state["incoming_text"]
        lines = [l.strip() for l in incoming.split("\n") if l.strip()]

        if len(lines) <= 1:
            return {"message_type": "single", "message_type_reason": "одно сообщение"}

        messages = [{"role": "system", "content": CLASSIFY_PROMPT}]
        messages.append({
            "role": "user",
            "content": f"СЕРИЯ СООБЩЕНИЙ ({len(lines)} шт.):\n{incoming}",
        })

        try:
            resp = _llm_call_with_retry(deps.llm,
                model=ASSISTANT_LLM_MODEL,
                messages=messages,
                temperature=0.1,
            )
            parsed = _parse_json(resp.choices[0].message.content)
            msg_type = parsed.get("message_type", "mixed")
            reason = parsed.get("reason", "")
        except Exception as e:
            print(f"[classify] LLM failed: {e}", file=sys.stderr)
            msg_type = "mixed"
            reason = f"ошибка классификации: {e}"

        if msg_type not in ("story", "questions", "mixed", "single"):
            msg_type = "mixed"

        return {"message_type": msg_type, "message_type_reason": reason}
    return _fn


def _parse_json(raw: str) -> dict:
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        if txt.startswith("json"):
            txt = txt[4:]
        txt = txt.strip()
    try:
        return json.loads(txt)
    except Exception:
        return {}


# ---- triage node --------------------------------------------------------- #

def _triage_node(deps: AgentDeps):
    def _fn(state: AgentState) -> AgentState:
        incoming = state["incoming_text"]

        if RISK_KEYWORDS.search(incoming):
            return {
                "triage_classification": "needs_attention",
                "triage_reason": "ключевое слово в сообщении",
            }

        messages = [{"role": "system", "content": TRIAGE_PROMPT}]
        if state.get("chat_prompt"):
            messages.append({
                "role": "system",
                "content": "КОНТЕКСТ ЧАТА:\n\n" + state["chat_prompt"],
            })
        if state.get("working_memory"):
            messages.append({
                "role": "system",
                "content": "РАБОЧАЯ ПАМЯТЬ (контекст переписки):\n\n" + state["working_memory"],
            })
        if state.get("raw_history"):
            messages.append({
                "role": "system",
                "content": "ПОСЛЕДНИЕ СООБЩЕНИЯ:\n\n" + state["raw_history"],
            })
        messages.append({
            "role": "user",
            "content": f"СОБЕСЕДНИК: {state['sender_name']}\nСООБЩЕНИЕ: {incoming}",
        })

        try:
            resp = _llm_call_with_retry(deps.llm,
                model=ASSISTANT_LLM_MODEL,
                messages=messages,
                temperature=0.2,
            )
            parsed = _parse_json(resp.choices[0].message.content)
            classification = parsed.get("classification", "needs_attention")
            reason = parsed.get("reason", "")
        except Exception as e:
            print(f"[triage] LLM failed: {e}", file=sys.stderr)
            classification = "needs_attention"
            reason = f"ошибка триажа: {e}"

        if classification not in ("auto_reply", "needs_attention"):
            classification = "needs_attention"

        return {"triage_classification": classification, "triage_reason": reason}
    return _fn


def _after_triage(state: AgentState) -> str:
    return "generate"


# ---- generate node ------------------------------------------------------- #

def _build_user_prompt(state: AgentState) -> str:
    msg_type = state.get("message_type", "single")
    type_hint = f"\nТИП СООБЩЕНИЙ: {msg_type}" if msg_type != "single" else ""
    return (
        f"СОБЕСЕДНИК: {state['sender_name']}\n"
        f"{type_hint}\n"
        f"НОВОЕ ВХОДЯЩЕЕ СООБЩЕНИЕ:\n{state['incoming_text']}\n\n"
        f"ОТВЕТЬ КАК ОБЫЧНО ТЫ."
    )


def _parse_llm_json(raw: str) -> dict:
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        if txt.startswith("json"):
            txt = txt[4:]
        txt = txt.strip()
    try:
        obj = json.loads(txt)
    except Exception:
        return {"reply": raw.strip(), "requires_approval": True,
                "approval_reason": "LLM did not return valid JSON"}
    obj.setdefault("reply", "")
    obj.setdefault("requires_approval", True)
    obj.setdefault("approval_reason", "")
    return obj


def _generate_node(deps: AgentDeps):
    def _fn(state: AgentState) -> AgentState:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if state.get("chat_prompt"):
            messages.append({
                "role": "system",
                "content": "ДОПОЛНИТЕЛЬНЫЕ ПРАВИЛА ДЛЯ ЭТОГО ЧАТА:\n\n" + state["chat_prompt"],
            })
        if state.get("working_memory"):
            messages.append({
                "role": "system",
                "content": "РАБОЧАЯ ПАМЯТЬ (контекст переписки):\n\n" + state["working_memory"],
            })
        if state.get("raw_history"):
            messages.append({
                "role": "system",
                "content": "ПОСЛЕДНИЕ СООБЩЕНИЯ В ЧАТЕ (для контекста, не отвечай на них — только на НОВОЕ ВХОДЯЩЕЕ):\n\n" + state["raw_history"],
            })
        messages.append({"role": "user", "content": _build_user_prompt(state)})

        try:
            resp = _llm_call_with_retry(deps.llm,
                model=ASSISTANT_LLM_MODEL,
                messages=messages,
                temperature=0.7,
            )
            parsed = _parse_llm_json(resp.choices[0].message.content)
        except Exception as e:
            print(f"[err] LLM failed: {e}", file=sys.stderr)
            return {"error": str(e), "reply_text": "",
                    "requires_approval": True, "approval_reason": "LLM error"}

        return {
            "reply_text": (parsed.get("reply") or "").strip(),
            "requires_approval": bool(parsed.get("requires_approval")),
            "approval_reason": parsed.get("approval_reason") or "",
        }
    return _fn


# ---- graph builder ------------------------------------------------------- #

def build_agent(
    qdrant=None,
    embedder=None,
    chat_prompts: dict[str, str] | None = None,
    llm: OpenAI | None = None,
):
    """qdrant / embedder are accepted for compatibility but ignored in simple mode."""
    if llm is None:
        llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    deps = AgentDeps(llm=llm, chat_prompts=chat_prompts or {})

    g = StateGraph(AgentState)
    g.add_node("load_chat_prompt", _load_chat_prompt_node(deps))
    g.add_node("load_working_memory", _load_working_memory_node(deps))
    g.add_node("classify_messages", _classify_messages_node(deps))
    g.add_node("triage", _triage_node(deps))
    g.add_node("generate", _generate_node(deps))

    g.set_entry_point("load_chat_prompt")
    g.add_edge("load_chat_prompt", "load_working_memory")
    g.add_edge("load_working_memory", "classify_messages")
    g.add_edge("classify_messages", "triage")
    g.add_conditional_edges("triage", _after_triage)
    g.add_edge("generate", END)

    return g.compile()


async def run_agent(agent, state: AgentState) -> AgentState:
    return await agent.ainvoke(state)
