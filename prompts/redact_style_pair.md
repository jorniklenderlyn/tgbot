You prepare chat examples for a reusable STYLE bank. Each example is a pair:
an incoming message from a friend and the user's reply. The bank is shared
across different conversations, so every example MUST keep the writing STYLE
while removing all specific facts — otherwise private details from one chat
would leak into another.

For each pair produce three fields:

- "situation": a short, FACT-FREE description in Russian of what is happening and
  what kind of reply this is — the intent/register, not the content. Examples:
  "приветствие", "ответ на вопрос о времени встречи", "реакция на хорошую новость",
  "отказ от предложения", "уточняющий вопрос", "благодарность". No names, no
  specifics. This is used later to decide whether the example fits a new case.

- "incoming_redacted": the incoming message with concrete facts replaced by
  placeholders, structure preserved.

- "reply_redacted": the user's reply with concrete facts replaced by
  placeholders, but STYLE PRESERVED EXACTLY — same casing, punctuation, emoji,
  bracket-smileys like ) and ((( , slang, abbreviations, message length and line
  breaks. Change ONLY the facts, never the styling.

Replace facts with placeholders:
  names/usernames -> [имя], places/orgs -> [место], dates/times -> [время],
  numbers/amounts -> [число], subjects/topics -> [тема], links -> [ссылка],
  other specific nouns -> [объект].
Keep interjections, filler words, emoji, smileys and slang untouched. If a
message has no facts (e.g. "Привет", ")", "норм", "хз"), return it unchanged.

INPUT: a JSON array of objects {"id": int, "incoming": str, "reply": str}.
OUTPUT: ONLY a JSON array of objects
  {"id": int, "situation": str, "incoming_redacted": str, "reply_redacted": str}
with the same ids, same order, no extra text.
