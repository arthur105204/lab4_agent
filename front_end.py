import html
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from agent import (
	deserialize_agent_state,
	new_session_state,
	run_turn_with_trace,
	serialize_agent_state,
)

SESSIONS_DIR = Path(__file__).resolve().parent / "chat_sessions"
SCHEMA_V1 = "travelbuddy_chat_session_v1"
_WELCOME_MSG = (
	"Xin chào! Mình có thể giúp bạn tìm chuyến bay, khách sạn và tính ngân sách chuyến đi."
)
_NEW_CHAT_GREETING = (
	"Đã tạo hội thoại mới. Bạn muốn đi đâu, ngân sách khoảng bao nhiêu?"
)

st.set_page_config(
	page_title="TravelBuddy",
	layout="wide",
	initial_sidebar_state="collapsed",
)

_INTENT_LABELS: dict[str, str] = {
	"react_langgraph": "ReAct (LangGraph)",
	"ngoài_phạm_vi": "Ngoài phạm vi",
	"lỗi": "Lỗi xử lý",
}
_ACTION_LABELS: dict[str, str] = {
	"gọi_công_cụ": "Đã gọi công cụ",
	"trả_lời_trực_tiếp": "Trả lời trực tiếp",
	"từ_chối": "Từ chối",
	"lỗi": "Lỗi",
}

_DEFAULT_TRACE: dict[str, Any] = {
	"intent": "",
	"action": "",
	"workflow": [],
	"tool_calls": [],
}

ChatEntry = tuple[str, str, dict[str, Any] | None]


def _normalize_turn(raw_ch: list) -> list[ChatEntry]:
	out: list[ChatEntry] = []
	for x in raw_ch:
		if not isinstance(x, (list, tuple)) or len(x) < 2:
			continue
		role = str(x[0])
		content = str(x[1]) if x[1] is not None else ""
		tr = x[2] if len(x) >= 3 and isinstance(x[2], dict) else None
		out.append((role, content, tr))
	return out


def _as_entry(role: str, content: str, trace: dict[str, Any] | None = None) -> ChatEntry:
	return (role, content, trace)


def _session_path(session_id: str) -> Path:
	return SESSIONS_DIR / f"{session_id}.json"


def _list_sessions_meta() -> list[tuple[str, str, float]]:
	SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
	out: list[tuple[str, str, float]] = []
	for path in SESSIONS_DIR.glob("*.json"):
		try:
			data = json.loads(path.read_text(encoding="utf-8"))
			sid = str(data.get("id") or path.stem)
			title = str(data.get("title") or sid[:8])
			mtime = path.stat().st_mtime
			out.append((sid, title, mtime))
		except (json.JSONDecodeError, OSError):
			continue
	return sorted(out, key=lambda x: -x[2])


def _session_title_from_history(history: list) -> str:
	for item in history:
		role = item[0]
		content = item[1] if len(item) > 1 else ""
		if role == "user" and isinstance(content, str) and content.strip():
			t = content.strip().replace("\n", " ")
			return (t[:48] + "...") if len(t) > 48 else t
	return "Hội thoại mới"


def _persist_current_session() -> None:
	sid = st.session_state.get("current_session_id") or ""
	if not sid:
		return
	title = _session_title_from_history(st.session_state.chat_history)
	ser_hist: list[list[Any]] = []
	for role, content, tr in st.session_state.chat_history:
		if tr is None:
			ser_hist.append([role, content])
		else:
			ser_hist.append([role, content, tr])
	payload = {
		"schema": SCHEMA_V1,
		"id": sid,
		"title": title,
		"updated_at": datetime.now(timezone.utc).isoformat(),
		"chat_history": ser_hist,
		"agent_state": serialize_agent_state(st.session_state.agent_state),
		"last_trace": dict(st.session_state.last_trace),
	}
	SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
	_session_path(sid).write_text(
		json.dumps(payload, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)


def _load_session_into_state(session_id: str) -> None:
	path = _session_path(session_id)
	if not path.is_file():
		return
	raw = json.loads(path.read_text(encoding="utf-8"))
	st.session_state.current_session_id = session_id
	st.session_state.chat_history = _normalize_turn(raw.get("chat_history") or [])
	st.session_state.agent_state = deserialize_agent_state(raw.get("agent_state") or {})
	st.session_state.last_trace = raw.get("last_trace") or dict(_DEFAULT_TRACE)
	st.session_state.agent_turn_phase = 0
	st.session_state.pending_user_text = ""


def _reset_to_fresh_session(new_id: str, greeting: str) -> None:
	st.session_state.current_session_id = new_id
	st.session_state.agent_state = new_session_state()
	st.session_state.chat_history = [_as_entry("assistant", greeting, None)]
	st.session_state.last_trace = dict(_DEFAULT_TRACE)
	st.session_state.agent_turn_phase = 0
	st.session_state.pending_user_text = ""


def _ensure_session() -> None:
	if "pending_input" not in st.session_state:
		st.session_state.pending_input = None
	if "agent_turn_phase" not in st.session_state:
		st.session_state.agent_turn_phase = 0
	if "pending_user_text" not in st.session_state:
		st.session_state.pending_user_text = ""

	if "current_session_id" not in st.session_state:
		meta = _list_sessions_meta()
		if meta:
			_load_session_into_state(meta[0][0])
		else:
			_reset_to_fresh_session(uuid.uuid4().hex, _WELCOME_MSG)
			_persist_current_session()

	if "chat_history" not in st.session_state:
		st.session_state.chat_history = [_as_entry("assistant", _WELCOME_MSG, None)]
	if "agent_state" not in st.session_state:
		st.session_state.agent_state = new_session_state()
	if "last_trace" not in st.session_state:
		st.session_state.last_trace = dict(_DEFAULT_TRACE)


def _display_intent(raw: str) -> str:
	return _INTENT_LABELS.get(raw, raw or "Chưa có")


def _display_action(raw: str) -> str:
	return _ACTION_LABELS.get(raw, raw or "—")


def _delete_session_file(session_id: str) -> None:
	p = _session_path(session_id)
	if p.is_file():
		p.unlink()


def _render_sidebar() -> None:
    st.markdown("### Chat")

    if st.button("＋ New chat", use_container_width=True):
        _persist_current_session()
        _reset_to_fresh_session(uuid.uuid4().hex, _NEW_CHAT_GREETING)
        _persist_current_session()
        st.rerun()

    meta = _list_sessions_meta()
    ids = [m[0] for m in meta]
    cur = st.session_state.current_session_id

    if not ids:
        _reset_to_fresh_session(uuid.uuid4().hex, _WELCOME_MSG)
        _persist_current_session()
        st.rerun()

    if cur not in ids:
        _persist_current_session()
        _load_session_into_state(ids[0])
        st.rerun()

    # ===== CSS kiểu ChatGPT =====
    st.markdown("""
    <style>
    .chat-item {
        padding: 10px 12px;
        border-radius: 10px;
        font-size: 0.9rem;
        line-height: 1.3;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.15s ease;
    }

    .chat-item:hover {
        background: rgba(255,255,255,0.08);
    }

    .chat-item.active {
        background: rgba(255,255,255,0.12);
    }

    .chat-title {
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        flex: 1;
    }

    .delete-btn {
        opacity: 0;
        transition: opacity 0.15s;
    }

    .chat-item:hover .delete-btn {
        opacity: 1;
    }

    button[data-testid="baseButton-secondary"] {
        background: transparent !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("##### Chats")

    for sid, title, _ in meta:
        short = (title[:50] + "...") if len(title) > 50 else title
        is_cur = sid == cur

        container = st.container()
        with container:
            col1, col2 = st.columns([0.85, 0.15])

            with col1:
                clicked = st.button(
                    short,
                    key=f"chat_{sid}",
                    use_container_width=True
                )
                if clicked:
                    _persist_current_session()
                    _load_session_into_state(sid)
                    st.rerun()

            with col2:
                delete = st.button(
                    "🗑",
                    key=f"del_{sid}",
                    help="Delete",
                    use_container_width=True
                )
                if delete:
                    was_cur = sid == cur
                    _delete_session_file(sid)

                    remaining = _list_sessions_meta()
                    if was_cur:
                        if remaining:
                            _load_session_into_state(remaining[0][0])
                        else:
                            _reset_to_fresh_session(uuid.uuid4().hex, _WELCOME_MSG)
                            _persist_current_session()
                    st.rerun()

    st.caption(f"Saved in `{SESSIONS_DIR.name}`")


def _render_trace_content(trace: dict[str, Any]) -> None:
	intent_raw = trace.get("intent") or ""
	action_raw = trace.get("action") or ""
	workflow = trace.get("workflow") or []
	tool_calls = trace.get("tool_calls") or []

	has_data = bool(intent_raw or workflow or tool_calls)
	if not has_data:
		st.caption("Chưa có dữ liệu trace cho lượt này.")
		return

	st.caption(
		f"{_display_intent(intent_raw)} | {_display_action(action_raw)} | {len(tool_calls)} tool"
	)
	for i, step in enumerate(workflow, start=1):
		st.caption(f"{i}. {step}")

	if not tool_calls:
		return

	with st.expander("Chi tiết công cụ (JSON)", expanded=False):
		names = [str(t.get("name", "?")) for t in tool_calls]
		st.caption("Thực thi: " + " -> ".join(names))
		for idx, item in enumerate(tool_calls, start=1):
			name = item.get("name", "?")
			args = item.get("args", {})
			try:
				args_text = json.dumps(args, ensure_ascii=False, indent=2)
			except TypeError:
				args_text = str(args)
			st.markdown(f"`{idx}. {name}`")
			st.code(args_text, language="json")
		combined = []
		for i, item in enumerate(tool_calls, start=1):
			combined.append({"buoc": i, "tool": item.get("name"), "args": item.get("args")})
		payload = json.dumps(combined, ensure_ascii=False, indent=2)
		st.download_button(
			label="Tải JSON công cụ",
			data=payload,
			file_name="travelbuddy_trace_tools.json",
			mime="application/json",
			use_container_width=True,
		)


def _trace_nonempty(tr: dict[str, Any] | None) -> bool:
	if not tr:
		return False
	return bool(tr.get("intent") or tr.get("workflow") or tr.get("tool_calls"))


def _process_agent_reply_only(user_text: str) -> None:
	try:
		_ans, st.session_state.agent_state, trace, steps = run_turn_with_trace(
			user_text,
			st.session_state.agent_state,
		)
		st.session_state.last_trace = trace
	except Exception as exc:
		msg = f"Có lỗi khi xử lý: {exc}"
		trace = {
			"intent": "lỗi",
			"action": "lỗi",
			"workflow": ["Gọi agent thất bại", str(exc)],
			"tool_calls": [],
		}
		st.session_state.last_trace = trace
		steps = [msg]
	last = len(steps) - 1
	for i, part in enumerate(steps):
		tr = trace if i == last else None
		st.session_state.chat_history.append(_as_entry("assistant", part, tr))
	_persist_current_session()


_ensure_session()

with st.sidebar:
	_render_sidebar()

st.title("TravelBuddy")
st.caption("Trợ lý du lịch (ReAct + LangGraph).")
st.markdown("#### Hội thoại")
chat_box = st.container()
with chat_box:
	for entry in st.session_state.chat_history:
		role, content, tr = entry
		with st.chat_message(role):
			st.markdown(content)
			if role == "assistant" and _trace_nonempty(tr):
				assert tr is not None
				with st.expander("Trace ReAct / LangGraph", expanded=False):
					_render_trace_content(tr)

_phase = int(st.session_state.agent_turn_phase)
if _phase == 1:
	st.session_state.agent_turn_phase = 2
	st.rerun()
elif _phase == 2:
	user_text = (st.session_state.pending_user_text or "").strip()
	st.session_state.pending_user_text = ""
	st.session_state.agent_turn_phase = 0
	if user_text:
		with st.spinner("Đang xử lý..."):
			_process_agent_reply_only(user_text)
		st.rerun()

prompt = st.chat_input("Nhập yêu cầu bằng tiếng Việt...")
if prompt:
	text = prompt.strip()
	if text:
		st.session_state.chat_history.append(_as_entry("user", text, None))
		st.session_state.pending_user_text = text
		st.session_state.agent_turn_phase = 1
		_persist_current_session()
		st.rerun()
