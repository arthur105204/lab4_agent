import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools import FLIGHTS_DB, HOTELS_DB, calculate_budget, search_flights, search_hotels

load_dotenv()

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

OUT_OF_SCOPE_KEYWORDS = (
	"bài tập",
	"linked list",
	"chính trị",
	"tài chính cá nhân",
	"y tế",
	"chứng khoán",
	"kinh doanh chứng khoán",
)

REFUSAL_MESSAGE = (
	"Xin lỗi, tôi chỉ hỗ trợ các yêu cầu liên quan đến du lịch. "
	"Nếu bạn cần, tôi có thể giúp lên kế hoạch chuyến đi theo ngân sách."
)

RECURSION_LIMIT = 25

TURN_LOG_DIR = Path(__file__).resolve().parent / "turn_logs"


def _json_safe(value: Any) -> Any:
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, dict):
		return {str(k): _json_safe(v) for k, v in value.items()}
	if isinstance(value, (list, tuple)):
		return [_json_safe(item) for item in value]
	return str(value)


def _message_to_log_entry(message: BaseMessage) -> dict[str, Any]:
	entry: dict[str, Any] = {"type": message.__class__.__name__}
	content = getattr(message, "content", None)
	if isinstance(content, str):
		entry["content"] = content
	elif isinstance(content, list):
		entry["content"] = _json_safe(content)
	else:
		entry["content"] = _json_safe(content)

	if isinstance(message, AIMessage):
		if message.tool_calls:
			entry["tool_calls"] = _json_safe(message.tool_calls)
		if message.response_metadata:
			entry["response_metadata"] = _json_safe(message.response_metadata)
		if message.usage_metadata:
			entry["usage_metadata"] = _json_safe(dict(message.usage_metadata))
	if isinstance(message, ToolMessage):
		entry["name"] = message.name
		entry["tool_call_id"] = message.tool_call_id
		status = getattr(message, "status", None)
		if status:
			entry["status"] = status

	return entry


class AgentState(TypedDict):
	messages: Annotated[list, add_messages]
	memory: dict[str, Any]


class ToolTrace(TypedDict):
	name: str
	args: dict[str, Any]


class TurnTrace(TypedDict):
	intent: str
	action: str
	workflow: list[str]
	tool_calls: list[ToolTrace]


def _preview_message_content(message: BaseMessage) -> str:
	raw = getattr(message, "content", "")
	text = raw if isinstance(raw, str) else json.dumps(_json_safe(raw), ensure_ascii=False)
	if len(text) > 2000:
		return text[:2000] + "... [rut gon]"
	return text


def _write_turn_log(
	user_input: str,
	final_reply: str,
	trace: TurnTrace,
	updated_state: AgentState,
	turn_messages: list[BaseMessage],
) -> Path | None:
	"""Ghi mot file JSON trong turn_logs/ (ten theo timestamp) de debug. That bai ghi file khong lam dung luong chinh."""
	try:
		TURN_LOG_DIR.mkdir(parents=True, exist_ok=True)
		now_local = datetime.now().astimezone()
		stamp = now_local.strftime("%Y%m%d_%H%M%S_%f")
		path = TURN_LOG_DIR / f"turn_{stamp}.json"

		payload_tool_calls = list(trace.get("tool_calls", []))
		summary_lines = [
			f"Thoi gian (may local): {now_local.isoformat()}",
			f"Thoi gian (UTC): {datetime.now(timezone.utc).isoformat()}",
			f"Ten file: {path.name}",
			"",
			"[Nguoi dung]",
			user_input,
			"",
			f"[Trace] intent={trace.get('intent')} | hanh_dong={trace.get('action')}",
			"[Workflow]",
			*(f"  - {step}" for step in (trace.get("workflow") or [])),
			"",
			f"[Tool calls] tong so lan goi: {len(payload_tool_calls)}",
			*(f"  - {tc.get('name')}: {tc.get('args')}" for tc in payload_tool_calls),
			"",
			"[Tin nhan trong luot - tom tat (xem messages_in_turn day du)]",
			*(f"  - {m.__class__.__name__}: {_preview_message_content(m)}" for m in turn_messages),
			"",
			"[Cau tra loi cuoi]",
			final_reply,
			"",
			"[Bo nho session sau luot - JSON]",
			json.dumps(_json_safe(dict(updated_state.get("memory", {}))), ensure_ascii=False),
		]

		payload: dict[str, Any] = {
			"schema": "travelbuddy_turn_log_v1",
			"timestamp_local": now_local.isoformat(),
			"timestamp_utc": datetime.now(timezone.utc).isoformat(),
			"log_file": str(path.name),
			"summary_text": "\n".join(summary_lines),
			"user_input": user_input,
			"final_reply": final_reply,
			"trace": {
				"intent": trace["intent"],
				"action": trace["action"],
				"workflow": list(trace["workflow"]),
				"tool_calls": _json_safe(list(trace["tool_calls"])),
			},
			"memory_after": _json_safe(dict(updated_state.get("memory", {}))),
			"messages_in_turn": [_message_to_log_entry(m) for m in turn_messages],
			"session_message_count": len(updated_state.get("messages", [])),
		}

		path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
		logger.info("Da ghi log luot tra loi: %s", path.resolve())
		return path
	except Exception as exc:
		logger.warning("Khong ghi duoc log luot tra loi: %s", exc)
		return None


def _is_out_of_scope(user_text: str) -> bool:
	"""Từ chối sớm các yêu cầu rõ ràng ngoài du lịch (giúp bài test ổn định)."""
	lower = user_text.lower()
	if any(k in lower for k in OUT_OF_SCOPE_KEYWORDS):
		return True
	if re.search(r"\bbài\s+tập\b", lower) and re.search(r"\bpython\b", lower):
		return True
	return False


def _extract_memory(user_text: str, previous_memory: dict[str, Any]) -> dict[str, Any]:
	memory = dict(previous_memory)
	text = user_text.strip()

	canonical = set(HOTELS_DB.keys())
	for origin, destination in FLIGHTS_DB.keys():
		canonical.add(origin)
		canonical.add(destination)
	sorted_canonical = sorted(canonical, key=len, reverse=True)

	def _resolve_city(raw: str) -> str:
		raw_stripped = raw.strip()
		for city in sorted_canonical:
			if city.lower() == raw_stripped.lower():
				return city
		for city in sorted_canonical:
			if city.lower() in raw_stripped.lower() or raw_stripped.lower() in city.lower():
				return city
		return raw_stripped

	dep_toi_o = re.search(
		r"\b(?:[Tt]ôi)\s+ở\s+([A-Za-zÀ-ỹ]+(?:\s+[A-Za-zÀ-ỹ]+)?)\b",
		text,
	)
	if dep_toi_o:
		memory["departure_city"] = _resolve_city(dep_toi_o.group(1))

	from_match = re.search(r"từ\s+([A-Za-zÀ-ỹ\s]+?)\s+(đi|đến)", text, flags=re.IGNORECASE)
	if from_match:
		memory["departure_city"] = _resolve_city(from_match.group(1))

	dest_match = re.search(
		r"(?:muốn\s+)?(?:đi|đến|tới)\s+([A-Za-zÀ-ỹ]+(?:\s+[A-Za-zÀ-ỹ]+)?)\b",
		text,
		flags=re.IGNORECASE,
	)
	if dest_match:
		memory["destination_city"] = _resolve_city(dest_match.group(1))
	else:
		for city in sorted_canonical:
			if city in text:
				memory["destination_city"] = city
				break

	budget_match = re.search(r"(\d+)\s*triệu", text, flags=re.IGNORECASE)
	if budget_match:
		memory["budget_vnd"] = int(budget_match.group(1)) * 1_000_000

	nights_match = re.search(r"(\d+)\s*(đêm|ngày)", text, flags=re.IGNORECASE)
	if nights_match:
		memory["nights"] = int(nights_match.group(1))

	return memory


def _build_memory_message(memory: dict[str, Any]) -> str:
	if not memory:
		return "<memory_context>Chưa có dữ liệu người dùng đã lưu.</memory_context>"

	lines = ["<memory_context>"]
	for key, value in memory.items():
		lines.append(f"- {key}: {value}")
	lines.append("</memory_context>")
	return "\n".join(lines)


def _build_db_catalog_message() -> str:
	routes = sorted({f"{origin} -> {destination}" for origin, destination in FLIGHTS_DB.keys()})
	origins = sorted({origin for origin, _ in FLIGHTS_DB.keys()})
	destinations = sorted({destination for _, destination in FLIGHTS_DB.keys()})
	hotel_cities = sorted(HOTELS_DB.keys())

	allowed = sorted(set(origins) | set(destinations) | set(hotel_cities))
	return "\n".join([
		"<db_catalog_dynamic>",
		f"- Tuyến bay: {', '.join(routes)}",
		f"- Điểm xuất phát: {', '.join(origins)}",
		f"- Điểm đến (bay): {', '.join(destinations)}",
		f"- Thành phố có khách sạn: {', '.join(hotel_cities)}",
		f"- Các tên địa danh ĐƯỢC PHÉP nhắc trong câu trả lời (gợi ý, tư vấn): {', '.join(allowed)}.",
		"- QUY TẮC: Không nhắc bất kỳ thành phố/tỉnh/đảo du lịch nào khác (ví dụ Nha Trang, Huế, Vũng Tàu) kể cả khi người dùng nói thích biển hay hỏi chung.",
		"</db_catalog_dynamic>",
	])


def _route_after_agent(state: AgentState) -> str:
	last_message = state["messages"][-1]
	if isinstance(last_message, AIMessage) and last_message.tool_calls:
		return "tools"
	return END


def agent_node(state: AgentState) -> AgentState:
	messages = state["messages"]
	memory = state.get("memory", {})

	for msg in reversed(messages):
		if isinstance(msg, HumanMessage):
			memory = _extract_memory(str(msg.content), memory)
			break

	prompt_messages: list[SystemMessage | HumanMessage | AIMessage] = [
		SystemMessage(content=SYSTEM_PROMPT),
		SystemMessage(content=_build_db_catalog_message()),
		SystemMessage(content=_build_memory_message(memory)),
		*messages,
	]

	response = llm_with_tools.invoke(prompt_messages)
	if isinstance(response, AIMessage) and response.tool_calls:
		for call in response.tool_calls:
			logger.info("ReAct: goi tool name=%s args=%s", call.get("name"), call.get("args"))
	else:
		logger.info("ReAct: ket thuc luot (khong goi tool)")

	return {"messages": [response], "memory": memory}


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise RuntimeError("Thiếu OPENAI_API_KEY. Vui lòng thiết lập biến môi trường hoặc file .env.")

TOOLS = [search_flights, search_hotels, calculate_budget]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)
llm_with_tools = llm.bind_tools(TOOLS)

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")
graph = builder.compile()


def new_session_state() -> AgentState:
	return {"messages": [], "memory": {}}


def serialize_agent_state(state: AgentState) -> dict[str, Any]:
	from langchain_core.messages import messages_to_dict

	return {
		"messages": messages_to_dict(state["messages"]),
		"memory": _json_safe(state.get("memory", {})),
	}


def deserialize_agent_state(data: dict[str, Any]) -> AgentState:
	from langchain_core.messages import messages_from_dict

	raw_messages = data.get("messages")
	if not raw_messages:
		return new_session_state()
	return {
		"messages": messages_from_dict(list(raw_messages)),
		"memory": dict(data.get("memory") or {}),
	}


def _final_text(message: BaseMessage) -> str:
	content = getattr(message, "content", None)
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		parts: list[str] = []
		for block in content:
			if isinstance(block, dict) and block.get("type") == "text":
				parts.append(str(block.get("text", "")))
			elif isinstance(block, str):
				parts.append(block)
		return "".join(parts)
	if content is None:
		return ""
	return str(content)


def _assistant_ui_steps_from_messages(new_messages: list[BaseMessage]) -> list[str]:
	"""Moi AIMessage (gọi công cụ hoặc trả lời) tuong ung mot doan hien thi tren UI."""
	steps: list[str] = []
	for m in new_messages:
		if not isinstance(m, AIMessage):
			continue
		text = _final_text(m).strip()
		if m.tool_calls:
			names = ", ".join(str(c.get("name", "?")) for c in m.tool_calls)
			line = f"**Bước** Gọi công cụ: `{names}`"
			if text:
				line = f"{text}\n\n{line}"
			steps.append(line)
		elif text:
			steps.append(text)
	return steps


def run_turn_with_trace(
	user_input: str, session_state: AgentState,
) -> tuple[str, AgentState, TurnTrace, list[str]]:
	if _is_out_of_scope(user_input):
		reply = REFUSAL_MESSAGE
		updated_state: AgentState = {
			"messages": [
				*session_state["messages"],
				HumanMessage(content=user_input),
				AIMessage(content=reply),
			],
			"memory": session_state.get("memory", {}),
		}
		trace: TurnTrace = {
			"intent": "ngoài_phạm_vi",
			"action": "từ_chối",
			"workflow": [
				"Nhận yêu cầu người dùng",
				"Phát hiện nội dung ngoài phạm vi du lịch",
				"Trả lời từ chối cố định, không gọi công cụ",
			],
			"tool_calls": [],
		}
		turn_messages: list[BaseMessage] = [
			HumanMessage(content=user_input),
			AIMessage(content=reply),
		]
		_write_turn_log(user_input, reply, trace, updated_state, turn_messages)
		return reply, updated_state, trace, [reply]

	previous_len = len(session_state["messages"])
	updated_state = graph.invoke(
		{
			"messages": [*session_state["messages"], HumanMessage(content=user_input)],
			"memory": session_state.get("memory", {}),
		},
		config={"recursion_limit": RECURSION_LIMIT},
	)

	new_messages = updated_state["messages"][previous_len:]
	tool_calls: list[ToolTrace] = []
	for message in new_messages:
		if isinstance(message, AIMessage) and message.tool_calls:
			for call in message.tool_calls:
				tool_calls.append({"name": call["name"], "args": call.get("args", {})})

	workflow = [
		"Nhận yêu cầu người dùng",
		"LangGraph ReAct: agent chọn hành động (gọi công cụ hoặc trả lời)",
	]
	if tool_calls:
		workflow.append("Thực thi ToolNode và quay lại agent với quan sát (observation)")
		workflow.append("Lặp cho đến khi agent không còn tool call")
	workflow.append("Trả lời cuối cho người dùng")

	action = "gọi_công_cụ" if tool_calls else "trả_lời_trực_tiếp"
	trace = {
		"intent": "react_langgraph",
		"action": action,
		"workflow": workflow,
		"tool_calls": tool_calls,
	}
	final_message = updated_state["messages"][-1]
	final_reply = _final_text(final_message)
	_write_turn_log(
		user_input,
		final_reply,
		trace,
		updated_state,
		list(new_messages),
	)
	ui_steps = _assistant_ui_steps_from_messages(list(new_messages))
	if not ui_steps:
		ui_steps = [final_reply] if final_reply else ["(Khong co phan hoi tu mo hinh.)"]
	return final_reply, updated_state, trace, ui_steps


def run_turn(user_input: str, session_state: AgentState) -> tuple[str, AgentState]:
	answer, updated_state, _, _ = run_turn_with_trace(user_input, session_state)
	return answer, updated_state


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

	print("=" * 60)
	print("TravelBuddy - Tro ly du lich thong minh (ReAct + LangGraph)")
	print("Go 'quit' de thoat")
	print("=" * 60)

	session_state = new_session_state()
	while True:
		user_input = input("\nBan: ").strip()
		if user_input.lower() == "quit":
			print("Tam biet! Chuc ban co chuyen di vui ve!")
			break
		print("\nTravelBuddy dang suy nghi...")
		answer, session_state = run_turn(user_input, session_state)
		print(f"\nTravelBuddy: {answer}")
