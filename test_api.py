import os

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
	raise SystemExit(
		"Thiếu OPENAI_API_KEY. Vui lòng thêm vào .env trước khi chạy test_api.py"
	)

from agent import graph
from langchain_core.messages import HumanMessage


def run_case(user_text: str) -> str:
	result = graph.invoke({"messages": [HumanMessage(content=user_text)], "memory": {}})
	return result["messages"][-1].content


if __name__ == "__main__":
	cases = [
		"Xin chào! Tôi muốn đi du lịch.",
		"Tìm giúp tôi chuyến bay từ Hà Nội đi Đà Nẵng",
		"Giải giúp tôi bài tập linked list Python",
	]

	for index, prompt in enumerate(cases, start=1):
		print(f"\n=== CASE {index} ===")
		print(f"User: {prompt}")
		print(f"Assistant: {run_case(prompt)}")
