from typing import Literal, Union

from langchain_core.tools import tool
from pydantic import BaseModel, Field

FLIGHTS_DB = {
	("Hà Nội", "Đà Nẵng"): [
		{"airline": "Vietnam Airlines", "departure": "06:00", "arrival": "07:20", "price": 1_450_000, "class": "economy"},
		{"airline": "Vietnam Airlines", "departure": "14:00", "arrival": "15:20", "price": 2_800_000, "class": "business"},
		{"airline": "VietJet Air", "departure": "08:30", "arrival": "09:50", "price": 890_000, "class": "economy"},
		{"airline": "Bamboo Airways", "departure": "11:00", "arrival": "12:20", "price": 1_200_000, "class": "economy"},
	],
	("Hà Nội", "Phú Quốc"): [
		{"airline": "Vietnam Airlines", "departure": "07:00", "arrival": "09:15", "price": 2_100_000, "class": "economy"},
		{"airline": "VietJet Air", "departure": "10:00", "arrival": "12:15", "price": 1_350_000, "class": "economy"},
		{"airline": "VietJet Air", "departure": "16:00", "arrival": "18:15", "price": 1_100_000, "class": "economy"},
	],
	("Hà Nội", "Hồ Chí Minh"): [
		{"airline": "Vietnam Airlines", "departure": "06:00", "arrival": "08:10", "price": 1_600_000, "class": "economy"},
		{"airline": "VietJet Air", "departure": "07:30", "arrival": "09:40", "price": 950_000, "class": "economy"},
		{"airline": "Bamboo Airways", "departure": "12:00", "arrival": "14:10", "price": 1_300_000, "class": "economy"},
		{"airline": "Vietnam Airlines", "departure": "18:00", "arrival": "20:10", "price": 3_200_000, "class": "business"},
	],
	("Hồ Chí Minh", "Đà Nẵng"): [
		{"airline": "Vietnam Airlines", "departure": "09:00", "arrival": "10:20", "price": 1_300_000, "class": "economy"},
		{"airline": "VietJet Air", "departure": "13:00", "arrival": "14:20", "price": 780_000, "class": "economy"},
	],
	("Hồ Chí Minh", "Phú Quốc"): [
		{"airline": "Vietnam Airlines", "departure": "08:00", "arrival": "09:00", "price": 1_100_000, "class": "economy"},
		{"airline": "VietJet Air", "departure": "15:00", "arrival": "16:00", "price": 650_000, "class": "economy"},
	],
}

HOTELS_DB = {
	"Đà Nẵng": [
		{"name": "Mường Thanh Luxury", "stars": 5, "price_per_night": 1_800_000, "area": "Mỹ Khê", "rating": 4.5},
		{"name": "Sala Danang Beach", "stars": 4, "price_per_night": 1_200_000, "area": "Mỹ Khê", "rating": 4.3},
		{"name": "Fivitel Danang", "stars": 3, "price_per_night": 650_000, "area": "Sơn Trà", "rating": 4.1},
		{"name": "Memory Hostel", "stars": 2, "price_per_night": 250_000, "area": "Hải Châu", "rating": 4.6},
		{"name": "Christina's Homestay", "stars": 2, "price_per_night": 350_000, "area": "An Thượng", "rating": 4.7},
	],
	"Phú Quốc": [
		{"name": "Vinpearl Resort", "stars": 5, "price_per_night": 3_500_000, "area": "Bãi Dài", "rating": 4.4},
		{"name": "Sol by Meliá", "stars": 4, "price_per_night": 1_500_000, "area": "Bãi Trường", "rating": 4.2},
		{"name": "Lahana Resort", "stars": 3, "price_per_night": 800_000, "area": "Dương Đông", "rating": 4.0},
		{"name": "9Station Hostel", "stars": 2, "price_per_night": 200_000, "area": "Dương Đông", "rating": 4.5},
	],
	"Hồ Chí Minh": [
		{"name": "Rex Hotel", "stars": 5, "price_per_night": 2_800_000, "area": "Quận 1", "rating": 4.3},
		{"name": "Liberty Central", "stars": 4, "price_per_night": 1_400_000, "area": "Quận 1", "rating": 4.1},
		{"name": "Cochin Zen Hotel", "stars": 3, "price_per_night": 550_000, "area": "Quận 3", "rating": 4.4},
		{"name": "The Common Room", "stars": 2, "price_per_night": 180_000, "area": "Quận 1", "rating": 4.6},
	],
}


def format_vnd(value: int) -> str:
	return f"{value:,}".replace(",", ".") + "₫"


class SearchFlightsInput(BaseModel):
	origin: str = Field(description="Thành phố khởi hành (đúng chính tả có dấu, ví dụ: Hà Nội, Hồ Chí Minh)")
	destination: str = Field(description="Thành phố đến (đúng chính tả có dấu, ví dụ: Đà Nẵng, Phú Quốc)")
	trip_type: Literal["one_way", "round_trip"] = Field(
		default="one_way",
		description="Loại vé: một chiều hoặc khứ hồi",
	)


@tool(args_schema=SearchFlightsInput)
def search_flights(origin: str, destination: str, trip_type: str = "one_way") -> str:
	"""Tìm các chuyến bay theo cặp thành phố; nếu không có chiều xuôi sẽ thử chiều ngược trong cùng dữ liệu."""
	try:
		route = (origin.strip(), destination.strip())
		reverse_route = (destination.strip(), origin.strip())

		flights = FLIGHTS_DB.get(route)
		route_note = ""
		if flights is None:
			flights = FLIGHTS_DB.get(reverse_route)
			if flights is not None:
				route_note = (
					f"Lưu ý: không có tuyến {origin} -> {destination}, hiển thị tuyến ngược {destination} -> {origin}.\n"
				)

		if not flights:
			return f"Không tìm thấy chuyến bay từ {origin} đến {destination}."

		sorted_flights = sorted(flights, key=lambda item: item["price"])
		lines = [f"Kết quả chuyến bay {origin} -> {destination} ({trip_type}):"]
		for index, flight in enumerate(sorted_flights, start=1):
			lines.append(
				f"{index}. {flight['airline']} | {flight['departure']}-{flight['arrival']} | {flight['class']} | {format_vnd(flight['price'])}"
			)
		return route_note + "\n".join(lines)
	except Exception as exc:
		return f"Lỗi khi tìm chuyến bay: {exc}"


class SearchHotelsInput(BaseModel):
	city: str = Field(description="Thành phố cần tìm khách sạn (đúng tên trong dữ liệu, ví dụ: Đà Nẵng)")
	max_price_per_night: int = Field(
		default=99_999_999,
		ge=0,
		description="Giá tối đa mỗi đêm (VNĐ)",
	)
	min_stars: int = Field(
		default=1,
		ge=1,
		le=5,
		description="Số sao tối thiểu (1-5)",
	)


@tool(args_schema=SearchHotelsInput)
def search_hotels(city: str, max_price_per_night: int = 99_999_999, min_stars: int = 1) -> str:
	"""Tìm khách sạn theo thành phố, lọc theo trần giá mỗi đêm và số sao tối thiểu."""
	try:
		hotels = HOTELS_DB.get(city.strip())
		if not hotels:
			return "Dữ liệu không có sẵn."

		matched = [
			hotel
			for hotel in hotels
			if hotel["price_per_night"] <= max_price_per_night and hotel["stars"] >= min_stars
		]
		matched.sort(key=lambda item: item["rating"], reverse=True)

		if not matched:
			return (
				f"Không tìm thấy khách sạn tại {city} với giá dưới "
				f"{format_vnd(max_price_per_night)}/đêm. Hãy thử tăng ngân sách."
			)

		lines = [
			f"Khách sạn phù hợp tại {city} (<= {format_vnd(max_price_per_night)}/đêm, >= {min_stars} sao):"
		]
		for index, hotel in enumerate(matched[:5], start=1):
			lines.append(
				f"{index}. {hotel['name']} | {hotel['stars']} sao | {hotel['area']} | "
				f"đánh giá {hotel['rating']}/5 | {format_vnd(hotel['price_per_night'])}/đêm"
			)
		return "\n".join(lines)
	except Exception as exc:
		return f"Lỗi khi tìm khách sạn: {exc}"


class ExpenseItem(BaseModel):
	name: str = Field(description="Tên khoản chi (ví dụ: vé máy bay, khách sạn)")
	amount: int = Field(ge=0, description="Số tiền (VNĐ)")


class CalculateBudgetInput(BaseModel):
	total_budget: int = Field(ge=0, description="Tổng ngân sách ban đầu (VNĐ)")
	expenses: Union[list[ExpenseItem], str] = Field(
		description="Danh sách khoản chi hoặc chuỗi 'ten:so_tien,...' ví dụ 've_may_bay:1100000,khach_san:1600000'"
	)


def _parse_expenses(expenses: Union[list[ExpenseItem], str]) -> list[ExpenseItem]:
	if isinstance(expenses, list):
		return [item if isinstance(item, ExpenseItem) else ExpenseItem.model_validate(item) for item in expenses]

	parsed: list[ExpenseItem] = []
	parts = [part.strip() for part in expenses.split(",") if part.strip()]
	if not parts:
		raise ValueError("Danh sách chi phí trống")

	for part in parts:
		if ":" not in part:
			raise ValueError("Sai định dạng khoản chi, cần dạng ten_khoan:so_tien")
		name, raw_amount = part.split(":", maxsplit=1)
		amount = int(raw_amount.strip())
		parsed.append(ExpenseItem(name=name.strip(), amount=amount))
	return parsed


@tool(args_schema=CalculateBudgetInput)
def calculate_budget(total_budget: int, expenses: Union[list[ExpenseItem], str]) -> str:
	"""Bắt buộc gọi sau search_flights và search_hotels khi user nêu tổng ngân sách và cần bảng chi phí / số tiền còn lại. Không thay thế bằng tính tay trong chat. expenses: chuỗi 'ten:so_tien,...' (ví dụ ve_may_bay:1100000,khach_san:3000000) — tiền phòng = đơn giá một đêm từ kết quả khách sạn nhân số đêm, hoặc danh sách ExpenseItem."""
	try:
		expense_items = _parse_expenses(expenses)
		total_expense = sum(item.amount for item in expense_items)
		remaining = total_budget - total_expense

		lines = ["Bảng chi phí:"]
		for item in expense_items:
			label = item.name.replace("_", " ").strip().title()
			lines.append(f"- {label}: {format_vnd(item.amount)}")

		lines.append(f"Tổng chi: {format_vnd(total_expense)}")
		lines.append(f"Ngân sách: {format_vnd(total_budget)}")

		if remaining < 0:
			lines.append(f"Vượt ngân sách {format_vnd(abs(remaining))}! Cần điều chỉnh.")
		else:
			lines.append(f"Còn lại: {format_vnd(remaining)}")

		return "\n".join(lines)
	except Exception as exc:
		return (
			"Lỗi định dạng expenses. Hãy dùng dạng 've_may_bay:1100000,khach_san:1600000' "
			f"hoặc mảng object. Chi tiết: {exc}"
		)
