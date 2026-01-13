"""
weather.py

A small, self-contained "Weather" class demonstrating basic functionality:
- store and manage daily forecasts
- query/filter forecasts
- compute simple statistics
- serialize/deserialize to JSON

This is not a full meteorological model—just a practical example class.
"""

"Comment by Fabi"
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Iterable
import json
import os


@dataclass(frozen=True)
class Forecast:
    """Represents a single day's forecast."""
    day: date
    high_c: float
    low_c: float
    precipitation_mm: float = 0.0
    condition: str = "Unknown"
    wind_kmh: float = 0.0

    def validate(self) -> None:
        if self.low_c > self.high_c:
            raise ValueError("low_c cannot be greater than high_c")
        if self.precipitation_mm < 0:
            raise ValueError("precipitation_mm cannot be negative")
        if self.wind_kmh < 0:
            raise ValueError("wind_kmh cannot be negative")


class Weather:
    """
    Basic weather forecast manager, changing another comment.

    Assumptions:
    - Forecasts are stored by calendar day (date).
    - Temperatures are stored internally in Celsius.
    - Wind speed is stored internally in km/h.
    """

    def __init__(self, location: str, timezone: str = "UTC") -> None:
        self.location = location
        self.timezone = timezone
        self._forecasts: Dict[date, Forecast] = {}

    # ----------------------------
    # CRUD operations
    # ----------------------------

    def add_forecast(self, forecast: Forecast, overwrite: bool = True) -> None:
        """Add a forecast for a day. By default, overwrites existing entries."""
        forecast.validate()
        if (forecast.day in self._forecasts) and (not overwrite):
            raise KeyError(f"Forecast for {forecast.day.isoformat()} already exists")
        self._forecasts[forecast.day] = forecast

    def upsert_forecast(
        self,
        day: date,
        high_c: float,
        low_c: float,
        precipitation_mm: float = 0.0,
        condition: str = "Unknown",
        wind_kmh: float = 0.0,
    ) -> None:
        """Convenience method to create/replace a forecast."""
        self.add_forecast(
            Forecast(
                day=day,
                high_c=high_c,
                low_c=low_c,
                precipitation_mm=precipitation_mm,
                condition=condition,
                wind_kmh=wind_kmh,
            ),
            overwrite=True,
        )

    def remove_forecast(self, day: date) -> None:
        """Remove a forecast for a specific day. Modifying comment."""
        if day not in self._forecasts:
            raise KeyError(f"No forecast found for {day.isoformat()}")
        del self._forecasts[day]

    def get_forecast(self, day: date) -> Optional[Forecast]:
        """Get the forecast for a day, or None if not present."""
        return self._forecasts.get(day)

    def list_days(self) -> List[date]:
        """Return all days with forecasts, sorted."""
        return sorted(self._forecasts.keys())

    def clear(self) -> None:
        """Remove all forecasts."""
        self._forecasts.clear()

    # ----------------------------
    # Query helpers
    # ----------------------------

    def iter_forecasts(self) -> Iterable[Forecast]:
        """Iterate forecasts in chronological order."""
        for d in self.list_days():
            yield self._forecasts[d]

    def range(self, start: date, end: date) -> List[Forecast]:
        """Get forecasts between start and end inclusive."""
        if start > end:
            raise ValueError("start must be <= end")
        out: List[Forecast] = []
        for f in self.iter_forecasts():
            if start <= f.day <= end:
                out.append(f)
        return out

    def filter_by_condition(self, condition: str) -> List[Forecast]:
        """Return forecasts where condition matches case-insensitively."""
        needle = condition.strip().lower()
        return [f for f in self.iter_forecasts() if f.condition.strip().lower() == needle]

    # ----------------------------
    # Stats
    # ----------------------------

    def average_high_low_c(self) -> Tuple[float, float]:
        """Compute average (high_c, low_c)."""
        forecasts = list(self.iter_forecasts())
        if not forecasts:
            raise ValueError("No forecasts available")
        avg_high = sum(f.high_c for f in forecasts) / len(forecasts)
        avg_low = sum(f.low_c for f in forecasts) / len(forecasts)
        return (avg_high, avg_low)

    def hottest_day(self) -> Forecast:
        """Return the day with the maximum high temperature."""
        forecasts = list(self.iter_forecasts())
        if not forecasts:
            raise ValueError("No forecasts available")
        return max(forecasts, key=lambda f: f.high_c)

    # ----------------------------
    # Unit conversions
    # ----------------------------

    @staticmethod
    def c_to_f(c: float) -> float:
        return (c * 9.0 / 5.0) + 32.0

    @staticmethod
    def f_to_c(f: float) -> float:
        return (f - 32.0) * 5.0 / 9.0

    @staticmethod
    def kmh_to_mph(kmh: float) -> float:
        return kmh * 0.621371

    @staticmethod
    def mph_to_kmh(mph: float) -> float:
        return mph / 0.621371

    # ----------------------------
    # Serialization
    # ----------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "location": self.location,
            "timezone": self.timezone,
            "country" : "Guatemala",
            "forecasts": [
                {
                    **asdict(f),
                    "day": f.day.isoformat(),
                }
                for f in self.iter_forecasts()
            ],
        }

    @staticmethod
    def _parse_date(value: str) -> date:
        return datetime.strptime(value, "%Y-%m-%d").date()

    @classmethod
    def from_dict(cls, payload: dict) -> "Weather":
        """Deserialize from a dict previously created by to_dict()."""
        obj = cls(location=payload["location"], timezone=payload.get("timezone", "UTC"))
        for item in payload.get("forecasts", []):
            f = Forecast(
                day=cls._parse_date(item["day"]),
                high_c=float(item["high_c"]),
                low_c=float(item["low_c"]),
                precipitation_mm=float(item.get("precipitation_mm", 0.0)),
                condition=str(item.get("condition", "Unknown")),
                wind_kmh=float(item.get("wind_kmh", 0.0)),
            )
            obj.add_forecast(f, overwrite=True)
        return obj

    def save_json(self, path: str) -> None:
        """Write forecasts to disk as JSON."""
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> "Weather":
        """Load forecasts from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    # ----------------------------
    # Display
    # ----------------------------

    def pretty_table(self, use_fahrenheit: bool = False) -> str:
        """Return a simple string table of forecasts."""
        rows = []
        header = ["Day", "High", "Low", "Precip (mm)", "Wind", "Condition"]
        rows.append(header)

        for f in self.iter_forecasts():
            if use_fahrenheit:
                high = f"{self.c_to_f(f.high_c):.1f}°F"
                low = f"{self.c_to_f(f.low_c):.1f}°F"
            else:
                high = f"{f.high_c:.1f}°C"
                low = f"{f.low_c:.1f}°C"

            wind = f"{f.wind_kmh:.0f} km/h"
            rows.append([f.day.isoformat(), high, low, f"{f.precipitation_mm:.1f}", wind, f.condition])

        # Basic text formatting without external dependencies
        col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
        lines = []
        for idx, row in enumerate(rows):
            line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            lines.append(line)
            if idx == 0:
                lines.append("-+-".join("-" * w for w in col_widths))
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    w = Weather(location="Guatemala City", timezone="America/Guatemala")

    w.upsert_forecast(date(2026, 1, 11), high_c=24.0, low_c=16.0, precipitation_mm=2.5, condition="Cloudy", wind_kmh=10)
    w.upsert_forecast(date(2026, 1, 12), high_c=25.5, low_c=15.5, precipitation_mm=0.0, condition="Sunny", wind_kmh=12)
    w.upsert_forecast(date(2026, 1, 13), high_c=23.0, low_c=14.0, precipitation_mm=8.2, condition="Rain", wind_kmh=18)

    print(f"Forecasts for: {w.location} ({w.timezone})")
    print(w.pretty_table())

    avg_high, avg_low = w.average_high_low_c()
    print(f"\nAverage high: {avg_high:.1f}°C, Average low: {avg_low:.1f}°C")

    hottest = w.hottest_day()
    wettest = w.wettest_day()
    print(f"Hottest day: {hottest.day.isoformat()} ({hottest.high_c:.1f}°C)")
    print(f"Wettest day: {wettest.day.isoformat()} ({wettest.precipitation_mm:.1f} mm)")

    path = "weather_sample.json"
    w.save_json(path)
    loaded = Weather.load_json(path)
    print("\nLoaded back from JSON:")
    print(loaded.pretty_table(use_fahrenheit=True))
