from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from models.domain import CoolingSiteRecord, SystemRecord

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
CONFIG_DIR = APP_ROOT / "config"


@lru_cache(maxsize=1)
def _load_json(file_name: str) -> dict:
    with (CONFIG_DIR / file_name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_systems() -> list[SystemRecord]:
    payload = _load_json("systems_registry.json")
    systems = []
    for item in payload.get("systems", []):
        systems.append(SystemRecord(**item))
    return systems


@lru_cache(maxsize=1)
def load_cooling_sites() -> list[CoolingSiteRecord]:
    payload = _load_json("cooling_sites.json")
    sites = []
    for item in payload.get("sites", []):
        site_core = {
            "site_id": item["site_id"],
            "name": item["name"],
            "city": item["city"],
            "lat": item["lat"],
            "lon": item["lon"],
            "surface_tilt": item["surface_tilt"],
            "surface_azimuth": item["surface_azimuth"],
            "design_cooling_load_kw": item["design_cooling_load_kw"],
            "status": item["status"],
        }
        sites.append(CoolingSiteRecord(**site_core))
    return sites


@lru_cache(maxsize=1)
def load_cooling_site_payloads() -> list[dict]:
    payload = _load_json("cooling_sites.json")
    return payload.get("sites", [])


@lru_cache(maxsize=1)
def load_model_defaults() -> dict:
    return _load_json("model_defaults.json")


def get_system_options() -> list[dict]:
    return [
        {
            "label": f"{system.name} | {system.location} | {system.capacity_kw:.3g} kW",
            "value": system.system_number,
        }
        for system in load_systems()
    ]


def get_system(system_number: int | None) -> SystemRecord | None:
    if system_number is None:
        return None
    for system in load_systems():
        if system.system_number == system_number:
            return system
    return None
