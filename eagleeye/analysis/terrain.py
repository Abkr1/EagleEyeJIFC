"""
Terrain analysis using OpenStreetMap/Overpass APIs for area threat briefings.
Fetches real geographic data (roads, waterways, settlements, etc.) for tactical analysis.
"""

import logging
import math
import time
from typing import Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "EagleEye/1.0"
REQUEST_TIMEOUT = 30


class TerrainAnalyzer:
    """Fetches and analyzes terrain data from OpenStreetMap for area threat briefings."""

    def __init__(self):
        self._last_nominatim_request = 0.0  # Rate limit: 1 req/sec

    def _nominatim_throttle(self):
        """Ensure at least 1 second between Nominatim requests."""
        elapsed = time.time() - self._last_nominatim_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_nominatim_request = time.time()

    def geocode_area(self, area_name: str, state: str) -> Optional[dict]:
        """
        Geocode an area name within a Nigerian state using Nominatim.
        Returns dict with lat, lon, bbox, display_name or None.
        """
        self._nominatim_throttle()
        query = f"{area_name}, {state} State, Nigeria"
        try:
            resp = requests.get(
                f"{NOMINATIM_URL}/search",
                params={
                    "q": query,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1,
                    "extratags": 1,
                },
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            results = resp.json()
            if not results:
                logger.warning(f"Nominatim: no results for '{query}'")
                return None

            r = results[0]
            return {
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "display_name": r.get("display_name", ""),
                "bbox": [float(x) for x in r["boundingbox"]] if "boundingbox" in r else None,
                "type": r.get("type", ""),
                "address": r.get("address", {}),
            }
        except Exception as e:
            logger.error(f"Nominatim geocode failed for '{query}': {e}")
            return None

    def geocode_coordinates(self, lat: float, lon: float) -> Optional[dict]:
        """Reverse geocode coordinates to get area name."""
        self._nominatim_throttle()
        try:
            resp = requests.get(
                f"{NOMINATIM_URL}/reverse",
                params={
                    "lat": lat,
                    "lon": lon,
                    "format": "json",
                    "addressdetails": 1,
                    "zoom": 10,
                },
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            result = resp.json()
            if "error" in result:
                return None

            addr = result.get("address", {})
            area_name = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("county")
                or addr.get("state_district")
                or result.get("display_name", "Unknown")
            )

            return {
                "lat": lat,
                "lon": lon,
                "area_name": area_name,
                "display_name": result.get("display_name", ""),
                "address": addr,
            }
        except Exception as e:
            logger.error(f"Nominatim reverse geocode failed: {e}")
            return None

    def fetch_terrain_features(self, lat: float, lon: float,
                               radius_km: float = 15) -> dict:
        """
        Query Overpass API for terrain features within radius of coordinates.
        Returns categorized terrain data.
        """
        radius_m = int(radius_km * 1000)

        # Overpass QL query for multiple feature types
        query = f"""
[out:json][timeout:25];
(
  // Roads
  way["highway"~"trunk|primary|secondary|tertiary|unclassified|track|path"](around:{radius_m},{lat},{lon});
  // Waterways
  way["waterway"~"river|stream|canal|drain"](around:{radius_m},{lat},{lon});
  node["waterway"="waterfall"](around:{radius_m},{lat},{lon});
  // Settlements
  node["place"~"city|town|village|hamlet"](around:{radius_m},{lat},{lon});
  // Land use
  way["landuse"~"farmland|forest|meadow|residential|industrial|commercial"](around:{radius_m},{lat},{lon});
  way["natural"~"wood|scrub|grassland|wetland|water|hill|peak|valley"](around:{radius_m},{lat},{lon});
  node["natural"~"peak|hill|saddle|spring"](around:{radius_m},{lat},{lon});
  // Infrastructure
  node["amenity"~"school|hospital|clinic|police|fire_station|marketplace|fuel"](around:{radius_m},{lat},{lon});
  way["bridge"="yes"](around:{radius_m},{lat},{lon});
  node["man_made"~"tower|water_tower|mast"](around:{radius_m},{lat},{lon});
);
out body;
>;
out skel qt;
"""

        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])
            logger.info(f"Overpass returned {len(elements)} elements for ({lat}, {lon})")
            return self._categorize_features(elements, lat, lon)
        except Exception as e:
            logger.error(f"Overpass query failed: {e}")
            return {
                "roads": [], "waterways": [], "settlements": [],
                "land_use": [], "natural_features": [], "infrastructure": [],
                "error": str(e),
            }

    def _categorize_features(self, elements: list, center_lat: float,
                             center_lon: float) -> dict:
        """Parse Overpass JSON elements into terrain categories."""
        roads = []
        waterways = []
        settlements = []
        land_use = []
        natural_features = []
        infrastructure = []

        for el in elements:
            tags = el.get("tags", {})
            if not tags:
                continue

            # Calculate distance & bearing if coordinates available
            el_lat = el.get("lat")
            el_lon = el.get("lon")
            dist = None
            bearing = None
            if el_lat and el_lon:
                dist = round(self._haversine(center_lat, center_lon, el_lat, el_lon), 1)
                bearing = self._bearing_label(
                    self._bearing(center_lat, center_lon, el_lat, el_lon)
                )

            name = tags.get("name", "")
            feature_info = {"name": name, "distance_km": dist, "direction": bearing}

            # Classify
            if "highway" in tags:
                roads.append({
                    **feature_info,
                    "road_type": tags["highway"],
                    "surface": tags.get("surface", "unknown"),
                })
            elif "waterway" in tags:
                waterways.append({
                    **feature_info,
                    "waterway_type": tags["waterway"],
                })
            elif "place" in tags:
                settlements.append({
                    **feature_info,
                    "place_type": tags["place"],
                    "population": tags.get("population", "unknown"),
                })
            elif "landuse" in tags:
                land_use.append({
                    **feature_info,
                    "use_type": tags["landuse"],
                })
            elif "natural" in tags:
                natural_features.append({
                    **feature_info,
                    "feature_type": tags["natural"],
                    "elevation": tags.get("ele", ""),
                })
            elif any(k in tags for k in ("amenity", "bridge", "man_made")):
                infra_type = tags.get("amenity") or tags.get("man_made") or "bridge"
                infrastructure.append({
                    **feature_info,
                    "infra_type": infra_type,
                })

        # Deduplicate and sort by distance
        for lst in [roads, waterways, settlements, land_use, natural_features, infrastructure]:
            lst.sort(key=lambda x: x.get("distance_km") or 999)

        # Summarize unique road types
        road_summary = {}
        for r in roads:
            rt = r["road_type"]
            road_summary[rt] = road_summary.get(rt, 0) + 1

        return {
            "roads": roads[:30],
            "road_summary": road_summary,
            "waterways": waterways[:20],
            "settlements": settlements[:25],
            "land_use": land_use[:20],
            "natural_features": natural_features[:20],
            "infrastructure": infrastructure[:20],
        }

    def analyze_terrain(self, area_name: str = "", state: str = "",
                        lat: float = None, lon: float = None) -> dict:
        """
        Main method: geocode + fetch terrain + build structured summary.
        Provide either (area_name, state) or (lat, lon, state).
        """
        area_info = None

        # Geocode
        if lat is not None and lon is not None:
            area_info = self.geocode_coordinates(lat, lon)
            if area_info:
                area_info["state"] = state
            else:
                area_info = {
                    "lat": lat, "lon": lon, "state": state,
                    "area_name": area_name or "Unknown",
                    "display_name": f"{lat}, {lon}",
                }
        elif area_name and state:
            geo = self.geocode_area(area_name, state)
            if geo:
                lat, lon = geo["lat"], geo["lon"]
                area_info = {
                    "lat": lat, "lon": lon,
                    "area_name": area_name,
                    "state": state,
                    "display_name": geo["display_name"],
                    "bbox": geo.get("bbox"),
                }
            else:
                return {
                    "error": f"Could not geocode '{area_name}' in {state} State",
                    "area_name": area_name,
                    "state": state,
                }
        else:
            return {"error": "Provide either (area_name, state) or (lat, lon)"}

        # Fetch terrain
        terrain = self.fetch_terrain_features(lat, lon)

        # Build tactical summary
        tactical = self._build_tactical_summary(terrain, area_info)

        return {
            "area_info": area_info,
            **terrain,
            "tactical_summary": tactical,
        }

    def _build_tactical_summary(self, terrain: dict, area_info: dict) -> dict:
        """Build a structured tactical summary from terrain data."""
        settlements = terrain.get("settlements", [])
        roads = terrain.get("roads", [])
        waterways = terrain.get("waterways", [])
        infrastructure = terrain.get("infrastructure", [])
        natural = terrain.get("natural_features", [])
        land = terrain.get("land_use", [])
        road_summary = terrain.get("road_summary", {})

        # Count nearby settlements by type
        settlement_counts = {}
        for s in settlements:
            pt = s.get("place_type", "other")
            settlement_counts[pt] = settlement_counts.get(pt, 0) + 1

        # Key infrastructure
        key_infra = [i for i in infrastructure if i.get("infra_type") in
                     ("police", "hospital", "clinic", "fire_station", "fuel")]

        # Obstacles (waterways, wetlands, forests)
        obstacles = []
        for w in waterways:
            if w.get("name"):
                obstacles.append(f"{w['waterway_type'].title()}: {w['name']}")
        for n in natural:
            if n.get("feature_type") in ("wetland", "water"):
                obstacles.append(f"{n['feature_type'].title()}: {n.get('name', 'unnamed')}")

        # Avenues of approach (major roads)
        avenues = []
        for r in roads:
            if r.get("road_type") in ("trunk", "primary", "secondary") and r.get("name"):
                avenues.append({
                    "route": r["name"],
                    "type": r["road_type"],
                    "direction": r.get("direction", ""),
                })

        # Cover & concealment (forests, scrub)
        cover_types = []
        for feat in natural + land:
            ft = feat.get("feature_type") or feat.get("use_type", "")
            if ft in ("wood", "forest", "scrub", "grassland"):
                cover_types.append(ft)

        return {
            "total_settlements": len(settlements),
            "settlement_breakdown": settlement_counts,
            "total_roads": len(roads),
            "road_types": road_summary,
            "total_waterways": len(waterways),
            "key_infrastructure": key_infra[:10],
            "obstacles": obstacles[:10],
            "avenues_of_approach": avenues[:10],
            "cover_concealment_types": list(set(cover_types)),
            "natural_features_count": len(natural),
            "area_name": area_info.get("area_name", ""),
        }

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in km between two points."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing in degrees from point 1 to point 2."""
        dlon = math.radians(lon2 - lon1)
        lat1r = math.radians(lat1)
        lat2r = math.radians(lat2)
        x = math.sin(dlon) * math.cos(lat2r)
        y = (math.cos(lat1r) * math.sin(lat2r) -
             math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    @staticmethod
    def _bearing_label(degrees: float) -> str:
        """Convert bearing degrees to compass direction label."""
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = round(degrees / 22.5) % 16
        return dirs[idx]
