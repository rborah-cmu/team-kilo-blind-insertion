"""Phase 1: find the box and the hole by touch."""

from .common import (
    SEARCH_MODE_HEIGHT,
    TRAVEL_MODE_HEIGHT,
    FORCE_THRESHOLD,
    SEARCH_SPACE_CORNERS,
    TABLETOP_HEIGHT,
    VIRTUAL_PROBE_LENGTH,
)
from .archive.contact import ContactDetector
from .archive.hole_filter import HoleSearchFilter
from .search import generate_snake_waypoints, get_ray_intersection

__all__ = [
    "ContactDetector",
    "SEARCH_MODE_HEIGHT",
    "TRAVEL_MODE_HEIGHT",
    "FORCE_THRESHOLD",
    "HoleSearchFilter",
    "SEARCH_SPACE_CORNERS",
    "TABLETOP_HEIGHT",
    "VIRTUAL_PROBE_LENGTH",
    "generate_snake_waypoints",
    "get_ray_intersection",
]
