"""
tracker.py
----------
Tracker configuration helper for Ultralytics built-in ByteTrack / BoT-SORT.

Ultralytics 8.x bundles both trackers as YAML config files that are passed
directly to model.track(). This module exposes a clean API to select and
describe each tracker without requiring any extra third-party packages.
"""

from dataclasses import dataclass


@dataclass
class TrackerProfile:
    """Describes a tracker option exposed in the Streamlit sidebar."""
    name: str
    config_file: str          # Passed to model.track(tracker=...)
    description: str
    strengths: list[str]
    best_for: str


# ── Available Trackers ───────────────────────────────────────────────────────

TRACKER_PROFILES: dict[str, TrackerProfile] = {
    "ByteTrack": TrackerProfile(
        name="ByteTrack",
        config_file="bytetrack.yaml",
        description=(
            "ByteTrack: high-performance multi-object tracker using a two-stage "
            "matching strategy. Keeps low-confidence detections in a secondary "
            "buffer to reduce ID switches during occlusion."
        ),
        strengths=[
            "Stable IDs through brief occlusion",
            "Low computational overhead",
            "Works well with dense crowds",
            "Very fast association step",
        ],
        best_for="General-purpose real-time tracking (webcam & video files)",
    ),
    "BoT-SORT": TrackerProfile(
        name="BoT-SORT",
        config_file="botsort.yaml",
        description=(
            "BoT-SORT: combines motion compensation (camera-motion aware) with "
            "re-ID appearance features for more robust long-term tracking. "
            "Slightly higher compute but better across camera movement."
        ),
        strengths=[
            "Camera-motion compensation",
            "Appearance-based re-identification",
            "Better for moving camera scenarios",
            "Handles longer occlusions",
        ],
        best_for="Surveillance footage or drone video with camera motion",
    ),
}

DEFAULT_TRACKER = "ByteTrack"


# ── Public API ───────────────────────────────────────────────────────────────

def get_tracker_config(tracker_name: str) -> str:
    """
    Return the YAML config file name for the chosen tracker.

    Parameters
    ----------
    tracker_name : str
        Human-readable name: 'ByteTrack' or 'BoT-SORT'.

    Returns
    -------
    str
        Config filename passed to model.track(tracker=...).
    """
    profile = TRACKER_PROFILES.get(tracker_name)
    if profile is None:
        # Graceful fallback to ByteTrack
        return TRACKER_PROFILES[DEFAULT_TRACKER].config_file
    return profile.config_file


def get_tracker_profile(tracker_name: str) -> TrackerProfile:
    """Return the full TrackerProfile for display in the sidebar."""
    return TRACKER_PROFILES.get(tracker_name, TRACKER_PROFILES[DEFAULT_TRACKER])


def list_tracker_names() -> list[str]:
    """Return all available tracker names for a dropdown widget."""
    return list(TRACKER_PROFILES.keys())
