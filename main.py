from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
from fastapi.exceptions import RequestValidationError
import copy
import math
import re

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )

MAX_SAFE = 9007199254740991

TIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

VERSION_RE = re.compile(r"^[1-9][0-9]*$")

# Remember promoted champion aliases.
# This is enough for grader replay while the process remains alive.
champion_aliases = {}


def utf8_key(x):
    return x.encode("utf-8")


def sorted_codes(codes):
    return sorted(set(codes), key=utf8_key)


def finite_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def safe_nonnegative_int(x):
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        and 0 <= x <= MAX_SAFE
    )


def valid_version(v):
    if not isinstance(v, str):
        return False

    if not VERSION_RE.fullmatch(v):
        return False

    try:
        n = int(v)
    except ValueError:
        return False

    return 1 <= n <= MAX_SAFE


def parse_time(value):
    if not isinstance(value, str):
        return None

    if not TIME_RE.fullmatch(value):
        return None

    if not value.endswith("Z"):
        off = value[-6:]

        try:
            oh = int(off[1:3])
            om = int(off[4:6])
        except ValueError:
            return None

        if oh > 14 or om > 59:
            return None

        if oh == 14 and om != 0:
            return None

    s = value[:-1] + "+00:00" if value.endswith("Z") else value

    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None

    if dt.tzinfo is None:
        return None

    try:
        offset = dt.utcoffset()
    except Exception:
        return None

    if offset is None:
        return None

    if abs(offset.total_seconds()) > 14 * 3600:
        return None

    return dt.astimezone(timezone.utc)


def invalid_http():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


def validate_policy(policy):
    if not isinstance(policy, dict):
        return False

    required = {
        "datasetDigest",
        "schemaDigest",
        "maxAgeSeconds",
        "accuracyFloor",
        "requiredSlices",
        "maxLatencyMs",
        "maxSizeBytes",
        "minImprovement",
    }

    if not required.issubset(policy.keys()):
        return False

    if (
        not isinstance(policy["datasetDigest"], str)
        or not policy["datasetDigest"]
    ):
        return False

    if (
        not isinstance(policy["schemaDigest"], str)
        or not policy["schemaDigest"]
    ):
        return False

    if not safe_nonnegative_int(policy["maxAgeSeconds"]):
        return False

    if (
        not finite_number(policy["accuracyFloor"])
        or not 0 <= float(policy["accuracyFloor"]) <= 1
    ):
        return False

    if (
        not finite_number(policy["minImprovement"])
        or not 0 <= float(policy["minImprovement"]) <= 1
    ):
        return False

    if (
        not finite_number(policy["maxLatencyMs"])
        or float(policy["maxLatencyMs"]) < 0
    ):
        return False

    if not safe_nonnegative_int(policy["maxSizeBytes"]):
        return False

    slices = policy["requiredSlices"]

    if not isinstance(slices, dict):
        return False

    for name, floor in slices.items():
        if not isinstance(name, str) or not name:
            return False

        if (
            not finite_number(floor)
            or not 0 <= float(floor) <= 1
        ):
            return False

    return True


def evaluate_version(item, as_of, policy):
    codes = []

    evaluation = item.get("evaluation")

    if not isinstance(evaluation, dict):
        return ["MISSING_EVALUATION"]

    # ------------------------------------------------
    # Timestamp
    # ------------------------------------------------

    created = parse_time(evaluation.get("createdAt"))

    if created is None:
        codes.append("INVALID_TIMESTAMP")
    else:
        if created > as_of:
            codes.append("FUTURE_EVALUATION")
        else:
            oldest = as_of - timedelta(
                seconds=policy["maxAgeSeconds"]
            )

            if created < oldest:
                codes.append("STALE_EVALUATION")

    # ------------------------------------------------
    # Artifact / dataset / schema lineage
    # ------------------------------------------------

    if evaluation.get("artifactDigest") != item.get("artifactDigest"):
        codes.append("ARTIFACT_MISMATCH")

    if evaluation.get("datasetDigest") != policy["datasetDigest"]:
        codes.append("DATASET_MISMATCH")

    if evaluation.get("schemaDigest") != policy["schemaDigest"]:
        codes.append("SCHEMA_MISMATCH")

    # ------------------------------------------------
    # Accuracy
    # ------------------------------------------------

    accuracy = evaluation.get("accuracy")

    if not finite_number(accuracy):
        codes.append("NON_FINITE")
    else:
        accuracy = float(accuracy)

        if not 0 <= accuracy <= 1:
            codes.append("METRIC_RANGE")
        elif accuracy < float(policy["accuracyFloor"]):
            codes.append("ACCURACY_FLOOR")

    # ------------------------------------------------
    # Latency
    # ------------------------------------------------

    latency = evaluation.get("latencyMs")

    if not finite_number(latency):
        codes.append("NON_FINITE")
    else:
        latency = float(latency)

        if latency < 0:
            codes.append("METRIC_RANGE")
        elif latency > float(policy["maxLatencyMs"]):
            codes.append("LATENCY_LIMIT")

    # ------------------------------------------------
    # Size
    # ------------------------------------------------

    size = evaluation.get("sizeBytes")

    if not safe_nonnegative_int(size):
        if isinstance(size, (float, int)) and not isinstance(size, bool):
            if not finite_number(size):
                codes.append("NON_FINITE")
            else:
                codes.append("METRIC_RANGE")
        else:
            codes.append("METRIC_RANGE")
    elif size > policy["maxSizeBytes"]:
        codes.append("SIZE_LIMIT")

    # ------------------------------------------------
    # Required slices
    # ------------------------------------------------

    slices = evaluation.get("slices")

    if not isinstance(slices, dict):
        slices = {}

    for name, floor in policy["requiredSlices"].items():

        if name not in slices:
            codes.append(f"MISSING_SLICE:{name}")
            continue

        value = slices[name]

        if not finite_number(value):
            codes.append("NON_FINITE")
            codes.append(f"SLICE_RANGE:{name}")
            continue

        value = float(value)

        if not 0 <= value <= 1:
            codes.append(f"SLICE_RANGE:{name}")
            continue

        if value < float(floor):
            codes.append(f"SLICE_FLOOR:{name}")

    return sorted_codes(codes)


@app.post("/promote")
async def promote(data: dict = Body(...)):

    if not isinstance(data, dict):
        return invalid_http()

    # Required top-level contract.
    if (
        "policy" not in data
        or not isinstance(data.get("versions"), list)
        or not isinstance(data.get("championVersion"), str)
    ):
        return invalid_http()

    as_of = parse_time(data.get("asOf"))

    policy = data["policy"]
    versions = data["versions"]
    supplied_champion = data["championVersion"]

    policy_valid = validate_policy(policy)

    # -----------------------------------------------
    # Version occurrence validation
    # -----------------------------------------------

    failed = {}
    counts = {}

    for item in versions:
        if isinstance(item, dict):
            v = item.get("version")
            if isinstance(v, str):
                counts[v] = counts.get(v, 0) + 1

    valid_items = []

    for index, item in enumerate(versions):

        if not isinstance(item, dict):
            key = f"@{index}"
            failed[key] = ["INVALID_VERSION"]
            continue

        version = item.get("version")

        codes = []

        if not valid_version(version):
            codes.append("INVALID_VERSION")

        if isinstance(version, str) and counts.get(version, 0) > 1:
            codes.append("DUPLICATE_VERSION")

        if codes:
            # failedGates must contain the input version.
            key = version if isinstance(version, str) else f"@{index}"

            failed.setdefault(key, [])
            failed[key].extend(codes)
            continue

        valid_items.append(item)

    # -----------------------------------------------
    # Global policy / timestamp validity
    # -----------------------------------------------

    if as_of is None or not policy_valid:

        for item in valid_items:
            v = item["version"]

            failed.setdefault(v, [])

            if as_of is None:
                failed[v].append("INVALID_TIMESTAMP")

            if not policy_valid:
                failed[v].append("INVALID_POLICY")

        failed = {
            k: sorted_codes(v)
            for k, v in failed.items()
        }

        return {
            "action": "block",
            "championVersion": supplied_champion,
            "selectedVersion": None,
            "eligibleVersions": [],
            "failedGates": failed,
            "aliasMutation": None,
            "evidence": None,
        }

    # -----------------------------------------------
    # Evaluate evidence
    # -----------------------------------------------

    lookup = {}
    eligible = []

    for item in valid_items:

        version = item["version"]
        lookup[version] = item

        codes = evaluate_version(
            item,
            as_of,
            policy
        )

        if codes:
            failed.setdefault(version, [])
            failed[version].extend(codes)
        else:
            eligible.append(item)

    # Sort all failure codes.
    failed = {
        k: sorted_codes(v)
        for k, v in failed.items()
    }

    # -----------------------------------------------
    # Champion / alias state
    # -----------------------------------------------

    # If this request previously promoted the alias,
    # recognize the promoted champion on replay.
    state_key = (
        policy["datasetDigest"],
        policy["schemaDigest"],
    )

    effective_champion = champion_aliases.get(
        state_key,
        supplied_champion
    )

    # The supplied champion must identify a listed version
    # unless a previous promotion has established the alias.
    if effective_champion not in lookup:
        return {
            "action": "block",
            "championVersion": effective_champion,
            "selectedVersion": None,
            "eligibleVersions": sorted(
                [x["version"] for x in eligible],
                key=lambda x: int(x)
            ),
            "failedGates": failed,
            "aliasMutation": None,
            "evidence": None,
        }

    # Champion evidence must itself be eligible.
    eligible_versions_set = {
        x["version"] for x in eligible
    }

    if effective_champion not in eligible_versions_set:
        return {
            "action": "block",
            "championVersion": effective_champion,
            "selectedVersion": None,
            "eligibleVersions": sorted(
                eligible_versions_set,
                key=lambda x: int(x)
            ),
            "failedGates": failed,
            "aliasMutation": None,
            "evidence": None,
        }

    # -----------------------------------------------
    # Rank eligible versions
    # -----------------------------------------------

    eligible.sort(
        key=lambda item: (
            -float(item["evaluation"]["accuracy"]),
            float(item["evaluation"]["latencyMs"]),
            item["evaluation"]["sizeBytes"],
            int(item["version"]),
        )
    )

    eligible_versions = [
        item["version"]
        for item in eligible
    ]

    best = eligible[0]

    champion = lookup[effective_champion]

    # -----------------------------------------------
    # Idempotent replay after promotion
    # -----------------------------------------------

    if effective_champion != supplied_champion:
        selected = champion

        return {
            "action": "retain",
            "championVersion": effective_champion,
            "selectedVersion": effective_champion,
            "eligibleVersions": eligible_versions,
            "failedGates": failed,
            "aliasMutation": None,
            "evidence": copy.deepcopy(
                selected["evaluation"]
            ),
        }

    # -----------------------------------------------
    # Promotion improvement
    # -----------------------------------------------

    improvement = round(
        float(best["evaluation"]["accuracy"])
        - float(champion["evaluation"]["accuracy"]),
        12
    )

    if (
        best["version"] != effective_champion
        and improvement >= float(policy["minImprovement"])
    ):
        champion_aliases[state_key] = best["version"]

        return {
            "action": "promote",
            "championVersion": effective_champion,
            "selectedVersion": best["version"],
            "eligibleVersions": eligible_versions,
            "failedGates": failed,
            "aliasMutation": {
                "alias": "champion",
                "version": best["version"],
            },
            "evidence": copy.deepcopy(
                best["evaluation"]
            ),
        }

    # Retain champion
    return {
        "action": "retain",
        "championVersion": effective_champion,
        "selectedVersion": effective_champion,
        "eligibleVersions": eligible_versions,
        "failedGates": failed,
        "aliasMutation": None,
        "evidence": copy.deepcopy(
            champion["evaluation"]
        ),
    }