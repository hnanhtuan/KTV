#!/usr/bin/env bash
set -euo pipefail

SEARCH_ROOT="${1:-outputs}"

if [[ ! -d "${SEARCH_ROOT}" ]]; then
  echo "Error: directory not found: ${SEARCH_ROOT}" >&2
  exit 1
fi

latest_link="$(find "${SEARCH_ROOT}" -type l -name latest.log -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"

if [[ -z "${latest_link}" ]]; then
  echo "No experiment logs found under ${SEARCH_ROOT}" >&2
  exit 1
fi

readlink -f "${latest_link}"
