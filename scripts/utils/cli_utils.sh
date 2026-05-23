#!/bin/bash

# Common CLI helpers for strict shell scripts.

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "Error: ${flag} requires a value." >&2
    exit 1
  fi
}

die_unknown_option() {
  local opt="$1"
  echo "Error: Unknown option '${opt}'." >&2
  echo "Run with --help for usage." >&2
  exit 1
}

require_bool_01() {
  local flag="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "Error: ${flag} must be 0 or 1." >&2
    exit 1
  fi
}
