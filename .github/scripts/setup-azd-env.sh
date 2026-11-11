#!/usr/bin/env bash
# Reconstruct the existing azd environment on an ephemeral CI runner.
#
# The checked-in env file contains only identifiers and endpoints. It never
# contains credentials, tokens, keys, connection strings, or the azd salt.
set -euo pipefail

ENV_FILE="${AZD_ENV_FILE:-.github/azd-env/rag-agent-dev.env}"
ENV_NAME="${AZD_ENV_NAME:-rag-agent-dev}"
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-}"
LOCATION="${AZURE_LOCATION:-southindia}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing azd environment template: $ENV_FILE" >&2
  exit 1
fi

if ! azd env select "$ENV_NAME" >/dev/null 2>&1; then
  azd env new "$ENV_NAME" \
    --subscription "${SUBSCRIPTION_ID:-$(awk -F= '$1=="AZURE_SUBSCRIPTION_ID" {print $2}' "$ENV_FILE")}" \
    --location "$LOCATION" \
    --no-prompt >/dev/null
fi

while IFS='=' read -r key value; do
  if [[ -z "$key" || "$key" == \#* ]]; then
    continue
  fi
  azd env set "$key" "$value" >/dev/null
done < "$ENV_FILE"

azd env get-values >/dev/null
echo "Configured azd environment '$ENV_NAME' from $ENV_FILE"
