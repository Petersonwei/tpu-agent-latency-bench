#!/usr/bin/env bash
# Money gate: make sure a GCP billing budget alert exists for this project BEFORE any TPU spend.
#   bash scripts/budget_guard.sh check                 # exit 0 if a guard budget exists, 1 if not (read-only)
#   AMOUNT=38 bash scripts/budget_guard.sh create      # AMOUNT is in the BILLING ACCOUNT's currency
#   DRY_RUN=1 AMOUNT=38 bash scripts/budget_guard.sh create   # print the commands only
# A budget only ALERTS (email), it never stops spending. The hard cost cap is flex-start's
# --max-run-duration (see tpu_create.sh). Lessons baked in (2026-11-11):
#   * the Budget API (billingbudgets.googleapis.com) is off by default -> enable it first
#   * the amount must be in the billing account's currency (an AUD account rejects "25USD"
#     with a bare INVALID_ARGUMENT)
#   * filter by project NUMBER, and exclude credits, or sprint credits hide all spend and
#     the alert never fires
set -euo pipefail
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
NAME=${BUDGET_NAME:-TPU sprint guard}
run(){ echo "+ $*"; [[ "${DRY_RUN:-0}" == "1" ]] || "$@"; }

[[ -n "$PROJECT" ]] || { echo "no project: set PROJECT or gcloud config set project"; exit 2; }
PN=$(gcloud projects describe "$PROJECT" --format="value(projectNumber)")
BA=$(gcloud billing projects describe "$PROJECT" --format="value(billingAccountName)" | sed 's#billingAccounts/##')
[[ -n "$BA" ]] || { echo "project $PROJECT has no billing account linked"; exit 2; }
CUR=$(gcloud billing accounts describe "$BA" --format="value(currencyCode)")
echo "project=$PROJECT (#$PN) billing_account=$BA currency=$CUR"

api_on(){ gcloud services list --enabled --project "$PROJECT" --filter="name:billingbudgets.googleapis.com" \
            --format="value(name)" 2>/dev/null | grep -q billingbudgets; }

find_guard(){  # prints matching budgets; succeeds if at least one covers this project and excludes credits
  gcloud billing budgets list --billing-account="$BA" --billing-project="$PROJECT" --format=json 2>/dev/null |
  PN="$PN" python3 -c '
import json, os, sys
pn = "projects/" + os.environ["PN"]
ok = False
for b in json.load(sys.stdin) or []:
    f = b.get("budgetFilter", {})
    projects = f.get("projects")
    covers = (not projects) or pn in projects          # no project filter = whole billing account
    excl = f.get("creditTypesTreatment") == "EXCLUDE_ALL_CREDITS"
    amt = b.get("amount", {}).get("specifiedAmount", {})
    th = [r.get("thresholdPercent") for r in b.get("thresholdRules", [])]
    name, units, cur = b.get("displayName"), amt.get("units"), amt.get("currencyCode")
    print(f"  budget [{name}]: {units} {cur}, covers_project={covers}, excludes_credits={excl}, thresholds={th}")
    ok |= covers and excl and bool(th)
sys.exit(0 if ok else 1)'
}

case "${1:-check}" in
  check)
    if ! api_on; then echo "BUDGET GUARD: MISSING (Budget API not enabled, so no budget can exist via API)"; exit 1; fi
    if find_guard; then echo "BUDGET GUARD: OK"; exit 0; fi
    echo "BUDGET GUARD: MISSING (no budget covering $PROJECT with credits excluded and alert thresholds)"; exit 1;;
  create)
    [[ -n "${AMOUNT:-}" ]] || { echo "set AMOUNT in $CUR (e.g. AMOUNT=38 for ~US\$25 when CUR=AUD)"; exit 2; }
    api_on || run gcloud services enable billingbudgets.googleapis.com --project "$PROJECT"
    if [[ "${DRY_RUN:-0}" != "1" ]] && find_guard; then echo "a guard budget already exists; not creating another"; exit 0; fi
    run gcloud billing budgets create --billing-account="$BA" --billing-project="$PROJECT" \
      --display-name="$NAME" --budget-amount="${AMOUNT}${CUR}" --filter-projects="projects/$PN" \
      --credit-types-treatment=exclude-all-credits \
      --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0
    [[ "${DRY_RUN:-0}" == "1" ]] || { find_guard && echo "BUDGET GUARD: OK"; };;
  *) echo "usage: $0 check|create"; exit 2;;
esac
