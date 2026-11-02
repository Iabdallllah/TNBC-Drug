# Optional BLIND_MODE suppression path (v1.2.7)

Status: **DEFAULT OFF / PUBLIC.** The dashboard and exported defence report are
unblinded by default. All tables, rankings, panel predictions, kill proxies and the
Live Predictor are visible without an environment variable or supervisor key.

To re-enable suppression for a private session, set `BLIND_MODE=1` or
`BLIND_MODE=true` explicitly. Only those two values activate the blind banner; any
other value (including `0`, `OFF`, or an unset variable) leaves the deployment public.

Steps for an explicitly suppressed session: (1) supervisor provides the key (not
stored in repo); (2) `BLIND_MODE=1` or `BLIND_MODE=true` is set; (3) a timestamped
entry is appended to `dashboard_access_log.template.json`; (4) the banner is visible
and predictions/panel ranking/kill proxies/Live Predictor are suppressed.

Requires: supervisor authorization (see `SUPERVISOR_SIGNOFF.md`).
