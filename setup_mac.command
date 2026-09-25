#!/usr/bin/env bash
# Double-click in Finder to set up Effect Factory (see setup.sh).
cd "$(dirname "$0")"
bash ./setup.sh "$@"
status=$?
echo
read -r -p "Press Return to close this window." _
exit $status
