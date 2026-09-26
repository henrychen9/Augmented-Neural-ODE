#!/bin/bash

set -e

DATES_FILE="half_dates.txt"
MARKER_FILE="activity_marker.txt"

if [ ! -f "$DATES_FILE" ]; then
  echo "Missing $DATES_FILE"
  exit 1
fi

count=0
toggle=0

while IFS= read -r DATE || [ -n "$DATE" ]; do
  [ -z "$DATE" ] && continue

  HOUR=$((10 + RANDOM % 14))
  MINUTE=$((RANDOM % 30))
  printf -v TIME "%02d:%02d:00" "$HOUR" "$MINUTE"

  TZ_OFFSET=$(date -j -f "%Y-%m-%d %H:%M:%S" "$DATE 12:00:00" "+%z")
  FULL_DATE="${DATE}T${TIME}${TZ_OFFSET}"

  if [ "$toggle" -eq 0 ]; then
    echo "1" >> "$MARKER_FILE"
    toggle=1
  else
    sed -i '' '$d' "$MARKER_FILE"
    toggle=0
  fi

  git add "$MARKER_FILE"

  GIT_AUTHOR_DATE="$FULL_DATE" \
  GIT_COMMITTER_DATE="$FULL_DATE" \
  git commit -m "project update"

  count=$((count + 1))
  echo "[$count] committed $FULL_DATE"

done < "$DATES_FILE"

echo
echo "Finished."
echo "Created $count commits."