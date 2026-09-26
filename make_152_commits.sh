#!/bin/bash

set -e

DATES_FILE="half_dates.txt"
MARKER_FILE="activity_marker.txt"

if [ ! -f "$DATES_FILE" ]; then
    echo "ERROR: half_dates.txt not found"
    exit 1
fi

grep -v '^$' "$DATES_FILE" > dates_clean.txt

TOTAL=$(wc -l < dates_clean.txt | tr -d ' ')

if [ "$TOTAL" -ne 152 ]; then
    echo "ERROR: Expected 152 dates, found $TOTAL"
    exit 1
fi

echo "Found 152 dates."
echo "Creating exactly ONE commit per date."
echo

count=0

while IFS= read -r DATE || [ -n "$DATE" ]; do

    count=$((count + 1))

    HOUR=$((10 + RANDOM % 13))
    MINUTE=$((RANDOM % 60))
    SECOND=$((RANDOM % 60))

    printf -v TIME "%02d:%02d:%02d" "$HOUR" "$MINUTE" "$SECOND"

    TZ_OFFSET=$(date -j -f "%Y-%m-%d %H:%M:%S" "$DATE 12:00:00" "+%z")

    FULL_DATE="${DATE}T${TIME}${TZ_OFFSET}"

    echo "activity-$count" >> "$MARKER_FILE"

    git add "$MARKER_FILE"

    GIT_AUTHOR_DATE="$FULL_DATE" \
    GIT_COMMITTER_DATE="$FULL_DATE" \
    git commit -m "project update"

    echo "[$count/152] $DATE -> 1 commit"

done < dates_clean.txt

echo
echo "Finished creating commits."
echo "Total created: $count"

echo
echo "Verifying dates..."

git log \
    --format="%ad" \
    --date=format:"%Y-%m-%d" \
    --grep="project update" \
    -n 152 \
    | sort \
    | uniq -c \
    > contribution_check.txt

cat contribution_check.txt

DUPES=$(awk '$1 != 1 {print}' contribution_check.txt)

if [ -n "$DUPES" ]; then
    echo
    echo "ERROR: Some dates do not have exactly 1 commit:"
    echo "$DUPES"
    echo "DO NOT PUSH."
    exit 1
fi

COUNT=$(wc -l < contribution_check.txt | tr -d ' ')

if [ "$COUNT" -ne 152 ]; then
    echo "ERROR: Expected 152 unique dates, found $COUNT"
    echo "DO NOT PUSH."
    exit 1
fi

echo
echo "SUCCESS."
echo "152 unique dates."
echo "Exactly 1 generated commit on every date."
echo "Safe to push."