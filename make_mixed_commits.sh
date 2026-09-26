#!/bin/bash

set -e

DATES_FILE="half_dates.txt"
MARKER_FILE="activity_marker.txt"

# Read non-empty dates into a temporary file
grep -v '^$' "$DATES_FILE" > dates_clean.txt

TOTAL=$(wc -l < dates_clean.txt | tr -d ' ')

if [ "$TOTAL" -ne 152 ]; then
    echo "Expected 152 dates, found $TOTAL"
    exit 1
fi

# Randomly choose exactly 76 dates for one commit.
# macOS has sort -R on many setups; if not, use Python fallback below.
python3 - <<'PY'
import random

with open("dates_clean.txt") as f:
    dates = [x.strip() for x in f if x.strip()]

random.shuffle(dates)

with open("one_commit_dates.txt", "w") as f:
    for d in dates[:76]:
        f.write(d + "\n")
PY

count=0

while IFS= read -r DATE || [ -n "$DATE" ]; do

    if grep -qx "$DATE" one_commit_dates.txt; then
        NUM=1
    else
        NUM=2
    fi

    echo "$DATE -> $NUM commit(s)"

    i=1
    while [ "$i" -le "$NUM" ]; do

        HOUR=$((10 + RANDOM % 14))
        MINUTE=$((RANDOM % 60))
        SECOND=$((RANDOM % 60))

        printf -v TIME "%02d:%02d:%02d" "$HOUR" "$MINUTE" "$SECOND"

        TZ_OFFSET=$(date -j -f "%Y-%m-%d %H:%M:%S" "$DATE 12:00:00" "+%z")
        FULL_DATE="${DATE}T${TIME}${TZ_OFFSET}"

        echo "1" >> "$MARKER_FILE"

        git add "$MARKER_FILE"

        GIT_AUTHOR_DATE="$FULL_DATE" \
        GIT_COMMITTER_DATE="$FULL_DATE" \
        git commit -m "project update"

        count=$((count + 1))
        i=$((i + 1))
    done

done < dates_clean.txt

echo
echo "DONE"
echo "152 dates total"
echo "76 dates have 1 commit"
echo "76 dates have 2 commits"
echo "Total new commits: $count"