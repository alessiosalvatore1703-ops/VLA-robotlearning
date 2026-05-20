# Diagnosis: Task3 Source Dataset Metadata / Video Timestamp Issues

Date: 2026-05-20

Datasets inspected:

- `ETHrobotlearning/task3-config1-TOY-clean`
- `ETHrobotlearning/task3-config2-TYO-clean`
- `ETHrobotlearning/task3-config3-YTO-clean`
- `ETHrobotlearning/task3-config4-YOT-clean`
- `ETHrobotlearning/task3-config5-OYT-clean`
- `ETHrobotlearning/task3-config6-OTY-clean`

The check decodes the actual MP4 timestamps with PyAV and compares them with the timestamps LeRobot will query:

```text
queried_video_timestamp = meta/episodes videos/<camera>/from_timestamp + data timestamp
```

Camera key:

```text
observation.images.front
```

Tolerance used:

```text
1e-4 seconds
```

## Summary

Four datasets are structurally OK:

```text
ETHrobotlearning/task3-config1-TOY-clean
ETHrobotlearning/task3-config3-YTO-clean
ETHrobotlearning/task3-config4-YOT-clean
ETHrobotlearning/task3-config6-OTY-clean
```

Two datasets have metadata problems:

```text
ETHrobotlearning/task3-config2-TYO-clean
ETHrobotlearning/task3-config5-OYT-clean
```

The main bugs are:

1. Duplicate rows in `meta/episodes` with the same `episode_index`.
2. Some duplicate rows disagree about the video timestamp range for the same episode.
3. Some `videos/observation.images.front/from_timestamp` values point beyond the end of the referenced MP4.
4. `ETHrobotlearning/task3-config2-TYO-clean` also has inconsistent dataset counts in `meta/info.json`.

## Dataset-Level Findings

### `task3-config1-TOY-clean`

```text
info total_episodes: 45
meta/episodes rows: 45
unique episode_index in meta: 45
unique episode_index in data: 45
data rows: 4314
duplicate episode_index rows: none
timestamp issues: none
```

### `task3-config2-TYO-clean`

```text
info total_episodes: 56
meta/episodes rows: 86
unique episode_index in meta: 45
unique episode_index in data: 45
data rows: 3764
```

This is inconsistent. `info.json` says 56 episodes, but the actual data has only 45 unique episodes. `meta/episodes` has 86 rows because many episode indices are duplicated.

Duplicated `episode_index` values:

```text
5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44
```

This looks like an append / resume / rewrite bug: episode metadata rows were appended without removing or replacing old rows.

Video files in this dataset have local timestamps starting at `0.0`. For example:

```text
file-001.mp4: 0.0..40.4s
file-003.mp4: 0.0..49.4s
file-004.mp4: 0.0..35.8s
file-005.mp4: 0.0..42.1s
file-006.mp4: 0.0..40.9s
file-007.mp4: 0.0..35.3s
file-008.mp4: 0.0..33.3s
file-009.mp4: 0.0..35.1s
```

Some metadata rows ask LeRobot to query beyond those MP4 durations.

Affected episodes:

```text
episode 5
bug: duplicate metadata rows
impact: both rows are identical and valid
reason: duplicate row should be removed, but it will not cause timestamp failure

episode 6
bug: duplicate metadata rows
impact: both rows are identical and valid
reason: duplicate row should be removed, but it will not cause timestamp failure

episode 7
bug: duplicate metadata rows + timestamp-empty
row 1: file-001, query 49.3..59.3s, video 0.0..40.4s, valid 0/101 frames
row 2: file-001, query 49.3..59.3s, video 0.0..40.4s, valid 0/101 frames
reason: metadata points completely past the end of file-001.mp4

episode 8
bug: duplicate metadata rows + timestamp-empty
row 1: file-001, query 69.8..80.1s, video 0.0..40.4s, valid 0/104 frames
row 2: file-001, query 69.8..80.1s, video 0.0..40.4s, valid 0/104 frames
reason: metadata points completely past the end of file-001.mp4

episode 9
bug: duplicate metadata rows with conflicting valid definitions
row 1: file-001, query 0.0..9.7s, valid 98/98 frames
row 2: file-002, query 0.0..9.7s, valid 98/98 frames
reason: two valid but different video mappings exist for the same episode_index

episode 10
bug: duplicate metadata rows with conflicting valid definitions
row 1: file-003, query 0.0..9.9s, valid 100/100 frames
row 2: file-002, query 0.0..9.9s, valid 100/100 frames
reason: two valid but different video mappings exist for the same episode_index

episode 11
bug: duplicate metadata rows
impact: both rows are identical and valid

episode 12
bug: duplicate metadata rows + partial timestamp mismatch
row 1: file-003, query 40.7..50.1s, video 0.0..49.4s, valid 88/95 frames
row 2: file-003, query 40.7..50.1s, video 0.0..49.4s, valid 88/95 frames
reason: last 7 frames request timestamps beyond the end of file-003.mp4

episode 13
bug: duplicate metadata rows + timestamp-empty
row 1: file-003, query 60.0..69.7s, video 0.0..49.4s, valid 0/98 frames
row 2: file-003, query 60.0..69.7s, video 0.0..49.4s, valid 0/98 frames
reason: metadata points completely past the end of file-003.mp4

episode 14
bug: duplicate metadata rows + timestamp-empty
row 1: file-003, query 78.3..86.7s, video 0.0..49.4s, valid 0/85 frames
row 2: file-003, query 78.3..86.7s, video 0.0..49.4s, valid 0/85 frames
reason: metadata points completely past the end of file-003.mp4

episode 15
bug: three metadata rows for one episode_index
impact: all rows are timestamp-valid, but metadata is structurally invalid
reason: duplicate append/rewrite bug

episodes 16, 17
bug: duplicate metadata rows with conflicting valid timestamp definitions
impact: timestamp-valid, but ambiguous metadata

episode 18
bug: duplicate metadata rows with one valid row and one partial-invalid row
row 1: file-004, query 26.0..34.9s, video 0.0..35.8s, valid 90/90 frames
row 2: file-004, query 34.1..43.0s, video 0.0..35.8s, valid 18/90 frames
reason: one duplicate row is usable; the other extends beyond file-004.mp4

episode 19
bug: duplicate metadata rows, no fully valid row
row 1: file-004, query 45.5..53.7s, video 0.0..35.8s, valid 0/83 frames
row 2: file-004, query 35.0..43.2s, video 0.0..35.8s, valid 9/83 frames
reason: both rows point mostly or fully beyond file-004.mp4

episodes 20, 21, 22, 23
bug: duplicate metadata rows with conflicting valid timestamp definitions
impact: timestamp-valid, but ambiguous metadata

episode 24
bug: duplicate metadata rows with one valid row and one timestamp-empty row
row 1: file-005, query 33.1..42.1s, video 0.0..42.1s, valid 91/91 frames
row 2: file-005, query 43.1..52.1s, video 0.0..42.1s, valid 0/91 frames
reason: one duplicate row is usable; the other starts after the MP4 ends

episodes 25, 26, 27, 28
bug: duplicate metadata rows with conflicting valid timestamp definitions
impact: timestamp-valid, but ambiguous metadata

episode 29
bug: duplicate metadata rows with one valid row and one timestamp-empty row
row 1: file-006, query 33.0..40.9s, video 0.0..40.9s, valid 80/80 frames
row 2: file-006, query 41.1..49.0s, video 0.0..40.9s, valid 0/80 frames
reason: one duplicate row is usable; the other starts after the MP4 ends

episodes 30, 31, 32, 33
bug: duplicate metadata rows with conflicting valid timestamp definitions
impact: timestamp-valid, but ambiguous metadata

episode 34
bug: duplicate metadata rows with one timestamp-empty row and one valid row
row 1: file-007, query 35.4..42.6s, video 0.0..35.3s, valid 0/73 frames
row 2: file-007, query 28.1..35.3s, video 0.0..35.3s, valid 73/73 frames
reason: one duplicate row is usable; the other starts after the MP4 ends

episodes 35, 36, 37, 38
bug: duplicate metadata rows with conflicting valid timestamp definitions
impact: timestamp-valid, but ambiguous metadata

episode 39
bug: duplicate metadata rows with one valid row and one timestamp-empty row
row 1: file-008, query 26.6..33.3s, video 0.0..33.3s, valid 68/68 frames
row 2: file-008, query 34.7..41.4s, video 0.0..33.3s, valid 0/68 frames
reason: one duplicate row is usable; the other starts after the MP4 ends

episodes 40, 41, 42, 43
bug: duplicate metadata rows with conflicting valid timestamp definitions
impact: timestamp-valid, but ambiguous metadata

episode 44
bug: duplicate metadata rows with one valid row and one timestamp-empty row
row 1: file-009, query 28.4..35.1s, video 0.0..35.1s, valid 68/68 frames
row 2: file-009, query 36.9..43.6s, video 0.0..35.1s, valid 0/68 frames
reason: one duplicate row is usable; the other starts after the MP4 ends
```

### `task3-config3-YTO-clean`

```text
info total_episodes: 45
meta/episodes rows: 45
unique episode_index in meta: 45
unique episode_index in data: 45
data rows: 3657
duplicate episode_index rows: none
timestamp issues: none
```

### `task3-config4-YOT-clean`

```text
info total_episodes: 45
meta/episodes rows: 45
unique episode_index in meta: 45
unique episode_index in data: 45
data rows: 3664
duplicate episode_index rows: none
timestamp issues: none
```

### `task3-config5-OYT-clean`

```text
info total_episodes: 45
meta/episodes rows: 50
unique episode_index in meta: 45
unique episode_index in data: 45
data rows: 3434
```

Duplicated `episode_index` values:

```text
40, 41, 42, 43, 44
```

Affected episodes:

```text
episodes 40, 41, 42, 43
bug: duplicate metadata rows with conflicting valid timestamp definitions
impact: timestamp-valid, but ambiguous metadata

episode 44
bug: duplicate metadata rows with one valid row and one timestamp-empty row
row 1: file-009, query 26.0..32.1s, video 0.0..32.1s, valid 62/62 frames
row 2: file-009, query 33.0..39.1s, video 0.0..32.1s, valid 0/62 frames
reason: one duplicate row is usable; the other starts after the MP4 ends
```

### `task3-config6-OTY-clean`

```text
info total_episodes: 45
meta/episodes rows: 45
unique episode_index in meta: 45
unique episode_index in data: 45
data rows: 3490
duplicate episode_index rows: none
timestamp issues: none
```

## Root Cause Hypothesis

The broken datasets look like episode metadata was appended multiple times without first deleting/rebuilding `meta/episodes`.

In `task3-config2-TYO-clean`, this is especially visible:

```text
data has 45 unique episodes
meta/episodes has 86 rows
info.json says total_episodes = 56
```

That means at least one part of the dataset writing pipeline updated `meta/info.json`, `meta/episodes`, and `data/` inconsistently.

The timestamp-empty rows look like the writer kept assigning increasing `from_timestamp` values while reusing a video file whose local timestamps reset to `0.0`. Example:

```text
episode 7:
metadata query: 49.3..59.3s
actual file-001.mp4: 0.0..40.4s
```

LeRobot then asks PyAV for frames that do not exist in the MP4, causing:

```text
FrameTimestampError
```

## What the Script Should Fix

The dataset writing/cleaning script should enforce these invariants before pushing:

```text
len(meta/episodes rows) == info.total_episodes
number of unique episode_index values == info.total_episodes
episode_index is unique in meta/episodes
episode_index is consecutive: 0..N-1
every data episode_index has exactly one meta/episodes row
every meta/episodes row has at least one data row
for each video key:
    from_timestamp + data.timestamp.min/max must be inside the actual MP4 timestamp range
frame_index starts at 0 and is consecutive inside each episode
global index starts at 0 and is consecutive
```

When duplicate metadata rows exist, the repair should not blindly keep the first row. It should:

```text
1. Score each duplicate metadata row by how many frame timestamps are valid.
2. Keep the row with 100% valid video queries if one exists.
3. If no 100% valid row exists but one row is mostly valid, either clip invalid tail frames or flag for manual review.
4. If all rows have 0 valid frames, the episode is unrecoverable unless the original raw video exists.
```

Episodes that are likely unrecoverable from the current Hub files:

```text
task3-config2-TYO-clean: episodes 7, 8, 13, 14
```

Episode that is technically recoverable but probably too damaged:

```text
task3-config2-TYO-clean: episode 19
```

Episodes recoverable by selecting the valid duplicate metadata row:

```text
task3-config2-TYO-clean: 18, 24, 29, 34, 39, 44
task3-config5-OYT-clean: 44
```

Episode recoverable by clipping a small invalid tail:

```text
task3-config2-TYO-clean: 12
```
