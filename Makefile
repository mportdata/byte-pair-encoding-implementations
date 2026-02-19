RUNNER ?= uv run

.PHONY: all incremental text 01 02 03 04 05 06 07 08 09

all: text 01 02 03 04 05 06 07 08 09

incremental: text 05 06 07 08 09

text:
	$(RUNNER) get_text.py

01:
	$(RUNNER) 01-naive-bpe/main.py

02:
	$(RUNNER) 02-type-frequency/main.py

03:
	$(RUNNER) 03-stream-type-counting/main.py

04:
	$(RUNNER) 04-regex-streaming/main.py

05:
	$(RUNNER) 05-incremental-updates/main.py

06:
	$(RUNNER) 06-index-pairs/main.py

07:
	$(RUNNER) 07-memory-lean-representations/main.py

08:
	$(RUNNER) 08-windowed-pair-updates/main.py

09:
	$(RUNNER) 09-lazy-index-compaction/main.py
