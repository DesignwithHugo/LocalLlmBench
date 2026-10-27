# LocalLlmBench

An unattended benchmark for every model in your LM Studio library on an Apple Silicon Mac. It loads one model at a time, measures speed, memory and context, runs a small quality set, then puts each model to work as an **agent** in a simulated workspace and grades what actually happened. Python standard library only; nothing leaves your Mac (the optional download queue asks LM Studio to fetch models, nothing else).

**Results so far:** [M1 Max 64 GB, October 2026](results/m1-max-64gb/README.md): 50+ models, 48 of them through the full agent suite and the real-world failure tier, three runs each.

## Why

Tokens per second doesn't tell you whether a model can run an agent. So besides the usual speed and context numbers, every model here works through 16 multi-step tool tasks (calendar, email, files, meeting rooms, sales data, two small Python repos) and 10 more tasks rebuilt from mistakes real assistants made (stamping invoices without checking the purchase order, "tidying" a chart by moving its data, deleting when the backup check timed out, acting before the plan was approved). Grading is deterministic code that checks the final state of the workspace, so every run is replayable.

## Run it on your own Mac

1. Install [LM Studio](https://lmstudio.ai), download some models, and start the local server (Developer tab, default port 1234). If the server asks for an API token, create one and paste it as the only line of `lmstudio_token.txt` in this folder.
2. Clone this repo and copy `config.example.json` to `config.json`.
3. `./run.sh --smoke` runs a 2-minute check on your smallest model.
4. `./run.sh --background --now` runs everything. It's resumable: stop it, restart the Mac, add models, run it again and it only does what's missing. `tail -f data/run.log` to watch.
5. Your report lands in `report/`, and a shareable package in `report/community/`.

Without `--now` it waits until the Mac has been idle for 15 minutes and stays inside the night window from `config.json`, and it redoes any test you interrupt by touching the keyboard.

## What it measures

- **Load:** first and repeat load at a 64k context (65,536), memory needed (LM Studio's estimate).
- **Speed:** time to first token, prompt reading speed (prefill) and generation speed with a short prompt and at 8k, 32k and 60k tokens. Medians of repeated runs.
- **Context:** the largest context each model loads (native max, stepping down), then one fill up to 128k with a hidden code the model has to find.
- **Quality:** 26 small tasks (21 text, 5 vision): instruction following, JSON extraction, single tool calls, Python with unit tests, reasoning, Spanish writing and summaries, OCR, charts, UI reading. Auto-graded; writing is also scored by two local judge models from a different family.
- **Agents:** 16 multi-step tool-use tasks, 3 runs each at temperature 0.6 with fixed seeds. Covers multi-step chains, error recovery, not inventing data when a tool is down, choosing among 40 tools, parallel calls, asking when ambiguous, multi-turn edits, prompt injection, drafting vs sending, long inputs and outputs, a coding loop with tests, and strict schemas.
- **Real-world failure tier:** 10 agent tasks rebuilt (anonymized) from real assistant mistakes, with a count of "serious slips" (hard-to-undo actions or false claims).
- **Prompt cache:** a ~12k-token agent prefix sent cold, then reused.
- **Embeddings:** speed and a simple retrieval check.

Half the tasks are in Spanish.

## Order of work

1. Base tests for every model, then the local judges score the writing.
2. Agent suite, first pass, smallest model first.
3. Agent reliability runs (two more per task), so "solved every time" can be computed.
4. Real-world tier, first pass then reliability runs.

You can stop after step 2 and still have a full agent leaderboard. Models up to `priority_small_gb` finish every phase before bigger ones continue.

## Safety

- Never raises the GPU memory limit, never asks for a password, never deletes a model unless you list it in `data/delete_when_done.json`.
- Two quality tasks and two agent tasks ask the model to write Python. That code is executed on your Mac, in a temporary folder, with `python -I`, no environment and a 30-second timeout. It is model-written code, so run the bench on a machine you would run untrusted scripts on.
- If swap grows more than 6 GB or free disk drops below 12 GB during a test, the model is unloaded and the test recorded as aborted. Testing pauses under 20 GB free.
- On a Mac that shares models over LM Link, it only unloads its own models.

## Files

- `bench.py` runs everything. `tasks.py` has the quality tasks and graders. `agent.py` has the agent workspace, tools, tasks and graders. `report.py` writes the report and the community package. `lsclient.py` talks to LM Studio and macOS.
- `assets/` public-domain Don Quijote text for long-context fills and summaries, and synthetic test images.
- `data/` (created on first run) raw results and logs. `data/download_queue.json`, `data/invalidate.json` and `data/delete_when_done.json` let you queue downloads, redo tests and schedule cleanups without touching the data file.
- `results/<machine>/` published runs: a README with every table, `results.csv`, `agent_tasks.csv`, charts, and `results.jsonl` with every run and full agent traces.

## Contributing results

Run it, then open a pull request adding `results/<your-chip>-<ram>gb/` with the contents of your `report/community/` folder (leave out the harness zip). Numbers from other Macs are the most useful thing this project can collect. Model requests and grader bugs go in Issues; every agent trace is in `results.jsonl`, so a wrong grade can be checked by anyone.

## How this was built

The harness, the graders, the report and this repository were written and operated by Claude (Opus 5.5 and Fable 5.1) working with the author over a couple of weeks: the author chose the models, set the rules and approved every download and deletion; Claude wrote and debugged the code, watched the runs and fixed grader mistakes as they turned up. The scenarios in the real-world tier are reconstructions of things assistants got wrong in actual use. There will be mistakes left; every agent trace is in `results.jsonl`, so please open an issue if a grade looks wrong.

## License

MIT.
