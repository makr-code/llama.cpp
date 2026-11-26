---
AI_Used: true
AI_Models: unknown
AI_Contributor: OpenAI API assistant
Human_Review: false
Human_Author: makr-code
---

# ThemisDB trainable LoRA patch

## Bounded implementation plan

Base: `1e8924fd65ad349d1d838412a2172292618f3bbf` (b7974).
Only the authorized public API, adapter implementation and optimizer changes
were imported. No parent repository content or model artifacts are included.

1. Review adapter ownership, target selection, parameter flags and backward
   propagation through frozen quantized weights and attention.
2. Add a generated, public-data-free tiny Llama fixture and focused tests for
   no-op initialization, deterministic serialization, invalid inputs, gradients,
   loss reduction, frozen base weights and export/reload parity.
3. Build and run focused CPU tests before publication.
4. Add reviewed weekly upstream-update automation in a separate commit:
   candidate rebase, tests, draft PR; no automatic merge.

Acceptance is real optimizer updates, not a simulated training path. CPU fixture
success does not establish GPU, all-architecture or production-model quality.
The parent owns integration and documentation in ThemisDB. Semantic C++ tools
were unavailable in this external checkout; source/reference inspection and
compilation/tests are the fallback. The ThemisDB wiki has no dedicated upstream
llama.cpp training contract; native upstream ownership conventions take precedence.

AI assisted this patch and narrative; exact model backend is not independently
verifiable. Human review remains required before upstream update promotion.

## Supported training contract

The capability macro is exactly `#define LLAMA_THEMIS_TRAINABLE_LORA 1`.
The public functions are documented in `include/llama.h`.

- Load a **dense Llama** model with `n_gpu_layers=0`. Build with
  `GGML_CPU_REPACK=OFF`: inference-only CPU repacking is incompatible with
  ordinary quantized backward kernels. Base tensors must be F32 or quantized;
  CPU OUT_PROD does not support F16/BF16 base weights.
- Create the trainable adapter before creating its context, select
  `llama_lora_target_filter_attn_mlp`, then attach with a finite nonzero scale.
  The model owns adapters and must outlive all using contexts.
- Set `n_batch` and `n_ubatch` to at least the **actual** `llama_n_ctx`
  (upstream rounds context sizes to multiples of 256). Each optimizer step sees
  an entire causal sequence; truncated-history training is not implemented.
- Zero-initialize `llama_opt_params`, set `train_lora_only=true`, and use
  `llama_opt_param_filter_all` for both adapter components. The target filter
  matches base names, not `.lora_a/.lora_b` optimizer tensor names.
- The training graph uses the existing no-cache causal attention path. This
  preserves Q/K/V gradients instead of reading detached inference cache tensors.
  Flash attention is disabled in optimization only. Inference is unchanged.
- Do not change attachments/scales or concurrently access the model/adapters
  during the optimizer context lifetime: parameter flags are shared on tensors.
  aLoRA training is rejected. Destroy the training context before using separate
  inference contexts. Synchronize adapter writes before export.
- Save is synchronous, deterministic for identical tensors/metadata on the same
  GGUF implementation/platform, and reloadable with the existing adapter loader.
  It does not preserve arbitrary metadata or atomically replace the file; a failed
  write may leave partial output. Use a staging filename and caller-controlled
  promotion if atomic publication is required.

## Focused validation

No test fixture is downloaded or committed. `tests/test-lora-train.cpp` generates
two real two-layer, grouped-query Llama models (F32 and Q4_0 matrices), 32 embedding
dimensions, 16 token IDs and 256-token sequences in a temporary directory.
Normal completion removes all generated models and adapters.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_HTTPLIB=OFF -DLLAMA_OPENSSL=OFF -DGGML_CCACHE=OFF \
  -DGGML_NATIVE=OFF -DGGML_CPU_REPACK=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF
cmake --build build --target test-lora-train --parallel 2
ctest --test-dir build --no-tests=error -R '^lora-train$' --output-on-failure
```

The test uses 64 real SGD steps, learning rate 0.1 and no weight decay/momentum.
Because dynamic upstream `ggml_opt_eval` discards its graph before the epoch
callback, `ggml_opt_grad_acc` cannot safely be used there. The test instead checks
the applied loss gradient as `(previous - updated) / learning_rate`, for every
A/B tensor of all 14 attention/MLP pairs. The first A gradient must be zero
(initial B is zero); all B gradients, then all subsequent A/B gradients, must
be finite and nonzero. This measures optimizer-applied gradients, not an
independent finite-difference accuracy proof.

Other assertions cover exact base-byte preservation (including norms and
embeddings), stale base-parameter flag removal, initial no-op logits, seed
determinism, invalid rank/alpha/model/filter/path arguments, I/O failure,
unregistered-on-error ownership, repeated-save and save/load/save byte parity,
trained tensor parity and reloaded inference logits.

Local validation on 2026-11-26:

- MSVC 19.44 / Ninja / Windows Release: build passed; `lora-train` passed.
  Loss F32: 2.493383 -> 2.382294; Q4_0: 2.423965 -> 2.333972.
- GCC 13.3 / Ninja / WSL Ubuntu Debug: build and `lora-train` passed.
- GCC 13.3 / Ninja / WSL Ubuntu Release: build and `lora-train` passed.
- Focused workflow actionlint and `git diff --check` passed.

The installed VS Insiders vcvars script failed to set VC include/library paths.
Windows validation used its SDK environment with explicit installed MSVC 14.44
include/library paths in the same process. Initial intermediate tests exposed
detached cache inputs, invalid dynamic-gradient accessor use and repacked-buffer
restrictions; those failures were investigated, not treated as passing evidence.
SGD 0.5 was unstable on the quantized fixture; 0.1 passed the unchanged loss gate.
No GPU, sanitizer, production-model quality or full upstream suite is claimed.
On Windows this internal-inspection test requires a static build.

## Reviewed upstream automation

`.github/workflows/themisdb-upstream-review.yml` runs weekly Monday at 06:23 UTC
or manually on default branch `themisdb`. Only that workflow should be enabled
in the fork; inherited workflows/schedules must remain disabled.

The read-only candidate job fetches public upstream master, checks a bounded,
linear series of at most 32 commits against an explicit focused-file allowlist,
and rebases with a 180-second timeout. Conflicts or scope/build/test failures
produce explicit diagnostics and prevent publication. It builds and runs the
focused Release CPU regression **before** transferring candidate code/history.
The artifact is a git bundle only; generated models/training context are excluded.

A separate fresh-runner publication job has `contents: write` and
`pull-requests: write`. Both checkouts use `actions/checkout@v5` with
`persist-credentials: false`. GH_TOKEN is exposed only to the final push/PR step;
no candidate build runs with write credentials. It checks the tested candidate
SHA and unchanged base, pushes a unique review branch and opens a **draft** PR
to `themisdb`. It never automatically merges or force-updates `themisdb`.

GITHUB_TOKEN-created PRs do not trigger ordinary PR workflows: review the linked
pre-PR run, and rerun validation after edits. Evidence covers only the focused
fixture. Preserve a linear patch series when manually promoting the reviewed
candidate; ordinary merge commits will intentionally block subsequent rebases
until a maintainer linearizes the series. Revisit the file allowlist for reviewed
future patch changes. Upstream workflow-file changes may require additional
GitHub publication authorization; do not introduce a job-wide PAT as a workaround.
Repository Actions settings may also need “Allow GitHub Actions to create and
approve pull requests”; publication failure explicitly reports this boundary.

### Manual runtime validation

`workflow_dispatch` offers `validation_only` (boolean, default false). In this
explicit manual mode the workflow fetches upstream to identify the fork's current
shared upstream ancestor, then force-replays the focused patch onto **that same
ancestor**, not newer upstream. The regular CPU build, focused tests, bundle
transfer and separate draft-PR publication jobs still run. The candidate adds
only a documentation smoke-test marker; its PR title/body explicitly say
**validation only — do not merge or promote**. Scheduled updates cannot select
this mode and still fail closed on real upstream conflicts.

```sh
gh workflow run themisdb-upstream-review.yml --repo makr-code/llama.cpp \
  --ref themisdb -f validation_only=true
```
