# Merged main validation — 15 November 2026

Repository: Anjali-Mishra05/BE_Major_Project.
Pulled main commit: `98d7e1fe9ed69a68b23ec3cfbd18638566d29731`.
Local main now tracks `anjali/main`. No commits or pushes were performed during validation.

## Initial merged-code result

The initial suite ran 20 tests: 19 passed and the model-fingerprint regression failed.
The merge replaced SHA-256 tree fingerprints with leaf-value sums, so different trees
could look identical to a cached client. It also removed base-score consistency and
forest-length checks, and omitted the empty forest from initial synchronization messages.

## Local fixes

- Restore SHA-256 fingerprints and byte-array transport on server and client together.
- Rebuild cached client scores when the base score changes.
- Reject a forest whose length differs from the declared tree index.
- Send the empty forest so initial synchronization can reset an old model.
- Add three tests covering changed base score, empty-forest reset and incorrect forest length.
- Make the Flower smoke test add the selected Python environment's executable directory
  to PATH. Without this, unactivated Windows environments could not launch Flower helpers.

## Verification after fixes

| Check | Result |
|---|---|
| Automated regression suite | 23 passed |
| React production build | Passed |
| Python dependency compatibility (`pip check`) | Passed |
| Real HTTP server + two clients | Plain, secure and DP modes each completed three depth-2 trees |
| Final-model delivery and local inference | Passed in all three HTTP runs |
| Saved centralized model evaluation | Completed on 49,904 held-out records |
| Real Flower deployment | Three trees, two clients, all six exchanges completed |
| Interrupted-client recovery | Passed; both clients acknowledged final model |
| Missing-client deadline | Passed; incomplete aggregate rejected |
| Whitespace/error check (`git diff --check`) | Passed |

Detailed HTTP results: `runtime/merged-main-comparison.json` (local, ignored).
Flower evidence: `reports/fl/smoke/run-pmwwhh6z/` (local, ignored).
Recovery evidence: `reports/network_acceptance.json`.

The passing result applies to merged main **plus the local fixes above**. GitHub still
needs the user to commit and push these fixes. Physical three-device LAN testing and
hardware acceptance remain outstanding. These short runs verify integration; they do
not replace the recorded 20-tree performance experiment or establish zero future bugs.
