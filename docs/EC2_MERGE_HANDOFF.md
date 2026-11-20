# EC2 conflict-resolution handoff

Branch: `aura-ec2-merge-prep`
Base: `feature/aura-voice-playground-ui`

## What this branch contains

- `backend/prompts/tutor.txt`: merged tutor prompt guidance. It combines the current GitHub teaching flow with the known EC2 academic guidance, missing-information behavior, qualification pronunciation, language/script handling, and full chronological conversation continuity.

## What is known about the EC2-only changes

The EC2 stash had these changes, which must be preserved when reconciling the three conflicted files:

### `backend/cloud_voice.py`

- Keep the GitHub full chronological `conversation_history` / `conversation_messages` behavior.
- Keep GitHub per-turn KB telemetry: `kb_status`, `kb_used`, `kb_chunks`, `kb_agent`, `kb_titles`, written to the active `conversation_turns` item.
- Preserve the EC2 selective retrieval gate `self._should_use_knowledge_base(user_text)`; when false, do not call KB status or retrieval and leave status as `not_needed`.
- When retrieval is needed, use these statuses consistently: `disabled`, `empty`, `used`, `no_match`, `unavailable`.
- Preserve the EC2 agent-specific fallback guidance for customer_service and tutor when retrieval was attempted but unavailable. Do not issue fallback warnings for `not_needed`.
- Do not duplicate the telemetry update or conversation history block.
- The current GitHub version does not contain `_should_use_knowledge_base`; the exact EC2 helper implementation was not available to transplant into this branch. Do not invent its intent rules.

### `frontend/src/App.tsx`

- Keep the current GitHub implementation of live conversation turns, per-turn KB telemetry, full transcript/search, and live audio/STT diagnostics.
- The EC2 stash also contained dynamic `integrationStatus` settings and a `newConversation` action.
- The exact supporting EC2 state, API fetch, callback, and endpoint code was not available in the GitHub file or the captured conflict excerpts. Those pieces must be recovered from the EC2 backup/stash before they can safely be added.
- Do not add references to `integrationStatus`, `newConversation`, or an integrations endpoint without their complete definitions and backend contract.

## EC2 safety checklist

Repository path: `/home/ubuntu/sarvam-voice-agent`

Before resolving anything on EC2:
1. Keep both existing stashes. Do not drop either stash.
2. Keep `backend/data/` and all existing backups.
3. Preserve local changes to `backend/api.py`, `backend/knowledge_base.py`, and `frontend/src/styles.css`.
4. Preserve `backend/knowledge_base.py` setting `MIN_RELEVANCE_SCORE = 0.80`.
5. Do not run `git reset --hard`, `git checkout --ours`, `git checkout --theirs`, `git add .`, or delete backup files.
6. Do not restart `aura.service` until conflicts are resolved, Python syntax checks pass, and the frontend build passes.

## Verification commands

```bash
cd /home/ubuntu/sarvam-voice-agent

grep -nE '^(<<<<<<<|=======|>>>>>>>)' \
  backend/cloud_voice.py \
  backend/prompts/tutor.txt \
  frontend/src/App.tsx

.venv/bin/python -m py_compile \
  backend/cloud_voice.py backend/api.py backend/knowledge_base.py

cd frontend
npm run build
```

## Important limitation

This branch does not claim that the EC2-only cloud voice and frontend code has been merged. Only the tutor prompt has been merged from the available source material. The EC2 stash and backup files remain the source of truth for the missing helper and integration-status implementation.
