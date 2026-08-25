# Deployment Guide

## Option A — Docker Compose (single server / pilot deployment)

Suitable for a departmental pilot on a single VM (e.g. an NIC/state data
centre VM, or any cloud VM with Docker installed).

```bash
git clone <this-repo>
cd lmcs
cp .env.example .env
```

Edit `.env`:
```
POSTGRES_PASSWORD=<generate a strong password>
SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_urlsafe(64))">
```

```bash
docker compose up --build -d
docker compose ps            # confirm db, backend, frontend are healthy
```

**Create the first admin account** (one-time; the endpoint disables itself
after this):
```bash
curl -X POST http://localhost:8000/api/v1/auth/bootstrap-admin \
  -H "Content-Type: application/json" \
  -d '{
    "full_name": "Director, Legal Metrology",
    "email": "admin@yourdept.gov.in",
    "password": "<a strong password>",
    "designation": "Director"
  }'
```

Then log in at `http://<server-ip>/` and use **Settings → (future) User
Management** or `POST /api/v1/auth/register` (as admin) to create officer
accounts.

**Database migrations**: the app auto-creates tables on first boot for
convenience. For any subsequent schema change, use Alembic instead of
relying on auto-create:
```bash
docker compose exec backend alembic revision --autogenerate -m "describe change"
docker compose exec backend alembic upgrade head
```

## Option B — Managed cloud services (recommended for state-wide rollout)

| Component | Suggested service |
|---|---|
| Backend (FastAPI container) | AWS ECS/Fargate, Azure Container Apps, GCP Cloud Run, or NIC Cloud (MeghRaj) container hosting |
| Database | AWS RDS for PostgreSQL / Azure Database for PostgreSQL / managed Postgres on MeghRaj |
| File storage | S3 / Azure Blob / GCS (see `docs/ARCHITECTURE.md` "Object storage") instead of local volumes |
| Frontend | Static hosting (S3+CloudFront, Azure Static Web Apps) or the same container behind a CDN |
| Task queue (at scale) | Managed Redis + Celery workers, or SQS/Cloud Tasks |
| Secrets | AWS Secrets Manager / Azure Key Vault — never commit `.env` |
| TLS | ACM/managed certificates behind an ALB, or Let's Encrypt via Nginx/Caddy |

## OCR engine notes (PaddleOCR)

- The backend uses PaddleOCR (PP-OCRv4) instead of Tesseract. The Docker
  image pre-downloads the English detection/recognition/angle-classification
  models at **build time**, so the running container needs no outbound
  internet access and there's no first-request download delay.
- CPU inference (`OCR_USE_GPU=false`, the default) is fine for a pilot —
  a few hundred ms to ~2s per thorough scan on a typical VM — but the live
  camera's real-time feedback loop is noticeably snappier on GPU. For a
  higher-volume deployment, provision a CUDA-capable host, swap
  `paddlepaddle` for `paddlepaddle-gpu` in `backend/requirements.txt`, and
  set `OCR_USE_GPU=true`.
- To support Hindi (or another) label text, set `OCR_LANG` (e.g. `hi`) — note
  PaddleOCR uses one model set per language, so mixed-language labels are
  best handled by picking the dominant script for your deployment region
  rather than trying to combine languages in one pass.
- `backend/app/main.py` warms the model up on startup in a background
  thread (`OCR_WARM_UP_ON_STARTUP=true`, the default) so the first officer
  to use the app after a deploy doesn't eat the model-load latency.

## Live camera feature

The "Live camera" capture mode on the scan page uses the browser's
`getUserMedia` API and polls `POST /api/v1/scans/live-check` every ~1.3s for
real-time blur/glare/framing feedback (nothing from that polling loop is
persisted — only the final "Capture" press feeds into the normal, persisted
`POST /api/v1/scans` flow). Two things to get right in production:

- **HTTPS is mandatory** (see the checklist below) — browsers block camera
  access on plain HTTP for any origin other than `localhost`.
- On mobile, the camera defaults to the rear-facing camera
  (`facingMode: "environment"`); the "🔄 Flip" button toggles to the
  front-facing camera. No extra native permissions/config are needed beyond
  the browser's own camera-permission prompt.

## Production checklist

- [ ] `SECRET_KEY` is a long random value, stored in a secrets manager, not in source control.
- [ ] `DATABASE_URL` points to a managed/backed-up PostgreSQL instance, not the SQLite default.
- [ ] HTTPS is enforced end-to-end (browser ↔ frontend ↔ backend). **Required** for the live-camera feature specifically — browsers only allow `getUserMedia` (camera access) on `https://` origins or `localhost`; over plain HTTP the "Live camera" button will fail with a permission error even if everything else works.
- [ ] `CORS_ORIGINS` is restricted to your actual frontend domain(s) — no wildcard.
- [ ] Default seed accounts (`scripts/seed.py`) are **not** used in production; the bootstrap-admin flow is used instead and the endpoint is confirmed disabled afterward (it self-disables once any user exists).
- [ ] File storage (uploads + generated reports) is backed by durable, backed-up storage (managed volume or object storage), not an ephemeral container filesystem.
- [ ] Database backups are scheduled (point-in-time recovery recommended given evidentiary use).
- [ ] `backend/app/rules/*.json` has been reviewed and signed off by a Legal Metrology / legal officer against the current Gazette-notified Rules (see the `_meta.disclaimer` field in each file).
- [ ] Log retention/monitoring is configured (the `audit_logs` table plus your platform's container logs).
- [ ] A vulnerability scan and dependency audit (`pip-audit`, `npm audit`) has been run before go-live, and periodically thereafter.
- [ ] Role assignment reviewed: only trusted staff hold `admin`; field officers use `officer`; external/read-only stakeholders use `viewer`.
- [ ] Rate limiting / WAF is placed in front of publicly reachable endpoints if this will be internet-facing rather than on a government intranet.

## Environment variables reference

See `backend/.env.example` for the full list. Key ones:

| Variable | Purpose | Default |
|---|---|---|
| `SECRET_KEY` | JWT signing secret | **must be overridden** |
| `DATABASE_URL` | SQLAlchemy connection string | `sqlite:///./lmcs.db` (dev only) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT access token lifetime | 480 (8h) |
| `OCR_LANG` | PaddleOCR language code, e.g. `en`, `hi` | `en` |
| `OCR_USE_GPU` | Use a CUDA GPU for OCR inference (needs `paddlepaddle-gpu`) | `false` |
| `OCR_WARM_UP_ON_STARTUP` | Load the OCR model at app boot instead of on first request | `true` |
| `CORS_ORIGINS` | Allowed frontend origins (JSON array) | localhost dev ports |
| `MAX_UPLOAD_SIZE_MB` | Upload size cap | 15 |

## Rolling back / disaster recovery

- Database: restore from the most recent PostgreSQL backup/snapshot.
- File storage: if using object storage with versioning enabled, restore
  affected objects; if using a Docker volume, restore from your volume
  backup schedule.
- Application: containers are stateless — redeploying the previous image tag
  is sufficient; no special migration-down process is required unless a
  schema migration also needs reverting (`alembic downgrade -1`).
