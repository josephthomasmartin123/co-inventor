# Deploy to Railway

## Steps

### 1. Push to GitHub

```bash
git init
git add -A
git commit -m "Initial commit — Co-Inventor"
git remote add origin https://github.com/YOUR_USERNAME/co-inventor.git
git push -u origin main
```

### 2. Create Railway project

- [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
- Select your repo — Railway auto-detects Python and uses `railway.toml`

### 3. Set environment variables

In Railway dashboard → your service → **Variables**, add:

| Variable | Value |
|---|---|
| `OPENROUTER_API_KEY` | your OpenRouter key |
| `EXA_API_KEY` | your Exa key |
| `DEFAULT_MODEL` | `deepseek/deepseek-v4-pro` |
| `DB_PATH` | `/data/co_inventor.db` |

### 4. Add persistent storage (recommended)

Without this, sessions are wiped on every redeploy.

- Railway dashboard → your project → **Add Service** → **Volume**
- Set mount path: `/data`
- Attach it to your app service

With a Volume, `DB_PATH=/data/co_inventor.db` persists across deploys.

### 5. Deploy

Railway builds and deploys automatically. You get a URL like:
`https://co-inventor-production.up.railway.app`

---

## Cost

Railway Hobby plan: **$5/month credit** included. A small always-on Python service costs ~$3–5/month.

## SQLite on Railway

SQLite is fine for demos and small teams. Sessions are stored in a single file.

- **No Volume**: sessions lost on redeploy (fine for quick demos)
- **With Volume**: sessions persist indefinitely (recommended for shared use)

For production-scale multi-user use, migrate to PostgreSQL — Railway has a Postgres plugin.
