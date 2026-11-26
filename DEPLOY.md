# Deploying the Smart Beta ETF Dashboard to Render (Free)

This uses the **cached-data** approach: you generate the results once on your own
computer (or Colab), commit the small cache file, and Render just serves the dashboard —
no live Yahoo Finance calls needed at runtime, so it works reliably on the free tier.

---

## Step 0 — Files you need in one folder

You should have all of these together locally:

```
smart-beta-etf-builder/
├── smart_beta_etf_builder.py
├── dash_app.py
├── precompute_cache.py
├── requirements.txt
├── Procfile
├── .gitignore
└── pipeline_cache.json        <- you'll generate this in Step 1
```

## Step 1 — Generate the data cache (run this locally, not on Render)

Open a terminal on your own computer in that folder:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python precompute_cache.py
```

This talks to Yahoo Finance and Kenneth French's data library and saves everything into
`pipeline_cache.json` (a few hundred KB, small enough to commit to Git). You'll see it appear
in the folder when it's done — that can take a minute or two.

> Don't have Python installed locally? Run the same steps in a Google Colab notebook instead,
> then download `pipeline_cache.json` from the Colab file browser to your computer.

## Step 2 — Put it on GitHub

If you don't already have a GitHub account, make one at github.com (free).

```bash
git init
git add .
git commit -m "Smart Beta ETF Builder dashboard"
```

Then on github.com: click **New repository** → name it (e.g. `smart-beta-etf-builder`) →
**Public** → Create repository. GitHub will show you commands like:

```bash
git remote add origin https://github.com/YOUR_USERNAME/smart-beta-etf-builder.git
git branch -M main
git push -u origin main
```

Run those. Refresh the GitHub page — your files (including `pipeline_cache.json`) should be there.

## Step 3 — Create the Render account and service

1. Go to **render.com** → **Get Started** → sign up (you can sign up with your GitHub account,
   which makes the next step easier).
2. On the Render dashboard, click **New +** → **Web Service**.
3. Connect your GitHub account if prompted, then select your `smart-beta-etf-builder` repo.
4. Fill in the settings:
   - **Name:** anything, e.g. `smart-beta-etf-builder` (this becomes part of your URL)
   - **Region:** closest to you
   - **Branch:** `main`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn dash_app:server`
   - **Instance Type:** Free
5. Click **Create Web Service**.

Render will build and deploy automatically. Watch the log — it should end with something like
`Loading cached pipeline results from ... pipeline_cache.json` followed by the server starting.
That log line confirms it found your cache and is *not* trying to hit Yahoo Finance live.

## Step 4 — Get your link

Once the deploy finishes (a few minutes), Render gives you a URL like:

```
https://smart-beta-etf-builder.onrender.com
```

That's your live, public, interactive dashboard. Put it in your resume, GitHub README (as a
"🔗 Live Demo" link), or LinkedIn project section.

---

## Notes

- **Free tier sleeps after 15 minutes of no traffic.** The first visitor after a quiet period
  waits ~30-60 seconds for it to wake up, then it's fast. This is normal and fine for a demo link.
- **To refresh the data** (e.g. show updated performance next quarter): re-run
  `python precompute_cache.py` locally, then:
  ```bash
  git add pipeline_cache.json
  git commit -m "Refresh data snapshot"
  git push
  ```
  Render auto-redeploys on every push to `main`.
- **If you'd rather have it pull live data** instead of a cached snapshot, that also works on
  Render's free tier (no whitelist restriction like PythonAnywhere) — just skip Step 1 and don't
  commit `pipeline_cache.json`; `dash_app.py` automatically falls back to running the pipeline
  live if the cache file isn't present.
