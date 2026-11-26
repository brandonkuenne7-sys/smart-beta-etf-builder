# Deploying the Smart Beta ETF Dashboard to PythonAnywhere (Free, No Git Required)

Since the free tier can't reach Yahoo Finance directly, we generate the data once on
your own computer, then just upload the files — no command line on your end except
one quick step, no GitHub needed.

---

## Step 1 — Generate the data file (on YOUR computer, not PythonAnywhere)

Open a terminal in the folder with all your project files, then:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python precompute_cache.py
```

Wait for it to finish — you'll see a new file appear: **`pipeline_cache.json`**. That's the only
file that needs live internet access to create. Everything after this step happens entirely on
PythonAnywhere's site with no command-line git.

> No Python on your computer? Upload `smart_beta_etf_builder.py`, `precompute_cache.py`, and
> `requirements.txt` to Google Colab, run `!pip install -r requirements.txt` then
> `!python precompute_cache.py` in a cell, and download the resulting `pipeline_cache.json` from
> Colab's file browser (folder icon on the left).

---

## Step 2 — Create your PythonAnywhere account

Go to **pythonanywhere.com** → **Pricing & signup** → **Create a Beginner account** (free).

---

## Step 3 — Upload your files

1. Once logged in, click the **Files** tab at the top.
2. In "New directory," type `mysite` and hit the arrow to create it, then click into it.
3. Use the **Upload a file** button to upload, one at a time:
   - `smart_beta_etf_builder.py`
   - `dash_app.py`
   - `pipeline_cache.json` (the file from Step 1)
   - `requirements.txt`

That's it for files — no zipping, no git.

---

## Step 4 — Install the packages

1. Click the **Consoles** tab → **Bash** (this opens a terminal in your browser).
2. Run these commands one at a time:

```bash
cd mysite
mkvirtualenv --python=/usr/bin/python3.10 mysite-venv
pip install -r requirements.txt
```

`mkvirtualenv` creates an isolated Python environment called `mysite-venv` and activates it
automatically — you'll see the prompt change to show `(mysite-venv)`. The `pip install` then
takes a minute or two to finish.

---

## Step 5 — Create the web app

1. Click the **Web** tab → **Add a new web app**.
2. Click **Next**, then instead of picking Flask/Django, scroll down and choose
   **"Manual configuration"**.
3. Pick **Python 3.10** (must match what you used in Step 4) → **Next**. It'll create the app.

You'll land on a configuration page for your new site. Fill in these two boxes:

- **Source code:** `/home/YOURUSERNAME/mysite`
- **Working directory:** `/home/YOURUSERNAME/mysite`
- **Virtualenv:** `/home/YOURUSERNAME/.virtualenvs/mysite-venv`

(Replace `YOURUSERNAME` with your actual PythonAnywhere username, shown in the URL/top-right.)

---

## Step 6 — Point it at your Dash app

Still on the **Web** tab, click the link to your **WSGI configuration file** (something like
`/var/www/yourusername_pythonanywhere_com_wsgi.py`). It opens an editor. **Delete everything in
it** and replace it with:

```python
import sys
import os

project_home = '/home/YOURUSERNAME/mysite'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

os.chdir(project_home)

from dash_app import server as application
```

Replace `YOURUSERNAME` with your real username again. Click **Save**.

---

## Step 7 — Reload and visit

1. Go back to the **Web** tab, click the big green **Reload** button.
2. Click the link at the top of the page — it'll be `https://yourusername.pythonanywhere.com`.

That's your live dashboard. No sleep/wake delay like Render's free tier either — PythonAnywhere
free apps stay up (they just get reset if idle for very long stretches, which a reload fixes).

---

## Recovering From a Disk-Quota Crisis (and the Real Fix)

If you've hit `OSError: [Errno 122] Disk quota exceeded` while installing packages, here's
what's actually going on and the permanent fix, not just a workaround.

**Root cause:** the deployed dashboard (`dash_app.py`) only ever reads the pre-generated
`pipeline_cache.json` — it never calls `yfinance`, the `scipy` optimizer, `matplotlib`/`seaborn`,
or the Fama-French regression at runtime. Those libraries are only needed locally, when you run
`precompute_cache.py` on your own machine. But the original `requirements.txt` installed *all*
of them on the server too, including `yfinance`'s heavy dependency chain (`curl_cffi`,
`protobuf`, etc.) — easily 150-200MB+ that the live site never actually uses.

As of this version, `smart_beta_etf_builder.py` imports those heavy libraries defensively (if
they're missing, it just notes that and moves on) rather than requiring them — so the server
only needs a **slim** dependency set.

### Step 1 — Wipe the broken venv and start clean

```bash
workon mysite-venv
deactivate
rmvirtualenv mysite-venv
mkvirtualenv --python=/usr/bin/python3.10 mysite-venv
```

### Step 2 — Upload `requirements-web.txt` alongside your other files

Upload this new file into `mysite` (via the Files tab, same as any other file).

### Step 3 — Install only the slim set

```bash
workon mysite-venv
pip install --no-cache-dir -r ~/mysite/requirements-web.txt
```

This installs only `pandas`, `numpy`, `plotly`, `dash`, `dash-bootstrap-components`, and
`gunicorn` — a small fraction of the disk footprint of the full research stack.

### Step 4 — Replace `smart_beta_etf_builder.py` with the current version

Make sure you're on the version where the heavy imports (`yfinance`, `matplotlib`, `seaborn`,
`scipy`, `statsmodels`, `pandas_datareader`) are wrapped in `try/except` rather than required —
otherwise the slim install won't be enough and you'll hit `ModuleNotFoundError` again.

### Step 5 — Reload

**Web** tab → **Reload** → visit your site.

**Going forward:** keep using the full `requirements.txt` locally (for `precompute_cache.py`,
which does need the full stack to actually pull data), and `requirements-web.txt` on the server.

## Troubleshooting

- **Blank page / error page:** click **Web tab → Error log** — it'll tell you exactly what broke
  (usually a typo in the file paths in Step 5/6, or a package missing from `requirements.txt`).
- **"No such file: pipeline_cache.json":** double check it's sitting directly inside `mysite/`
  next to `dash_app.py`, not in a subfolder.
- **Want fresher data later:** repeat Step 1 on your computer, then just re-upload the new
  `pipeline_cache.json` in the Files tab (overwrite the old one) and hit **Reload** on the Web tab.
