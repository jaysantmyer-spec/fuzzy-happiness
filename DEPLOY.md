# Put UFC Oracle online (Streamlit Community Cloud, free)

You end up with a permanent link like `https://jody-ufc-oracle.streamlit.app` that works from your phone
on cellular, can be added to your home screen, and can be shared.

## 1. Create a GitHub account
github.com → Sign up (free). Verify the email.

## 2. Create a repository
1. Top-right **+** → **New repository**.
2. Name: `ufc-oracle`. Leave it **Public** (Streamlit's free tier needs public repos, or connect a private one
   if you upgrade). Do **not** tick "Add a README".
3. **Create repository**.

## 3. Upload the app
On the empty-repository page, click the link **"uploading an existing file"**.
1. In Finder, open the `ufc_oracle_v2` folder. Select everything inside it **except the `.venv` folder**
   (press ⌘A, then ⌘-click `.venv` to deselect it). If you can't see `.streamlit`, press ⌘⇧. in Finder
   to show hidden files; it must be included.
2. Drag the selection into the browser's upload box. Wait for every file to finish (the model file is 12 MB).
3. Scroll down, click **Commit changes**.

If the browser upload complains about too many files, install **GitHub Desktop** instead
(desktop.github.com): File → Add local repository → choose the folder → Publish repository.

## 4. Deploy
1. Go to **share.streamlit.io** → **Sign in with GitHub** → authorise it.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repository: `yourname/ufc-oracle`. Branch: `main`. Main file path: `app.py`.
4. **Advanced settings** → Python version **3.11**. In the **Secrets** box, paste (with your real key):
   ```
   ODDS_API_KEY = "your-odds-api-key"
   ```
   (Skip this if you don't have a key yet; you can add it later under the app's Settings → Secrets.)
5. **Deploy**. First build takes 3 to 5 minutes. XGBoost and LightGBM install automatically on the server.

## 5. Use it
Open the link on your phone. In Safari, tap Share → **Add to Home Screen** to get an app icon.

## Things to know
- **Saving data.** The server's disk is wiped whenever the app restarts or you redeploy. Anything the app writes
  (scraped results, ledger, retrained model, odds history) lasts until then. To keep changes permanently, run
  them on your Mac and upload the changed files in `data/` and `models/` to GitHub (Add file → Upload files);
  the site redeploys itself in about a minute.
- **Speed.** The free server is much slower than your Mac. Viewing cards, pricing and the export is fast;
  a full retrain or a 10-event track record can take several minutes. Do those on the Mac when you can.
- **Sleeping.** Free apps go to sleep after a few days without visitors and take ~30 s to wake on the next visit.
- **Privacy.** A public repo means anyone can read the code and data (there is nothing sensitive in it; your
  API key stays in Secrets, not in the repo). The app link itself is unlisted but not password-protected.
- **Updating the app code.** Upload the changed file to GitHub and it redeploys.
