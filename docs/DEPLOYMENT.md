# ReliTrace release instructions (v0.4)

## A. Local verification (Windows)

Open VS Code in the folder that **directly contains `app.py`**.

```powershell
Get-Location
Get-ChildItem -Name
py -m pip install -r requirements-dev.txt
py -m pytest tests/ -q
py -m streamlit run app.py
```

The Streamlit app should open at `http://localhost:8501` (or the port printed by Streamlit).
Check CSV comparison with an original and a deliberately shifted dataset. Choose automatic calibration and check that the app reports its cutoffs and control warnings. **Do not claim any scientific validation not supported by the saved results.**

## B. Create a GitHub repository

1. Sign in at GitHub and create an empty repository named `relitrace` (or an available variant). Do not initialize it with a README; this project already has one.
2. Open a terminal in the folder containing `app.py`.
3. Run, replacing the username and repository name below:

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Prepare ReliTrace experimental dashboard release"
git remote add origin https://github.com/YOUR_USERNAME/relitrace.git
git push -u origin main
```

If Git reports your identity is unknown, configure your Git display name/email. When GitHub asks for authentication, use its browser/credential-manager flow, not your account password pasted into the terminal.

**Before pushing:** Check `git status` and `git diff --cached --stat` for unintended data, secrets, private reports or large files. Do not commit confidential user data. This package excludes example data and report CSVs by default.

## C. Verify continuous integration

On GitHub open **Actions > Tests**. The workflow runs the test suite on Python 3.12 and 3.13 each time you push and for pull requests. Wait for both jobs to pass. Failures require inspection; do not simply skip failing tests.

## D. Deploy to Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in.
2. Choose **Create app** (wording may vary) and select the repository, branch `main`, and entrypoint `app.py` at repository root.
3. Deploy, then open the public URL Streamlit provides.
4. Smoke-test the public URL with **non-sensitive synthetic or public benchmark CSVs** and test both automatic and manual modes.
5. Confirm no private training files, credentials, or raw sensitive uploads were committed. Review the service's access controls and data-handling documentation before inviting external users.

**This project is a research preview, not a production security-reviewed service.** File content is processed on the server; do not upload personal, regulated, or confidential data to the hosted app.

## E. Release criteria

- Both Python CI jobs pass.
- An unmodified control and a controlled numerical/missingness change run on the public URL.
- No committed private data, tokens, or model artifacts.
- README shows research limitations, input size limits, and clear non-certification claims.
- Choose an explicit open-source license before describing the repository as open source.
- Keep the original v0.3/v0.4 backups while testing deployment.
