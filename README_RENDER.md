# S2Cool Render Deployment

## Deploy

1. Create a new GitHub repository for the trimmed S2Cool deployment.
2. Copy the runtime files listed in `DEPLOYMENT_FILE_MANIFEST.md` into that repository.
3. Push the repository to GitHub.
4. In Render, create a new Web Service from that repository.
5. Use the settings in `render.yaml`, or set:
   - Build command: `pip install -r s2cool_python_app/requirements.txt gunicorn`
   - Start command: `cd s2cool_python_app && gunicorn --bind 0.0.0.0:$PORT app:server`
6. Choose the Free plan and deploy.

Render will provide a public URL ending in `onrender.com`.

## Local Render-mode smoke test

From the repository root:

```powershell
$env:HOST = "0.0.0.0"
$env:PORT = "8050"
$env:RENDER = "true"
python s2cool_python_app/app.py
```

For production-like serving after installing Gunicorn:

```powershell
cd s2cool_python_app
gunicorn --bind 0.0.0.0:8050 app:server
```

The free Render service may sleep when idle, so the first request after inactivity can take longer. Generated files are local to the running instance and should be downloaded through the app when needed.
