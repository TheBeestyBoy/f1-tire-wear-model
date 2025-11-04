# Deployment Guide - F1 Tire Wear AI Predictor

This guide covers deploying the backend to Railway and the frontend to Vercel.

## Architecture Overview

- **Backend (FastAPI)**: Deployed on Railway
- **Frontend (React)**: Deployed on Vercel
- **Communication**: Frontend calls backend API via environment variable

---

## Part 1: Deploy Backend to Railway

### Prerequisites
- Railway account (https://railway.app)
- GitHub repository pushed with all code

### Step 1: Create Railway Project

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your `f1-tire-wear-model` repository
5. Railway will detect the backend automatically

### Step 2: Configure Railway Service

1. **Root Directory**: Set to `backend`
   - In Railway dashboard → Settings → Root Directory: `backend`

2. **Environment Variables**: Add these in Settings → Variables
   ```
   PORT=8000
   ALLOWED_ORIGINS=https://your-vercel-app.vercel.app
   PYTHON_VERSION=3.11
   ```
   (You'll update `ALLOWED_ORIGINS` after deploying frontend)

3. **Build Settings**: Railway will auto-detect Python
   - Start command is defined in `Procfile`: `uvicorn app.f1_main:app --host 0.0.0.0 --port $PORT`
   - Build will install from `requirements.txt`

### Step 3: Deploy Backend

1. Railway will automatically deploy after configuration
2. Wait for build to complete (5-10 minutes for PyTorch)
3. Once deployed, copy your Railway URL (looks like: `https://your-app.up.railway.app`)

### Step 4: Verify Backend

Visit your Railway URL in browser - you should see:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "available_races": 149
}
```

### Important Notes

- **First deployment takes 5-10 minutes** (PyTorch is large)
- Railway automatically installs dependencies from `requirements.txt`
- The `ai_model/` folder with trained models must be in the repository
- Railway free tier: $5 credit/month, should be sufficient for low traffic

---

## Part 2: Deploy Frontend to Vercel

### Prerequisites
- Vercel account (https://vercel.com)
- Backend deployed on Railway (from Part 1)

### Step 1: Create Vercel Project

1. Go to https://vercel.com/new
2. Import your GitHub repository
3. Vercel will auto-detect React app

### Step 2: Configure Vercel

1. **Root Directory**: Set to `frontend`
   - Framework Preset: Create React App (auto-detected)
   - Root Directory: `frontend`
   - Build Command: `npm run build` (auto-filled)
   - Output Directory: `build` (auto-filled)

2. **Environment Variables**: Add in Settings → Environment Variables
   ```
   REACT_APP_API_URL=https://your-railway-backend.up.railway.app
   ```
   (Use your Railway URL from Part 1, **without trailing slash**)

### Step 3: Deploy Frontend

1. Click "Deploy"
2. Wait 2-3 minutes for build
3. Vercel will provide your app URL (e.g., `https://your-app.vercel.app`)

### Step 4: Update Backend CORS

Go back to Railway and update the `ALLOWED_ORIGINS` variable:
```
ALLOWED_ORIGINS=https://your-vercel-app.vercel.app
```

Railway will auto-redeploy with new CORS settings.

### Step 5: Verify Full Stack

1. Visit your Vercel URL
2. You should see the F1 Tire Wear AI Predictor interface
3. Select a race/driver and click "Run Prediction"
4. Charts should load with data from Railway backend

---

## Environment Variables Summary

### Backend (Railway)
| Variable | Value | Required |
|----------|-------|----------|
| `PORT` | `8000` | Yes (Railway provides automatically) |
| `ALLOWED_ORIGINS` | `https://your-vercel-app.vercel.app` | Yes |
| `PYTHON_VERSION` | `3.11` | Recommended |

### Frontend (Vercel)
| Variable | Value | Required |
|----------|-------|----------|
| `REACT_APP_API_URL` | `https://your-railway-backend.up.railway.app` | Yes |

---

## Troubleshooting

### Backend Issues

**"Model not loaded" error**
- Ensure `ai_model/` folder with trained models is in the repository
- Check Railway logs for model loading errors
- Verify folder structure: `ai_model/models/best_model.pth` exists

**"503 Service Unavailable"**
- Railway may still be building (check build logs)
- PyTorch takes 5-10 minutes to install
- Check Railway usage limits (free tier: $5/month)

**CORS errors in browser**
- Update `ALLOWED_ORIGINS` in Railway with your Vercel URL
- Ensure no trailing slash in URL
- Wait for Railway to redeploy (1-2 minutes)

### Frontend Issues

**"Failed to connect to backend"**
- Verify `REACT_APP_API_URL` is set correctly in Vercel
- Check Railway backend is deployed and healthy
- Test Railway URL directly in browser

**Charts not showing**
- Open browser console (F12) for errors
- Check if API calls are returning data
- Verify backend endpoints work: `https://your-railway-backend.up.railway.app/races/available`

**Environment variables not working**
- Vercel requires rebuild after changing env vars
- Go to Deployments → Click "..." → Redeploy
- Environment variables must start with `REACT_APP_`

---

## Local Development After Deployment

To test locally with deployed backend:

1. Create `frontend/.env.local`:
   ```
   REACT_APP_API_URL=https://your-railway-backend.up.railway.app
   ```

2. Run frontend:
   ```bash
   cd frontend
   npm start
   ```

To test locally with local backend:

1. Remove or comment out `.env.local`
2. Run backend: `python backend/app/f1_main.py`
3. Run frontend: `cd frontend && npm start`

---

## Deployment Costs (Estimated)

- **Railway**: $5 free credit/month
  - Backend uses ~1GB RAM, minimal CPU
  - Should stay within free tier for low traffic
  - Overage: $0.000231/GB-hour

- **Vercel**: Free tier (Hobby)
  - 100GB bandwidth/month
  - Unlimited deployments
  - Should stay free for most usage

**Total estimated cost: $0-5/month**

---

## Production Checklist

- [ ] Backend deployed on Railway with health check passing
- [ ] Frontend deployed on Vercel and loads correctly
- [ ] Environment variables configured on both platforms
- [ ] CORS configured to allow Vercel domain
- [ ] Test prediction with real race data
- [ ] Check all 5 charts load properly
- [ ] Verify custom tooltip shows tire info
- [ ] Test comparison mode works
- [ ] Monitor Railway logs for errors

---

## Updating After Deployment

### Update Backend
1. Push code changes to GitHub
2. Railway auto-deploys from `main` branch
3. Check Railway logs for deployment status

### Update Frontend
1. Push code changes to GitHub
2. Vercel auto-deploys from `main` branch
3. Check Vercel deployment logs

### Update Environment Variables
- Railway: Settings → Variables → Add/Edit → Save (auto-redeploys)
- Vercel: Settings → Environment Variables → Add/Edit → Redeploy manually

---

## Support

- Railway Docs: https://docs.railway.app
- Vercel Docs: https://vercel.com/docs
- FastAPI Docs: https://fastapi.tiangolo.com
- React Docs: https://react.dev

---

**Last Updated**: 2025-11-03
