# Deployment Guide for Streamlit Community Cloud

This guide will help you deploy the PigeonPhoto Automator to Streamlit Community Cloud.

## Prerequisites

1. A GitHub account
2. A Streamlit Community Cloud account (free)
3. Git installed on your local machine

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right, then "New repository"
3. Name it `PigeonPhoto_Automator` (or any name you prefer)
4. Choose **Public** (required for free Streamlit Cloud)
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Push Your Code to GitHub

Run these commands in your project directory:

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/PigeonPhoto_Automator.git

# Rename branch to main (if needed)
git branch -M main

# Push your code
git push -u origin main
```

If you're prompted for credentials:
- Use a **Personal Access Token** instead of your password
- Create one at: https://github.com/settings/tokens
- Give it `repo` permissions

## Step 3: Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `YOUR_USERNAME/PigeonPhoto_Automator`
5. Select branch: `main`
6. Main file path: `app.py`
7. Click "Deploy"

## Step 4: Configure the App (if needed)

The app should work out of the box, but you may need to:

1. **Add model file** (if you have a trained YOLOv8 model):
   - Go to your GitHub repository
   - Upload `pigeon_pose_v1.pt` to the `models/` folder
   - Commit the changes

2. **Add template file** (if not already included):
   - Ensure `Pigeon Template.jpg` is in the repository root or `assets/templates/`
   - The app will use a white background if template is missing

3. **Add optional assets**:
   - `assets/overlays/glass_reflection.png` (for eye enhancement)
   - `assets/fonts/TrajanPro.ttf` (for custom text font)

## Important Notes

- **Model files are large**: The `.gitignore` excludes `.pt` files. If you have a trained model, consider using Git LFS or hosting it separately.
- **First run**: The app will download rembg models automatically (may take a minute)
- **Free tier limits**: Streamlit Community Cloud has resource limits. For production, consider upgrading.

## Troubleshooting

### App fails to start
- Check the logs in Streamlit Cloud dashboard
- Ensure all dependencies are in `requirements.txt`
- Verify `app.py` is in the root directory

### Missing dependencies
- Add any missing packages to `requirements.txt`
- Push changes to GitHub
- Streamlit Cloud will automatically redeploy

### Model not found
- The app will use default YOLOv8n-pose model if `pigeon_pose_v1.pt` is missing
- This model needs fine-tuning for best results

### Template not found
- The app creates a white background if template is missing
- Upload `Pigeon Template.jpg` to fix this

## Updating the App

After making changes:

```bash
git add .
git commit -m "Your commit message"
git push origin main
```

Streamlit Cloud will automatically redeploy your app.

## Support

For Streamlit Cloud issues, see: https://docs.streamlit.io/streamlit-community-cloud

