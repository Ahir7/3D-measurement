# User Guide - Flutter 3D Measurement App

Step-by-step guide for using the 3D measurement app.

---

## 🎬 Getting Started

### First Time Setup

1. **Install the App**
   - Get the APK from your admin
   - Enable "Install from Unknown Sources" if needed
   - Install and open

2. **Grant Permissions**
   - When prompted, allow **Camera** permission
   - This is required for photo capture

3. **Configure Server**
   - Tap the **Settings** icon (⚙️) at top right
   - Enter server URL provided by admin (e.g., `http://192.168.1.100:8000`)
   - Tap **Test Connection**
   - Wait for green checkmark
   - Tap **Save Settings**
   - Go back to home screen

---

## 📱 Home Screen

### What You'll See:

**Status Indicator**
- 🟢 **Green dot** + "Server Ready" = Ready to measure
- 🔴 **Red dot** + "Cannot connect" = Server issue
- 🟠 **Orange dot** + "Checking..." = Testing connection

**GPU Information** (if connected)
- GPU name (e.g., "NVIDIA GeForce GTX 1650")
- Available memory

**Buttons**
- **View Instructions** - See how to capture properly
- **Start Measurement** - Begin capture process (only if server ready)
- **Retry Connection** - Test server again if failed

**Top Bar**
- ⚙️ **Settings** - Change server URL
- 🔄 **Refresh** - Check server status again

---

## 📸 Capture Process

### Step 1: Prepare Object
- Place object on flat, stable surface
- Clear background (plain wall or table)
- Ensure good lighting (bright, no harsh shadows)
- Remove nearby clutter

### Step 2: Start Capture
1. From home screen, tap **Start Measurement**
2. Camera opens with instructions overlay
3. Read the on-screen instructions:
   - "Position the object in frame"
   - "Stand about 50-70cm away"
   - "Walk slowly in a complete circle"

### Step 3: Position Yourself
- Stand **50-70cm** (arm's length) from object
- Keep object centered in camera view
- Hold phone at comfortable height
- Make sure object is fully visible

### Step 4: Begin Auto-Capture
- Tap **Start Auto-Capture** button
- Countdown begins
- **Start walking slowly** in a circle around object

### Step 5: During Capture (30 seconds)
**What You'll See:**
- Circular progress indicator
- Percentage (0% → 100%)
- Photo counter (e.g., "5 / 20 photos")
- Guidance messages:
  - "Hold steady" - Stop moving briefly
  - "Move slower" - You're going too fast
  - "Move faster" - Speed up a bit
  - "Perfect! Keep going" - Good pace

**What You Should Do:**
- Walk in a smooth circle
- Keep same distance from object
- Hold phone steady (don't shake)
- Maintain same height
- Complete full 360° circle
- Let app capture automatically (no need to tap)

### Step 6: Upload & Processing
**After 20 photos captured:**
- Screen shows "Uploading... X%"
- Photos compress automatically
- Upload progress: 0% → 100%
- Then: "Processing on server..."
- Time estimate: "This may take 60-90 seconds"

**What's Happening:**
1. Images compressed (~70% size reduction)
2. Uploaded to server
3. Server processes with GPU
4. 3D reconstruction created
5. Measurements calculated

---

## 📊 Results Screen

### What You'll See:

**Confidence Score** (top card, color-coded)
- 🟢 **Green (70%+)**: High Accuracy - Great job!
- 🟠 **Orange (40-70%)**: Medium Accuracy - Consider recapture
- 🔴 **Red (<40%)**: Low Accuracy - Recapture recommended

**Dimensions Section**
- 📏 **Width**: Horizontal measurement (cm)
- 📐 **Height**: Vertical measurement (cm)
- 🔲 **Depth**: Front-to-back measurement (cm)

**Calculated Values**
- 📦 **Volume**: Total space occupied (cm³)
- 📄 **Surface Area**: Total surface (cm²)

**Processing Information**
- **3D Points**: Number of points in 3D model
- **Quality**: Point cloud quality percentage
- **Processing Time**: Total time taken
- **GPU Time**: GPU computation time

### Actions Available:

- **Share** (top right) - Share results text
- **New Measurement** - Start over with new object
- **Home** (top left) - Return to home screen

---

## ⚙️ Settings Screen

### How to Access:
- Tap ⚙️ icon on home screen

### What You Can Do:

**Change Server URL**
1. Tap in text field
2. Enter new URL (must start with `http://` or `https://`)
3. Tap **Test Connection**
4. Wait for result (green checkmark or red X)
5. If successful, tap **Save Settings**

**Test Connection**
- Verifies server is reachable
- Shows success/failure message
- No changes saved until you tap "Save Settings"

**Reset to Default**
- Tap 🔄 icon in URL field
- Restores original default URL

### Help Section:
Shows how to find your computer's IP address:
- Windows: Run `ipconfig`
- Mac/Linux: Run `ifconfig`
- Format: `http://192.168.1.100:8000`

### Current Configuration:
Displays your current settings:
- Server URL
- Images per capture (20)
- Capture interval (1500ms)
- Request timeout (180s)

---

## 💡 Tips for Best Results

### Lighting
✅ **Good**: Bright, even lighting (natural daylight or multiple lamps)  
❌ **Bad**: Dim lighting, single harsh light, direct sunlight creating shadows

### Distance
✅ **Good**: 50-70cm (arm's length), consistent throughout  
❌ **Bad**: Too close (<30cm), too far (>1m), changing distance

### Movement
✅ **Good**: Slow, steady walk in smooth circle  
❌ **Bad**: Fast movement, jerky stops, incomplete circle

### Object Placement
✅ **Good**: Stable surface, plain background, clear space around  
❌ **Bad**: Cluttered background, unstable placement, multiple objects

### Camera Technique
✅ **Good**: Hold steady, same height, object centered  
❌ **Bad**: Shaking, changing height, object at edge of frame

---

## 🔧 Troubleshooting

### "Cannot connect to server"

**What it means**: App can't reach the server  

**Check:**
1. Is server running? (Ask admin)
2. Are you on the same WiFi as server?
3. Is the URL correct in Settings?

**Try:**
1. Tap Settings → Test Connection
2. Try different WiFi network
3. Ask admin for correct server URL
4. Restart app

---

### "Server Not Ready"

**What it means**: Server is reachable but not ready to process  

**Check:**
1. Just started server? Wait 10-15 seconds for GPU initialization
2. Is GPU properly detected? (Ask admin)

**Try:**
1. Wait 30 seconds and tap Refresh
2. Ask admin to check server logs
3. Ask admin to restart server

---

### "Camera permission denied"

**What it means**: App doesn't have camera access  

**Fix:**
1. Go to phone Settings
2. Tap Apps or Applications
3. Find "3D Measurement" app
4. Tap Permissions
5. Enable Camera
6. Return to app and restart

---

### "Processing timeout"

**What it means**: Server took too long to process  

**Reasons:**
- First measurement (normal - GPU warmup)
- Server busy with other request
- Too many/large images

**Try:**
1. Wait full 3 minutes for first measurement
2. Recapture with better lighting
3. Contact admin if persists

---

### Low Confidence Score (<40%)

**What it means**: Measurements may not be accurate  

**Common causes:**
- Poor lighting
- Incomplete 360° coverage
- Inconsistent distance
- Object too reflective/transparent
- Cluttered background

**Fix:**
1. Review "Tips for Best Results" section above
2. Recapture with:
   - Better lighting
   - Complete circle walk
   - Consistent distance
   - Plain background
3. For critical measurements, capture again and compare

---

## 📋 Quick Reference

### Capture Checklist
Before starting capture:
- [ ] Object on stable surface
- [ ] Plain background
- [ ] Good lighting
- [ ] Clear space around object (1-2m circle)
- [ ] You're standing 50-70cm away
- [ ] Object fully visible in camera

During capture:
- [ ] Walk in smooth circle
- [ ] Same distance throughout
- [ ] Same height throughout
- [ ] Complete full 360°
- [ ] Let app capture automatically (30 seconds)

---

### When to Recapture

Recapture if:
- Confidence score < 40%
- Measurements seem obviously wrong
- You moved too fast
- Didn't complete full circle
- Lighting was poor
- You changed distance during capture

---

### Expected Accuracy

**With good technique:**
- Confidence: 20-40%
- Accuracy: ±5-15% (before calibration)
- Accuracy: ±2-5% (after calibration with known dimension)

**Best practices:**
- Multiple captures of same object
- Average the results
- Use manual calibration if available

---

## 🎯 Typical Workflow

### For Routine Measurements:
1. Open app (5 seconds)
2. Check server ready (automatic)
3. Tap Start Measurement
4. Position object and self
5. Tap Start Auto-Capture
6. Walk in circle (30 seconds)
7. Wait for processing (60-90 seconds)
8. View results
9. Tap New Measurement for next object

**Total time per object: ~2-3 minutes**

---

### For Critical Measurements:
1. Set up object carefully (good lighting, plain background)
2. Capture once (2-3 minutes)
3. Note results
4. Capture again (2-3 minutes)
5. Compare results
6. If similar (within 5%), average them
7. If different (>10%), capture third time
8. Use median of three captures

**Total time: 6-10 minutes for high confidence**

---

## 📞 Getting Help

### In-App Help:
- **Home screen**: Tap "View Instructions"
- **Settings screen**: See "How to find your server IP" section

### Contact Admin If:
- Server not responding
- Results consistently wrong
- App crashes
- Camera not working
- Need different server URL

### Self-Help:
1. Read this guide fully
2. Check "Troubleshooting" section above
3. Try capturing different object
4. Restart app
5. Restart phone
6. Reinstall app (last resort)

---

## 🔒 Privacy & Data

### What's Stored Locally:
- Server URL (in app settings)
- No images saved permanently
- No measurement history (in current version)

### What's Sent to Server:
- Captured photos (deleted after processing)
- No personal information
- No location data

### Data Retention:
- Photos: Deleted from phone after upload
- Results: Only displayed, not saved locally
- Server: Ask admin about server data retention

---

## 📱 App Information

**Version**: 1.0.0  
**Platform**: Android (iOS version available on request)  
**Required Android**: 5.0+ (API 21+)  
**Permissions**: Camera, Internet, Storage  
**Size**: ~50 MB  
**Developer**: [Your Organization]  

---

**Need more help?** Contact your system administrator.

**Ready to start?** Open the app and tap "View Instructions"!

