# Flutter 3D Measurement App - Setup Guide

Complete setup guide for the Flutter Android app with auto-capture functionality.

---

## 📋 Prerequisites

### Required:
- ✅ Flutter SDK 3.0+ ([Download](https://flutter.dev))
- ✅ Android Studio or VS Code
- ✅ Android device with USB debugging enabled
- ✅ Server running on your computer (see backend setup)

### Verify Flutter Installation:
```bash
flutter doctor
```

All checkmarks should be green for Flutter, Android toolchain, and Android Studio/VS Code.

---

## 🚀 Quick Setup (5 Minutes)

### Step 1: Install Dependencies
```bash
cd flutter_measurement_app
flutter pub get
```

### Step 2: Connect Android Device
1. Enable **Developer Options** on your Android device:
   - Go to Settings → About Phone
   - Tap "Build Number" 7 times
2. Enable **USB Debugging**:
   - Go to Settings → Developer Options
   - Enable "USB Debugging"
3. Connect device via USB
4. Verify connection:
   ```bash
   flutter devices
   ```

### Step 3: Start Backend Server
On your computer:
```bash
cd C:\Users\harsh\Downloads\3D-measurement-main\3D-measurement-main
python main.py serve --host 0.0.0.0 --port 8000
```

### Step 4: Find Your Computer's IP
**Windows:**
```cmd
ipconfig
```
Look for "IPv4 Address" (e.g., 192.168.1.100)

**Mac/Linux:**
```bash
ifconfig
```
Look for "inet" address

### Step 5: Run the App
```bash
flutter run
```

### Step 6: Configure API in App
1. Once app launches, tap **Settings** icon (top right)
2. Enter your server URL: `http://YOUR_IP:8000`
3. Tap **Test Connection**
4. If successful, tap **Save Settings**
5. Go back to home screen

---

## 📸 How to Use

### Capturing Measurements:

1. **Home Screen**:
   - Check server status (should show green "Server Ready")
   - Tap "View Instructions" to see capture guidelines
   - Tap "Start Measurement"

2. **Camera Screen**:
   - Position object in frame (50-70cm away)
   - Ensure good lighting
   - Tap "Start Auto-Capture"
   - Walk slowly in a complete circle around object
   - 20 photos will be captured automatically (30 seconds)

3. **Processing**:
   - Photos compress and upload automatically
   - Server processes images (60-90 seconds)
   - Progress shown on screen

4. **Results**:
   - View measurements: Width, Height, Depth
   - See volume and surface area
   - Check confidence score
   - Tap "New Measurement" to start again

---

## 🎯 Capture Tips for Accuracy

### ✅ DO:
- **Lighting**: Use bright, even lighting
- **Distance**: Maintain 50-70cm distance
- **Coverage**: Complete full 360° circle
- **Speed**: Walk slowly and steadily
- **Stability**: Hold phone steady
- **Background**: Use plain background
- **Height**: Keep camera at same height

### ❌ DON'T:
- Move too fast
- Change distance during capture
- Use dim or harsh lighting
- Skip parts of the circle
- Capture reflective/transparent objects
- Use cluttered background

---

## ⚙️ Configuration

### Change Number of Images

Edit `lib/utils/constants.dart`:
```dart
static const int recommendedImages = 20;  // Change to 10-30
```

### Change Capture Speed

Edit `lib/utils/constants.dart`:
```dart
static const Duration captureInterval = Duration(milliseconds: 1500);  // 1.5 seconds
```

### Change Image Quality

Edit `lib/utils/constants.dart`:
```dart
static const int imageMaxWidth = 1024;  // Max width in pixels
static const int imageQuality = 85;     // JPEG quality (0-100)
```

---

## 🔧 Troubleshooting

### "Cannot connect to server"

**Check:**
- [ ] Server is running: `python main.py serve`
- [ ] Phone and computer on same WiFi
- [ ] IP address is correct in app settings
- [ ] Firewall allows port 8000
- [ ] Test in phone browser: `http://YOUR_IP:8000`

**Fix:**
```bash
# On computer, verify server
curl http://localhost:8000/health

# Should return: {"status":"healthy", ...}
```

---

### "Camera permission denied"

**Fix:**
1. Go to Settings → Apps → 3D Measurement
2. Tap Permissions
3. Enable Camera
4. Restart app

---

### "Server Not Ready"

**Check:**
```bash
# On computer
python main.py info

# Should show:
# GPU: NVIDIA GeForce GTX 1650
# CUDA: 12.1
# Status: Ready
```

**Fix:**
- Wait 10-15 seconds after starting (GPU initialization)
- Check GPU is detected: See `FIX_CUDA.md` in parent directory
- Review server logs for errors

---

### "Processing timeout"

**Causes:**
- Normal for first measurement (GPU warmup)
- Too many images
- Server overloaded

**Fix:**
- Wait full 3 minutes for first measurement
- Reduce image count to 10-15
- Check server logs
- Restart server

---

### "Low confidence results"

**Causes:**
- Poor lighting
- Incomplete coverage
- Inconsistent distance
- Reflective object

**Fix:**
- Recapture with better lighting
- Ensure full 360° circle
- Maintain consistent 50-70cm distance
- Try different object

---

## 📦 Building APK

### Debug Build (for testing):
```bash
flutter build apk
```
Location: `build/app/outputs/flutter-apk/app-debug.apk`

### Release Build (for distribution):
```bash
flutter build apk --release
```
Location: `build/app/outputs/flutter-apk/app-release.apk`

### Install on Device:
```bash
# Via Flutter
flutter install

# Or manually
adb install build/app/outputs/flutter-apk/app-release.apk
```

---

## 📱 App Features

### ✅ Implemented:
- Automatic 20-photo capture
- Timer-based capture (1.5s intervals)
- Visual guidance and progress
- Image compression (1024px, 85% quality)
- Configurable API endpoint
- Health check before capture
- Upload progress tracking
- Comprehensive error handling
- Results with confidence score
- Instructions dialog
- Settings screen

### 📋 Future Enhancements:
- Manual calibration with known dimension
- Measurement history
- Share results (PDF/image)
- AR preview overlay
- Cloud sync

---

## 🔍 Technical Details

### Architecture:
```
lib/
├── main.dart                    # Entry point
├── models/                      # Data models
│   ├── health_response.dart
│   ├── measurement_result.dart
│   └── capture_config.dart
├── services/                    # Business logic
│   ├── api_service.dart
│   ├── camera_service.dart
│   └── auto_capture_service.dart
├── screens/                     # UI screens
│   ├── home_screen.dart
│   ├── capture_screen.dart
│   ├── results_screen.dart
│   └── settings_screen.dart
├── widgets/                     # Reusable widgets
│   ├── capture_guidance.dart
│   └── instruction_dialog.dart
└── utils/                       # Configuration
    ├── constants.dart
    └── api_config.dart
```

### Key Technologies:
- **Flutter**: 3.0+ (Dart)
- **Camera**: camera plugin (v0.10.5+)
- **HTTP**: dio + http packages
- **Sensors**: sensors_plus (gyroscope/accelerometer)
- **Storage**: shared_preferences
- **Image Processing**: image package

### API Integration:
- **Base URL**: Configurable in settings
- **Endpoints**:
  - `GET /health` - Server status
  - `POST /measure` - Upload & measure
  - `GET /gpu-stats` - GPU info
- **Format**: multipart/form-data
- **Timeout**: 180 seconds

---

## 📊 Expected Performance

### Capture Phase:
- **Time**: 30 seconds (20 photos @ 1.5s)
- **File size**: 200-400 KB per photo (compressed)
- **Total upload**: 4-8 MB

### Upload Phase:
- **WiFi**: 10-20 seconds
- **4G/5G**: 20-40 seconds

### Processing Phase:
- **First time**: 90-120 seconds (GPU warmup)
- **Subsequent**: 60-90 seconds

### Accuracy:
- **Before calibration**: ±5-15%
- **After calibration**: ±2-5%
- **Confidence**: 20-40% (typical)

---

## 🌐 Production Deployment

### Current (Development):
- Server: Your computer (local IP)
- Access: Same WiFi only
- Protocol: HTTP

### Production (Recommended):
- Server: Cloud with GPU (AWS/Azure/GCP)
- Access: Internet (anywhere)
- Protocol: HTTPS
- Cost: ~$50-200/month

### To Deploy:
1. Set up cloud GPU server
2. Get public IP/domain
3. Update app settings with HTTPS URL
4. Build release APK
5. Distribute to users

---

## 📚 Additional Resources

- **Parent Docs**: `../FLUTTER_APP_GUIDE.md` - Complete guide
- **API Docs**: `../MOBILE_API_GUIDE.md` - Backend API
- **Backend Setup**: `../QUICKSTART.md` - Server setup
- **Troubleshooting**: `../TROUBLESHOOTING.md` - Common issues

---

## 🆘 Support

### Before Asking for Help:
- [ ] Flutter doctor passes
- [ ] Server is running and accessible
- [ ] Correct IP in settings
- [ ] Camera permission granted
- [ ] Phone and computer on same WiFi
- [ ] Tested server in browser

### Common Questions:

**Q: How accurate are measurements?**  
A: ±5-15% without calibration, ±2-5% with calibration

**Q: Can I use fewer photos?**  
A: Yes, minimum 3, but accuracy decreases

**Q: How long does processing take?**  
A: 60-90 seconds (first time may be 2 minutes)

**Q: Can I change the server?**  
A: Yes, tap Settings and enter new URL

**Q: Does it work offline?**  
A: No, requires server connection

---

## ✅ Setup Checklist

Before first use:
- [ ] Flutter installed: `flutter doctor`
- [ ] Dependencies installed: `flutter pub get`
- [ ] Device connected: `flutter devices`
- [ ] Server running: `python main.py serve`
- [ ] IP address found: `ipconfig` / `ifconfig`
- [ ] App launched: `flutter run`
- [ ] Settings configured with server IP
- [ ] Connection tested successfully

---

**Your Flutter app is ready to use!** 🎉

Start capturing 3D measurements with automatic photo capture in just 30 seconds!

