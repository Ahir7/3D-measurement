# Quick Start - 10 Minutes ⚡

Get the Flutter 3D Measurement app running in 10 minutes!

---

## Step 1: Install Flutter (if not already)
```bash
# Download from https://flutter.dev
# Follow installation instructions for your OS
# Verify installation:
flutter doctor
```

---

## Step 2: Setup Project
```bash
cd flutter_measurement_app
flutter pub get
```

**Expected output:**
```
Running "flutter pub get" in measurement_app...
Resolving dependencies... (15.2s)
+ camera 0.10.5+5
+ dio 5.3.3
+ http 1.1.0
...
Got dependencies!
```

---

## Step 3: Connect Android Device
1. Enable Developer Options (Settings → About → Tap Build Number 7x)
2. Enable USB Debugging (Settings → Developer Options)
3. Connect via USB
4. Verify: `flutter devices`

---

## Step 4: Start Server
```bash
# In another terminal
cd C:\Users\harsh\Downloads\3D-measurement-main\3D-measurement-main
python main.py serve --host 0.0.0.0 --port 8000
```

---

## Step 5: Get Your IP
**Windows:** `ipconfig` → Look for IPv4 Address  
**Mac/Linux:** `ifconfig` → Look for inet address

Example: `192.168.1.100`

---

## Step 6: Run App
```bash
flutter run
```

Wait for app to install and launch on your device.

---

## Step 7: Configure
1. Tap **Settings** icon (⚙️ top right)
2. Enter: `http://YOUR_IP:8000`
3. Tap **Test Connection**
4. If green checkmark appears, tap **Save Settings**
5. Go back

---

## Step 8: Test!
1. Tap **Start Measurement**
2. Allow camera permission
3. Position a small object in frame
4. Tap **Start Auto-Capture**
5. Walk slowly in a circle
6. Wait for results!

---

## Expected Timeline

| Step | Time |
|------|------|
| Flutter setup | 2 min |
| Dependencies | 1 min |
| Device connection | 1 min |
| Server start | 30 sec |
| App run | 2 min |
| Configuration | 1 min |
| First test | 2 min |
| **Total** | **~10 min** |

---

## Troubleshooting Quick Fixes

### "flutter: command not found"
```bash
# Add Flutter to PATH
export PATH="$PATH:/path/to/flutter/bin"
```

### "No devices found"
- Check USB debugging enabled
- Try different USB cable/port
- Run: `adb devices`

### "Cannot connect to server"
- Both on same WiFi?
- Server running?
- Correct IP?
- Test in browser: `http://YOUR_IP:8000`

### "Camera permission denied"
- Settings → Apps → 3D Measurement → Permissions → Camera → Allow

---

## Next Steps

✅ **Read full guide**: See `SETUP_GUIDE.md`  
✅ **Understand captures**: See `README.md`  
✅ **Customize settings**: Edit `lib/utils/constants.dart`  
✅ **Build APK**: `flutter build apk --release`

---

**Done!** 🎉 You now have a working 3D measurement app!

Capture time: 30 seconds | Processing: 60-90 seconds | Results: Instant

