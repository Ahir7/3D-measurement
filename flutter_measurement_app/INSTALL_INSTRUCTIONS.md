# Installation Instructions

Simple step-by-step installation guide for the 3D Measurement app.

---

## 📱 For End Users (Installing APK)

### Step 1: Get the APK File
Your administrator will provide the APK file in one of these ways:
- Email attachment
- Download link
- USB transfer
- Shared folder

The file will be named something like: `app-release.apk` or `3d-measurement.apk`

### Step 2: Enable Installation from Unknown Sources

**On Android 8.0+ (Oreo and newer):**
1. When you try to install, Android will ask for permission
2. Tap **Settings**
3. Enable **Allow from this source**
4. Tap back button
5. Continue installation

**On Android 7.1 and older:**
1. Go to **Settings**
2. Tap **Security** or **Lock screen and security**
3. Enable **Unknown sources**
4. Tap **OK** to confirm

### Step 3: Install the App
1. Locate the APK file:
   - In **Downloads** folder
   - In **File Manager**
   - In notification if just downloaded
2. Tap the APK file
3. Tap **Install**
4. Wait for installation (5-10 seconds)
5. Tap **Open** when done

### Step 4: First Launch
1. App opens to home screen
2. Tap **Settings** icon (⚙️) at top right
3. Enter server URL (get from admin):
   - Example: `http://192.168.1.100:8000`
4. Tap **Test Connection**
5. If successful (green checkmark), tap **Save Settings**
6. Go back to home screen

### Step 5: Grant Permissions
When you first try to capture:
1. Android will ask: "Allow 3D Measurement to take pictures and record video?"
2. Tap **Allow** or **While using the app**
3. You're ready!

---

## 👨‍💻 For Developers (Building from Source)

### Prerequisites
```bash
# Verify Flutter installation
flutter doctor
# All checkmarks should be green for Flutter, Android, and IDE
```

**Required:**
- Flutter SDK 3.0+
- Android Studio or VS Code with Flutter plugin
- Git (optional, for version control)

### Step 1: Get the Code
```bash
# If you received a ZIP file:
unzip flutter_measurement_app.zip
cd flutter_measurement_app

# Or if using git:
git clone <repository-url>
cd flutter_measurement_app
```

### Step 2: Install Dependencies
```bash
flutter pub get
```

Expected output:
```
Running "flutter pub get" in flutter_measurement_app...
+ camera 0.10.5+5
+ dio 5.3.3
+ http 1.1.0
[... more packages ...]
Got dependencies!
```

### Step 3: Connect Device or Setup Emulator

**Physical Device (Recommended):**
1. Enable Developer Options:
   - Settings → About Phone
   - Tap "Build Number" 7 times
2. Enable USB Debugging:
   - Settings → Developer Options
   - Enable "USB Debugging"
3. Connect via USB
4. Verify: `flutter devices`

**Emulator (Alternative):**
1. Open Android Studio
2. Tools → AVD Manager
3. Create Virtual Device
4. Select device (e.g., Pixel 4)
5. Download and select system image (API 30+)
6. Finish and start emulator

### Step 4: Run in Debug Mode
```bash
flutter run
```

This will:
- Build the app
- Install on device/emulator
- Launch automatically
- Enable hot reload for development

**Hot reload:** Make code changes and press `r` in terminal to reload instantly

### Step 5: Build Release APK

**For Distribution:**
```bash
flutter build apk --release
```

Output location:
```
build/app/outputs/flutter-apk/app-release.apk
```

**For Testing (smaller, faster):**
```bash
flutter build apk
```

Output location:
```
build/app/outputs/flutter-apk/app-debug.apk
```

### Step 6: Install APK

**Using Flutter:**
```bash
flutter install
```

**Using ADB:**
```bash
adb install build/app/outputs/flutter-apk/app-release.apk
```

**Manual:**
- Copy APK to device
- Open file manager and tap APK
- Follow installation prompts

---

## 🔧 Development Setup

### VS Code (Recommended for quick edits)
1. Install VS Code
2. Install extensions:
   - Flutter
   - Dart
   - (Optional) Flutter Widget Snippets
3. Open project folder
4. Press F5 to run

### Android Studio (Full IDE experience)
1. Install Android Studio
2. Install Flutter and Dart plugins
3. File → Open → Select project folder
4. Wait for Gradle sync
5. Select device/emulator
6. Click Run (▶️) button

---

## 📦 Building for Distribution

### Signing the APK (for Play Store)

**1. Create keystore:**
```bash
keytool -genkey -v -keystore ~/upload-keystore.jks -keyalg RSA -keysize 2048 -validity 10000 -alias upload
```

**2. Create `android/key.properties`:**
```properties
storePassword=<password>
keyPassword=<password>
keyAlias=upload
storeFile=<path-to-jks-file>
```

**3. Update `android/app/build.gradle`:**
```gradle
// Add before android block
def keystoreProperties = new Properties()
def keystorePropertiesFile = rootProject.file('key.properties')
if (keystorePropertiesFile.exists()) {
    keystoreProperties.load(new FileInputStream(keystorePropertiesFile))
}

android {
    // ...
    
    signingConfigs {
        release {
            keyAlias keystoreProperties['keyAlias']
            keyPassword keystoreProperties['keyPassword']
            storeFile keystoreProperties['storeFile'] ? file(keystoreProperties['storeFile']) : null
            storePassword keystoreProperties['storePassword']
        }
    }
    
    buildTypes {
        release {
            signingConfig signingConfigs.release
        }
    }
}
```

**4. Build signed APK:**
```bash
flutter build apk --release
```

### App Bundle (for Play Store)
```bash
flutter build appbundle --release
```

Output: `build/app/outputs/bundle/release/app-release.aab`

---

## 🌐 Cloud Deployment Preparation

### Update for Production Server

**1. Change default URL in `lib/utils/constants.dart`:**
```dart
static const String defaultBaseUrl = 'https://your-domain.com';
```

**2. Update network security config (for HTTPS):**
Edit `android/app/src/main/res/xml/network_security_config.xml`:
```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <base-config cleartextTrafficPermitted="false">
        <trust-anchors>
            <certificates src="system" />
        </trust-anchors>
    </base-config>
</network-security-config>
```

**3. Build release:**
```bash
flutter build apk --release
```

---

## ✅ Pre-Release Checklist

Before distributing to users:

### Functionality
- [ ] App launches without errors
- [ ] Can connect to test server
- [ ] Camera opens and captures
- [ ] Auto-capture completes (20 photos)
- [ ] Upload shows progress
- [ ] Results display correctly
- [ ] Settings save and persist
- [ ] Instructions dialog opens
- [ ] Permissions requested properly

### Testing
- [ ] Tested on at least 3 different devices
- [ ] Tested with different Android versions (8.0+)
- [ ] Tested with poor network connection
- [ ] Tested with server offline
- [ ] Tested with camera permission denied
- [ ] Tested end-to-end workflow 10+ times

### Configuration
- [ ] Default server URL is correct
- [ ] App name is correct
- [ ] Icon is set (if custom)
- [ ] Version number updated in pubspec.yaml
- [ ] Network security config allows your servers

### Documentation
- [ ] Installation instructions provided to users
- [ ] Server URL provided to users
- [ ] Admin contact information shared
- [ ] User guide distributed

---

## 🆘 Installation Troubleshooting

### "App not installed"

**Causes:**
- Insufficient storage
- Conflicting package name
- Corrupted APK
- Android version too old

**Solutions:**
1. Free up storage space (need ~100MB)
2. Uninstall old version first
3. Re-download APK
4. Check Android version (need 5.0+)

---

### "Parse error"

**Cause:** APK file is corrupted or incompatible

**Solutions:**
1. Re-download APK from reliable source
2. Check Android version compatibility
3. Ensure download completed fully
4. Try different file transfer method

---

### "Installation blocked"

**Cause:** Security settings prevent installation

**Solutions:**
1. Enable "Unknown sources" (see Step 2 above)
2. Temporarily disable antivirus
3. Use device's file manager to install (not browser)

---

### App crashes on first open

**Solutions:**
1. Grant camera permission when asked
2. Restart device
3. Clear app cache: Settings → Apps → 3D Measurement → Storage → Clear Cache
4. Reinstall app
5. Check Android version (need 5.0+)

---

## 📊 System Requirements

### Minimum Requirements
- **Android**: 5.0 (Lollipop, API 21)
- **RAM**: 2 GB
- **Storage**: 100 MB free space
- **Camera**: Any working camera
- **Network**: WiFi or mobile data

### Recommended Requirements
- **Android**: 8.0+ (Oreo)
- **RAM**: 4 GB
- **Storage**: 500 MB free space
- **Camera**: 8MP+ with autofocus
- **Network**: WiFi (for faster upload)

### Not Supported
- Android below 5.0
- Devices without camera
- Tablets (not optimized, but should work)

---

## 📝 Version History

### Version 1.0.0 (Current)
- Initial release
- Automatic 20-photo capture
- Configurable API endpoint
- Real-time progress tracking
- Comprehensive error handling

### Future Versions
- Manual calibration
- Measurement history
- Share functionality
- AR preview
- iOS support

---

## 📞 Support

### For Installation Issues:
1. Check "Installation Troubleshooting" section above
2. Verify system requirements
3. Contact your administrator

### For Usage Help:
- See `USER_GUIDE.md`
- Tap "View Instructions" in app
- Contact your administrator

### For Administrators:
- See `SETUP_GUIDE.md` for server setup
- See `PROJECT_SUMMARY.md` for technical details
- See `FEATURES.md` for feature list

---

**Ready to install?** Follow Step 1 above!

**Already installed?** See `USER_GUIDE.md` for usage instructions.

