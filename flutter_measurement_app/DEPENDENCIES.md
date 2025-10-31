# Dependencies

This document lists runtime dependencies added or used by the Flutter app and how to verify installation.

## Core
- `dio` (HTTP client) — unified networking across the app.
- `http_parser` — sets explicit `Content-Type` for multipart file uploads.

## Camera & Images
- `camera` — device camera access.
- `image` — image resizing/compression and EXIF handling.
- `path_provider` — access to temporary and documents directories.

## Sensors & Location
- `geolocator` — GPS location, permissions and service status.
- `sensors_plus` — IMU streams (accelerometer and gyroscope).

## State & UI
- `provider` — state management.
- `flutter_spinkit`, `percent_indicator` — UI widgets.
- `intl` — formatting utilities.
- `shared_preferences` — persistent key-value storage.
- `permission_handler` — runtime permissions helper.

## Installation
Dependencies are declared in `pubspec.yaml`. Run:

```
flutter pub get
```

If `flutter` is not available, install Flutter SDK and ensure `flutter doctor` passes.

## Android Setup
- Location permissions are defined in `android/app/src/main/AndroidManifest.xml`:
  - `android.permission.ACCESS_COARSE_LOCATION`
  - `android.permission.ACCESS_FINE_LOCATION`
- No additional permissions required for `sensors_plus`.

## iOS Setup (if applicable)
- Add `NSLocationWhenInUseUsageDescription` to `Info.plist` if using location.
- Ensure pods are installed: `cd ios && pod install`.

## Verifying Installation
- Run `flutter pub get` without errors.
- Build the app: `flutter build apk` (Android) or `flutter run` (device/emulator).
- At runtime:
  - Upload process succeeds and server receives `image/jpeg` files.
  - Location is requested once; if denied or unavailable, app continues without error.
  - IMU samples are collected quickly and included in payload.