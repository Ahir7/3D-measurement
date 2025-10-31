# Changelog

This document records changes implemented to improve performance, reliability, and user experience while preserving backward compatibility and existing functionality.

## 2025-10-31

- Offloaded image compression to a background isolate in `CameraService` to prevent UI jank during capture and ensure smooth responsiveness.
  - Behavior unchanged: images are still resized to `AppConstants.imageMaxWidth` and compressed to `AppConstants.imageQuality` while preserving EXIF when available.
  - Impact: Faster, smoother capture experience; lower risk of dropped frames.

- Unified all HTTP interactions onto `Dio` in `ApiService` and centralized timeout configuration via `AppConstants`.
  - Health, GPU stats, measurements, and connection tests now consistently use `Dio` with appropriate per-request timeouts.
  - Introduced `AppConstants.connectTimeout` and `AppConstants.sendTimeout` and applied them across `ApiService`.

- Fixed base URL reinitialization logic in `ApiService`.
  - The service no longer overrides a dynamically updated base URL when calling `checkHealth()` or `testConnection()`.
  - Prevents false-negative connection tests when testing a URL prior to saving it in Settings.

- Clarified Test Mode dialog copy in `CaptureReviewScreen`.
  - Messaging now specifies that images are stored temporarily and may be removed when the app closes, and instructs users to disable Test Mode and recapture once the backend is available.

- Added explicit `Content-Type: image/jpeg` for multipart image uploads.
  - Ensures backend accepts files and avoids content-type validation issues.

- Included optional location metadata and short IMU sample burst in uploads.
  - Added `LocationService` (uses `geolocator`) and `ImuService` (uses `sensors_plus`).
  - `CaptureReviewScreen` now attaches GPS coordinates when available and 800ms of accelerometer/gyroscope samples.
  - Backend continues to receive fields `files`, `metadata`, and `imu_data` as JSON strings.

### Compatibility

- No breaking changes to public APIs or data models.
- `HealthResponse`, `MeasurementResult`, and server endpoints (`/health`, `/gpu-stats`, `/measure`) remain unchanged.
- `ApiService.measureDimensions` now accepts `imuData` as a list of sample maps.
- `CaptureConfig` and constants retain existing defaults and semantics.

### Testing Guidance

- Core workflows: Capture → Review → Upload/Process → Results.
- Edge cases: Slow server response, connection timeouts, upload progress updates, and test mode active.
- Integration points: `ApiService` with backend endpoints; `CameraService` compression; Settings base URL/testing flow.
- UI: Verify capture responsiveness, progress indicators, and dialog messaging.

Suggested manual checks:
- Settings: Enter a temporary URL and use “Test connection” without saving; it should test the typed URL correctly.
- Capture: Take a set of images; ensure UI remains responsive during compression.
- Upload: Watch upload progress and processing steps; validate that timeouts behave as configured.

### Rollback Procedures

If issues are discovered post-deployment:
1. Revert `flutter_measurement_app/lib/services/camera_service.dart` to the previous compression method (remove `Isolate.run` and restore inline processing).
2. Revert `flutter_measurement_app/lib/services/api_service.dart` to previous HTTP usage (restore `package:http` calls and prior `_initialize()` behavior).
   - If uploads fail due to new content type or metadata, remove `http_parser` import and `contentType` argument, and drop `location`/`imu_data` fields.
3. Remove newly added timeouts (`connectTimeout`, `sendTimeout`) from `AppConstants` and restore prior values.
4. Restore Test Mode dialog text to the previous copy if messaging regressions are reported.

Additionally, to disable new metadata collection:
- Delete `lib/services/location_service.dart` and remove `geolocator` from `pubspec.yaml`.
- Delete `lib/services/imu_service.dart` and remove `sensors_plus` from `pubspec.yaml`.

All changes are isolated to the Flutter app (`flutter_measurement_app`) and do not affect Python backend modules.