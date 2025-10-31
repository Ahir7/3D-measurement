# Test Plan

This test plan covers functional verification of the Flutter app changes to ensure zero regressions and consistent behavior across modules and screens.

## Environments
- Device: Android (mid-range) and iOS (optional) phones
- Backend: Local server (`main.py serve`) or Docker container
- Network: Wi-Fi with typical latency; optionally simulate slow/unstable connections

## Core Workflows
- Launch app ➜ Health check ➜ Start capture ➜ Review ➜ Upload ➜ Results
- Repeat capture with different image counts (min, recommended, max)

## Settings & Connectivity
- Set base URL and save; confirm UI reflects URL
- Test connection using a typed URL without saving (should test typed URL)
- Toggle Test Mode ON/OFF and verify:
  - Home screen allows capture in Test Mode
  - Review screen shows Test Mode dialog with updated messaging

## Capture Responsiveness
- Take sequential photos quickly and slowly; verify no UI stutter
- Confirm progress indicators update smoothly during upload
- Verify compression does not block UI (no visible jank)

## Upload & Processing
- Observe upload progress percentage and transitions between steps
- Handle server unavailability gracefully (clear error messaging)
- Validate timeouts:
  - Health check times out appropriately
  - Upload respects `requestTimeout` and errors informatively
 - Verify multipart image content type is `image/jpeg` on server side
 - Confirm metadata includes device, capture, camera, and `location` when available
 - Confirm `imu_data` contains a summary entry and sample events

## Results Screen
- Validate measurement fields (width, height, depth, volume, surface area)
- Confirm confidence label/color and warning tips for low confidence
- Ensure “New Measurement” resets state correctly

## Edge Cases
- Minimal images (`minImages`)
- Maximal images (`maxImages`)
- Invalid server URL formats (e.g., missing protocol)
- Slow GPU stats or health response
- Intermittent network drops during upload

## Integration Points
- `ApiService` with `/health`, `/measure`, `/gpu-stats`
- `CameraService` compression pipeline and EXIF preservation
- `ApiConfig` persistence of base URL and Test Mode via `SharedPreferences`
 - `LocationService` permissions, service state, and fallback behavior
 - `ImuService` sample collection timing and payload formatting

## Layout & UI Consistency
- Verify layout across portrait/landscape orientations
- Ensure overlays (guidance, orientation) do not overlap controls
- Validate adaptive text and button visibility across different screen sizes

## Backward Compatibility
- Confirm `HealthResponse` and `MeasurementResult` JSON mapping unchanged
- Verify server interaction contracts remain consistent
 - Ensure uploads succeed even if location or IMU data are unavailable

## Pass/Fail Criteria
- All flows complete without crashes or dead-ends
- Errors are handled with clear, actionable messaging
- No UI jank during capture/compression
- No regressions observed compared to previous build