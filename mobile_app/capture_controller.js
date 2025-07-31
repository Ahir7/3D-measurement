class AdvancedCaptureController {
    constructor() {
        this.sensorManager = new SensorManager();
        this.captureData = {
            images: [],
            imuData: [],
            metadata: [],
            timestamps: []
        };
    }

    async initializeCapture() {
        await this.sensorManager.startIMU();
        await this.sensorManager.requestPermissions();
        console.log("📱 Advanced capture system initialized");
    }

    async captureImageSequence(targetCount = 15) {
        console.log(`📸 Starting capture sequence for ${targetCount} images`);
        
        for (let i = 0; i < targetCount; i++) {
            const timestamp = Date.now();
            
            // Capture IMU data burst around image capture
            const imuBurst = await this.sensorManager.captureIMUBurst(timestamp);
            
            // Capture image with full metadata
            const image = await this.captureImageWithMetadata();
            
            // Store synchronized data
            this.captureData.images.push(image);
            this.captureData.imuData.push(imuBurst);
            this.captureData.metadata.push(image.exif);
            this.captureData.timestamps.push(timestamp);
            
            // Guide user movement
            await this.guideUserMovement(i, targetCount);
            
            console.log(`📸 Captured image ${i + 1}/${targetCount}`);
        }
        
        return this.captureData;
    }

    async captureImageWithMetadata() {
        const options = {
            quality: 1.0,
            base64: false,
            exif: true,
            skipProcessing: true
        };

        const image = await Camera.takePictureAsync(options);
        
        const enhancedMetadata = {
            ...image.exif,
            deviceInfo: await this.getDeviceInfo(),
            cameraInfo: await this.getCameraInfo(),
            environmentalData: await this.getEnvironmentalData()
        };

        return {
            ...image,
            exif: enhancedMetadata,
            captureTimestamp: Date.now()
        };
    }

    async guideUserMovement(currentIndex, totalCount) {
        const movements = [
            "Move slightly to the right",
            "Move slightly higher",
            "Move slightly to the left", 
            "Move slightly lower",
            "Move closer to the object",
            "Move farther from the object",
            "Rotate slightly clockwise",
            "Rotate slightly counter-clockwise"
        ];
        
        const movement = movements[currentIndex % movements.length];
        await this.showGuidance(movement);
        await this.wait(2000);
    }

    async sendToServer(serverUrl) {
        const formData = new FormData();
        
        // Add images
        this.captureData.images.forEach((image, index) => {
            formData.append('images', {
                uri: image.uri,
                type: 'image/jpeg',
                name: `image_${index}.jpg`
            });
        });
        
        // Add sensor data
        formData.append('imu_data', JSON.stringify(this.captureData.imuData));
        formData.append('metadata', JSON.stringify(this.captureData.metadata));
        formData.append('timestamps', JSON.stringify(this.captureData.timestamps));
        
        try {
            const response = await fetch(`${serverUrl}/calculate-dimensions-advanced`, {
                method: 'POST',
                body: formData,
                headers: {
                    'Content-Type': 'multipart/form-data',
                }
            });
            
            const results = await response.json();
            return results;
        } catch (error) {
            console.error('Error sending data to server:', error);
            throw error;
        }
    }

    async getDeviceInfo() {
        return {
            model: Device.modelName,
            os: Device.osName,
            osVersion: Device.osVersion,
            screenScale: Device.screenScale
        };
    }

    async getCameraInfo() {
        return {
            availableCameras: await Camera.getAvailableCamerasAsync(),
            cameraType: Camera.Constants.Type.back,
            flashMode: Camera.Constants.FlashMode.off
        };
    }

    async getEnvironmentalData() {
        return {
            lightLevel: await this.estimateLightLevel(),
            deviceOrientation: await this.getDeviceOrientation(),
            stabilityScore: await this.assessStability()
        };
    }

    async wait(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async showGuidance(message) {
        // Implement UI guidance display
        console.log(`Guidance: ${message}`);
    }

    async estimateLightLevel() {
        // Placeholder - implement actual light level estimation
        return 'normal';
    }

    async getDeviceOrientation() {
        // Placeholder - implement device orientation detection
        return 'portrait';
    }

    async assessStability() {
        // Placeholder - implement stability assessment
        return 0.8;
    }
}
