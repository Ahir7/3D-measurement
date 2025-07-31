class SensorManager {
    constructor() {
        this.imuBuffer = [];
        this.isRecording = false;
        this.samplingRate = 100; // Hz
    }

    async startIMU() {
        // Configure sensors
        Accelerometer.setUpdateInterval(1000 / this.samplingRate);
        Gyroscope.setUpdateInterval(1000 / this.samplingRate);
        Magnetometer.setUpdateInterval(1000 / this.samplingRate);
        
        this.isRecording = true;
        this.startContinuousRecording();
    }

    startContinuousRecording() {
        // Accelerometer data
        this.accelSubscription = Accelerometer.addListener(data => {
            if (this.isRecording) {
                this.imuBuffer.push({
                    type: 'accelerometer',
                    timestamp: Date.now(),
                    x: data.x,
                    y: data.y,
                    z: data.z
                });
            }
        });

        // Gyroscope data
        this.gyroSubscription = Gyroscope.addListener(data => {
            if (this.isRecording) {
                this.imuBuffer.push({
                    type: 'gyroscope',
                    timestamp: Date.now(),
                    x: data.x,
                    y: data.y,
                    z: data.z
                });
            }
        });

        // Magnetometer data
        this.magnetSubscription = Magnetometer.addListener(data => {
            if (this.isRecording) {
                this.imuBuffer.push({
                    type: 'magnetometer',
                    timestamp: Date.now(),
                    x: data.x,
                    y: data.y,
                    z: data.z
                });
            }
        });

        // Device motion (if available)
        if (DeviceMotion.isAvailableAsync()) {
            DeviceMotion.setUpdateInterval(1000 / this.samplingRate);
            this.motionSubscription = DeviceMotion.addListener(data => {
                if (this.isRecording) {
                    this.imuBuffer.push({
                        type: 'deviceMotion',
                        timestamp: Date.now(),
                        acceleration: data.acceleration,
                        accelerationIncludingGravity: data.accelerationIncludingGravity,
                        rotation: data.rotation,
                        orientation: data.orientation
                    });
                }
            });
        }
    }

    async captureIMUBurst(timestamp, windowMs = 500) {
        const startTime = timestamp - windowMs / 2;
        const endTime = timestamp + windowMs / 2;
        
        // Extract IMU data within time window
        const burstData = this.imuBuffer.filter(entry => 
            entry.timestamp >= startTime && entry.timestamp <= endTime
        );

        // Organize by sensor type
        const organized = {
            accelerometer: burstData.filter(d => d.type === 'accelerometer'),
            gyroscope: burstData.filter(d => d.type === 'gyroscope'),
            magnetometer: burstData.filter(d => d.type === 'magnetometer'),
            deviceMotion: burstData.filter(d => d.type === 'deviceMotion'),
            timestamp: timestamp
        };

        return organized;
    }

    async requestPermissions() {
        // Request necessary permissions for sensors
        const { status } = await Accelerometer.requestPermissionsAsync();
        if (status !== 'granted') {
            throw new Error('Accelerometer permission not granted');
        }
        
        const { status: gyroStatus } = await Gyroscope.requestPermissionsAsync();
        if (gyroStatus !== 'granted') {
            throw new Error('Gyroscope permission not granted');
        }
        
        const { status: magnetStatus } = await Magnetometer.requestPermissionsAsync();
        if (magnetStatus !== 'granted') {
            throw new Error('Magnetometer permission not granted');
        }
    }

    async stopIMU() {
        this.isRecording = false;
        
        if (this.accelSubscription) this.accelSubscription.remove();
        if (this.gyroSubscription) this.gyroSubscription.remove();
        if (this.magnetSubscription) this.magnetSubscription.remove();
        if (this.motionSubscription) this.motionSubscription.remove();
        
        this.imuBuffer = [];
    }
}
