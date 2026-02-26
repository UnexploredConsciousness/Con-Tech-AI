package com.guardianai.app.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat

/**
 * Foreground service for monitoring phone calls.
 * Runs in background to detect incoming calls and analyze them for scam patterns.
 *
 * Usage:
 * 1. Start this service when the user enables call monitoring
 * 2. Service listens for call state changes
 * 3. When a call is detected, it records audio and sends to backend for analysis
 * 4. Shows overlay alert if scam is detected
 */
class CallMonitorService : Service() {

    companion object {
        const val CHANNEL_ID = "guardian_ai_call_monitor"
        const val NOTIFICATION_ID = 1001
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = createNotification()
        startForeground(NOTIFICATION_ID, notification)

        // TODO: Register phone state listener
        // TODO: Start audio recording when call is detected
        // TODO: Send audio to backend for analysis
        // TODO: Show overlay alert if scam detected

        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        // TODO: Unregister listeners, cleanup
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Call Monitor",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Guardian AI call monitoring service"
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Guardian AI")
            .setContentText("Monitoring calls for scam protection")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
    }
}
