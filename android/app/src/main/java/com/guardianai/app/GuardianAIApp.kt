package com.guardianai.app

import android.app.Application

/**
 * Guardian AI Application class.
 * Initializes app-wide services and configurations.
 */
class GuardianAIApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // Initialize crash reporting, analytics, etc. here
    }
}
