package com.guardianai.app.receiver

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.telephony.TelephonyManager
import android.util.Log

/**
 * Broadcast receiver for phone call state changes.
 * Triggers analysis when an incoming call is detected.
 */
class CallStateReceiver : BroadcastReceiver() {

    companion object {
        private const val TAG = "CallStateReceiver"
    }

    override fun onReceive(context: Context?, intent: Intent?) {
        if (intent?.action != TelephonyManager.ACTION_PHONE_STATE_CHANGED) return

        val state = intent.getStringExtra(TelephonyManager.EXTRA_STATE)
        val phoneNumber = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER)

        when (state) {
            TelephonyManager.EXTRA_STATE_RINGING -> {
                Log.d(TAG, "Incoming call from: $phoneNumber")
                // TODO: Start recording and analysis
                // TODO: Show overlay with caller information
            }
            TelephonyManager.EXTRA_STATE_OFFHOOK -> {
                Log.d(TAG, "Call answered")
                // TODO: Begin real-time analysis
            }
            TelephonyManager.EXTRA_STATE_IDLE -> {
                Log.d(TAG, "Call ended")
                // TODO: Stop recording, finalize analysis
                // TODO: Show analysis results notification
            }
        }
    }
}
