package com.guardianai.app.network

import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import java.util.concurrent.TimeUnit

/**
 * API Service for Guardian AI Backend communication.
 * Sends audio/image/video files to the Flask backend for analysis.
 */
interface ApiService {

    @Multipart
    @POST("/api/analyze/audio")
    suspend fun analyzeAudio(
        @Part file: MultipartBody.Part
    ): Response<AnalysisResponse>

    @Multipart
    @POST("/api/analyze/image")
    suspend fun analyzeImage(
        @Part file: MultipartBody.Part
    ): Response<AnalysisResponse>

    @Multipart
    @POST("/api/analyze/video")
    suspend fun analyzeVideo(
        @Part file: MultipartBody.Part
    ): Response<AnalysisResponse>

    @GET("/api/health")
    suspend fun healthCheck(): Response<HealthResponse>

    companion object {
        // Change this to your backend server URL
        private const val BASE_URL = "http://10.0.2.2:5000/" // Android emulator → localhost

        fun create(): ApiService {
            val logging = HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            }

            val client = OkHttpClient.Builder()
                .addInterceptor(logging)
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)
                .writeTimeout(60, TimeUnit.SECONDS)
                .build()

            return Retrofit.Builder()
                .baseUrl(BASE_URL)
                .client(client)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
                .create(ApiService::class.java)
        }
    }
}

// ─── Response Models ─────────────────────────────────────────

data class AnalysisResponse(
    val success: Boolean,
    val timestamp: String?,
    val data: AnalysisData?,
    val error: String?
)

data class AnalysisData(
    val type: String?,
    val filename: String?,
    val threat_score: Double?,
    val threat_level: String?,
    val threat_color: String?,
    val threat_description: String?,
    val confidence: Double?,
    val classification: String?,
    val processing_time: Double?,
    val transcript: String?,
    val recommendations: List<String>?,
    val detected_patterns: List<PatternData>?,
    val detected_indicators: List<IndicatorData>?,
    val analyses: List<SubAnalysisData>?
)

data class SubAnalysisData(
    val name: String?,
    val score: Double?,
    val weight: Double?,
    val details: String?
)

data class PatternData(
    val name: String?,
    val description: String?,
    val severity: String?
)

data class IndicatorData(
    val type: String?,
    val detail: String?,
    val severity: String?
)

data class HealthResponse(
    val success: Boolean,
    val data: HealthData?
)

data class HealthData(
    val status: String?,
    val service: String?,
    val version: String?
)
