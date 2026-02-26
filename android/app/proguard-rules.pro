# Proguard rules for Guardian AI

# Retrofit
-keepattributes Signature
-keepattributes *Annotation*
-keep class com.guardianai.app.network.** { *; }
-dontwarn retrofit2.**
-keep class retrofit2.** { *; }

# Gson
-keep class com.google.gson.** { *; }
-keepattributes EnclosingMethod

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**

# TensorFlow Lite
-keep class org.tensorflow.** { *; }

# ML Kit
-keep class com.google.mlkit.** { *; }
