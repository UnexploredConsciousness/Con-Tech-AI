package com.guardianai.app.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

// ─── Guardian AI Brand Colors ────────────────────────────────

val GuardianPrimary = Color(0xFF6366F1)       // Indigo 500
val GuardianSecondary = Color(0xFF8B5CF6)      // Violet 500
val GuardianTertiary = Color(0xFFA855F7)       // Purple 500

val GuardianDarkSurface = Color(0xFF0F0F23)
val GuardianDarkSurfaceVariant = Color(0xFF1A1A2E)

private val DarkColorScheme = darkColorScheme(
    primary = GuardianPrimary,
    secondary = GuardianSecondary,
    tertiary = GuardianTertiary,
    surface = GuardianDarkSurface,
    surfaceVariant = GuardianDarkSurfaceVariant,
    background = GuardianDarkSurface,
    onPrimary = Color.White,
    onSecondary = Color.White,
    onTertiary = Color.White,
    onSurface = Color.White,
    onBackground = Color.White,
    error = Color(0xFFEF4444),
    errorContainer = Color(0xFF3B1111)
)

private val LightColorScheme = lightColorScheme(
    primary = GuardianPrimary,
    secondary = GuardianSecondary,
    tertiary = GuardianTertiary,
    surface = Color(0xFFFAFAFF),
    surfaceVariant = Color(0xFFF0F0FF),
    background = Color.White,
    onPrimary = Color.White,
    error = Color(0xFFDC2626)
)

@Composable
fun GuardianAITheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography(),
        content = content
    )
}
