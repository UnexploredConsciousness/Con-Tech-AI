package com.guardianai.app.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.guardianai.app.ui.screens.AudioAnalyzerScreen
import com.guardianai.app.ui.screens.ImageAnalyzerScreen
import com.guardianai.app.ui.screens.VideoAnalyzerScreen
import com.guardianai.app.ui.screens.DashboardScreen
import com.guardianai.app.ui.theme.GuardianAITheme

/**
 * Main Activity — entry point for Guardian AI Android app.
 * Uses Jetpack Compose with bottom navigation for:
 *   - Dashboard (overview & stats)
 *   - Audio Analysis (call scam detection)
 *   - Image Analysis (deepfake detection)
 *   - Video Analysis (deepfake detection)
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            GuardianAITheme {
                GuardianAIApp()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GuardianAIApp() {
    val navController = rememberNavController()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route

    val bottomNavItems = listOf(
        BottomNavItem("dashboard", "Dashboard", Icons.Filled.Home),
        BottomNavItem("audio", "Audio", Icons.Filled.Mic),
        BottomNavItem("image", "Image", Icons.Filled.Image),
        BottomNavItem("video", "Video", Icons.Filled.Videocam),
    )

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Guardian AI",
                        style = MaterialTheme.typography.titleLarge
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                    titleContentColor = MaterialTheme.colorScheme.onSurface
                )
            )
        },
        bottomBar = {
            NavigationBar {
                bottomNavItems.forEach { item ->
                    NavigationBarItem(
                        icon = { Icon(item.icon, contentDescription = item.label) },
                        label = { Text(item.label) },
                        selected = currentRoute == item.route,
                        onClick = {
                            navController.navigate(item.route) {
                                popUpTo(navController.graph.startDestinationId) { saveState = true }
                                launchSingleTop = true
                                restoreState = true
                            }
                        }
                    )
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = "dashboard",
            modifier = Modifier.padding(innerPadding)
        ) {
            composable("dashboard") { DashboardScreen() }
            composable("audio") { AudioAnalyzerScreen() }
            composable("image") { ImageAnalyzerScreen() }
            composable("video") { VideoAnalyzerScreen() }
        }
    }
}

data class BottomNavItem(
    val route: String,
    val label: String,
    val icon: androidx.compose.ui.graphics.vector.ImageVector
)
