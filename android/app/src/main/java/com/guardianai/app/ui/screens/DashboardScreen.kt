package com.guardianai.app.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Dashboard Screen — overview of Guardian AI capabilities and stats.
 */
@Composable
fun DashboardScreen() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.height(24.dp))

        // Hero
        Icon(
            imageVector = Icons.Filled.Shield,
            contentDescription = "Shield",
            modifier = Modifier.size(72.dp),
            tint = MaterialTheme.colorScheme.primary
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Guardian AI",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Bold
        )

        Text(
            text = "Multi-Modal Threat Detection",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(32.dp))

        // Stats Row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            StatCard("94.7%", "Audio\nAccuracy")
            StatCard("91.3%", "Image\nAccuracy")
            StatCard("89.6%", "Video\nAccuracy")
        }

        Spacer(modifier = Modifier.height(32.dp))

        // Feature Cards
        FeatureCard(
            icon = Icons.Filled.Mic,
            title = "Audio Scam Detection",
            description = "Analyze phone calls for scam patterns using speech recognition, keyword detection, and behavioral analysis."
        )

        Spacer(modifier = Modifier.height(12.dp))

        FeatureCard(
            icon = Icons.Filled.Image,
            title = "Image Deepfake Detection",
            description = "Detect AI-generated images using metadata forensics, noise analysis, and face artifact detection."
        )

        Spacer(modifier = Modifier.height(12.dp))

        FeatureCard(
            icon = Icons.Filled.Videocam,
            title = "Video Deepfake Detection",
            description = "Identify manipulated videos through temporal consistency, face tracking, and frame-by-frame analysis."
        )

        Spacer(modifier = Modifier.height(24.dp))
    }
}

@Composable
fun StatCard(value: String, label: String) {
    Card(
        modifier = Modifier.width(100.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = value,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = label,
                style = MaterialTheme.typography.bodySmall,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
                lineHeight = 14.sp
            )
        }
    }
}

@Composable
fun FeatureCard(icon: ImageVector, title: String, description: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.Top
        ) {
            Icon(
                imageVector = icon,
                contentDescription = title,
                modifier = Modifier.size(32.dp),
                tint = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.width(16.dp))
            Column {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = description,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}
