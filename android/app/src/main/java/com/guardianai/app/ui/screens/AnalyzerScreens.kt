package com.guardianai.app.ui.screens

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * Audio Analyzer Screen — upload audio files for scam detection.
 */
@Composable
fun AudioAnalyzerScreen() {
    AnalyzerScreen(
        title = "Audio Scam Detection",
        description = "Upload a phone call recording to detect scam patterns",
        icon = Icons.Filled.Mic,
        fileTypes = arrayOf("audio/*"),
        analyzeType = "audio"
    )
}

/**
 * Image Analyzer Screen — upload images for deepfake detection.
 */
@Composable
fun ImageAnalyzerScreen() {
    AnalyzerScreen(
        title = "Image Deepfake Detection",
        description = "Upload an image to check for AI generation or manipulation",
        icon = Icons.Filled.Image,
        fileTypes = arrayOf("image/*"),
        analyzeType = "image"
    )
}

/**
 * Video Analyzer Screen — upload videos for deepfake detection.
 */
@Composable
fun VideoAnalyzerScreen() {
    AnalyzerScreen(
        title = "Video Deepfake Detection",
        description = "Upload a video to check for deepfake manipulation",
        icon = Icons.Filled.Videocam,
        fileTypes = arrayOf("video/*"),
        analyzeType = "video"
    )
}

/**
 * Shared analyzer screen component used by all three analysis types.
 */
@Composable
fun AnalyzerScreen(
    title: String,
    description: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    fileTypes: Array<String>,
    analyzeType: String
) {
    var selectedUri by remember { mutableStateOf<Uri?>(null) }
    var isAnalyzing by remember { mutableStateOf(false) }
    var resultText by remember { mutableStateOf<String?>(null) }
    var threatLevel by remember { mutableStateOf<String?>(null) }
    var threatScore by remember { mutableStateOf<Double?>(null) }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        selectedUri = uri
        resultText = null
        threatLevel = null
        threatScore = null
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    ) {
        // Header
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer
            )
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = icon,
                    contentDescription = title,
                    modifier = Modifier.size(40.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.width(16.dp))
                Column {
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = description,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // Upload Button
        OutlinedCard(
            modifier = Modifier.fillMaxWidth(),
            onClick = { launcher.launch(fileTypes.first()) }
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    imageVector = Icons.Filled.CloudUpload,
                    contentDescription = "Upload",
                    modifier = Modifier.size(48.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = if (selectedUri != null) "File Selected ✓" else "Tap to Select File",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Medium
                )
                if (selectedUri != null) {
                    Text(
                        text = selectedUri.toString().substringAfterLast("/"),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Analyze Button
        Button(
            onClick = {
                isAnalyzing = true
                // TODO: Call ApiService.analyzeAudio/Image/Video with the selected URI
                // For now, simulate analysis
                resultText = "Analysis would be performed via the Flask backend API.\n\nConnect to http://localhost:5000 and upload the file for real analysis."
                threatLevel = "MEDIUM"
                threatScore = 45.0
                isAnalyzing = false
            },
            modifier = Modifier.fillMaxWidth(),
            enabled = selectedUri != null && !isAnalyzing
        ) {
            if (isAnalyzing) {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp),
                    color = MaterialTheme.colorScheme.onPrimary,
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Analyzing...")
            } else {
                Icon(Icons.Filled.Analytics, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Analyze ${analyzeType.replaceFirstChar { it.uppercase() }}")
            }
        }

        // Results
        if (resultText != null) {
            Spacer(modifier = Modifier.height(20.dp))

            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = when (threatLevel) {
                        "CRITICAL", "HIGH" -> MaterialTheme.colorScheme.errorContainer
                        "MEDIUM" -> MaterialTheme.colorScheme.tertiaryContainer
                        else -> MaterialTheme.colorScheme.secondaryContainer
                    }
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Threat Level: ${threatLevel ?: "N/A"}",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        threatScore?.let {
                            Text(
                                text = "${it.toInt()}%",
                                style = MaterialTheme.typography.headlineMedium,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(8.dp))

                    // Progress bar for threat score
                    threatScore?.let { score ->
                        LinearProgressIndicator(
                            progress = { (score / 100).toFloat() },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(8.dp),
                        )
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Text(
                        text = resultText ?: "",
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
    }
}
