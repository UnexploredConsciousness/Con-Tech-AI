package com.guardianai.app.ui.screens

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Upload
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

/**
 * Audio Analyzer Screen — Upload and analyze audio for scam detection.
 */
@Composable
fun AudioAnalyzerScreen() {
    var selectedFileUri by remember { mutableStateOf<Uri?>(null) }
    var selectedFileName by remember { mutableStateOf("") }
    var isAnalyzing by remember { mutableStateOf(false) }

    val filePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        selectedFileUri = uri
        selectedFileName = uri?.lastPathSegment ?: "audio_file"
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        Icon(
            Icons.Filled.Mic,
            contentDescription = null,
            modifier = Modifier.size(48.dp),
            tint = MaterialTheme.colorScheme.primary
        )
        Text(
            "Audio Scam Detection",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )
        Text(
            "Upload a phone call recording to analyze for fraud patterns, scam keywords, and suspicious audio characteristics.",
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Upload area
        OutlinedCard(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                if (selectedFileUri != null) {
                    Text("Selected: $selectedFileName", fontWeight = FontWeight.SemiBold)
                } else {
                    Text("No file selected", color = MaterialTheme.colorScheme.onSurfaceVariant)
                }

                Button(
                    onClick = { filePicker.launch("audio/*") },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(Icons.Filled.Upload, contentDescription = null, modifier = Modifier.size(20.dp))
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Select Audio File")
                }

                Button(
                    onClick = {
                        // TODO: Implement API call via ApiService
                        isAnalyzing = true
                    },
                    enabled = selectedFileUri != null && !isAnalyzing,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.tertiary
                    )
                ) {
                    if (isAnalyzing) {
                        CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Analyzing...")
                    } else {
                        Text("Analyze Audio")
                    }
                }
            }
        }

        // Results placeholder
        Text(
            "Results will appear here after analysis.",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            style = MaterialTheme.typography.bodySmall
        )
    }
}
