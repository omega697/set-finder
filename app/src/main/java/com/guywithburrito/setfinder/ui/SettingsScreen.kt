package com.guywithburrito.setfinder.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Menu
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.guywithburrito.setfinder.tracking.SettingsManager

@Composable
fun SettingsScreen(onBackClicked: () -> Unit) {
    val context = LocalContext.current
    val settingsManager = remember { SettingsManager(context) }
    val highlightColors = remember { settingsManager.highlightColors.toMutableStateList() }
    var sensitivity by remember { mutableFloatStateOf(settingsManager.sensitivity) }
    var showLabels by remember { mutableStateOf(settingsManager.showLabels) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = onBackClicked) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                backgroundColor = MaterialTheme.colors.surface,
                contentColor = MaterialTheme.colors.onSurface,
                elevation = 0.dp
            )
        },
        backgroundColor = MaterialTheme.colors.background
    ) { padding ->
        Column(modifier = Modifier.padding(padding).padding(16.dp)) {
            Text(
                text = "Highlight Colors",
                style = MaterialTheme.typography.h6,
                modifier = Modifier.padding(bottom = 8.dp)
            )
            Text(
                text = "Priority list for set highlights. Use buttons to reorder.",
                style = MaterialTheme.typography.body2,
                color = MaterialTheme.colors.onSurface.copy(alpha = 0.6f),
                modifier = Modifier.padding(bottom = 16.dp)
            )
            
            Card(
                elevation = 2.dp,
                backgroundColor = MaterialTheme.colors.surface,
                contentColor = MaterialTheme.colors.onSurface
            ) {
                LazyColumn(modifier = Modifier.fillMaxWidth()) {
                    itemsIndexed(highlightColors) { index, item ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(24.dp)
                                    .background(item.second, MaterialTheme.shapes.small)
                            )
                            Spacer(modifier = Modifier.width(16.dp))
                            Text(text = item.first, modifier = Modifier.weight(1f))
                            
                            Row {
                                if (index > 0) {
                                    IconButton(
                                        onClick = {
                                            val temp = highlightColors[index]
                                            highlightColors[index] = highlightColors[index - 1]
                                            highlightColors[index - 1] = temp
                                            settingsManager.saveColors(highlightColors)
                                        }
                                    ) {
                                        Text("↑", fontWeight = FontWeight.Bold, color = MaterialTheme.colors.primary)
                                    }
                                }
                                if (index < highlightColors.size - 1) {
                                    IconButton(
                                        onClick = {
                                            val temp = highlightColors[index]
                                            highlightColors[index] = highlightColors[index + 1]
                                            highlightColors[index + 1] = temp
                                            settingsManager.saveColors(highlightColors)
                                        }
                                    ) {
                                        Text("↓", fontWeight = FontWeight.Bold, color = MaterialTheme.colors.primary)
                                    }
                                }
                            }
                        }
                        if (index < highlightColors.size - 1) {
                            Divider(modifier = Modifier.padding(horizontal = 16.dp))
                        }
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(32.dp))
            
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(text = "Show Card Labels", style = MaterialTheme.typography.h6, modifier = Modifier.weight(1f))
                Switch(
                    checked = showLabels,
                    onCheckedChange = {
                        showLabels = it
                        settingsManager.showLabels = it
                    }
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Text(text = "Detection Sensitivity", style = MaterialTheme.typography.h6)
            Slider(
                value = sensitivity,
                onValueChange = { 
                    sensitivity = it
                    settingsManager.sensitivity = it
                },
                modifier = Modifier.fillMaxWidth()
            )
            Text(
                text = "Higher sensitivity detects cards faster but may be noisier.",
                style = MaterialTheme.typography.caption,
                color = MaterialTheme.colors.onSurface.copy(alpha = 0.6f)
            )

            Spacer(modifier = Modifier.height(32.dp))
            
            OutlinedButton(
                onClick = {
                    com.guywithburrito.setfinder.tracking.HistoryPersistence(context).clearHistory()
                },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = Color.Red)
            ) {
                Text("Clear Scan History")
            }
        }
    }
}
