package com.guywithburrito.setfinder.ui

import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGesturesAfterLongPress
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Menu
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.guywithburrito.setfinder.tracking.SettingsManager
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

@Composable
fun SettingsScreen(onBackClicked: () -> Unit) {
    val context = LocalContext.current
    val settingsManager = remember { SettingsManager(context) }
    val highlightColors = remember { mutableStateListOf<Pair<String, Color>>().apply { addAll(settingsManager.highlightColors) } }
    var sensitivity by remember { mutableFloatStateOf(settingsManager.sensitivity) }
    var showLabels by remember { mutableStateOf(settingsManager.showLabels) }
    var useYOLO by remember { mutableStateOf(settingsManager.useYOLO) }
    var arMode by remember { mutableStateOf(settingsManager.arMode) }

    // Switch implementation when setting changes
    LaunchedEffect(useYOLO) {
        com.guywithburrito.setfinder.cv.QuadFinder.setImplementation(context, useYOLO)
    }

    val scope = rememberCoroutineScope()
    val lazyListState = rememberLazyListState()
    var draggedIndex by remember { mutableStateOf<Int?>(null) }
    var draggingOffset by remember { mutableFloatStateOf(0f) }
    var autoscrollJob by remember { mutableStateOf<Job?>(null) }

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
                text = "Priority list for set highlights. Long-press to drag and reorder.",
                style = MaterialTheme.typography.body2,
                color = MaterialTheme.colors.onSurface.copy(alpha = 0.6f),
                modifier = Modifier.padding(bottom = 16.dp)
            )
            
            Card(
                elevation = 2.dp,
                shape = RoundedCornerShape(8.dp),
                backgroundColor = MaterialTheme.colors.surface,
                contentColor = MaterialTheme.colors.onSurface
            ) {
                LazyColumn(
                    state = lazyListState,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    itemsIndexed(highlightColors, key = { _, item -> item.first }) { index, item ->
                        val isDragging = draggedIndex == index
                        val elevation by animateDpAsState(if (isDragging) 8.dp else 0.dp)

                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(56.dp)
                                .zIndex(if (isDragging) 1f else 0f)
                                .shadow(elevation)
                                .background(if (isDragging) MaterialTheme.colors.surface else Color.Transparent)
                                .offset(y = if (isDragging) draggingOffset.dp else 0.dp)
                                .pointerInput(Unit) {
                                    detectDragGesturesAfterLongPress(
                                        onDragStart = { 
                                            draggedIndex = index
                                            draggingOffset = 0f
                                        },
                                        onDragEnd = {
                                            draggedIndex = null
                                            draggingOffset = 0f
                                            autoscrollJob?.cancel()
                                            settingsManager.saveColors(highlightColors)
                                        },
                                        onDragCancel = {
                                            draggedIndex = null
                                            draggingOffset = 0f
                                            autoscrollJob?.cancel()
                                        },
                                        onDrag = { change, dragAmount ->
                                            change.consume()
                                            draggingOffset += dragAmount.y / density
                                            
                                            // Live Reordering Logic
                                            val currentOffset = draggingOffset
                                            val threshold = 28 // Half height of row
                                            
                                            if (currentOffset > threshold && index < highlightColors.size - 1) {
                                                highlightColors.add(index + 1, highlightColors.removeAt(index))
                                                draggedIndex = index + 1
                                                draggingOffset -= 56
                                            } else if (currentOffset < -threshold && index > 0) {
                                                highlightColors.add(index - 1, highlightColors.removeAt(index))
                                                draggedIndex = index - 1
                                                draggingOffset += 56
                                            }
                                        }
                                    )
                                }
                                .padding(horizontal = 12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Default.Menu,
                                contentDescription = "Reorder ${item.first}",
                                tint = MaterialTheme.colors.onSurface.copy(alpha = 0.4f)
                            )
                            Spacer(modifier = Modifier.width(16.dp))
                            Box(
                                modifier = Modifier
                                    .size(24.dp)
                                    .background(item.second, RoundedCornerShape(4.dp))
                            )
                            Spacer(modifier = Modifier.width(16.dp))
                            Text(text = item.first, modifier = Modifier.weight(1f))
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

            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = "Use YOLO Model", style = MaterialTheme.typography.h6)
                    Text(
                        text = "Use pose-based neural network for robust corner detection. More stable in variable lighting.",
                        style = MaterialTheme.typography.caption,
                        color = MaterialTheme.colors.onSurface.copy(alpha = 0.6f)
                    )
                }
                Switch(
                    checked = useYOLO,
                    onCheckedChange = {
                        useYOLO = it
                        settingsManager.useYOLO = it
                    }
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(text = "Augmented Reality Mode", style = MaterialTheme.typography.h6)
                    Text(
                        text = "Use ARCore for world-anchored stability. Experimental.",
                        style = MaterialTheme.typography.caption,
                        color = MaterialTheme.colors.onSurface.copy(alpha = 0.6f)
                    )
                }
                Switch(
                    checked = arMode,
                    onCheckedChange = {
                        arMode = it
                        settingsManager.arMode = it
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
